"""Data preparation (pre-registration §3).

Steps (run in order):
  python scripts/prepare_data.py download   --n-files 12          # FineWeb-Edu sample-100BT parquet shards
  python scripts/prepare_data.py tokenizer  --chars 2e9           # train a 16k byte-level BPE
  python scripts/prepare_data.py tokenize   --workers 12          # fixed-order uint16 shards (100M tokens each)
  python scripts/prepare_data.py valsets                          # FWE held-out + second distribution + exploratory sets

Any FineWeb-Edu parquet directory can be used via --src (e.g. an existing sample-10BT download); the LAST file
in sorted order is always reserved for validation.

Layout:
  data/raw/fineweb-edu/sample/100BT/*.parquet   downloaded shards (sorted; the LAST one is reserved for validation)
  data/tokenizer/bpe16k.json
  data/fwe_train/shard_XXXX.bin                  training stream, fixed order
  data/val/{fwe_val,second_val,finemath_val,code_val}.bin
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RAW = "data/raw/fineweb-edu"
SRC = None   # parquet directory override (--src); default RAW/sample/100BT
TOK = "data/tokenizer/bpe16k.json"
EOT = "<|endoftext|>"
SHARD_TOKENS = 100_000_000


def fwe_files() -> list[str]:
    d = SRC or f"{RAW}/sample/100BT"
    files = sorted(glob.glob(f"{d}/*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files in {d}; pass --src <dir> or run download")
    return files


def cmd_download(a):
    from huggingface_hub import snapshot_download, list_repo_files
    if not a.skip_fwe:
        files = sorted(f for f in list_repo_files("HuggingFaceFW/fineweb-edu", repo_type="dataset")
                       if f.startswith("sample/100BT/") and f.endswith(".parquet"))
        pick = files[: a.n_files]
        print(f"{len(files)} shards available; downloading {len(pick)} (last one reserved for validation)")
        snapshot_download("HuggingFaceFW/fineweb-edu", repo_type="dataset", local_dir=RAW, allow_patterns=pick)
    os.makedirs("data/raw/extra", exist_ok=True)
    # second distribution + exploratory sets (small; all ungated parquet)
    for repo, pats in [("DKYoon/SlimPajama-6B", ["data/validation-*", "data/test-*"]),      # ~2 x 10M tokens
                       ("HuggingFaceTB/finemath", ["finemath-4plus/train-00000-*"]),
                       ("codeparrot/github-code-clean", ["data/train-00000-of-00880.parquet"])]:
        try:
            snapshot_download(repo, repo_type="dataset", local_dir=f"data/raw/extra/{repo.split('/')[-1]}",
                              allow_patterns=pats)
            print("downloaded", repo, pats)
        except Exception as e:  # noqa: BLE001
            print("WARNING could not download", repo, pats, "->", str(e)[:200])


def iter_texts(path: str, column: str = "text", batch_rows: int = 2048):
    pf = pq.ParquetFile(path)
    for rb in pf.iter_batches(batch_size=batch_rows, columns=[column]):
        for t in rb.column(0).to_pylist():
            if t:
                yield t


# GPT-4 / Llama-3 style pre-tokenization: letters, numbers in groups of at most 3 digits, punctuation, whitespace
SPLIT_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
                 r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def _refuse_overwrite(paths: list[str], force: bool, what: str) -> None:
    """The tokenizer, training stream and validation sets are frozen artifacts (sha256 in 03_偏离记录.md)."""
    existing = [p for p in paths if os.path.exists(p)]
    if existing and not force:
        raise SystemExit(f"refusing to overwrite the frozen {what} ({existing[0]} ...). "
                         f"Delete it deliberately or pass --force; every run must use the same data.")


def cmd_tokenizer(a):
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers, decoders, trainers
    _refuse_overwrite([TOK], a.force, "tokenizer")
    os.makedirs(os.path.dirname(TOK), exist_ok=True)
    files = fwe_files()[:-1]
    budget = int(float(a.chars))

    def gen():
        n = 0
        for f in files:
            for t in iter_texts(f):
                yield t
                n += len(t)
                if n >= budget:
                    return
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(SPLIT_PATTERN), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=a.vocab, special_tokens=[EOT], min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    tok.train_from_iterator(gen(), trainer=trainer)
    tok.save(TOK)
    print("saved", TOK, "vocab", tok.get_vocab_size())


_tok = None


def _encode_chunk(args):
    global _tok
    if _tok is None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"   # one process per core already; avoid thread oversubscription
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(TOK)
    texts, = args
    eot = _tok.token_to_id(EOT)
    out = []
    for enc in _tok.encode_batch(texts):
        out.extend(enc.ids); out.append(eot)
    return np.asarray(out, dtype=np.uint16)


def tokenize_files(files: list[str], out_dir: str, workers: int, max_tokens: int | None, prefix: str,
                   column: str = "text"):
    os.makedirs(out_dir, exist_ok=True)
    buf, shard_id, written = [], 0, 0
    total = 0

    def flush(force=False):
        nonlocal buf, shard_id, written
        cat = np.concatenate(buf) if buf else np.empty(0, dtype=np.uint16)
        while len(cat) >= SHARD_TOKENS or (force and len(cat) > 0):
            n = min(SHARD_TOKENS, len(cat))
            cat[:n].tofile(os.path.join(out_dir, f"{prefix}_{shard_id:04d}.bin"))
            shard_id += 1; written += n
            cat = cat[n:]
        buf = [cat] if len(cat) else []

    with Pool(workers) as pool:
        for f in files:
            chunks = []
            texts = []
            for t in iter_texts(f, column=column):
                texts.append(t)
                if len(texts) == 512:
                    chunks.append((texts,)); texts = []
            if texts:
                chunks.append((texts,))
            for arr in pool.imap(_encode_chunk, chunks, chunksize=1):   # ordered
                buf.append(arr); total += len(arr)
                if sum(len(b) for b in buf) >= SHARD_TOKENS:
                    flush()
                if max_tokens and total >= max_tokens:
                    break
            print(f"{f}: cumulative {total/1e9:.2f}B tokens", flush=True)
            if max_tokens and total >= max_tokens:
                break
    flush(force=True)
    print(f"wrote {written/1e9:.2f}B tokens to {out_dir}")


def cmd_tokenize(a):
    _refuse_overwrite(sorted(glob.glob("data/fwe_train/*.bin")), a.force, "training stream")
    files = fwe_files()[:-1]  # last file reserved
    tokenize_files(files, "data/fwe_train", a.workers, None, "shard")


def _write_val(name: str, files: list[str], n_tokens: int, workers: int, column: str = "text"):
    tmp = f"data/val/_tmp_{name}"
    os.makedirs(tmp, exist_ok=True)
    tokenize_files(files, tmp, workers, n_tokens, name, column=column)
    parts = sorted(glob.glob(f"{tmp}/*.bin"))
    arr = np.concatenate([np.fromfile(p, dtype=np.uint16) for p in parts])[:n_tokens]
    arr.tofile(f"data/val/{name}.bin")
    for p in parts:
        os.remove(p)
    os.rmdir(tmp)
    print(f"data/val/{name}.bin: {len(arr)/1e6:.1f}M tokens")


def cmd_valsets(a):
    _refuse_overwrite(sorted(glob.glob("data/val/*_val.bin")), a.force, "validation sets")
    os.makedirs("data/val", exist_ok=True)
    _write_val("fwe_val", fwe_files()[-1:], 50_000_000, a.workers)
    extra = "data/raw/extra"
    second = sorted(glob.glob(f"{extra}/SlimPajama-6B/**/*.parquet", recursive=True))
    if second:
        _write_val("second_val", second, 20_000_000, a.workers)
    fm = sorted(glob.glob(f"{extra}/finemath/**/*.parquet", recursive=True))
    if fm:
        _write_val("finemath_val", fm, 20_000_000, a.workers)
    code = sorted(glob.glob(f"{extra}/github-code-clean/**/*.parquet", recursive=True))
    if code:
        _write_val("code_val", code, 20_000_000, a.workers, column="code")
    for name in ("fwe_val", "second_val", "finemath_val", "code_val"):
        path = f"data/val/{name}.bin"
        print(f"{name:14s}", f"{os.path.getsize(path) // 2 / 1e6:8.1f}M tokens" if os.path.exists(path) else "   MISSING")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ap.add_argument("--src", default=None, help="directory of FineWeb-Edu parquet files (default data/raw/fineweb-edu/sample/100BT)")
    d = sub.add_parser("download"); d.add_argument("--n-files", type=int, default=12)
    d.add_argument("--skip-fwe", action="store_true", help="only fetch the small extra validation datasets")
    t = sub.add_parser("tokenizer"); t.add_argument("--chars", default="2e9"); t.add_argument("--vocab", type=int, default=16384)
    k = sub.add_parser("tokenize"); k.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    v = sub.add_parser("valsets"); v.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    for sp in (t, k, v):
        sp.add_argument("--force", action="store_true", help="overwrite existing frozen outputs")
    a = ap.parse_args()
    global SRC
    SRC = a.src
    {"download": cmd_download, "tokenizer": cmd_tokenizer, "tokenize": cmd_tokenize, "valsets": cmd_valsets}[a.cmd](a)


if __name__ == "__main__":
    main()
