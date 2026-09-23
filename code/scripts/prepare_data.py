"""Data preparation (pre-registration §3).

Steps (run in order):
  python scripts/prepare_data.py download   --n-files 12          # FineWeb-Edu sample-100BT parquet shards
  python scripts/prepare_data.py tokenizer  --chars 2e9           # train a 16k byte-level BPE
  python scripts/prepare_data.py tokenize   --workers 12          # fixed-order uint16 shards (100M tokens each)
  python scripts/prepare_data.py valsets                          # FWE held-out + second distribution + exploratory sets

Layout:
  data/raw/fineweb-edu/sample/100BT/*.parquet   downloaded shards (sorted; the LAST one is reserved for validation)
  data/tokenizer/bpe16k.json
  data/fwe_train/shard_XXXX.bin                  training stream, fixed order
  data/val/{fwe_val,second_val,finemath_val,stack_val}.bin
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
TOK = "data/tokenizer/bpe16k.json"
EOT = "<|endoftext|>"
SHARD_TOKENS = 100_000_000


def fwe_files() -> list[str]:
    return sorted(glob.glob(f"{RAW}/sample/100BT/*.parquet"))


def cmd_download(a):
    from huggingface_hub import snapshot_download, list_repo_files
    files = sorted(f for f in list_repo_files("HuggingFaceFW/fineweb-edu", repo_type="dataset")
                   if f.startswith("sample/100BT/") and f.endswith(".parquet"))
    pick = files[: a.n_files]
    print(f"{len(files)} shards available; downloading {len(pick)} (last one reserved for validation)")
    snapshot_download("HuggingFaceFW/fineweb-edu", repo_type="dataset", local_dir=RAW, allow_patterns=pick)
    os.makedirs("data/raw/extra", exist_ok=True)
    # second distribution + exploratory sets (small, first file of each)
    for repo, pat in [("DKYoon/SlimPajama-6B", "data/validation-*"),
                      ("HuggingFaceTB/finemath", "finemath-4plus/train-00000-*"),
                      ("bigcode/the-stack-smol", "data/python/*")]:
        try:
            snapshot_download(repo, repo_type="dataset", local_dir=f"data/raw/extra/{repo.split('/')[-1]}",
                              allow_patterns=[pat])
            print("downloaded", repo, pat)
        except Exception as e:  # noqa: BLE001
            print("WARNING could not download", repo, pat, "->", str(e)[:200])


def iter_texts(path: str, column: str = "text", batch_rows: int = 2048):
    pf = pq.ParquetFile(path)
    for rb in pf.iter_batches(batch_size=batch_rows, columns=[column]):
        for t in rb.column(0).to_pylist():
            if t:
                yield t


def cmd_tokenizer(a):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
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
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=a.vocab, special_tokens=[EOT], min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    tok.train_from_iterator(gen(), trainer=trainer)
    tok.save(TOK)
    print("saved", TOK, "vocab", tok.get_vocab_size())


_tok = None


def _encode_chunk(args):
    global _tok
    from tokenizers import Tokenizer
    if _tok is None:
        _tok = Tokenizer.from_file(TOK)
    texts, = args
    eot = _tok.token_to_id(EOT)
    out = []
    for enc in _tok.encode_batch(texts):
        out.extend(enc.ids); out.append(eot)
    return np.asarray(out, dtype=np.uint16)


def tokenize_files(files: list[str], out_dir: str, workers: int, max_tokens: int | None, prefix: str):
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
            for t in iter_texts(f):
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
    files = fwe_files()[:-1]  # last file reserved
    tokenize_files(files, "data/fwe_train", a.workers, None, "shard")


def _write_val(name: str, files: list[str], n_tokens: int, workers: int, column: str = "text"):
    tmp = f"data/val/_tmp_{name}"
    os.makedirs(tmp, exist_ok=True)
    tokenize_files(files, tmp, workers, n_tokens, name)
    parts = sorted(glob.glob(f"{tmp}/*.bin"))
    arr = np.concatenate([np.fromfile(p, dtype=np.uint16) for p in parts])[:n_tokens]
    arr.tofile(f"data/val/{name}.bin")
    for p in parts:
        os.remove(p)
    os.rmdir(tmp)
    print(f"data/val/{name}.bin: {len(arr)/1e6:.1f}M tokens")


def cmd_valsets(a):
    os.makedirs("data/val", exist_ok=True)
    _write_val("fwe_val", fwe_files()[-1:], 50_000_000, a.workers)
    extra = "data/raw/extra"
    second = sorted(glob.glob(f"{extra}/SlimPajama-6B/**/*.parquet", recursive=True))
    if second:
        _write_val("second_val", second, 20_000_000, a.workers)
    fm = sorted(glob.glob(f"{extra}/finemath/**/*.parquet", recursive=True))
    if fm:
        _write_val("finemath_val", fm, 20_000_000, a.workers)
    st = sorted(glob.glob(f"{extra}/the-stack-smol/**/*.parquet", recursive=True))
    if st:
        _write_val("stack_val", st, 20_000_000, a.workers, column="content")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download"); d.add_argument("--n-files", type=int, default=12)
    t = sub.add_parser("tokenizer"); t.add_argument("--chars", default="2e9"); t.add_argument("--vocab", type=int, default=16384)
    k = sub.add_parser("tokenize"); k.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    v = sub.add_parser("valsets"); v.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    {"download": cmd_download, "tokenizer": cmd_tokenizer, "tokenize": cmd_tokenize, "valsets": cmd_valsets}[a.cmd](a)


if __name__ == "__main__":
    main()
