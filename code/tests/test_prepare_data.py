"""Data pipeline on a tiny synthetic FineWeb-Edu directory: tokenizer, training stream, validation sets.
Checks that the last parquet file is held out (no train/val leakage) and that non-'text' columns work."""
import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import scripts.prepare_data as pdp

WORDS = ("the model loops over the same layers again and learning rate schedule data token "
         "education science history water energy cell number equation").split()


def _docs(rng, n, extra=""):
    return [" ".join(rng.choice(WORDS, size=int(rng.integers(20, 60)))) + extra for _ in range(n)]


def _write(path, col, texts):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.table({col: texts}), path)


def test_pipeline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    src = "data/raw/fineweb-edu/sample/10BT"
    _write(f"{src}/000_00000.parquet", "text", _docs(rng, 300))
    _write(f"{src}/001_00000.parquet", "text", _docs(rng, 300))
    _write(f"{src}/002_00000.parquet", "text", _docs(rng, 200, " zebraquux"))   # held-out file, sentinel word
    _write("data/raw/extra/SlimPajama-6B/data/validation-0.parquet", "text", _docs(rng, 100))
    _write("data/raw/extra/finemath/finemath-4plus/train-00000-0.parquet", "text", _docs(rng, 100))
    _write("data/raw/extra/github-code-clean/data/train-00000-of-00880.parquet", "code",
           ["def f(x):\n    return x + 1\n"] * 100)
    monkeypatch.setattr(pdp, "SRC", src)
    pdp.cmd_tokenizer(argparse.Namespace(chars="2e5", vocab=400, force=False))
    pdp.cmd_tokenize(argparse.Namespace(workers=2, force=False))
    pdp.cmd_valsets(argparse.Namespace(workers=2, force=False))
    # frozen artifacts are never overwritten by accident
    import pytest
    for fn, ns in ((pdp.cmd_tokenizer, argparse.Namespace(chars="2e5", vocab=400, force=False)),
                   (pdp.cmd_tokenize, argparse.Namespace(workers=2, force=False)),
                   (pdp.cmd_valsets, argparse.Namespace(workers=2, force=False))):
        with pytest.raises(SystemExit):
            fn(ns)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(pdp.TOK)
    train = np.concatenate([np.fromfile(f"data/fwe_train/{f}", dtype=np.uint16)
                            for f in sorted(os.listdir("data/fwe_train"))])
    assert len(train) > 1000 and train.max() < tok.get_vocab_size()
    for name in ("fwe_val", "second_val", "finemath_val", "code_val"):
        arr = np.fromfile(f"data/val/{name}.bin", dtype=np.uint16)
        assert len(arr) > 100, name
    val_text = tok.decode(np.fromfile("data/val/fwe_val.bin", dtype=np.uint16).tolist())
    train_text = tok.decode(train[:200_000].tolist())
    assert "zebraquux" in val_text and "zebraquux" not in train_text
    assert "return" in tok.decode(np.fromfile("data/val/code_val.bin", dtype=np.uint16).tolist())
    # numbers are split into groups of at most three digits; text round-trips exactly
    sample = "In 2024 the model saw 1234567 tokens, x = 3.14159!"
    enc = tok.encode(sample)
    assert tok.decode(enc.ids) == sample
    digit_tokens = [t for t in enc.tokens if t.isdigit()]
    assert digit_tokens and all(len(t) <= 3 for t in digit_tokens), enc.tokens
    assert not any(d.startswith("_tmp") for d in os.listdir("data/val"))
