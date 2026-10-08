"""Per-loop probes on a tiny looped model: the lens at the training r is the model itself; shapes and bookkeeping."""
import argparse
import json
import os

import numpy as np
import torch

from fit import loop_probe as lp
from looped.data import TokenStream, ValSet, write_shard
from looped.evaluate import evaluate
from looped.model import LoopedLM, ModelConfig

T, V = 16, 64


def _setup(tmp_path):
    rng = np.random.default_rng(0)
    write_shard(str(tmp_path / "train" / "shard_000.bin"), rng.integers(0, V, 20_000))
    write_shard(str(tmp_path / "val" / "fwe.bin"), rng.integers(0, V, 2_000))
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=V, d_model=32, head_dim=16, seq_len=T, n_layers=4, n_prelude=1, n_coda=1,
                      placement="middle", r=2, k_bwd=0)
    model = LoopedLM(cfg)
    stream = TokenStream.from_glob(str(tmp_path / "train" / "*.bin"))
    val = ValSet(str(tmp_path / "val" / "fwe.bin"), T, 2 * 3 * T)
    return model, stream, val


def test_lens_at_training_r_is_the_model_and_bookkeeping_is_consistent(tmp_path):
    model, stream, val = _setup(tmp_path)
    device = torch.device("cpu")
    ref = evaluate(model, val, 2, device)
    a = argparse.Namespace(r_max=4, probe_tokens=4 * 2 * T, batch=2, lr_adapter=1e-3, lr_linear=3e-4)
    rep = lp.analyze(model, 2, a, stream, 0, val, device, None)
    assert abs(rep["loss"]["lens"][1] - ref) < 1e-4            # loop 2 = the model's own forward at r = 2
    assert len(rep["loss"]["adapter"]) == 4 and len(rep["cos_next"]) == 3 and len(rep["agree_next"]) == 3
    assert rep["match_final"][1] == 1.0 and abs(rep["cos_to_final"][1] - 1.0) < 1e-5
    n = rep["n_tokens"]
    assert n == val.n_windows * T                                 # 2 x 3 x T tokens -> 5 whole windows of T
    assert sum(rep["settle_hist"]["all"]) == n
    assert sum(rep["settle_hist"]["correct"]) + sum(rep["settle_hist"]["wrong"]) == n
    assert all(1 <= s <= 2 for s in rep["settle_mean_by_decile"])
    assert rep["probe_tokens"] == 4 * 2 * T and len(rep["probe_log"]) == 1
    # a settle loop of 1 means the loop-1 prediction already equals the loop-2 one: it must match agree_next[0]
    assert abs(rep["settle_hist"]["all"][0] / n - rep["agree_next"][0]) < 1e-9


def test_probe_training_only_touches_the_probes(tmp_path):
    model, stream, val = _setup(tmp_path)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    a = argparse.Namespace(r_max=3, probe_tokens=2 * 2 * T, batch=2, lr_adapter=1e-2, lr_linear=1e-2)
    lp.analyze(model, 2, a, stream, 0, val, torch.device("cpu"), None)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_summary_rows(tmp_path):
    rep = dict(run="x", ckpt="branch40", r_train=2, r_max=3, loss=dict(lens=[1, 2, 3], adapter=[1, 2, 3], linear=[1, 2, 3]),
               cos_to_final=[0, 1, 0], match_final=[0, 1, 0], cos_next=[0.5, 0.5], rel_change=[1, 1], agree_next=[0.1, 0.2])
    json.dump(rep, open(tmp_path / "x__branch40.json", "w"))
    lp.write_summary(str(tmp_path))
    rows = open(tmp_path / "summary.csv").read().strip().splitlines()
    assert len(rows) == 4 and rows[-1].endswith(",,,")                 # the last loop has no "next" columns
