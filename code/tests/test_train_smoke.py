"""End-to-end smoke test: trunk + two cooldown branches + results rows, on synthetic data, CPU."""
import csv
import json
import math
import os

import numpy as np
import yaml

from looped.data import write_shard
from looped.train import Runner, load_run_config


T, V = 16, 64
batch_tokens = 4 * T


def _config(tmp_path, **train_overrides):
    rng = np.random.default_rng(0)
    write_shard(str(tmp_path / "train" / "shard_000.bin"), rng.integers(0, V, 60_000))
    write_shard(str(tmp_path / "val" / "fwe.bin"), rng.integers(0, V, 5_000))
    write_shard(str(tmp_path / "val" / "second.bin"), rng.integers(0, V, 3_000))
    cfg = dict(
        name="smoke", out_dir=str(tmp_path / "run"),
        model=dict(vocab_size=V, d_model=32, head_dim=16, seq_len=T, n_layers=4, n_prelude=1, n_coda=1,
                   placement="middle", r=2, k_bwd=1),
        train=dict(seed=1, lr0=1e-3, batch_tokens=batch_tokens, micro_seqs=2, warmup_tokens=2 * batch_tokens,
                   budgets=[20 * batch_tokens, 40 * batch_tokens], cooldown_frac=0.2, eval_every_steps=5,
                   eval_windows=4, final_eval_points=2, final_eval_gap_steps=2, log_every_steps=5,
                   dtype="fp32", eval_batch_seqs=4, **train_overrides),
        data=dict(train_shards=str(tmp_path / "train" / "*.bin"),
                  val_sets=dict(fwe=str(tmp_path / "val" / "fwe.bin"), second=str(tmp_path / "val" / "second.bin"),
                                code=str(tmp_path / "val" / "missing.bin")),   # optional set absent -> skipped
                  val_main="fwe", val_max_tokens=None),
    )
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return load_run_config(str(cfg_path))


def test_max_steps_stops_after_the_trunk_prefix(tmp_path):
    name, out_dir, m, t, d = _config(tmp_path, max_steps=10)
    Runner(name, out_dir, m, t, d, device="cpu").run()
    log = list(csv.DictReader(open(os.path.join(out_dir, "train_log.csv"))))
    assert [int(r["step"]) for r in log] == [5, 10] and {r["phase"] for r in log} == {"trunk"}
    assert log[1]["quick_val"] != "nan"                   # the step-10 line carries a quick validation
    assert not os.path.exists(os.path.join(out_dir, "results.jsonl"))
    assert not os.path.exists(os.path.join(out_dir, "TRUNK_DONE"))


def test_trunk_and_branches(tmp_path):
    name, out_dir, m, t, d = _config(tmp_path)
    Runner(name, out_dir, m, t, d, device="cpu").run()
    rows = [json.loads(l) for l in open(os.path.join(out_dir, "results.jsonl"))]
    assert [r["budget_tokens"] for r in rows] == [20 * batch_tokens, 40 * batch_tokens]
    for r in rows:
        assert r["tokens_seen"] == r["budget_tokens"]
        assert set(r["val"]) == {"fwe", "second"}
        assert r["n_avg_points"] == 2                       # one tail point + the end point, no double counting
        assert set(r["val_avg"]) == {"fwe", "second"}
        assert r["git_commit"] and "torch_version" in r
    assert os.path.exists(os.path.join(out_dir, "TRUNK_DONE"))
    assert os.path.exists(os.path.join(out_dir, f"branch_{16 * batch_tokens}.pt"))
    # branches start at steps 16 and 32, so their first log lines cover 4 and 3 steps, not log_every_steps;
    # on uniform random tokens every logged loss must stay near ln V (was 0.8x and 0.6x before the fix)
    log = list(csv.DictReader(open(os.path.join(out_dir, "train_log.csv"))))
    assert {r["phase"] for r in log} == {"trunk", f"cool_{20 * batch_tokens}", f"cool_{40 * batch_tokens}"}
    for r in log:
        assert abs(float(r["loss"]) / math.log(V) - 1) < 0.1, r
    # re-running is a no-op (all branches done)
    Runner(name, out_dir, m, t, d, device="cpu").run()
    rows2 = [json.loads(l) for l in open(os.path.join(out_dir, "results.jsonl"))]
    assert len(rows2) == 2
