"""End-to-end smoke test: trunk + two cooldown branches + results rows, on synthetic data, CPU."""
import csv
import json
import math
import os

import numpy as np
import torch
import yaml

from looped import model as model_mod
from looped.data import write_shard
from looped.train import Runner, load_run_config


T, V = 16, 64
batch_tokens = 4 * T


def _config(tmp_path, out="run", model=None, **train_overrides):
    rng = np.random.default_rng(0)
    write_shard(str(tmp_path / "train" / "shard_000.bin"), rng.integers(0, V, 60_000))
    write_shard(str(tmp_path / "val" / "fwe.bin"), rng.integers(0, V, 5_000))
    write_shard(str(tmp_path / "val" / "second.bin"), rng.integers(0, V, 3_000))
    cfg = dict(
        name="smoke", out_dir=str(tmp_path / out),
        model={**dict(vocab_size=V, d_model=32, head_dim=16, seq_len=T, n_layers=4, n_prelude=1, n_coda=1,
                      placement="middle", r=2, k_bwd=1), **(model or {})},
        train={**dict(seed=1, lr0=1e-3, batch_tokens=batch_tokens, micro_seqs=2, warmup_tokens=2 * batch_tokens,
                      budgets=[20 * batch_tokens, 40 * batch_tokens], cooldown_frac=0.2, eval_every_steps=5,
                      eval_windows=4, final_eval_points=2, final_eval_gap_steps=2, log_every_steps=5,
                      dtype="fp32", eval_batch_seqs=4), **train_overrides},
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


def test_init_from_continues_an_earlier_trunk(tmp_path):
    name, out_a, m, t, d = _config(tmp_path)                            # budgets 20 and 40 batches: trunk ends at 32
    Runner(name, out_a, m, t, d, device="cpu").run()
    src = os.path.join(out_a, f"branch_{32 * batch_tokens}.pt")
    cfg_b = _config(tmp_path, out="run_b", budgets=[80 * batch_tokens], init_from=src)
    name, out_b, m, t, d = cfg_b
    Runner(*cfg_b, device="cpu").run()
    log = list(csv.DictReader(open(os.path.join(out_b, "train_log.csv"))))
    assert min(int(r["step"]) for r in log) > 32                         # the inherited trunk is not trained again
    assert os.path.exists(os.path.join(out_b, f"branch_{64 * batch_tokens}.pt"))
    rows = [json.loads(l) for l in open(os.path.join(out_b, "results.jsonl"))]
    assert [r["budget_tokens"] for r in rows] == [80 * batch_tokens]
    assert rows[0]["tokens_seen"] == 80 * batch_tokens and rows[0]["step"] == 80 and rows[0]["init_from"] == src
    assert rows[0]["mean_r_train"] == 2 and rows[0]["r_sample"] == ""  # fixed r: every step, inherited or not, ran r loops
    # equivalent to one run with budgets 20/40/80: the 64-batch trunk checkpoints are identical
    name, out_c, m, t, d = _config(tmp_path, out="run_c", budgets=[20 * batch_tokens, 40 * batch_tokens, 80 * batch_tokens])
    Runner(name, out_c, m, t, d, device="cpu").run()
    b = torch.load(os.path.join(out_b, f"branch_{64 * batch_tokens}.pt"), weights_only=False)
    c = torch.load(os.path.join(out_c, f"branch_{64 * batch_tokens}.pt"), weights_only=False)
    assert b["tokens_seen"] == c["tokens_seen"] and b["sampler"] == c["sampler"]
    assert all(torch.equal(b["model"][k], c["model"][k]) for k in c["model"])
    # re-running B is a no-op
    Runner(*cfg_b, device="cpu").run()
    assert len(open(os.path.join(out_b, "results.jsonl")).readlines()) == 1


def test_r_sample_draws_one_loop_count_per_step(tmp_path, monkeypatch):
    calls = []
    orig = model_mod.LoopedLM.forward

    def rec(self, idx, targets=None, r=None, k_bwd=None):
        calls.append((self.training, r, k_bwd))
        return orig(self, idx, targets, r=r, k_bwd=k_bwd)
    monkeypatch.setattr(model_mod.LoopedLM, "forward", rec)
    name, out_dir, m, t, d = _config(tmp_path, model=dict(r=4, k_bwd=2), max_steps=10, r_sample="uniform:1:4")
    runner = Runner(name, out_dir, m, t, d, device="cpu")
    runner.run()
    train = [(r, k) for training, r, k in calls if training]
    assert len(train) == 10 * runner.n_micro
    assert all(1 <= r <= 4 and k == min(2, r) for r, k in train)       # truncation never asks for more loops than run
    per_step = [train[i * runner.n_micro:(i + 1) * runner.n_micro] for i in range(10)]
    assert all(len(set(s)) == 1 for s in per_step)                      # one draw per optimizer step
    assert len({s[0][0] for s in per_step}) > 1                         # and it varies
    assert all(r is None and k is None for training, r, k in calls if not training)   # evaluation at the nominal r
    assert runner.loops_executed == sum(s[0][0] for s in per_step)
    # the draw sequence survives a checkpoint round trip
    runner.save("trunk_latest")
    other = Runner(name, out_dir, m, t, d, device="cpu")
    other.load("trunk_latest")
    assert other.loops_executed == runner.loops_executed
    assert [other.r_rng.randint(1, 4) for _ in range(5)] == [runner.r_rng.randint(1, 4) for _ in range(5)]
