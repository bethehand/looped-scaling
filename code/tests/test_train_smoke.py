"""End-to-end smoke test: trunk + two cooldown branches + results rows, on synthetic data, CPU."""
import json
import os

import numpy as np
import yaml

from looped.data import write_shard
from looped.train import Runner, load_run_config


def test_trunk_and_branches(tmp_path):
    rng = np.random.default_rng(0)
    T, V = 16, 64
    write_shard(str(tmp_path / "train" / "shard_000.bin"), rng.integers(0, V, 60_000))
    write_shard(str(tmp_path / "val" / "fwe.bin"), rng.integers(0, V, 5_000))
    write_shard(str(tmp_path / "val" / "second.bin"), rng.integers(0, V, 3_000))
    batch_tokens = 4 * T
    cfg = dict(
        name="smoke", out_dir=str(tmp_path / "run"),
        model=dict(vocab_size=V, d_model=32, head_dim=16, seq_len=T, n_layers=4, n_prelude=1, n_coda=1,
                   placement="middle", r=2, k_bwd=1),
        train=dict(seed=1, lr0=1e-3, batch_tokens=batch_tokens, micro_seqs=2, warmup_tokens=2 * batch_tokens,
                   budgets=[20 * batch_tokens, 40 * batch_tokens], cooldown_frac=0.2, eval_every_steps=5,
                   eval_windows=4, final_eval_points=2, final_eval_gap_steps=2, log_every_steps=5,
                   dtype="fp32", eval_batch_seqs=4),
        data=dict(train_shards=str(tmp_path / "train" / "*.bin"),
                  val_sets=dict(fwe=str(tmp_path / "val" / "fwe.bin"), second=str(tmp_path / "val" / "second.bin"),
                                code=str(tmp_path / "val" / "missing.bin")),   # optional set absent -> skipped
                  val_main="fwe", val_max_tokens=None),
    )
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    name, out_dir, m, t, d = load_run_config(str(cfg_path))
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
    # re-running is a no-op (all branches done)
    Runner(name, out_dir, m, t, d, device="cpu").run()
    rows2 = [json.loads(l) for l in open(os.path.join(out_dir, "results.jsonl"))]
    assert len(rows2) == 2
