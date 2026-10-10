"""Synthetic reasoning tasks: generators are correct, batches are laid out as documented, every condition trains."""
import argparse
import json
import os

import numpy as np
import pytest

from looped import model as model_mod
from synthetic import tasks
from synthetic.train_synth import CONDITIONS, train_and_eval


def test_chain_prompts_evaluate_to_their_answers():
    rng = np.random.default_rng(0)
    for d in (1, 3, 8, 24):
        toks, pos, xs = tasks.gen_chain(rng, d)
        assert toks[pos] == tasks.ARROW and len(toks) == 3 * d + 4 and len(xs) == d + 1
        assert tasks.parse_chain(toks) == xs[-1]
        assert [toks[p] for p in tasks.op_positions(d)] and all(toks[p] in tasks.OPS for p in tasks.op_positions(d))
        # intermediates follow the listed operations step by step
        x = xs[0]
        for j, p in enumerate(tasks.op_positions(d), start=1):
            x = tasks.apply_op(toks[p], toks[p + 1], x)
            assert x == xs[j]


def test_hops_prompts_follow_the_listed_function():
    rng = np.random.default_rng(1)
    for k in (1, 5, 16):
        toks, pos, xs = tasks.gen_hops(rng, k)
        assert toks[pos] == tasks.ARROW and toks[pos - 1] == k and toks[pos - 2] == xs[0] and toks[pos - 3] == tasks.QRY
        f = {}
        i = 1
        while toks[i] != tasks.QRY:
            a, gt, b, sep = toks[i:i + 4]
            assert gt == tasks.GT and sep == tasks.SEP
            f[a] = b
            i += 4
        assert len(f) == tasks.N_NODES
        x = xs[0]
        for j in range(1, k + 1):
            x = f[x]
            assert x == xs[j]


def test_batches_are_padded_after_the_answer():
    rng = np.random.default_rng(2)
    for task, d in (("chain", 6), ("hops", 4)):
        x, pos, ans, inter = tasks.make_batch(task, d, 7, rng)
        assert x.shape == (7, tasks.SEQ_LEN[task]) and inter.shape == (7, d + 1)
        for i in range(7):
            assert x[i, pos[i]] == tasks.ARROW and x[i, pos[i] + 1] == ans[i] == inter[i, -1]
            assert (x[i, pos[i] + 2:] == tasks.PAD).all()
        assert x.max() < tasks.VOCAB_SIZE


def _args(tmp_path, task, cond, **kw):
    a = dict(task=task, cond=cond, seed=1, width=32, steps=3, batch=4, lr=1e-3, warmup=1, weight_decay=0.1,
             grad_clip=1.0, d_train=3, d_test="1..4", r_eval="1,2", eval_n=8, quick_n=4, eval_every=2, log_every=1,
             tol=0.02, device="cpu", out=str(tmp_path / "results"), ckpt_dir=str(tmp_path / "runs"), p=97,
             curriculum_frac=0.0)
    a.update(kw)
    return argparse.Namespace(**a)


@pytest.mark.parametrize("cond", ["dense8", "dense20", "loop_r2", "loop_rand", "loop_step"])
def test_every_condition_trains_and_evaluates(tmp_path, cond, monkeypatch):
    calls = []
    orig = model_mod.LoopedLM.forward

    def rec(self, idx, targets=None, r=None, k_bwd=None):
        calls.append((self.training, r))
        return orig(self, idx, targets, r=r, k_bwd=k_bwd)
    monkeypatch.setattr(model_mod.LoopedLM, "forward", rec)
    rep = train_and_eval(_args(tmp_path, "chain", cond))
    assert set(rep["final"]) == {1, 2, 3, 4} and rep["executed_layers"] == {"dense8": 8, "dense20": 20, "loop_r2": 12}.get(cond, 36)
    path = tmp_path / "results" / "chain" / f"{cond}_lr0.001_s1.json"
    assert json.load(open(path))["cond"] == cond and os.path.exists(tmp_path / "runs" / "chain" / f"{cond}_lr0.001_s1.pt")
    train_r = [r for training, r in calls if training]
    if cond == "loop_step":
        assert all(r is not None and 1 <= r <= 3 for r in train_r)          # loops = step count of the batch
    elif cond == "loop_rand":
        assert all(r is not None and 1 <= r <= 8 for r in train_r)
    else:
        assert all(r is None for r in train_r)
    if cond.startswith("loop"):
        row = rep["final"][4]
        assert {"r=1", "r=2", "oracle", "adaptive", "adaptive_loops"} <= set(row) and 1 <= row["adaptive_loops"] <= 32
    else:
        assert set(rep["final"][1]) == {"acc"}


def test_hops_condition_runs(tmp_path):
    rep = train_and_eval(_args(tmp_path, "hops", "loop_r2", d_test="1..2"))
    assert set(rep["final"]) == {1, 2}


def test_intermediate_value_probes_have_the_documented_shape():
    import torch
    from synthetic.probe_synth import probe_checkpoint
    from synthetic.train_synth import build_model
    torch.manual_seed(0)
    for cond, unit, n_units in (("loop_r2", "loop", 3), ("dense8", "layer", 8)):
        model = build_model(cond, "chain", 32)
        rep = probe_checkpoint(model, d=3, unit=unit, r_max=3, n_train=40, n_test=16, device=torch.device("cpu"), amp=None, steps=5)
        assert rep["units"] == n_units and len(rep["answer_table"]) == n_units and len(rep["answer_table"][0]) == 4
        assert len(rep["local_table"]) == n_units and all(0 <= v <= 1 for row in rep["local_table"] for v in row)
        assert len(rep["first_unit_decodable_answer"]) == 4


def test_curriculum_grows_the_step_count(tmp_path, monkeypatch):
    seen = []
    import synthetic.train_synth as ts
    orig = ts.make_batch

    def rec(task, d, batch, rng):
        seen.append(d)
        return orig(task, d, batch, rng)
    monkeypatch.setattr(ts, "make_batch", rec)
    train_and_eval(_args(tmp_path, "chain", "loop_step", steps=8, d_train=4, curriculum_frac=0.5, eval_every=100))
    train_d = seen[:8]                                   # the evaluation batches come after the 8 training steps
    assert max(train_d[:2]) <= 2 and max(train_d[:4]) <= 4 and all(1 <= d <= 4 for d in train_d)
