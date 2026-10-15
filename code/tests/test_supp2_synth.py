"""S5 (chain of thought, pause tokens) and S9 (loops = steps + random extra loops) in the synthetic trainer and the
suite runner (docs/supplementary-experiments-2.md); the existing conditions are unchanged."""
import json
import os

import numpy as np
import pytest
import torch

from looped import model as model_mod
from synthetic import suite, tasks
from synthetic import train_synth as ts
from synthetic.tasks import ARROW, PAUSE, make_batch
from tests.test_synthetic import _args


def _record_r(monkeypatch):
    """Every forward call as (training, r, d), d read off the prompt (3 tokens per step before ARROW)."""
    calls = []
    orig = model_mod.LoopedLM.forward

    def rec(self, idx, targets=None, r=None, k_bwd=None):
        toks = idx[0].tolist()
        calls.append((self.training, r, (toks.index(ARROW) - 3) // 3))
        return orig(self, idx, targets, r=r, k_bwd=k_bwd)
    monkeypatch.setattr(model_mod.LoopedLM, "forward", rec)
    return calls


def test_step_extra_loops_between_d_and_d_plus_4(tmp_path, monkeypatch):
    calls = _record_r(monkeypatch)
    rep = ts.train_and_eval(_args(tmp_path, "chain", "loop_step_extra", steps=12))
    train = [(r, d) for training, r, d in calls if training]
    assert len(train) == 12 and all(d <= r <= d + ts.EXTRA_LOOPS for r, d in train)
    assert any(r > d for r, d in train)                                 # extra loops do occur
    assert {"oracle", "adaptive", "r=1", "r=2"} <= set(rep["final"][4])
    assert os.path.exists(tmp_path / "results" / "chain" / "loop_step_extra_lr0.001_s1.json")


def test_existing_random_loop_sequence_is_unchanged(tmp_path, monkeypatch):
    calls = _record_r(monkeypatch)
    ts.train_and_eval(_args(tmp_path, "chain", "loop_rand", steps=8))
    train_r = [r for training, r, _ in calls if training]
    expected = [int(v) for v in np.random.default_rng(1 + 2000).integers(1, 9, size=8)]
    assert train_r == expected                                           # one draw per step, as before 2026-10-15


def test_cot_and_pause_batches_have_the_documented_layout():
    tasks.set_family("lookup", 32)
    rng = np.random.default_rng(0)
    x, pos, ans, inter = make_batch("chain", 3, 4, rng, "cot")
    assert x.shape[1] == tasks.SEQ_LEN_LONG
    for i in range(4):
        assert x[i, pos[i]] == ARROW and x[i, pos[i] + 1: pos[i] + 4].tolist() == inter[i, 1:].tolist() and ans[i] == inter[i, -1]
        assert x[i, pos[i] + 4] == tasks.PAD
    x, pos, ans, inter = make_batch("chain", 3, 4, rng, "pause")
    for i in range(4):
        assert x[i, pos[i]] == ARROW and x[i, pos[i] - 3: pos[i]].tolist() == [PAUSE] * 3 and x[i, pos[i] + 1] == ans[i]
        assert x[i, pos[i] - 4] != PAUSE
    with pytest.raises(ValueError):
        make_batch("hops", 2, 1, rng, "cot")
    xs, _, _, _ = make_batch("chain", 3, 4, np.random.default_rng(0))
    assert xs.shape[1] == tasks.SEQ_LEN["chain"]                       # the single-answer format is unchanged
    tasks.set_family("arith", 32)


@pytest.mark.parametrize("fmt", ["cot", "pause"])
def test_dense_trains_and_evaluates_with_the_new_formats(tmp_path, fmt):
    rep = ts.train_and_eval(_args(tmp_path, "chain", "dense8", steps=3, format=fmt))
    assert rep["format"] == fmt and set(rep["final"]) == {1, 2, 3, 4}
    keys = {"acc", "chain_acc"} if fmt == "cot" else {"acc"}
    assert set(rep["final"][2]) == keys and all(0 <= v <= 1 for v in rep["final"][2].values())
    path = tmp_path / "results" / "chain" / f"dense8_{fmt}_lr0.001_s1.json"
    assert json.load(open(path))["format"] == fmt and os.path.exists(tmp_path / "runs" / "chain" / f"dense8_{fmt}_lr0.001_s1.pt")


def test_cot_generation_scores_the_last_generated_token(monkeypatch):
    """With an oracle that writes the true next value, every chain is right; with one that is wrong at the last step
    only, the answer is wrong but the earlier tokens are not."""
    tasks.set_family("lookup", 32)
    device = torch.device("cpu")

    def oracle(wrong_last):
        def f(model, x, pos, r, device, amp):
            B, V = x.size(0), tasks.VOCAB_SIZE
            out = torch.full((B, V), -1e9)
            for b in range(B):
                toks = x[b].tolist()
                arrow = toks.index(ARROW)
                d = (arrow - 3) // 3
                j = int(pos[b]) - arrow                     # predicting x_{j+1}
                v = toks[1]
                for step in range(j + 1):
                    v = tasks.apply_op(tasks.LUT, toks[3 * step + 4], v)
                if wrong_last and j == d - 1:
                    v = (v + 1) % tasks.P
                out[b, v] = 0.0
            return out
        return f
    model = ts.build_model("dense8", "chain", 32, "cot")
    monkeypatch.setattr(ts, "answer_logits", oracle(False))
    res = ts.evaluate_cot(model, "chain", [1, 3], None, 6, device, None, batch=3)
    assert res[1] == {"acc": 1.0, "chain_acc": 1.0} and res[3] == {"acc": 1.0, "chain_acc": 1.0}
    monkeypatch.setattr(ts, "answer_logits", oracle(True))
    res = ts.evaluate_cot(model, "chain", [3], None, 6, device, None, batch=3)
    assert res[3] == {"acc": 0.0, "chain_acc": 0.0}
    tasks.set_family("arith", 32)


def test_supp2_suite_jobs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    jobs = suite.supp2_sweep_jobs("py")
    assert len(jobs) == 12 and jobs[0]["cost"] >= jobs[-1]["cost"]
    names = {j["name"] for j in jobs}
    assert "chain/dense8_cot_lr0.0005_s1_lut32" in names and "chain/loop_step_extra_lr0.001_s1_lut32" in names
    cot = next(j for j in jobs if "dense36_cot" in j["name"])
    assert cot["cmd"][cot["cmd"].index("--format") + 1] == "cot" and "--format" not in next(j for j in jobs if "loop_step_extra" in j["name"])["cmd"]
    # a finished sweep: fake results, choose, formal and probe jobs
    for j in jobs:
        os.makedirs(os.path.dirname(j["done"]), exist_ok=True)
        cond = j["cmd"][j["cmd"].index("--cond") + 1]
        key = suite.own_key(cond)
        lr = float(j["cmd"][j["cmd"].index("--lr") + 1])
        acc = 0.9 if lr == 5e-4 else 0.5
        json.dump(dict(final={str(d): {key: acc} for d in range(1, 9)}), open(j["done"], "w"))
    assert len(suite.supp2_sweep_jobs("py")) == 12                       # best at 5e-4: no edge point
    choice = suite.supp2_choose("py")
    assert json.load(open(suite.SUPP2_CHOICE))["choice"] == choice
    assert choice["dense8_cot"]["lr"] == 5e-4 and choice["loop_step_extra"]["metric"] == "oracle"
    formal = suite.supp2_formal_jobs("py")
    assert len(formal) == 4 * 5 and all(j["stage"] == "supp2_formal" for j in formal)
    probes = suite.supp2_probe_jobs("py")
    assert len(probes) == 5 and all("loop_step_extra" in j["name"] for j in probes)


def test_suite_run_takes_a_gpu_only_below_the_memory_threshold(tmp_path, monkeypatch):
    """run(): a card with 12 GB in use (a text run) is skipped at the default threshold and used at --gpu-free-mb 16000."""
    import sys

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(suite, "gpu_memory", lambda: {0: 12000})
    done = tmp_path / "job.json"
    jobs = [dict(stage="t", name="j", done=str(done), cmd=[sys.executable, "-c", f"open({str(done)!r}, 'w').write('x')"], cost=1.0)]

    class Tick(Exception):
        pass

    def no_sleep(_):
        raise Tick

    monkeypatch.setattr(suite.time, "sleep", no_sleep)
    with pytest.raises(Tick):
        suite.run(lambda: jobs, [0], poll=0.0)                    # default 1000 MiB: the card is busy, nothing starts
    assert not done.exists()
    monkeypatch.setattr(suite.time, "sleep", lambda _: None)
    suite.run(lambda: jobs, [0], poll=0.0, gpu_free_mb=16000)  # the job runs and the loop returns once it is done
    assert done.exists()
