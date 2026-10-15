"""Stopping rules and voting (synthetic/halting.py, S8): each rule stops where the plan says, and the state rule
reproduces the trainer's adaptive stop on the same problems."""
import numpy as np
import torch

from synthetic import halting as h
from synthetic.train_synth import evaluate, train_and_eval
from tests.test_synthetic import _args


def test_rules_stop_where_the_plan_says():
    # three problems, six loops; answers 7, 2, 9
    preds = np.array([[5, 1, 9], [5, 2, 9], [5, 2, 9], [7, 2, 9], [7, 2, 9], [7, 2, 9]])
    top = np.array([[0.3, 0.2, 0.99], [0.5, 0.2, 0.99], [0.95, 0.2, 0.99], [0.6, 0.2, 0.99], [0.99, 0.2, 0.99], [0.99, 0.2, 0.99]])
    rel = np.array([[0.5, 0.5, 0.5], [0.5, 0.001, 0.5], [0.5, 0.5, 0.5], [0.01, 0.5, 0.5], [0.001, 0.5, 0.5]])
    ans = np.array([7, 2, 9])
    out = h.apply_rules(preds, top, rel, ans, own=2, d=3)
    assert out["pred_1"] == (2, 7) and out["pred_2"] == (2, 10) and out["pred_3"] == (3, 15)
    assert out["pred_2_min_own"] == (2, 10)                       # own = 2 adds no constraint
    assert out["conf_0.9"] == (2, 10) and out["conf_0.99"] == (3, 12)
    assert out["state_0.05"] == (3, 14) and out["state_0.02"] == (3, 14) and out["state_0.01"] == (3, 15)
    assert out["own"] == (2, 6) and out["oracle"] == (2, 9)
    assert out["fixed_1"] == (1, 3) and out["fixed_2"] == (2, 6) and out["fixed_4"] == (3, 12) and "fixed_8" not in out
    assert not any(k.startswith("vote") for k in out)            # no vote loop fits in six loops
    out4 = h.apply_rules(preds, top, rel, ans, own=4, d=3)
    assert out4["pred_2_min_own"] == (3, 14)                      # not before loop 4


def test_votes_take_the_majority_and_break_ties_at_the_later_loop():
    preds = np.zeros((32, 2), dtype=np.int64)
    preds[:, 0] = 4; preds[31, 0] = 9                             # 12, 16, 24 say 4; 32 says 9 -> 4
    preds[:, 1] = 1; preds[23:, 1] = 6                            # 12, 16 say 1; 24, 32 say 6 -> tie -> later loop: 6
    assert h._vote(preds, (12, 16, 24, 32)).tolist() == [4, 6]
    top, rel = np.full((32, 2), 0.5), np.full((31, 2), 0.5)
    out = h.apply_rules(preds, top, rel, np.array([4, 6]), own=8, d=5)
    assert out["vote_12_16_24_32"] == (2, 64) and out["vote_8_32"][1] == 64


def test_state_rule_matches_the_trainer_adaptive_stop(tmp_path):
    train_and_eval(_args(tmp_path, "chain", "loop_r2", steps=3))
    model, ck = h.load_checkpoint(str(tmp_path / "runs" / "chain" / "loop_r2_lr0.001_s1.pt"), torch.device("cpu"))
    ref = evaluate(model, "chain", [1], [2], 8, torch.device("cpu"), None, tol=0.02)[1]
    res = h.evaluate_checkpoint(model, "loop_r2", [1], 8, 32, torch.device("cpu"), None, batch=8,
                                seeds={"report": 12345})["report"]["1"]
    assert res["state_0.02"]["acc"] == ref["adaptive"] and res["state_0.02"]["loops"] == ref["adaptive_loops"]
    assert res["own"]["acc"] == ref["r=2"] and res["oracle"]["acc"] == ref["oracle"]
    assert res["best_fixed"]["acc"] >= max(res["own"]["acc"], res["fixed_32"]["acc"])
    assert all(1 <= v["loops"] <= 32 for v in res.values())
