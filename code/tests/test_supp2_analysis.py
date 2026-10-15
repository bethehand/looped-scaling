"""Readings of the supplementary experiments S4-S9 (fit/supp2_analysis.py) on made-up result files: S6 verdict bins and
the S8 selection rule of docs/supplementary-experiments-2.md."""
import json
import os

import pytest

from fit import supp2_analysis as sa

RULES = ["best_fixed", "conf_0.9", "conf_0.99", "fixed_1", "fixed_8", "oracle", "own", "pred_1", "pred_2", "pred_2_min_own",
         "pred_3", "state_0.005", "state_0.02", "vote_12_16_24_32", "vote_8_32"]
# accuracy in range (d <= 8) and out of range (d >= 9) per rule, the same on both problem sets
ACC = {"own": (1.0, 0.04), "oracle": (1.0, 0.8), "best_fixed": (1.0, 0.84), "fixed_1": (0.3, 0.03), "fixed_8": (1.0, 0.04),
       "state_0.02": (1.0, 0.65), "state_0.005": (1.0, 0.82), "pred_1": (0.6, 0.1), "pred_2": (0.99, 0.5), "pred_2_min_own": (1.0, 0.52),
       "pred_3": (1.0, 0.67), "conf_0.9": (0.97, 0.71), "conf_0.99": (0.998, 0.832), "vote_12_16_24_32": (1.0, 0.83), "vote_8_32": (1.0, 0.835)}
LOOPS = {"own": 4, "oracle": 6, "best_fixed": 7, "fixed_1": 1, "fixed_8": 8, "state_0.02": 10, "state_0.005": 25, "pred_1": 3,
         "pred_2": 5, "pred_2_min_own": 5.5, "pred_3": 7, "conf_0.9": 6, "conf_0.99": 9, "vote_12_16_24_32": 32, "vote_8_32": 32}


def _halting_file(path, cond, seed, acc=ACC):
    res = {s: {str(d): {r: dict(acc=acc[r][0] if d <= 8 else acc[r][1], loops=float(LOOPS[r])) for r in RULES} for d in range(1, 25)}
           for s in ("select", "report")}
    json.dump(dict(ckpt="x.pt", cond=cond, seed=seed, lr=1e-4, n=2000, r_max=32, d=list(range(1, 25)),
                   seeds=dict(select=777, report=888), rules=RULES, results=res), open(path, "w"))


def test_s8_selects_the_most_extrapolating_eligible_rule(tmp_path, monkeypatch):
    d = tmp_path / "halting"
    d.mkdir()
    for seed in (2, 3):
        _halting_file(d / f"loop_r4_lr0.0001_s{seed}_lut32.json", "loop_r4", seed)
    step = dict(ACC, **{"conf_0.99": (0.93, 0.76), "pred_2_min_own": (0.84, 0.6), "state_0.02": (0.04, 0.03), "vote_8_32": (0.04, 0.04),
                        "vote_12_16_24_32": (0.04, 0.04), "pred_2": (0.76, 0.21), "pred_3": (0.2, 0.1), "conf_0.9": (0.8, 0.66),
                        "state_0.005": (0.04, 0.03), "own": (1.0, 0.957), "oracle": (1.0, 0.957), "best_fixed": (0.87, 0.957)})
    _halting_file(d / "loop_step_lr0.0001_s2_lut32.json", "loop_step", 2, step)
    monkeypatch.setattr(sa, "HALTING", str(d))
    monkeypatch.setattr(sa, "OUT", str(tmp_path / "out"))
    res = sa.s8()
    r4 = res["conditions"]["loop_r4"]
    # the votes are eligible (in range within 0.01 of own) and extrapolate best, so the plan's selection picks vote_8_32;
    # without the votes the confidence rule wins; conf_0.9 (0.97 in range) is not eligible
    assert r4["n_seeds"] == 2 and r4["selected"] == "vote_8_32" and r4["selected_without_votes"] == "conf_0.99"
    assert not r4["candidates"]["conf_0.9"]["eligible"] and r4["candidates"]["conf_0.99"]["eligible"]
    assert r4["report"]["conf_0.99"]["acc_12"] == 0.832 and r4["report"]["conf_0.99"]["loops_12"] == 9.0
    assert r4["report"]["conf_0.99"]["recovered_share_12"] == pytest.approx((0.832 - 0.04) / (0.84 - 0.04), abs=1e-3)
    assert r4["best_rule_12"] == "vote_8_32" and r4["best_rule_acc_12"] == 0.835
    assert r4["pred_2_loops_over_own_1_8"] == 1.25 and not r4["votes_beat_best_fixed"]
    st = res["conditions"]["loop_step"]
    assert st["selected"] is None and st["selected_without_votes"] is None           # every rule loses in-range accuracy
    assert st["best_rule_12"] == "conf_0.99" and st["report"]["own"]["recovered_share_12"] is None
    p = res["predictions"]
    assert p == dict(pred_2_beats_state_rule_r4_r8=False, pred_2_recovers_85pct_r4_r8=False, pred_2_loops_within_1_5x_own=True,
                     votes_do_not_beat_best_fixed=True, no_rule_rescues_step_matched=False)
    assert os.path.exists(tmp_path / "out" / "s8_readings.json")


def test_s6_verdict_bins_and_predictions(tmp_path, monkeypatch):
    def row(d, **kw):
        base = dict(r_train=8, r_eval=8, k_bwd=0, r_sample="", lr0=1e-3, loss_forward=3.4, loss_full_cache=3.4,
                    loss_shared_cache=round(3.4 + d, 4), delta_shared_minus_full=d)
        return dict(base, **kw)
    kv = {"20M_middle_r8_full_s42": row(0.091), "20M_middle_r8_k4_s42": row(0.14, k_bwd=4), "20M_middle_r8_full_rs1to8_s42": row(0.0),
          "20M_middle_r4_full_s42": row(0.17, r_train=4, r_eval=4), "20M_middle_r4_full_lrs0.5_s42": row(0.1, r_train=4, r_eval=4)}
    f = tmp_path / "kv_share.json"
    json.dump(kv, open(f, "w"))
    monkeypatch.setattr(sa, "KV_SHARE", str(f))
    monkeypatch.setattr(sa, "OUT", str(tmp_path / "out"))
    out = sa.s6()
    verdict = {m["run"]: m["verdict"] for m in out["models"]}
    assert verdict == {"20M_middle_r8_full_s42": "degrades", "20M_middle_r8_k4_s42": "breaks", "20M_middle_r8_full_rs1to8_s42": "usable",
                       "20M_middle_r4_full_s42": "breaks", "20M_middle_r4_full_lrs0.5_s42": "degrades"}
    assert out["predictions"] == dict(fixed_r8_full_breaks_gt_0_3=False, truncated_k4_lt_0_1=False, random_loops_lt_0_05=True,
                                      r4_half_rate_smaller_than_1x=True)
