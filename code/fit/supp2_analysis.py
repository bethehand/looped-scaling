"""Readings of the supplementary experiments S4–S9, as fixed in docs/supplementary-experiments-2.md (frozen
2026-10-15). Each part is written before its results exist and left unchanged afterwards; a part whose inputs are
missing is skipped.

    python fit/supp2_analysis.py            # every part whose inputs exist -> results/supp2/*.json and tables

S4  the equal-compute comparison with the dense curves extended to 640 (10M) and 320 (20M) tokens per parameter
S8  stopping rules and voting on the formal synthetic checkpoints
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from synthetic.halting import OWN_LOOPS  # noqa: E402

OUT = "results/supp2"
S4_RESULTS = "results/supp2_s4.csv"                     # fit/collect_results.py --manifest configs/manifest_supp2_s4.csv
EXT = ["results/ext_data.csv", "results/ext_160m.csv", "results/ext_160m_dense.csv"]
S4_TOP = {"10M": 160.0, "20M": 80.0}                    # the top looped budget per rung (tokens per parameter)
HALTING = "results/synthetic/halting"
PY = sys.executable


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------- S4
def s4() -> dict | None:
    if not os.path.exists(S4_RESULTS):
        print("S4: no results yet"); return None
    os.makedirs(OUT, exist_ok=True)
    s4r = pd.read_csv(S4_RESULTS)
    dense = s4r[s4r.placement == "dense"]
    dense_csv = os.path.join(OUT, "s4_dense_points.csv")
    dense.to_csv(dense_csv, index=False)
    raw = os.path.join(OUT, "iso_flop_raw_s4.csv")
    _run([PY, "fit/iso_flop_raw.py", "--ext", *EXT, dense_csv, "--out", raw])
    _run([PY, "fit/iso_flop_bounds.py", "--ext", *EXT, dense_csv, "--out", os.path.join(OUT, "iso_flop_bounds_s4.csv"),
          "--summary", os.path.join(OUT, "iso_flop_bounds_s4.json")])
    _run([PY, "fit/local_phi.py", "--files", "results/all_results.csv", *EXT, dense_csv,
          "--out", os.path.join(OUT, "local_phi_s4.csv"), "--trends-out", os.path.join(OUT, "local_phi_s4_trends.json")])
    r = pd.read_csv(raw)
    top = r[[m == S4_TOP.get(g, -1) for g, m in zip(r.rung, r.mult)]].copy()
    # reading rule, per (rung, cell): both seeds measured; gap > +0.01 dense wins; within +-0.01 parity; < -0.01 looped
    verdicts = {}
    for (rung, cell), g in top.groupby(["rung", "cell"]):
        gaps = g.gap_flop.values
        meas = (~g.extrapolated).all()
        if (gaps > 0.01).all():
            v = "dense wins"
        elif (np.abs(gaps) <= 0.01).all():
            v = "parity"
        elif (gaps < -0.01).all():
            v = "looping overtakes"
        else:
            v = "mixed"
        verdicts[f"{rung} {cell}"] = dict(gap_by_seed={int(s): float(x) for s, x in zip(g.seed, gaps)}, measured=bool(meas),
                                          verdict=v)
    gaps_all = top[~top.extrapolated].gap_flop.values
    overall = ("dense wins throughout the measured range" if (gaps_all > 0.01).all() else
               "parity at the largest data per parameter" if (np.abs(gaps_all) <= 0.01).all() else
               "looping overtakes somewhere" if (gaps_all < -0.01).any() else "mixed")
    prediction_held = bool(((gaps_all >= -0.01) & (gaps_all <= 0.04)).all() and (gaps_all >= -0.02).all())
    # the tuned looped model (0.35x, seeds 42 and 43) against the extended dense curve, S1's Q3
    tuned = _tuned_gap(s4r)
    # multipliers at the budgets that gained dense rungs
    lp = pd.read_csv(os.path.join(OUT, "local_phi_s4.csv"))
    lp_old = pd.read_csv("results/fit/local_phi.csv")
    k = ["cell", "r", "rung", "tpp"]
    m = lp_old.merge(lp, on=k, suffixes=("_old", "_s4"))
    changed = m[(m.rung == "10M") & (m.tpp >= 80) | (m.rung == "20M") & (m.tpp >= 80)]
    out = dict(top_budget_comparisons=verdicts, overall=overall, prediction_held=prediction_held,
               n_measured=int((~top.extrapolated).sum()), n_extrapolated=int(top.extrapolated.sum()),
               tuned_looped_vs_dense=tuned,
               multipliers_changed=changed[k + ["M_old", "status_old", "M_s4", "status_s4"]].round(3).to_dict("records"))
    json.dump(out, open(os.path.join(OUT, "s4_readings.json"), "w"), indent=1)
    pd.set_option("display.width", 200)
    print(top[["rung", "cell", "seed", "mult", "rho", "gap_token", "gap_flop", "extrapolated"]].round(3).to_string(index=False))
    print(f"S4: {overall}; prediction held: {prediction_held}")
    for kk, v in verdicts.items():
        print(f"  {kk:22s} {v['verdict']:18s} gaps {v['gap_by_seed']}")
    print("  tuned looped (0.35x) vs dense at equal FLOPs:", tuned)
    return out


def _tuned_gap(s4r: pd.DataFrame) -> dict:
    """10M middle r=4 at 0.35x (S1 seed 42, S4 seed 43) against the same seed's dense curve extended by S4."""
    sys.path.insert(0, "fit")
    from iso_flop_raw import dense_at
    frames = [pd.read_csv(f) for f in ["results/all_results.csv", *EXT, S4_RESULTS] if os.path.exists(f)]
    if os.path.exists("results/supp_lr.csv"):
        frames.append(pd.read_csv("results/supp_lr.csv"))
    df = pd.concat(frames, ignore_index=True)
    df = df[~df.diverged.astype(str).str.lower().eq("true")]
    df["mult"] = (df.budget_tokens / df.N_rung).round(1)
    out = {}
    d10 = df[(df.rung == "10M")]
    dense_fpt = d10[(d10.placement == "dense")].train_flops_per_token.iloc[0]
    for seed in (42, 43):
        tuned = d10[d10.name == f"10M_middle_r4_full_lrs0.35_s{seed}"]
        ruler = (d10[(d10.placement == "dense") & (d10.seed == seed) & ~d10.name.str.contains("lrs")]
                 .groupby("mult").val_fwe.mean().sort_index())
        if tuned.empty or len(ruler) < 3:
            continue
        rho = tuned.train_flops_per_token.iloc[0] / dense_fpt
        for _, row in tuned.iterrows():
            if row.mult in (80.0, 160.0):
                d, extra = dense_at(ruler, row.mult * rho)
                out[f"seed {seed}, {int(row.mult)} tokens per parameter"] = dict(
                    looped=round(float(row.val_fwe), 4), dense_equal_flops=round(d, 4), gap=round(float(row.val_fwe) - d, 4),
                    dense_equiv_mult=round(row.mult * rho, 1), extrapolated=bool(extra))
    return out


# ---------------------------------------------------------------- S8
REFERENCES = ("own", "oracle", "best_fixed")


def s8() -> dict | None:
    files = sorted(glob.glob(os.path.join(HALTING, "*.json")))
    if not files:
        print("S8: no results yet"); return None
    os.makedirs(OUT, exist_ok=True)
    reps = [json.load(open(f)) for f in files]
    out = {}
    for cond in ("loop_r2", "loop_r4", "loop_r8", "loop_rand", "loop_step"):
        rs = [r for r in reps if r["cond"] == cond]
        if not rs:
            continue
        rules = [k for k in rs[0]["rules"] if k not in REFERENCES and not k.startswith("fixed_")]

        def mean(set_name, rule, ds, key="acc"):
            return float(np.mean([np.mean([r["results"][set_name][str(d)][rule][key] for d in ds]) for r in rs]))
        own_in = mean("select", "own", range(1, 9))
        cand = {}
        for rule in rules:
            in_range = mean("select", rule, range(1, 9))
            cand[rule] = dict(select_in_range=round(in_range, 4), select_extrap_9_16=round(mean("select", rule, range(9, 17)), 4),
                              eligible=bool(in_range >= own_in - 0.01))
        eligible = [k for k, v in cand.items() if v["eligible"]]
        best = max(eligible, key=lambda k: cand[k]["select_extrap_9_16"]) if eligible else None

        def report(rule):
            return dict(acc_1_8=round(mean("report", rule, range(1, 9)), 4), acc_12=round(mean("report", rule, [12]), 4),
                        acc_16=round(mean("report", rule, [16]), 4), acc_20=round(mean("report", rule, [20]), 4),
                        loops_1_8=round(mean("report", rule, range(1, 9), "loops"), 2),
                        loops_12=round(mean("report", rule, [12], "loops"), 2), loops_16=round(mean("report", rule, [16], "loops"), 2),
                        acc_12_by_seed=[r["results"]["report"]["12"][rule]["acc"] for r in rs])
        rep = {rule: report(rule) for rule in ("own", "best_fixed", "state_0.02", "pred_2", "pred_2_min_own",
                                               "vote_12_16_24_32", "vote_8_32") + ((best,) if best else ())}
        span12 = rep["best_fixed"]["acc_12"] - rep["own"]["acc_12"]
        rec = (rep["pred_2"]["acc_12"] - rep["own"]["acc_12"]) / span12 if span12 > 0.05 else None
        own_loops = float(np.mean([OWN_LOOPS.get(cond, d) for d in range(1, 9)]))
        out[cond] = dict(n_seeds=len(rs), candidates=cand, selected=best, report=rep,
                         pred_2_recovered_share_12=(round(rec, 3) if rec is not None else None),
                         pred_2_loops_over_own_1_8=round(rep["pred_2"]["loops_1_8"] / own_loops, 2),
                         votes_beat_best_fixed=bool(max(rep["vote_12_16_24_32"]["acc_12"], rep["vote_8_32"]["acc_12"]) > rep["best_fixed"]["acc_12"] + 0.01
                                                    or max(rep["vote_12_16_24_32"]["acc_16"], rep["vote_8_32"]["acc_16"]) > rep["best_fixed"]["acc_16"] + 0.01),
                         best_rule_acc_12=max(v["acc_12"] for k, v in rep.items() if k not in REFERENCES))
    pred = dict(
        pred_2_beats_state_rule_r4_r8=all(out[c]["report"]["pred_2"]["acc_12"] > out[c]["report"]["state_0.02"]["acc_12"]
                                          for c in ("loop_r4", "loop_r8") if c in out),
        pred_2_recovers_85pct_r4_r8=all((out[c]["pred_2_recovered_share_12"] or 0) >= 0.85 for c in ("loop_r4", "loop_r8") if c in out),
        pred_2_loops_within_1_5x_own=all(out[c]["pred_2_loops_over_own_1_8"] <= 1.5 for c in ("loop_r4", "loop_r8") if c in out),
        votes_do_not_beat_best_fixed=not any(v["votes_beat_best_fixed"] for v in out.values()),
        no_rule_rescues_step_matched=bool(out.get("loop_step", {}).get("best_rule_acc_12", 0) <= 0.2))
    res = dict(conditions=out, predictions=pred, selection="highest mean accuracy over d = 9-16 on the selection set among "
                                                             "rules within 0.01 of the own-loop in-range accuracy; reported on the reporting set")
    json.dump(res, open(os.path.join(OUT, "s8_readings.json"), "w"), indent=1)
    print(f"{'condition':10s} {'selected':16s} {'own12':>6s} {'state12':>8s} {'pred2_12':>9s} {'best_fix12':>10s} {'sel12':>6s} {'sel16':>6s} {'loops1-8':>9s}")
    for c, v in out.items():
        rp = v["report"]; b = v["selected"]
        print(f"{c:10s} {str(b):16s} {rp['own']['acc_12']:6.2f} {rp['state_0.02']['acc_12']:8.2f} {rp['pred_2']['acc_12']:9.2f} "
              f"{rp['best_fixed']['acc_12']:10.2f} {(rp[b]['acc_12'] if b else float('nan')):6.2f} {(rp[b]['acc_16'] if b else float('nan')):6.2f} "
              f"{(rp[b]['loops_1_8'] if b else float('nan')):9.1f}")
    print("S8 predictions:", pred)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "s4", "s8"])
    a = ap.parse_args()
    if a.part in ("all", "s4"):
        s4()
    if a.part in ("all", "s8"):
        s8()


if __name__ == "__main__":
    main()
