"""Readings of the supplementary experiments S4–S9, as fixed in docs/supplementary-experiments-2.md (frozen
2026-10-15). Each part is written before its results exist and left unchanged afterwards; a part whose inputs are
missing is skipped.

    python fit/supp2_analysis.py            # every part whose inputs exist -> results/supp2/*.json and tables

S4  the equal-compute comparison with the dense curves extended to 640 (10M) and 320 (20M) tokens per parameter
S5  chain-of-thought and pause-token baselines against the looped models on the chain task
S6  KV-cache sharing across loops at inference (scripts/kv_share_eval.py)
S7  seeds that change the data order (collect with fit/collect_results.py --manifest configs/manifest_supp2_s7.csv)
S8  stopping rules and voting on the formal synthetic checkpoints
S9  the condition trained with loops = steps + random extra loops
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
from synthetic.analyze import max_solvable, probe_frontier  # noqa: E402

OUT = "results/supp2"
SUPP2_FORMAL = "results/synthetic/supp2/formal/chain"
SUPP2_PROBES = "results/synthetic/supp2/probes"
HALTING_SUPP2 = "results/synthetic/halting_supp2"            # synthetic/halting.py on the S9 checkpoints
FORMAL_SUMMARY = "results/synthetic/formal_summary.csv"
KV_SHARE = "results/supp2/kv_share.json"
S7_RESULTS = "results/supp2_s7.csv"
VALSETS = ("val_fwe", "val_second", "val_finemath", "val_code")
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


# ---------------------------------------------------------------- S6
def s6() -> dict | None:
    if not os.path.exists(KV_SHARE):
        print("S6: no results yet"); return None
    kv = json.load(open(KV_SHARE))
    rows = []
    for key, v in kv.items():
        d = v["delta_shared_minus_full"]
        rows.append(dict(run=key, r_train=v["r_train"], r_eval=v["r_eval"], k_bwd=v["k_bwd"], r_sample=v["r_sample"], lr0=v["lr0"],
                         loss_full=v["loss_full_cache"], loss_shared=v["loss_shared_cache"], delta=d,
                         full_vs_forward=round(v["loss_full_cache"] - v["loss_forward"], 4),
                         verdict=("usable" if d <= 0.02 else "degrades" if d <= 0.1 else "breaks")))
    df = pd.DataFrame(rows)

    def delta(name):
        m = df[df.run == name]
        return float(m.delta.iloc[0]) if len(m) else None
    pred = dict(fixed_r8_full_breaks_gt_0_3=(delta("20M_middle_r8_full_s42") or 0) > 0.3,
                truncated_k4_lt_0_1=(delta("20M_middle_r8_k4_s42") if delta("20M_middle_r8_k4_s42") is not None else 9) < 0.1,
                random_loops_lt_0_05=(delta("20M_middle_r8_full_rs1to8_s42") if delta("20M_middle_r8_full_rs1to8_s42") is not None else 9) < 0.05,
                r4_half_rate_smaller_than_1x=((delta("20M_middle_r4_full_lrs0.5_s42") or 9) < (delta("20M_middle_r4_full_s42") or 0)))
    out = dict(models=rows, predictions=pred, rule="delta <= 0.02 usable; <= 0.1 degrades; > 0.1 breaks")
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "s6_readings.json"), "w"), indent=1)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False)); print("S6 predictions:", pred)
    return out


# ---------------------------------------------------------------- S7
def s7() -> dict | None:
    if not os.path.exists(S7_RESULTS):
        print("S7: no results yet"); return None
    s7r = pd.read_csv(S7_RESULTS)
    base = pd.read_csv("results/all_results.csv")
    base = base[(base.rung == "20M") & (base.accounting == "iso_token")]
    for df in (s7r, base):
        df["tpp"] = (df.budget_tokens / df.N_rung).round(0)
        df["cellname"] = df.placement + "_r" + df.r.astype(str) + np.where(df.k_bwd > 0, "_k" + df.k_bwd.astype(str), "")
    offs = {m.group(0) if (m := __import__("re").search(r"do\d+b", n)) else "0": n for n in s7r.name.unique()}
    out = {}
    for cell in ("dense_r1", "middle_r4"):
        for tpp in (10.0, 20.0, 40.0):
            b = base[(base.cellname == cell) & (base.tpp == tpp)]
            s = s7r[(s7r.cellname == cell) & (s7r.tpp == tpp)]
            entry = {}
            for vs in VALSETS:
                by_order = {"0": float(b[b.seed == 42][vs].iloc[0])} if len(b[b.seed == 42]) else {}
                for _, row in s.iterrows():
                    tag = __import__("re").search(r"do(\d+b)", row["name"]).group(1)
                    by_order[tag] = float(row[vs])
                seeds = b[b.seed.isin([42, 43, 44])][vs].astype(float).tolist()
                entry[vs] = dict(loss_by_data_order=by_order, data_order_spread=round(max(by_order.values()) - min(by_order.values()), 4) if len(by_order) > 1 else None,
                                 init_seed_spread=round(max(seeds) - min(seeds), 4) if len(seeds) > 1 else None, init_seed_losses=[round(x, 4) for x in seeds])
            out[f"{cell} {int(tpp)}N"] = entry
    # the paired looped - dense gap under each data order
    gaps = {}
    for tpp in (10.0, 20.0, 40.0):
        e_d, e_l = out[f"dense_r1 {int(tpp)}N"], out[f"middle_r4 {int(tpp)}N"]
        gaps[f"{int(tpp)}N"] = {vs: {o: round(e_l[vs]["loss_by_data_order"][o] - e_d[vs]["loss_by_data_order"][o], 4)
                                     for o in e_l[vs]["loss_by_data_order"] if o in e_d[vs]["loss_by_data_order"]} for vs in VALSETS}
    main40 = {c: out[f"{c} 40N"]["val_fwe"] for c in ("dense_r1", "middle_r4")}
    ratio = {c: (v["data_order_spread"] / v["init_seed_spread"] if v["init_seed_spread"] else None) for c, v in main40.items()}
    gap40 = gaps["40N"]["val_fwe"]
    pred = dict(main_set_ratio_1_to_3=all(r is not None and 1 <= r <= 3 for r in ratio.values()),
                main_set_spread_0_005_to_0_02=all(0.005 <= v["data_order_spread"] <= 0.02 for v in main40.values()),
                paired_gap_varies_less_than_losses=(max(gap40.values()) - min(gap40.values())) < min(v["data_order_spread"] for v in main40.values()),
                code_spread_ge_0_1=all(out[f"{c} 40N"]["val_code"]["data_order_spread"] >= 0.1 for c in ("dense_r1", "middle_r4")))
    caveat = any(r is not None and r > 2 for r in ratio.values())
    res = dict(by_cell_and_budget=out, paired_gap_by_order=gaps, main_set_ratio_40N=ratio, predictions=pred,
               caveat_needed=caveat, rule="caveat if the data-order spread exceeds 2x the init-seed spread on the main set at 40N")
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "s7_readings.json"), "w"), indent=1)
    for k, e in out.items():
        print(f"{k:16s} " + "  ".join(f"{vs[4:]}: orders {e[vs]['loss_by_data_order']} spread {e[vs]['data_order_spread']} seeds {e[vs]['init_seed_spread']}" for vs in ("val_fwe", "val_code")))
    print("S7 paired gaps 40N main:", gap40, "| ratio:", ratio, "| predictions:", pred, "| caveat:", caveat)
    return res


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
        rep = {rule: report(rule) for rule in rs[0]["rules"]}          # every rule, the references and the fixed counts
        span12 = rep["best_fixed"]["acc_12"] - rep["own"]["acc_12"]
        for rule, v in rep.items():                                     # share of the best-fixed gain at 12 steps recovered
            v["recovered_share_12"] = round((v["acc_12"] - rep["own"]["acc_12"]) / span12, 3) if span12 > 0.05 else None
        own_loops = float(np.mean([OWN_LOOPS.get(cond, d) for d in range(1, 9)]))
        # the same selection without the votes, which run all 32 loops and so are not stopping rules (exploratory)
        no_vote = [k for k in eligible if not k.startswith("vote_")]
        best_stop = max(no_vote, key=lambda k: cand[k]["select_extrap_9_16"]) if no_vote else None
        best_12 = max((k for k in rules), key=lambda k: rep[k]["acc_12"])
        out[cond] = dict(n_seeds=len(rs), candidates=cand, selected=best, selected_without_votes=best_stop, report=rep,
                         pred_2_recovered_share_12=rep["pred_2"]["recovered_share_12"],
                         pred_2_loops_over_own_1_8=round(rep["pred_2"]["loops_1_8"] / own_loops, 2),
                         votes_beat_best_fixed=bool(max(rep["vote_12_16_24_32"]["acc_12"], rep["vote_8_32"]["acc_12"]) > rep["best_fixed"]["acc_12"] + 0.01
                                                    or max(rep["vote_12_16_24_32"]["acc_16"], rep["vote_8_32"]["acc_16"]) > rep["best_fixed"]["acc_16"] + 0.01),
                         best_rule_12=best_12, best_rule_acc_12=rep[best_12]["acc_12"])
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
    print(f"{'condition':10s} {'selected':16s} {'own12':>6s} {'state12':>8s} {'pred2_12':>9s} {'conf99_12':>9s} {'best_fix12':>10s} "
          f"{'sel12':>6s} {'sel16':>6s} {'loops1-8':>9s}  {'best rule @12 (no votes: selected)':s}")
    for c, v in out.items():
        rp = v["report"]; b = v["selected"]
        print(f"{c:10s} {str(b):16s} {rp['own']['acc_12']:6.2f} {rp['state_0.02']['acc_12']:8.2f} {rp['pred_2']['acc_12']:9.2f} "
              f"{rp['conf_0.99']['acc_12']:9.2f} {rp['best_fixed']['acc_12']:10.2f} {(rp[b]['acc_12'] if b else float('nan')):6.2f} "
              f"{(rp[b]['acc_16'] if b else float('nan')):6.2f} {(rp[b]['loops_1_8'] if b else float('nan')):9.1f}  "
              f"{v['best_rule_12']} {v['best_rule_acc_12']:.2f} (no votes: {v['selected_without_votes']})")
    print("S8 predictions:", pred)
    return res


# ---------------------------------------------------------------- S5
def _runs(prefix: str) -> list[dict]:
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(SUPP2_FORMAL, prefix + "_lr*_s*_lut32.json")))]


def _acc(runs: list[dict], d: int, key: str) -> dict:
    v = [r["final"][str(d)][key] for r in runs]
    return dict(mean=round(float(np.mean(v)), 4), min=round(float(min(v)), 4), max=round(float(max(v)), 4))


def s5() -> dict | None:
    conds = {"dense8_cot": "acc", "dense36_cot": "acc", "dense8_pause": "acc", "loop_r8_cot": "r=8"}
    runs = {c: _runs(c) for c in conds}
    if not any(runs.values()):
        print("S5: no results yet"); return None
    os.makedirs(OUT, exist_ok=True)
    fs = pd.read_csv(FORMAL_SUMMARY)
    ref = {c: dict(acc12=float(g.acc12_own.mean()), acc16=float(g.acc16_own.mean()),
                   acc12_more=float(g.acc12_more.mean()) if "acc12_more" in g and g.acc12_more.notna().any() else None,
                   acc16_more=float(g.acc16_more.mean()) if "acc16_more" in g and g.acc16_more.notna().any() else None)
           for c, g in fs.groupby("cond")}
    out = {"reference_formal_suite": ref, "conditions": {}}
    for c, rs in runs.items():
        if not rs:
            continue
        key = conds[c]
        res = dict(n_seeds=len(rs), lr=rs[0]["lr"],
                   acc_by_d={d: _acc(rs, d, key) for d in range(1, 25)},
                   maxd=[max_solvable(r["final"], key) for r in rs],
                   acc12=_acc(rs, 12, key), acc16=_acc(rs, 16, key), in_range=round(float(np.mean(
                       [np.mean([r["final"][str(d)][key] for d in range(1, 9)]) for r in rs])), 4))
        ck = "chain_acc" if key == "acc" else "chain_r=8"
        if ck in rs[0]["final"]["1"]:
            res["chain_acc12"] = _acc(rs, 12, ck); res["chain_acc16"] = _acc(rs, 16, ck)
        out["conditions"][c] = res
    c = out["conditions"]
    q = {}
    if "dense8_cot" in c:
        a12, a16 = c["dense8_cot"]["acc12"]["mean"], c["dense8_cot"]["acc16"]["mean"]
        q["Q1_prediction_cot_ge_0.7_at_12_and_0.4_at_16"] = bool(a12 >= 0.7 and a16 >= 0.4)
        q["Q1_cot8_vs_loop_r8_more_loops_12_16"] = [round(a12 - ref["loop_r8"]["acc12_more"], 3), round(a16 - ref["loop_r8"]["acc16_more"], 3)]
        q["Q1_cot8_vs_loop_rand_more_loops_12_16"] = [round(a12 - ref["loop_rand"]["acc12_more"], 3), round(a16 - ref["loop_rand"]["acc16_more"], 3)]
        q["Q1_cot8_vs_loop_step_own_12_16"] = [round(a12 - ref["loop_step"]["acc12"], 3), round(a16 - ref["loop_step"]["acc16"], 3)]
        q["Q1_position_extrapolation_suspected"] = bool(a12 < 0.4)
    if "dense8_pause" in c:
        q["Q2_pause_minus_dense8_at_12"] = round(c["dense8_pause"]["acc12"]["mean"] - ref["dense8"]["acc12"], 4)
        q["Q2_prediction_below_0.1"] = bool(q["Q2_pause_minus_dense8_at_12"] < 0.1)
    if "dense36_cot" in c:
        q["Q3_cot36_at_12"] = c["dense36_cot"]["acc12"]["mean"]
        q["Q3_prediction_ge_0.8"] = bool(c["dense36_cot"]["acc12"]["mean"] >= 0.8)
    comp = {}
    for d in (12, 16):
        L = 3 * d + 4
        comp[str(d)] = {"looped_r8_own_36L": 36 * L, "looped_r8_16_loops_68L": 68 * L, "cot8_kv_cache_8(L+d-1)": 8 * (L + d - 1),  # the answer is the last written value, not processed (corrected 2026-10-28)
                        "cot8_no_cache_8*sum(L+i)": 8 * sum(L + i for i in range(d)), "cot36_kv_cache_36(L+d-1)": 36 * (L + d - 1),
                        "pause8_8(L+d)": 8 * (L + d)}
    out.update(readings=q, inference_layer_passes_per_problem=comp)
    json.dump(out, open(os.path.join(OUT, "s5_readings.json"), "w"), indent=1)
    print(f"{'condition':14s} {'n':>2s} {'in-range':>8s} {'acc12':>18s} {'acc16':>18s} {'maxd':>12s}")
    for name, v in c.items():
        print(f"{name:14s} {v['n_seeds']:2d} {v['in_range']:8.3f} {v['acc12']['mean']:6.2f} [{v['acc12']['min']:.2f},{v['acc12']['max']:.2f}]"
              f"   {v['acc16']['mean']:6.2f} [{v['acc16']['min']:.2f},{v['acc16']['max']:.2f}]   {str(v['maxd']):>12s}")
    print("S5 readings:", json.dumps(q))
    return out


# ---------------------------------------------------------------- S9
def s9() -> dict | None:
    rs = _runs("loop_step_extra")
    if not rs:
        print("S9: no results yet"); return None
    os.makedirs(OUT, exist_ok=True)
    fs = pd.read_csv(FORMAL_SUMMARY)
    step = fs[fs.cond == "loop_step"]
    res = dict(n_seeds=len(rs), lr=rs[0]["lr"],
               in_range_oracle=round(float(np.mean([np.mean([r["final"][str(d)]["oracle"] for d in range(1, 9)]) for r in rs])), 4),
               in_range_r8=round(float(np.mean([np.mean([r["final"][str(d)]["r=8"] for d in range(1, 9)]) for r in rs])), 4),
               maxd_oracle=[max_solvable(r["final"], "oracle") for r in rs],
               maxd_more=[max(max_solvable(r["final"], k) for k in r["final"]["1"] if k == "oracle" or (k.startswith("r=") and int(k[2:]) > 8)) for r in rs],
               acc12_oracle=_acc(rs, 12, "oracle"), acc16_oracle=_acc(rs, 16, "oracle"),
               acc12_adaptive=_acc(rs, 12, "adaptive"), acc16_adaptive=_acc(rs, 16, "adaptive"),
               adaptive_loops_1_8=round(float(np.mean([np.mean([r["final"][str(d)]["adaptive_loops"] for d in range(1, 9)]) for r in rs])), 2),
               gain_over_8_loops={d: round(float(np.mean([r["final"][str(d)]["oracle"] - r["final"][str(d)]["r=8"] for r in rs])), 4)
                                  for d in range(9, 25)})
    pr = probe_frontier(SUPP2_PROBES) if os.path.isdir(SUPP2_PROBES) else pd.DataFrame()
    if len(pr):
        pr = pr[pr.cond == "loop_step_extra"]
        res["probes"] = dict(n=len(pr), persistence_16=round(float(pr.persistence.mean()), 3),
                             persistence_own=round(float(pr.persistence_own.mean()), 3),
                             diag_share=round(float(pr.diag_share.mean()), 3), slope=round(float(pr.slope.mean()), 3),
                             persistence_16_by_seed=[round(float(x), 3) for x in pr.persistence])
    halt = sorted(glob.glob(os.path.join(HALTING_SUPP2, "loop_step_extra_*.json")))
    if halt:
        hs = [json.load(open(f)) for f in halt]
        rules = [k for k in hs[0]["rules"] if k not in REFERENCES and not k.startswith("fixed_")]
        res["halting_report_set_acc12"] = {r: round(float(np.mean([h["results"]["report"]["12"][r]["acc"] for h in hs])), 4) for r in rules + ["own", "best_fixed"]}
    pred = dict(i_in_range_ge_0_99=bool(res["in_range_oracle"] >= 0.99),
                ii_persistence_16_ge_0_8=(bool(res["probes"]["persistence_16"] >= 0.8) if "probes" in res else None),
                iii_adaptive_12_ge_0_6=bool(res["acc12_adaptive"]["mean"] >= 0.6),
                iv_maxd_oracle_ge_11=bool(np.mean(res["maxd_oracle"]) >= 11))
    if pred["ii_persistence_16_ge_0_8"] is None:
        reading = "probes missing"
    elif pred["ii_persistence_16_ge_0_8"] and pred["iii_adaptive_12_ge_0_6"] and pred["iv_maxd_oracle_ge_11"]:
        reading = "a model can both extrapolate and stop"
    elif pred["ii_persistence_16_ge_0_8"] and pred["iii_adaptive_12_ge_0_6"]:
        reading = "a trade-off: holding the answer costs extrapolation"
    elif not pred["ii_persistence_16_ge_0_8"]:
        reading = "random extra loops do not teach persistence"
    else:
        reading = "persistence without a usable stop"
    out = dict(loop_step_extra=res, loop_step_reference=dict(maxd_own=step.maxd_own.tolist(), acc12_own=float(step.acc12_own.mean()),
                                                              acc16_own=float(step.acc16_own.mean())),
               predictions=pred, reading=reading)
    json.dump(out, open(os.path.join(OUT, "s9_readings.json"), "w"), indent=1)
    print(f"S9 loop_step_extra: in-range {res['in_range_oracle']}, maxd {res['maxd_oracle']} (step-matched {step.maxd_own.tolist()}), "
          f"acc12 oracle {res['acc12_oracle']['mean']} adaptive {res['acc12_adaptive']['mean']}, "
          f"persistence16 {res.get('probes', {}).get('persistence_16')}; reading: {reading}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "s4", "s5", "s6", "s7", "s8", "s9"])
    a = ap.parse_args()
    if a.part in ("all", "s4"):
        s4()
    if a.part in ("all", "s5"):
        s5()
    if a.part in ("all", "s6"):
        s6()
    if a.part in ("all", "s7"):
        s7()
    if a.part in ("all", "s8"):
        s8()
    if a.part in ("all", "s9"):
        s9()


if __name__ == "__main__":
    main()
