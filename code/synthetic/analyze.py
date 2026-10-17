"""Analysis of the formal synthetic suite (extension C): per-run summary, verdicts on P1-P5 of the frozen design v2
(docs/synthetic-reasoning-design.md, section 4), and figures 13-15.

    python synthetic/analyze.py
Reads results/synthetic/formal/chain/*.json and results/synthetic/formal/probes/*.json; writes
results/synthetic/formal_summary.csv, results/synthetic/formal_verdicts.json and results/figures/fig13-15.

Definitions (as in the design): "largest solvable step count" = largest d such that every step count 1..d reaches
accuracy >= 0.9. "Own loop count" = the trained r for fixed loops and random loops (r = 8), loops = d for the
step-matched condition. "Best over loop counts" (maxd_best, acc*_best) also tries r_eval = 12, 16, 24, 32 and loops = d;
it is kept as it was because the frozen S2 analysis (fit/supp_analysis.py) reads it. "More loops" (maxd_more, acc*_more,
added 2026-10-15) tries every inference loop count above the trained one, which for r = 2 also includes 4 and 8 loops and
for r = 4 also 8 loops; for the other conditions it equals "best over loop counts".
"Adaptive" (added 2026-10-15, step 2 of the plan): the label-free stop that the training script already evaluates on
every looped model (stop at the first loop whose relative change of the answer-position state is below tol = 0.02, at
most 32 loops). The frozen design pre-specified it for the step-matched condition; for the other conditions it is
exploratory. It is reported next to best_fixed{d}, the best accuracy over the fixed inference loop counts, which is
chosen with the answers and is an upper bound for any fixed loop count.
"""
from __future__ import annotations

import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ORDER = ["dense8", "dense20", "dense36", "loop_r2", "loop_r4", "loop_r8", "loop_rand", "loop_step"]
LABEL = {"dense8": "dense 8 layers", "dense20": "dense 20 layers", "dense36": "dense 36 layers",
         "loop_r2": "looped r=2", "loop_r4": "looped r=4", "loop_r8": "looped r=8",
         "loop_rand": "random loop count (1-8)", "loop_step": "step-matched loops (= d)"}
EXTRA = ("r=12", "r=16", "r=24", "r=32", "oracle")
OWN_LOOPS = {"loop_r2": 2, "loop_r4": 4, "loop_r8": 8, "loop_rand": 8}   # the step-matched model runs loops = d


def own_key(cond: str, rep: dict) -> str:
    if cond.startswith("dense"):
        return "acc"
    return "oracle" if cond == "loop_step" else f"r={rep['condition']['r']}"


def max_solvable(final: dict, key: str, thr: float = 0.9) -> int:
    m = 0
    for d in sorted(final, key=int):
        if final[d].get(key, 0.0) >= thr:
            m = int(d)
        else:
            break
    return m


def acc(final: dict, d: int, keys) -> float:
    return max(final[str(d)].get(k, 0.0) for k in keys)


def summarise(path: str = "results/synthetic/formal/chain") -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(os.path.join(path, "*.json"))):
        r = json.load(open(p))
        f, c = r["final"], r["cond"]
        own = own_key(c, r)
        keys_best = [own] + ([k for k in EXTRA] if not c.startswith("dense") else [])
        best = max(max_solvable(f, k) for k in keys_best)
        row = dict(cond=c, seed=r["seed"], lr=r["lr"], n_params=r["n_params"], executed_layers=r["executed_layers"],
                   in_dist=np.mean([f[str(d)].get(own, 0.0) for d in range(1, 9)]), maxd_own=max_solvable(f, own),
                   maxd_best=best)
        for d in (9, 10, 12, 16, 20, 24):
            row[f"acc{d}_own"] = f[str(d)].get(own, 0.0)
            row[f"acc{d}_best"] = acc(f, d, keys_best)
        if not c.startswith("dense"):
            r_tr = r["condition"]["r"]
            for re_ in (1, 2, 4, 8, 16, 32):
                k = f"r={re_}"
                if k in f["1"]:
                    row[f"in_dist_r{re_}"] = np.mean([f[str(d)][k] for d in range(1, 9)])
            row["r_train"] = r_tr
            for d in (10, 12, 16, 20, 24):
                row[f"acc{d}_r8"] = f[str(d)].get("r=8", np.nan)
                row[f"acc{d}_oracle"] = f[str(d)].get("oracle", np.nan)   # loops = d
        rows.append(row)
    df = pd.DataFrame(rows)
    df["o"] = df.cond.map(lambda c: ORDER.index(c) if c in ORDER else len(ORDER))
    return df.sort_values(["o", "seed"]).drop(columns="o").reset_index(drop=True)


def more_loops(path: str = "results/synthetic/formal/chain") -> pd.DataFrame:
    """Per looped run (added 2026-10-15): the solvable step count and the 12- and 16-step accuracies over the own loop
    count and every larger inference loop count (r_eval > r_train, and loops = d), and the decision quantity of P3:
    gain{d} = accuracy with loops = d minus accuracy with 8 loops, for d = 9..24."""
    rows = []
    for p in sorted(glob.glob(os.path.join(path, "*.json"))):
        r = json.load(open(p))
        f, c = r["final"], r["cond"]
        if c.startswith("dense"):
            continue
        r_tr = r["condition"]["r"]
        keys = [own_key(c, r)] + [k for k in f["1"] if k == "oracle" or (k.startswith("r=") and int(k[2:]) > r_tr)]
        row = dict(cond=c, seed=r["seed"], lr=r["lr"], maxd_more=max(max_solvable(f, k) for k in keys),
                   acc12_more=acc(f, 12, keys), acc16_more=acc(f, 16, keys))
        for d in range(9, 25):
            row[f"gain{d}"] = f[str(d)].get("oracle", np.nan) - f[str(d)].get("r=8", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def adaptive(path: str = "results/synthetic/formal/chain") -> pd.DataFrame:
    """Per looped run: accuracy and mean loops of the adaptive stop in the training range (d = 1..8) and at 12 and 16
    steps, its solvable step count, and best_fixed{d} (see the module docstring)."""
    rows = []
    for p in sorted(glob.glob(os.path.join(path, "*.json"))):
        r = json.load(open(p))
        f, c = r["final"], r["cond"]
        if c.startswith("dense") or "adaptive" not in f["1"]:
            continue
        fixed = [k for k in f["1"] if k.startswith("r=")]
        row = dict(cond=c, seed=r["seed"], lr=r["lr"], tol=r.get("tol"),
                   in_dist_adaptive=np.mean([f[str(d)]["adaptive"] for d in range(1, 9)]),
                   loops_in_dist_adaptive=np.mean([f[str(d)]["adaptive_loops"] for d in range(1, 9)]),
                   maxd_adaptive=max_solvable(f, "adaptive"))
        for d in (12, 16):
            row[f"acc{d}_adaptive"] = f[str(d)]["adaptive"]
            row[f"loops{d}_adaptive"] = f[str(d)]["adaptive_loops"]
            row[f"best_fixed{d}"] = max(f[str(d)][k] for k in fixed)
        rows.append(row)
    return pd.DataFrame(rows)


def adaptive_summary(s: pd.DataFrame) -> dict:
    """Seed means per looped condition, and the share of the gain from the own loop count to the best fixed loop count
    that the adaptive stop recovers at 12 and 16 steps (descriptive; not one of P1-P5)."""
    out = {"note": "label-free stop: relative state change < tol between loops, at most 32 loops; pre-specified in the "
                   "frozen design for the step-matched condition, exploratory for the others; best_fixed is chosen "
                   "with the answers (upper bound for any fixed loop count)"}
    for c, g in s[s.cond.str.startswith("loop") & s.acc12_adaptive.notna()].groupby("cond", sort=False):
        row = {k: float(g[k].mean()) for k in ("in_dist_adaptive", "loops_in_dist_adaptive", "acc12_own", "acc12_adaptive",
                                                "best_fixed12", "loops12_adaptive", "acc16_own", "acc16_adaptive",
                                                "best_fixed16", "loops16_adaptive")}
        row["acc12_adaptive_by_seed"] = [float(x) for x in g.acc12_adaptive]
        row["acc16_adaptive_by_seed"] = [float(x) for x in g.acc16_adaptive]
        for d in (12, 16):
            span = row[f"best_fixed{d}"] - row[f"acc{d}_own"]
            row[f"recovered_share{d}"] = (float((row[f"acc{d}_adaptive"] - row[f"acc{d}_own"]) / span)
                                          if span > 0.05 else None)
        out[c] = row
    return out


def probe_frontier(path: str = "results/synthetic/formal/probes") -> pd.DataFrame:
    """Per checkpoint: first unit at which x_j (j = 2..8) is decodable at >= 0.5 in its own segment, the slope of that
    frontier (units per step, least squares over the steps that are reached) and the persistence (minimum accuracy
    after the frontier passes, averaged over j) over all probed units.
    Added 2026-10-15: persistence_own, the same over the model's own loop count only (r for fixed loops, 8 for random
    loops, d for the step-matched model, every layer for dense models; the step-matched model keeps its values for d
    loops and loses them only when it is run past d); diag_share, the share of the steps x_1..x_d that first become
    decodable (>= 0.5) at unit j, the design's "x_j is decoded at loop j"; diag_share_argmax, the share whose most
    accurate unit (the first within 0.01 of the maximum) is unit j, which is unstable because computed values stay
    near 1.0 for many loops."""
    rows = []
    for p in sorted(glob.glob(os.path.join(path, "*.json"))):
        r = json.load(open(p))
        t = np.array(r["local_table"])
        name = os.path.basename(p).split("__")[0]
        cond, seed = name.split("_lr")[0], int(name.split("_s")[-1].split("_")[0])
        lr = float(name.split("_lr")[1].split("_s")[0])
        n_own = t.shape[0] if r["unit"] == "layer" else OWN_LOOPS.get(cond, r["d"])
        first, keep, keep_own = [], [], []
        for j in range(2, t.shape[1]):
            u = np.where(t[:, j] >= 0.5)[0]
            first.append(int(u[0]) + 1 if len(u) else np.nan)
            keep.append(float(t[u[0]:, j].min()) if len(u) else np.nan)
            keep_own.append(float(t[u[0]:n_own, j].min()) if len(u) and u[0] < n_own else np.nan)
        js = np.arange(2, t.shape[1])
        ok = ~np.isnan(first)
        slope = float(np.polyfit(js[ok], np.array(first)[ok], 1)[0]) if ok.sum() >= 3 else np.nan
        steps = range(1, t.shape[1])
        diag = np.mean([(t[:, j] >= 0.5).any() and int(np.argmax(t[:, j] >= 0.5)) + 1 == j for j in steps])
        diag_max = np.mean([int(np.argmax(t[:, j] >= t[:, j].max() - 0.01)) + 1 == j for j in steps])
        rows.append(dict(cond=cond, seed=seed, lr=lr, unit=r["unit"], first=first, slope=slope, reached=int(ok.sum()),
                         persistence=float(np.nanmean(keep)) if ok.any() else np.nan,
                         persistence_own=float(np.nanmean(keep_own)) if not np.isnan(keep_own).all() else np.nan,
                         diag_share=float(diag), diag_share_argmax=float(diag_max)))
    df = pd.DataFrame(rows)
    df["o"] = df.cond.map(lambda c: ORDER.index(c) if c in ORDER else len(ORDER))
    return df.sort_values(["o", "seed"]).drop(columns="o").reset_index(drop=True)


def train_coverage(steps: int = 32000, batch: int = 512, d_train: int = 8, curriculum_frac: float = 0.5,
                   n_values: int = 97, n_tables: int = 32) -> dict:
    """How many distinct chain problems exist at each step count d, how many the formal training runs draw at d (one d per
    batch, uniform in 1..d_max, d_max growing over the first curriculum_frac of training, as in train_synth.py), and the
    expected share of uniformly drawn test problems at d that also occur in training. Problems are not unique: every
    1-step problem is seen many times; from d = 5 on almost every test problem is new (added 2026-10-14)."""
    counts = np.zeros(d_train + 1)
    for step in range(1, steps + 1):
        d_max = min(d_train, max(1, math.ceil(d_train * step / (curriculum_frac * steps))))
        counts[1:d_max + 1] += batch / d_max
    out = {}
    for d in range(1, d_train + 1):
        space = n_values * n_tables ** d
        out[d] = dict(distinct_problems=int(space), training_draws=int(round(counts[d])),
                      seen_share_of_test=float(1 - np.exp(-counts[d] / space)))
    return out


def verdicts(s: pd.DataFrame, pr: pd.DataFrame) -> dict:
    m = s.groupby("cond").agg(maxd_own=("maxd_own", "mean"), maxd_best=("maxd_best", "mean"),
                              maxd_own_min=("maxd_own", "min"), maxd_own_max=("maxd_own", "max"),
                              maxd_best_min=("maxd_best", "min"), maxd_best_max=("maxd_best", "max"))
    g = lambda c, k="maxd_own": float(m.loc[c, k])  # noqa: E731
    out = {}
    order = ["dense8", "loop_r2", "loop_r4", "loop_r8"]
    vals = [g(c) for c in order]
    out["P1"] = dict(
        statement="largest solvable step count increases dense8 < r2 < r4 < r8; dense8 <= 8; r8 >= 16",
        means={c: round(v, 2) for c, v in zip(order, vals)},
        increasing=bool(all(a < b for a, b in zip(vals, vals[1:]))), dense8_le_8=bool(g("dense8") <= 8),
        r8_ge_16=bool(g("loop_r8") >= 16), r8_best_over_loop_counts=round(g("loop_r8", "maxd_best"), 2))
    pairs = [("dense20", "loop_r4"), ("dense36", "loop_r8")]
    p2 = {}
    for dn, lp in pairs:
        gap_iso = g(lp) - g(dn)
        gap_param = g(lp) - g("dense8")
        p2[f"{dn}_vs_{lp}"] = dict(dense=round(g(dn), 2), looped=round(g(lp), 2), dense_not_lower=bool(g(dn) >= g(lp)),
                                   gap_within_quarter=bool(abs(gap_iso) <= 0.25 * abs(gap_param)))
    out["P2"] = dict(statement="iso-depth dense >= looped, and |gap| <= 1/4 of the looped-vs-dense8 gap", pairs=p2,
                     supported=bool(all(v["dense_not_lower"] and v["gap_within_quarter"] for v in p2.values())))
    r8 = s[s.cond == "loop_r8"]
    rr = s[s.cond == "loop_rand"]
    st = s[s.cond == "loop_step"]
    out["P3"] = dict(
        fixed_r_drops_off_train_r=dict(
            statement="fixed-r models lose accuracy sharply when r_eval != r_train",
            in_dist_r4=round(float(r8.in_dist_r4.mean()), 3), in_dist_r8=round(float(r8.in_dist_r8.mean()), 3),
            in_dist_r16=round(float(r8.in_dist_r16.mean()), 3),
            supported=bool(r8.in_dist_r16.mean() < r8.in_dist_r8.mean() - 0.2)),
        # Decision quantity of the frozen design (docs/synthetic-reasoning-design.md, section 4): the accuracy difference
        # between loops = d and loops = 8 on d > 8. Until 2026-10-14 this verdict used the largest solvable step count
        # instead (maxd_best - maxd_own < 1), which cannot see gains below the 90% line; corrected after an external
        # review. "No gain" = the seed-mean difference is at most 0.1 at both d = 12 and d = 16 (the margin P3's third
        # item uses for "significantly higher").
        random_r_robust_no_gain=dict(
            statement="random-r models are robust to r_eval but do not improve on d > 8 with more loops",
            in_dist_r4=round(float(rr.in_dist_r4.mean()), 3), in_dist_r8=round(float(rr.in_dist_r8.mean()), 3),
            in_dist_r16=round(float(rr.in_dist_r16.mean()), 3),
            robust=bool(abs(rr.in_dist_r16.mean() - rr.in_dist_r8.mean()) < 0.05),
            acc_loops_d_minus_loops_8={d: round(float((rr[f"acc{d}_oracle"] - rr[f"acc{d}_r8"]).mean()), 3)
                                       for d in (10, 12, 16, 20, 24)},
            acc16_loops_8_per_seed=[round(float(v), 3) for v in rr.acc16_r8],
            acc16_loops_d_per_seed=[round(float(v), 3) for v in rr.acc16_oracle],
            no_gain=bool(all((rr[f"acc{d}_oracle"] - rr[f"acc{d}_r8"]).mean() <= 0.1 for d in (12, 16))),
            maxd_own=round(float(rr.maxd_own.mean()), 2), maxd_best=round(float(rr.maxd_best.mean()), 2),
            maxd_own_per_seed=[int(v) for v in rr.maxd_own], maxd_best_per_seed=[int(v) for v in rr.maxd_best],
            superseded_criterion_maxd_gain_below_1=bool(rr.maxd_best.mean() - rr.maxd_own.mean() < 1)),
        # Decision quantity of the frozen design: the accuracy difference between loops = d and loops = 8 on d > 8, which
        # must be "significantly higher" and extend "to at least 16". Read as: the seed-mean difference exceeds 0.1 (the
        # margin used for "no gain" above) at every d from 10 to 16. At d = 9 eight loops already solve the chain
        # (0.96-1.00: the unshared layers around the loop add a step), so there is no room for a gain; it is only
        # required not to be lower. Until 2026-10-15 the verdict also required every seed to solve 16 steps at the 90%
        # line, which the design does not ask for (deviation log 2026-10-15).
        step_matched_extrapolates=dict(
            statement="on d > 8, step-matched models with loops = d beat loops = 8, extending to at least d = 16",
            gain_loops_d_minus_loops_8={d: round(float(st[f"gain{d}"].mean()), 3) for d in range(9, 25)},
            gain_min_over_seeds={d: round(float(st[f"gain{d}"].min()), 3) for d in range(9, 17)},
            acc9_loops_8=[round(float(v), 3) for v in st.acc9_own - st.gain9],
            maxd_oracle=[int(v) for v in st.maxd_own], acc12_oracle=[round(float(v), 2) for v in st.acc12_own],
            acc12_r8=[round(float(v), 2) for v in st.acc12_r8], acc16_oracle=[round(float(v), 2) for v in st.acc16_own],
            acc16_r8=[round(float(v), 2) for v in st.acc16_r8],
            supported=bool(all(st[f"gain{d}"].mean() > 0.1 for d in range(10, 17)) and st.gain9.min() >= -0.01),
            superseded_criterion_every_seed_solves_16=bool((st.maxd_own >= 16).all()
                                                           and (st.acc12_own > st.acc12_r8 + 0.1).all())))
    p3b = out["P3"]["random_r_robust_no_gain"]
    p3b["supported"] = bool(p3b["robust"] and p3b["no_gain"])
    fr = pr.groupby("cond").agg(slope=("slope", "mean"), persistence=("persistence", "mean"),
                                persistence_own=("persistence_own", "mean"), diag_share=("diag_share", "mean"),
                                diag_share_argmax=("diag_share_argmax", "mean"), reached=("reached", "mean"))
    loops = [c for c in fr.index if c.startswith("loop")]
    # Decision quantity of the frozen design: the diagonal share of the probe table ("x_i is decoded most accurately at
    # loop i"). Read on 2026-10-15, after the results were known, as the share of x_1..x_8 that first become decodable
    # at their own loop (diag_share); near-diagonal = share >= 0.5. The verdict until then used the slope of the frontier
    # (|slope - 1| < 0.3), a substitute that was not logged; it gives the same reading (deviation log 2026-10-15).
    out["P4"] = dict(
        statement="the probe table of the step-matched model is near-diagonal (one step per loop); the fixed-r model's is not",
        diagonal_share={c: round(float(fr.loc[c, "diag_share"]), 2) for c in loops},
        diagonal_share_min_over_seeds={c: round(float(pr[pr.cond == c].diag_share.min()), 2) for c in loops},
        diagonal_share_argmax={c: round(float(fr.loc[c, "diag_share_argmax"]), 2) for c in loops},
        slope_units_per_step={c: round(float(fr.loc[c, "slope"]), 2) for c in loops},
        persistence_own_loops={c: round(float(fr.loc[c, "persistence_own"]), 2) for c in loops},
        persistence_16_loops={c: round(float(fr.loc[c, "persistence"]), 2) for c in loops},
        step_matched_diagonal=bool(fr.loc["loop_step", "diag_share"] >= 0.5),
        fixed_r_not_diagonal=bool(fr.loc["loop_r8", "diag_share"] < 0.5),
        superseded_slope_criterion=dict(step_matched_diagonal=bool(abs(fr.loc["loop_step", "slope"] - 1) < 0.3),
                                        fixed_r_not_diagonal=bool(abs(fr.loc["loop_r8", "slope"] - 1) >= 0.3)))
    out["P4"]["supported"] = bool(out["P4"]["step_matched_diagonal"] and out["P4"]["fixed_r_not_diagonal"])
    out["P5"] = dict(statement="accuracy gaps of tens of points on the synthetic task, versus 0.01-0.07 nats on text",
                     acc12_best_dense_max=round(float(s[s.cond.str.startswith("dense")].acc12_best.max()), 2),
                     acc12_best_looped_mean=round(float(s[s.cond.str.startswith("loop")].acc12_best.mean()), 2),
                     acc12_more_looped_mean=round(float(s[s.cond.str.startswith("loop")].acc12_more.mean()), 2),
                     acc12_own_looped={c: round(float(s[s.cond == c].acc12_own.mean()), 2)
                                       for c in ORDER if c.startswith("loop")})
    out["per_condition"] = {c: {k: round(float(v), 2) for k, v in m.loc[c].items()} for c in m.index}
    return out


def figures(s: pd.DataFrame, pr: pd.DataFrame, out_dir: str = "results/figures",
            path: str = "results/synthetic/formal/chain") -> None:
    reps = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(path, "*.json")))]
    ds = np.arange(1, 25)
    # fig13: accuracy against step count at each condition's own loop count
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, c in enumerate(ORDER):
        rs = [r for r in reps if r["cond"] == c]
        a = np.array([[r["final"][str(d)].get(own_key(c, r), 0.0) for d in ds] for r in rs])
        ls = "--" if c.startswith("dense") else "-"
        ax.plot(ds, a.mean(0), ls=ls, marker="o", ms=3, color=f"C{i}", label=f"{LABEL[c]} ({len(rs)} seeds)")
        ax.fill_between(ds, a.min(0), a.max(0), color=f"C{i}", alpha=0.12)
    ax.axvspan(0.5, 8.5, color="grey", alpha=0.08); ax.text(4.5, 1.02, "training range", ha="center", fontsize=8)
    ax.set_xlabel("steps in the chain (d)"); ax.set_ylabel("answer accuracy"); ax.set_ylim(-0.02, 1.08)
    ax.set_title("Lookup chains: accuracy at each model's own loop count (mean, band = min-max over seeds)", fontsize=9)
    ax.legend(fontsize=6.5, ncol=1, loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig13_synthetic_accuracy.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "fig13_synthetic_accuracy.pdf")); plt.close(fig)
    # fig14: inference-time loops for looped r=8 (fixed training) and random loops; dense 36 for reference
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, c in zip(axes, ["loop_r8", "loop_rand"]):
        rs = [r for r in reps if r["cond"] == c]
        for i, re_ in enumerate([4, 8, 16, 32]):
            a = np.array([[r["final"][str(d)].get(f"r={re_}", np.nan) for d in ds] for r in rs])
            ax.plot(ds, np.nanmean(a, 0), marker="o", ms=3, color=f"C{i}", label=f"{re_} loops at inference")
            ax.fill_between(ds, np.nanmin(a, 0), np.nanmax(a, 0), color=f"C{i}", alpha=0.12)
        dn = np.array([[r["final"][str(d)]["acc"] for d in ds] for r in reps if r["cond"] == "dense36"])
        ax.plot(ds, dn.mean(0), ls="--", color="k", label="dense 36 layers (same depth as 8 loops)")
        ax.axvspan(0.5, 8.5, color="grey", alpha=0.08)
        ax.set_title(f"{LABEL[c]} (trained with 8 loops{' on average 4.5' if c == 'loop_rand' else ''})", fontsize=9)
        ax.set_xlabel("steps in the chain (d)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("answer accuracy (mean over 5 seeds; band = min-max)"); axes[0].legend(fontsize=7)
    fig.suptitle("More loops at inference solve longer chains than seen in training (trained on d = 1-8)", y=1.02, fontsize=9.5)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig14_synthetic_inference_loops.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig14_synthetic_inference_loops.pdf"), bbox_inches="tight"); plt.close(fig)
    # fig15: probe frontier: the loop (or layer) at which x_j becomes decodable in its own segment
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    js = np.arange(2, 9)
    for i, c in enumerate(["loop_r2", "loop_r4", "loop_r8", "loop_rand", "loop_step"]):
        g = pr[pr.cond == c]
        f = np.array(g["first"].tolist(), dtype=float)
        ls, mk = ("--", "s") if c == "loop_step" else ("-", "o")
        axes[0].plot(js, np.nanmean(f, 0), ls=ls, marker=mk, ms=7 if c == "loop_step" else 5, mfc="none" if c == "loop_step" else None,
                     color=f"C{i + 3}", label=f"{LABEL[c]} (persistence {g.persistence_own.mean():.2f} within its own "
                                               f"loops, {g.persistence.mean():.2f} over 16)")
    axes[0].plot(js, js - 0, ls=":", color="grey", label="one step per loop")
    axes[0].set_xlabel("step j"); axes[0].set_ylabel("first loop at which x_j is decodable (>= 0.5)")
    axes[0].set_title("Looped models: the computed frontier advances loop by loop", fontsize=9); axes[0].legend(fontsize=6.5)
    axes[0].grid(alpha=0.3)
    for i, c in enumerate(["dense20", "dense36"]):
        g = pr[pr.cond == c]
        for _, row in g.iterrows():
            axes[1].plot(js, row["first"], marker="o", ms=3, color=f"C{i + 1}", alpha=0.7,
                         label=f"{LABEL[c]}" if row.seed == g.seed.min() else None)
    axes[1].set_xlabel("step j"); axes[1].set_ylabel("first layer at which x_j is decodable (>= 0.5)")
    axes[1].set_title("Dense models: late, and absent in some seeds", fontsize=9)
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig15_synthetic_probes.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "fig15_synthetic_probes.pdf")); plt.close(fig)


def sweep_table(res_dir: str = "results/synthetic/chain", probe_dir: str = "results/synthetic/sweep_probes") -> pd.DataFrame:
    """Learning-rate sensitivity (seed 1, 16000 steps): per condition and learning rate, the in-distribution score used
    for the choice, the solvable step count at the own loop count and over loop counts, and the probe frontier."""
    s = summarise(res_dir)
    pr = probe_frontier(probe_dir) if os.path.isdir(probe_dir) and glob.glob(os.path.join(probe_dir, "*.json")) else None
    s = s[["cond", "lr", "in_dist", "maxd_own", "maxd_best", "acc12_best", "acc16_best"]]
    if pr is not None:
        s = s.merge(pr[["cond", "lr", "slope", "persistence", "reached"]], on=["cond", "lr"], how="left")
    s["o"] = s.cond.map(lambda c: ORDER.index(c) if c in ORDER else len(ORDER))
    return s.sort_values(["o", "lr"]).drop(columns="o").reset_index(drop=True)


def main():
    s = (summarise().merge(more_loops(), on=["cond", "seed", "lr"], how="left")
         .merge(adaptive(), on=["cond", "seed", "lr"], how="left"))
    pr = probe_frontier()
    s.to_csv("results/synthetic/formal_summary.csv", index=False)
    pr.to_csv("results/synthetic/formal_probe_frontier.csv", index=False)
    v = verdicts(s, pr)
    v["adaptive_stopping"] = adaptive_summary(s)
    json.dump(v, open("results/synthetic/formal_verdicts.json", "w"), indent=1)
    figures(s, pr)
    pd.set_option("display.width", 220)
    cols = ["cond", "seed", "lr", "maxd_own", "maxd_best", "maxd_more", "acc12_best", "acc16_best", "in_dist"]
    print(s[cols].round(2).to_string(index=False))
    print(json.dumps({k: v[k] for k in ("P1", "P2", "P3", "P4", "P5")}, indent=1))
    ad = pd.DataFrame({c: d for c, d in v["adaptive_stopping"].items() if c != "note"}).T
    print(ad[["in_dist_adaptive", "acc12_own", "acc12_adaptive", "best_fixed12", "loops12_adaptive", "recovered_share12",
              "acc16_own", "acc16_adaptive", "best_fixed16", "loops16_adaptive", "recovered_share16"]].to_string())
    print(pr[["cond", "seed", "slope", "persistence", "reached"]].round(2).to_string(index=False))
    sw = sweep_table()
    sw.to_csv("results/synthetic/sweep_lr_table.csv", index=False)
    print(sw.assign(lr=sw.lr.map(lambda v: f"{v:g}")).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
