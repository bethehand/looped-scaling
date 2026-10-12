"""Analysis of the formal synthetic suite (extension C): per-run summary, verdicts on P1-P5 of the frozen design v2
(docs/synthetic-reasoning-design.md, section 4), and figures 13-15.

    python synthetic/analyze.py
Reads results/synthetic/formal/chain/*.json and results/synthetic/formal/probes/*.json; writes
results/synthetic/formal_summary.csv, results/synthetic/formal_verdicts.json and results/figures/fig13-15.

Definitions (as in the design): "largest solvable step count" = largest d such that every step count 1..d reaches
accuracy >= 0.9. "Own loop count" = the trained r for fixed loops and random loops (r = 8), loops = d for the
step-matched condition. "Best over loop counts" also tries r_eval = 12, 16, 24, 32 and loops = d.
"""
from __future__ import annotations

import glob
import json
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
            for d in (10, 12, 16):
                row[f"acc{d}_r8"] = f[str(d)].get("r=8", np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["o"] = df.cond.map(ORDER.index)
    return df.sort_values(["o", "seed"]).drop(columns="o").reset_index(drop=True)


def probe_frontier(path: str = "results/synthetic/formal/probes") -> pd.DataFrame:
    """Per checkpoint: first unit at which x_j (j = 2..8) is decodable at >= 0.5 in its own segment, the slope of that
    frontier (units per step, least squares over the steps that are reached) and the persistence (minimum accuracy
    after the frontier passes, averaged over j)."""
    rows = []
    for p in sorted(glob.glob(os.path.join(path, "*.json"))):
        r = json.load(open(p))
        t = np.array(r["local_table"])
        name = os.path.basename(p).split("__")[0]
        cond, seed = name.split("_lr")[0], int(name.split("_s")[-1].split("_")[0])
        first, keep = [], []
        for j in range(2, t.shape[1]):
            u = np.where(t[:, j] >= 0.5)[0]
            first.append(int(u[0]) + 1 if len(u) else np.nan)
            keep.append(float(t[u[0]:, j].min()) if len(u) else np.nan)
        js = np.arange(2, t.shape[1])
        ok = ~np.isnan(first)
        slope = float(np.polyfit(js[ok], np.array(first)[ok], 1)[0]) if ok.sum() >= 3 else np.nan
        rows.append(dict(cond=cond, seed=seed, unit=r["unit"], first=first, slope=slope, reached=int(ok.sum()),
                         persistence=float(np.nanmean(keep)) if ok.any() else np.nan))
    df = pd.DataFrame(rows)
    df["o"] = df.cond.map(ORDER.index)
    return df.sort_values(["o", "seed"]).drop(columns="o").reset_index(drop=True)


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
        random_r_robust_no_gain=dict(
            statement="random-r models are robust to r_eval but do not improve on d > 8 with more loops",
            in_dist_r4=round(float(rr.in_dist_r4.mean()), 3), in_dist_r16=round(float(rr.in_dist_r16.mean()), 3),
            maxd_own=round(float(rr.maxd_own.mean()), 2), maxd_best=round(float(rr.maxd_best.mean()), 2),
            maxd_own_per_seed=[int(v) for v in rr.maxd_own], maxd_best_per_seed=[int(v) for v in rr.maxd_best],
            supported=bool(abs(rr.in_dist_r16.mean() - rr.in_dist_r8.mean()) < 0.05
                           and rr.maxd_best.mean() - rr.maxd_own.mean() < 1)),
        step_matched_extrapolates=dict(
            statement="step-matched models with loops = d beat loops = 8 on d > 8 and reach at least 16 steps",
            maxd_oracle=[int(v) for v in st.maxd_own], acc12_oracle=[round(float(v), 2) for v in st.acc12_own],
            acc12_r8=[round(float(v), 2) for v in st.acc12_r8], acc16_oracle=[round(float(v), 2) for v in st.acc16_own],
            acc16_r8=[round(float(v), 2) for v in st.acc16_r8],
            beats_r8_on_long=bool((st.acc12_own > st.acc12_r8 + 0.1).all()),
            supported=bool((st.maxd_own >= 16).all() and (st.acc12_own > st.acc12_r8 + 0.1).all())))
    fr = pr.groupby("cond").agg(slope=("slope", "mean"), persistence=("persistence", "mean"), reached=("reached", "mean"))
    out["P4"] = dict(
        statement="the probe table of the step-matched model is near-diagonal (one step per loop); the fixed-r model's is not",
        slope_units_per_step={c: round(float(fr.loc[c, "slope"]), 2) for c in fr.index if c.startswith("loop")},
        persistence={c: round(float(fr.loc[c, "persistence"]), 2) for c in fr.index if c.startswith("loop")},
        step_matched_diagonal=bool(abs(fr.loc["loop_step", "slope"] - 1) < 0.3),
        fixed_r_not_diagonal=bool(abs(fr.loc["loop_r8", "slope"] - 1) >= 0.3))
    out["P4"]["supported"] = bool(out["P4"]["step_matched_diagonal"] and out["P4"]["fixed_r_not_diagonal"])
    out["P5"] = dict(statement="accuracy gaps of tens of points on the synthetic task, versus 0.01-0.07 nats on text",
                     acc12_best_dense_max=round(float(s[s.cond.str.startswith("dense")].acc12_best.max()), 2),
                     acc12_best_looped_mean=round(float(s[s.cond.str.startswith("loop")].acc12_best.mean()), 2))
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
                     color=f"C{i + 3}", label=f"{LABEL[c]} (persistence {g.persistence.mean():.2f})")
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


def main():
    s = summarise()
    pr = probe_frontier()
    s.to_csv("results/synthetic/formal_summary.csv", index=False)
    pr.to_csv("results/synthetic/formal_probe_frontier.csv", index=False)
    v = verdicts(s, pr)
    json.dump(v, open("results/synthetic/formal_verdicts.json", "w"), indent=1)
    figures(s, pr)
    pd.set_option("display.width", 220)
    cols = ["cond", "seed", "lr", "maxd_own", "maxd_best", "acc12_best", "acc16_best", "in_dist"]
    print(s[cols].round(2).to_string(index=False))
    print(json.dumps({k: v[k] for k in ("P1", "P2", "P3", "P4", "P5")}, indent=1))
    print(pr[["cond", "seed", "slope", "persistence", "reached"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
