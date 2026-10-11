"""Figures for the paper, from results/all_results.csv, results/spikes.csv and results/fit/.

    python fit/make_figures.py            # -> results/figures/*.png and *.pdf

fig1_gap_to_dense     : loss gap of every looped setting to the dense model at the same N and tokens, per rung (iso-token)
fig2_phi_by_rung      : exploratory phi per rung and cell (ruler fixed), with the pre-registered pooled phi as dashes
fig3_loss_vs_tokens   : loss vs training tokens per rung, dense ruler and the looped settings (seed means)
fig4_spikes           : loss spikes and collapses per setting and rung
fig5_gap_by_exam      : looped-minus-dense gap at 40N on the four validation sets (web, web-2, math, code)
fig6_depth_curve      : loss vs inference loop count for the 80M looped models (token-matched checkpoints)
fig7_gain_by_difficulty: seed-averaged per-token gain by decile of average loss, with the leave-one-out dense null band
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RUNGS = ["10M", "20M", "40M", "80M"]
CELL_ORDER = ["middle_r2", "middle_r4", "middle_r4_k2", "middle_r8", "middle_r8_k4",
              "whole_r2", "whole_r4", "whole_r4_k2", "whole_r8", "whole_r8_k4"]
LABEL = {"middle_r2": "middle r=2", "middle_r4": "middle r=4", "middle_r4_k2": "middle r=4 trunc",
         "middle_r8": "middle r=8", "middle_r8_k4": "middle r=8 trunc", "whole_r2": "whole r=2",
         "whole_r4": "whole r=4", "whole_r4_k2": "whole r=4 trunc", "whole_r8": "whole r=8",
         "whole_r8_k4": "whole r=8 trunc"}
GROUP_LABEL = {"middle/full": "middle-block, full backprop", "middle/trunc": "middle-block, truncated",
               "whole/full": "whole-stack, full backprop", "whole/trunc": "whole-stack, truncated"}


def load(results: str) -> pd.DataFrame:
    df = pd.read_csv(results)
    df = df[~df.diverged.astype(str).str.lower().eq("true")].copy()
    df["bm"] = df.budget_mult.round().astype(int)
    df["cellname"] = np.where(df.placement == "dense", "dense",
                              df.placement + "_r" + df.r.astype(str) + np.where(df.k_bwd > 0, "_k" + df.k_bwd.astype(str), ""))
    return df


def fig1_gap(df: pd.DataFrame, out: str) -> None:
    d = df[df.accounting == "iso_token"]
    m = d.groupby(["cellname", "rung", "bm"]).val_fwe.mean()
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for ax, rung in zip(axes, RUNGS):
        for c in CELL_ORDER:
            xs, ys = [], []
            for bm in (10, 20, 40):
                if (c, rung, bm) in m and ("dense", rung, bm) in m:
                    xs.append(bm); ys.append(m[(c, rung, bm)] - m[("dense", rung, bm)])
            if xs:
                ax.plot(xs, ys, marker="o", ms=3, lw=1.2, ls="--" if "k" in c else "-",
                        color="C0" if c.startswith("middle") else "C3", alpha=0.35 + 0.15 * int(c.split("_r")[1][0]) / 2,
                        label=LABEL[c])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xscale("log", base=2); ax.set_xticks([10, 20, 40]); ax.set_xticklabels(["10N", "20N", "40N"])
        ax.set_title(rung); ax.set_xlabel("training tokens")
    axes[0].set_ylabel("loss gap to dense (nats, <0 = looped better)")
    axes[-1].legend(fontsize=7, ncol=2, loc="upper right")
    fig.suptitle("Looped minus dense at equal parameters and tokens (seed means; collapsed segments removed)", y=1.02)
    fig.tight_layout(); fig.savefig(out + ".png", dpi=200, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight"); plt.close(fig)


def fig2_phi(expl: dict, primary: dict, out: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(RUNGS))
    for i, (cell, lab) in enumerate(GROUP_LABEL.items()):
        y = [expl["by_rung"].get(f"{cell}@{r}", np.nan) for r in RUNGS]
        ax.plot(x, y, marker="o", label=lab, color=f"C{i}")
        ax.axhline(primary["F1"]["stage2_per_cell"]["theta"][cell], color=f"C{i}", ls=":", lw=1)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(RUNGS); ax.set_xlabel("loop-cell size")
    ax.set_ylabel("recurrence-equivalence exponent φ")
    ax.set_title("φ per size (exploratory, ruler fixed; dotted = pre-registered pooled φ)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf"); plt.close(fig)


def fig3_loss(df: pd.DataFrame, out: str) -> None:
    d = df[df.accounting == "iso_token"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, rung in zip(axes, RUNGS):
        g = d[d.rung == rung].groupby(["cellname", "D" if "D" in d else "budget_tokens"]).val_fwe.mean().reset_index()
        g.columns = ["cellname", "D", "L"]
        dense = g[g.cellname == "dense"].sort_values("D")
        ax.plot(dense.D, dense.L, color="k", marker="s", ms=4, lw=2, label="dense")
        for c in CELL_ORDER:
            gc = g[g.cellname == c].sort_values("D")
            if len(gc):
                ax.plot(gc.D, gc.L, marker="o", ms=3, lw=1, ls="--" if "k" in c else "-",
                        color="C0" if c.startswith("middle") else "C3", alpha=0.35 + 0.15 * int(c.split("_r")[1][0]) / 2, label=LABEL[c])
        ax.set_xscale("log"); ax.set_title(rung); ax.set_xlabel("training tokens"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("validation loss (nats/token)")
    axes[-1].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(out + ".png", dpi=200, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight"); plt.close(fig)


def fig4_spikes(spikes: pd.DataFrame, names: pd.Series, out: str) -> None:
    s = spikes[spikes.name.isin(names)].copy()
    s["rung"] = s.name.str.extract(r"^(\d+M)_")[0]
    s["cellname"] = s.name.str.replace(r"^\d+M_", "", regex=True).str.replace(r"_s\d+$", "", regex=True)
    s["cellname"] = s.cellname.str.replace("_full", "", regex=False).str.replace("dense_r1", "dense", regex=False)
    tab = s.pivot_table(index="cellname", columns="rung", values="n_spikes", aggfunc="sum").reindex(["dense"] + CELL_ORDER)[RUNGS]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(tab.fillna(0).values, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(RUNGS))); ax.set_xticklabels(RUNGS)
    ax.set_yticks(range(len(tab.index))); ax.set_yticklabels([LABEL.get(c, c) for c in tab.index])
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, int(v), ha="center", va="center", fontsize=8, color="w" if v > tab.values[~np.isnan(tab.values)].max() / 2 else "k")
    collapsed = s[s.collapse_step.notna() & (s.collapse_step.astype(str) != "")]
    ax.set_title(f"loss spikes (>1 nat) summed over seeds; collapses: {len(collapsed)} ({', '.join(collapsed.name)})", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="spikes")
    fig.tight_layout(); fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf"); plt.close(fig)


def fig5_gap_by_exam(df: pd.DataFrame, out: str) -> None:
    exams = [("val_fwe", "FineWeb-Edu (main)"), ("val_second", "SlimPajama"), ("val_finemath", "FineMath"), ("val_code", "code")]
    d = df[(df.accounting == "iso_token") & (df.bm == 40)]
    m = d.groupby(["cellname", "rung"])[[e for e, _ in exams]].mean()
    cells = ["middle_r4", "middle_r8", "middle_r8_k4", "whole_r8_k4"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, rung in zip(axes, ["20M", "40M", "80M"]):
        x = np.arange(len(exams)); w = 0.2
        for i, c in enumerate(cells):
            if (c, rung) in m.index:
                gap = [m.loc[(c, rung), e] - m.loc[("dense", rung), e] for e, _ in exams]
                ax.bar(x + (i - 1.5) * w, gap, w, label=LABEL[c], color=f"C{i}")
        ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels([n for _, n in exams], rotation=20, fontsize=8)
        ax.set_title(f"{rung}, 40N tokens")
    axes[0].set_ylabel("looped minus dense (nats, <0 = looped better)")
    axes[-1].legend(fontsize=7)
    fig.suptitle("The loop benefit is largest on math and code (exploratory; seed means)", y=1.02)
    fig.tight_layout(); fig.savefig(out + ".png", dpi=200, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight"); plt.close(fig)


def fig6_depth(curve: pd.DataFrame, out: str) -> None:
    c = curve[(curve.rung == "80M") & (curve.valset == "fwe")]
    fig, ax = plt.subplots(figsize=(6, 4))
    rs = [1, 2, 4, 8, 16]
    for i, (_, row) in enumerate(c.iterrows()):
        ys = [row[f"loss_r{r}"] for r in rs]
        ax.plot(rs, ys, marker="o", label=f"{LABEL.get(row.cell, row.cell)} (trained r={int(row.r_train)})", color=f"C{i}")
        ax.plot([row.r_train], [row[f"loss_r{int(row.r_train)}"]], marker="*", ms=14, color=f"C{i}")
    ax.set_xscale("log", base=2); ax.set_xticks(rs); ax.set_xticklabels([str(r) for r in rs])
    ax.set_xlabel("loop count at inference"); ax.set_ylabel("validation loss (FineWeb-Edu, 2M tokens)")
    ax.set_title("Fixed-r models only work at their training depth; truncated training is robust (80M)", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf"); plt.close(fig)


def fig7_difficulty(diff: pd.DataFrame, out: str) -> None:
    """Seed-averaged gain per difficulty decile against the leave-one-out dense null (by_difficulty_seedavg.csv)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, vs in zip(axes, ["fwe", "code"]):
        d = diff[diff.valset == vs]
        for i, rung in enumerate(["20M", "40M"]):
            g = d[(d.rung == rung) & (d.kind == "looped_vs_dense") & (d.cell == "middle_r8")].sort_values("avg_loss_decile")
            if len(g):
                ax.plot(g.avg_loss_decile, g.mean_delta, marker="o", ms=4, color=f"C{i}", label=f"{rung} middle r=8 vs dense (3 seeds each)")
            n = d[(d.rung == rung) & (d.kind == "null_loo")].groupby("avg_loss_decile").mean_delta.agg(["mean", "min", "max"])
            if len(n):
                ax.plot(n.index, n["mean"], ls="--", color=f"C{i}", alpha=0.7, label=f"{rung} dense seed vs other seeds (null)")
                ax.fill_between(n.index, n["min"], n["max"], color=f"C{i}", alpha=0.12)
        ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(1, 11))
        ax.set_xlabel("decile of the two models' average per-token loss (1 = easiest)")
        ax.set_title({"fwe": "FineWeb-Edu (main set)", "code": "code (out of distribution)"}[vs])
    axes[0].set_ylabel("dense minus looped loss (nats, >0 = looped better)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Per-token gain by difficulty, seed-averaged, with the dense-only null band", y=1.02)
    fig.tight_layout(); fig.savefig(out + ".png", dpi=200, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight"); plt.close(fig)


def _probe_title(rep: dict) -> str:
    return rep["run"].split("_s")[0].replace("_full", "").replace("_", " ") + f" (trained r={rep['r_train']})"


def fig8_probes(reps: list[dict], out: str) -> None:
    """Loss readable from the state after each loop: frozen read-out, trained adapter, trained linear head."""
    fig, axes = plt.subplots(1, len(reps), figsize=(4.2 * len(reps), 3.8))
    axes = np.atleast_1d(axes)
    for ax, rep in zip(axes, reps):
        loops = np.arange(1, rep["r_max"] + 1)
        L = rep["loss"]
        ax.plot(loops, L["lens"], marker="o", ms=3, label="frozen read-out (= run the model with this many loops)")
        ax.plot(loops, L["adapter"], marker="s", ms=3, ls="--", label="trained d x d adapter, frozen read-out")
        ax.plot(loops, L["linear"], marker="^", ms=3, ls=":", label="trained linear head on the state")
        ax.axvline(rep["r_train"], color="k", lw=0.8, alpha=0.5)
        lo = min(min(L["adapter"]), min(L["linear"]), min(L["lens"]))
        ax.set_ylim(lo - 0.1, max(max(L["adapter"]), max(L["linear"]), L["lens"][rep["r_train"] - 1] + 1.0) + 0.1)
        ax.set_title(_probe_title(rep), fontsize=9)
        ax.set_xlabel("loops run at inference"); ax.set_xticks([1, 2, 4, 8, 12, 16]); ax.grid(alpha=0.3)
    axes[0].set_ylabel("validation loss (FineWeb-Edu, 2M tokens)"); axes[0].legend(fontsize=6.5)
    fig.suptitle("What the state after each loop says about the next token (frozen read-out points above the frame are off-scale; see fig 6)", y=1.03, fontsize=9)
    fig.tight_layout(); fig.savefig(out + ".png", dpi=200, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight"); plt.close(fig)


def fig9_settle(reps: list[dict], out: str) -> None:
    """When does each token's prediction stop changing, and does that depend on difficulty?"""
    fig, axes = plt.subplots(2, len(reps), figsize=(4.2 * len(reps), 6.6), squeeze=False)
    for j, rep in enumerate(reps):
        r, n, h = rep["r_train"], rep["n_tokens"], rep["settle_hist"]
        loops = np.arange(1, r + 1)
        ax = axes[0, j]
        ax.bar(loops, np.array(h["all"]) / n, color="C0", alpha=0.5, label="all tokens")
        for k, c in (("correct", "C2"), ("wrong", "C3")):
            if sum(h[k]):
                ax.plot(loops, np.array(h[k]) / sum(h[k]), marker="o", ms=3, color=c, label=f"final top-1 {k} ({sum(h[k]) / n:.0%})")
        ax.set_title(_probe_title(rep), fontsize=9); ax.set_xticks(loops)
        ax.set_xlabel("loop after which the top-1 prediction no longer changes"); ax.legend(fontsize=7)
        if j == 0:
            ax.set_ylabel("fraction of tokens")
        ax = axes[1, j]
        ax.plot(range(1, 11), rep["settle_mean_by_decile"], marker="o", color="C0")
        ax.set_xlabel("difficulty decile (1 = easiest)"); ax.set_xticks(range(1, 11))
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel("mean settling loop", color="C0")
        ax2 = ax.twinx()
        ax2.plot(range(1, 11), rep["late_frac_by_decile"], marker="s", ms=3, color="C1")
        if j == len(reps) - 1:
            ax2.set_ylabel("fraction settling in the last two loops", color="C1")
    fig.suptitle("Prediction settling across loops (exploratory; frozen read-out after each loop, 2M validation tokens; "
                 "difficulty = decile of the token's loss at the training r)", y=1.01, fontsize=9)
    fig.tight_layout(); fig.savefig(out + ".png", dpi=200, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight"); plt.close(fig)


def fig10_random_r(reps: list[dict], extra, dense_ref: float | None, out: str) -> None:
    """20M: validation loss against the loop count used at inference, for fixed-r training and for random
    loop-count training (full and truncated backprop). Loops 1-16 come from the probe lens, 24/32 from the sweep."""
    style = {"20M_middle_r8_full_s42": ("fixed r=8 training", "C0", "o"),
             "20M_middle_r8_full_rs1to8_s42": ("r drawn from 1..8 at every step, full backprop", "C1", "s"),
             "20M_middle_r8_k4_rs1to8_s42": ("r drawn from 1..8 at every step, truncated k=4", "C2", "^")}
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for rep in reps:
        if rep["run"] not in style:
            continue
        label, color, marker = style[rep["run"]]
        loops, ys = list(range(1, rep["r_max"] + 1)), list(rep["loss"]["lens"])
        if extra is not None:
            e = extra[(extra.run == rep["run"]) & (extra.valset == "fwe") & (extra.r_eval > rep["r_max"])].sort_values("r_eval")
            loops += e.r_eval.tolist(); ys += e.mean_loss.tolist()
        ax.plot(loops, ys, marker=marker, ms=4, color=color, label=label)
        ax.plot([rep["r_train"]], [rep["loss"]["lens"][rep["r_train"] - 1]], marker="*", ms=13, color=color)
    if dense_ref is not None:
        ax.axhline(dense_ref, color="k", ls="--", lw=1, label=f"dense 20M at the same tokens (seed mean {dense_ref:.3f})")
    ticks = [1, 2, 4, 8, 16, 32]
    ax.set_xscale("log", base=2); ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlabel("loop count at inference"); ax.set_ylabel("validation loss (FineWeb-Edu, 2M tokens)")
    ax.set_title("20M middle block, 32N-token trunk checkpoints; star = nominal r=8\n"
                 "random loop-count training converges to a fixed point: robust, but no gain beyond ~4 loops", fontsize=8.5)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf"); plt.close(fig)


def fig11_data_extension(df: pd.DataFrame, ext: pd.DataFrame, out: str) -> None:
    """Looped minus dense at the same seed and budget against tokens per parameter: main grid to 40N (3 seeds), the
    2026-10-08 extension to 80N / 160N (seeds as available)."""
    allr = pd.concat([df, ext], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, rung in zip(axes, ["10M", "20M"]):
        d = allr[allr.rung == rung].copy()
        N = d.N_rung.iloc[0]
        d["mult"] = (d.budget_tokens / N).round(0).astype(int)
        dense = d[d.placement == "dense"].groupby(["mult", "seed"]).val_fwe.first()
        cells = [("middle", 4, 0, "middle r=4", "-"), ("middle", 8, 0, "middle r=8", "-")]
        if rung == "10M":     # the whole-stack cells were extended at 10M only (seed 42)
            cells += [("whole", 4, 0, "whole r=4", "--"), ("whole", 8, 4, "whole r=8 truncated", "--")]
        for i, (pl, r, k, label, ls) in enumerate(cells):
            g = d[(d.placement == pl) & (d.r == r) & (d.k_bwd == k)]
            gaps = [(row.mult, row.seed, row.val_fwe - dense[(row.mult, row.seed)]) for _, row in g.iterrows()
                    if (row.mult, row.seed) in dense.index]
            gp = pd.DataFrame(gaps, columns=["mult", "seed", "gap"]).groupby("mult").gap.agg(["mean", "min", "max", "count"])
            ax.plot(gp.index, gp["mean"], marker="o", ms=4, ls=ls, color=f"C{i}", label=label)
            ax.fill_between(gp.index, gp["min"], gp["max"], color=f"C{i}", alpha=0.12)
            if pl == "middle":
                for mult, row in gp.iterrows():
                    ax.annotate(f"{int(row['count'])}s", (mult, row["mean"]), textcoords="offset points",
                                xytext=(0, 6 if i else -12), ha="center", fontsize=6, color=f"C{i}")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xscale("log", base=2); ticks = [10, 20, 40, 80, 160]; ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlabel("training tokens per parameter (D/N)"); ax.set_title(f"{rung} rung", fontsize=10); ax.grid(alpha=0.3)
    axes[0].set_ylabel("looped minus dense loss (nats; < 0 = looped better)"); axes[0].legend(fontsize=8)
    fig.suptitle("The loop advantage keeps growing with tokens per parameter (same-seed gaps; band = min-max over seeds; "
                 "'ns' = seeds of the middle-block cells; whole-stack extension: seed 42; 80N/160N exploratory)", y=1.03, fontsize=8.5)
    fig.tight_layout(); fig.savefig(out + ".png", dpi=200, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight"); plt.close(fig)


MAIN_PROBE_RUNS = ["20M_middle_r8_full_s42", "40M_middle_r8_full_s42", "80M_middle_r8_full_s42", "80M_whole_r8_k4_s42"]


def fig8b_probe_seeds(reps: list[dict], out: str) -> None:
    """Replication of the per-loop read-out curves over seeds: one panel per rung and cell, one line per seed."""
    groups = {}
    for rep in reps:
        if "rs1to8" in rep["run"]:
            continue
        key = rep["run"].rsplit("_s", 1)[0]
        groups.setdefault(key, []).append(rep)
    keys = sorted(groups, key=lambda k: (int(k.split("M")[0]), k))
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.4))
    for ax, key in zip(np.atleast_1d(axes), keys):
        for i, rep in enumerate(sorted(groups[key], key=lambda r: r["run"])):
            loops = np.arange(1, rep["r_max"] + 1)
            seed = rep["run"].rsplit("_s", 1)[1]
            ax.plot(loops, rep["loss"]["lens"], marker="o", ms=2.5, color=f"C{i}", label=f"seed {seed}: frozen read-out")
            ax.plot(loops, rep["loss"]["adapter"], ls="--", color=f"C{i}", alpha=0.8, label=f"seed {seed}: trained adapter")
        ax.axvline(groups[key][0]["r_train"], color="k", lw=0.8, alpha=0.5)
        lo = min(min(r["loss"]["adapter"]) for r in groups[key])
        ax.set_ylim(lo - 0.1, lo + 1.6)
        ax.set_title(key.replace("_full", "").replace("_", " "), fontsize=9); ax.set_xticks([1, 4, 8, 12, 16]); ax.grid(alpha=0.3)
        ax.legend(fontsize=5.5)
    np.atleast_1d(axes)[0].set_ylabel("validation loss (2M tokens)")
    fig.suptitle("Per-loop read-out curves replicate across seeds (frozen read-out off-scale points are clipped)", y=1.03, fontsize=9)
    fig.tight_layout(); fig.savefig(out + ".png", dpi=200, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight"); plt.close(fig)


def fig12_scale(df: pd.DataFrame, ext: list[pd.DataFrame], out: str) -> None:
    """Ruler-free view of scale: the middle-block r=4 cell minus the dense model of the same rung and seed, at 10N,
    20N and 40N, from 10M to 160M (160M from the 2026-10-08 extension: two seeds of each)."""
    allr = pd.concat([df] + ext, ignore_index=True)
    allr["mult"] = (allr.budget_tokens / allr.N_rung).round(0).astype(int)
    rungs = ["10M", "20M", "40M", "80M", "160M"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, m in enumerate([10, 20, 40]):
        xs, ys, lo, hi = [], [], [], []
        for rung in rungs:
            x = allr[allr.rung == rung]
            loop = x[(x.placement == "middle") & (x.r == 4) & (x.k_bwd == 0) & (x.mult == m)].set_index("seed").val_fwe
            dense = x[(x.placement == "dense") & (x.mult == m)].set_index("seed").val_fwe
            gap = (loop - dense).dropna()
            if len(gap):
                xs.append(float(x.N_rung.iloc[0])); ys.append(gap.mean()); lo.append(gap.min()); hi.append(gap.max())
        ax.plot(xs, ys, marker="o", color=f"C{i}", label=f"{m}N tokens")
        ax.fill_between(xs, lo, hi, color=f"C{i}", alpha=0.15)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log"); ax.minorticks_off()
    ax.set_xticks([15.4e6, 26.8e6, 50.2e6, 92.7e6, 179.6e6]); ax.set_xticklabels(rungs)
    ax.set_xlabel("model size (rung)"); ax.set_ylabel("looped minus dense loss (nats; < 0 = looped better)")
    ax.set_title("Middle block looped 4 times vs the dense model of the same size\n(seed mean; band = min-max over seeds; "
                 "3 seeds at 10M-40M, 1 at 80M, 2 at 160M)", fontsize=8.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/all_results.csv")
    ap.add_argument("--spikes", default="results/spikes.csv")
    ap.add_argument("--fit", default="results/fit/primary/fit_iso_token_with_head.json")
    ap.add_argument("--exploratory", default="results/fit/exploratory_phi_by_rung.json")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load(a.results)
    fig1_gap(df, os.path.join(a.out, "fig1_gap_to_dense"))
    fig2_phi(json.load(open(a.exploratory)), json.load(open(a.fit)), os.path.join(a.out, "fig2_phi_by_rung"))
    fig3_loss(df, os.path.join(a.out, "fig3_loss_vs_tokens"))
    fig4_spikes(pd.read_csv(a.spikes), df.name.unique(), os.path.join(a.out, "fig4_spikes"))
    fig5_gap_by_exam(df, os.path.join(a.out, "fig5_gap_by_exam"))
    ext160 = [pd.read_csv(f) for f in ("results/ext_160m.csv", "results/ext_160m_dense.csv") if os.path.exists(f)]
    if ext160:
        fig12_scale(df, ext160, os.path.join(a.out, "fig12_scale_middle_r4"))
    if os.path.exists("results/ext_data.csv"):
        fig11_data_extension(df, pd.read_csv("results/ext_data.csv"), os.path.join(a.out, "fig11_data_extension"))
    reps = [json.load(open(p)) for p in sorted(glob.glob(os.path.join("results", "loop_probe", "*.json")))]
    fixed = [r for r in reps if r["run"] in MAIN_PROBE_RUNS]
    fixed.sort(key=lambda r: MAIN_PROBE_RUNS.index(r["run"]))
    if fixed:
        fig8_probes(fixed, os.path.join(a.out, "fig8_loop_probes"))
        fig9_settle(fixed, os.path.join(a.out, "fig9_settling"))
    if len([r for r in reps if "rs1to8" not in r["run"]]) > len(fixed):
        fig8b_probe_seeds(reps, os.path.join(a.out, "fig8b_probe_seeds"))
    if any("rs1to8" in r["run"] for r in reps):
        ml, dc = "results/token_analysis/mean_loss.csv", "results/token_analysis/depth_curve_randr.csv"
        dense_ref = None
        if os.path.exists(ml):
            m = pd.read_csv(ml)
            d = m[(m.rung == "20M") & m.cell.str.startswith("dense") & (m.valset == "fwe") & (m.r_eval == 1)]
            dense_ref = float(d.mean_loss.mean()) if len(d) else None
        fig10_random_r(reps, pd.read_csv(dc) if os.path.exists(dc) else None, dense_ref, os.path.join(a.out, "fig10_random_r_depth"))
    tok = "results/token_analysis"
    if os.path.exists(os.path.join(tok, "depth_curve.csv")):
        fig6_depth(pd.read_csv(os.path.join(tok, "depth_curve.csv")), os.path.join(a.out, "fig6_depth_curve"))
    if os.path.exists(os.path.join(tok, "by_difficulty_seedavg.csv")):
        fig7_difficulty(pd.read_csv(os.path.join(tok, "by_difficulty_seedavg.csv")), os.path.join(a.out, "fig7_gain_by_difficulty"))
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
