"""Figures for the paper, from results/all_results.csv, results/spikes.csv and results/fit/.

    python fit/make_figures.py            # -> results/figures/*.png and *.pdf

fig1_gap_to_dense     : loss gap of every looped setting to the dense model at the same N and tokens, per rung (iso-token)
fig2_phi_by_rung      : exploratory phi per rung and cell (ruler fixed), with the pre-registered pooled phi as dashes
fig3_loss_vs_tokens   : loss vs training tokens per rung, dense ruler and the looped settings (seed means)
fig4_spikes           : loss spikes and collapses per setting and rung
"""
from __future__ import annotations

import argparse
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
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
