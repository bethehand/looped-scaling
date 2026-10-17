"""Three-panel overview figure for the README and the report.

    python fit/make_overview.py           # -> results/figures/fig0_overview.png and .pdf

A: looped (middle block, r=4, full backprop) minus dense loss at equal parameters and tokens against model size,
   10M-160M, at 10N / 20N / 40N tokens (same data and logic as fig12 in make_figures.py).
B: the same gap against tokens per parameter, 10N-160N, for the 10M and 20M rungs, middle r=4 and r=8
   (same data and logic as fig11 in make_figures.py).
C: lookup-chain task: answer accuracy against chain length for dense models, the looped r=8 model at its training
   loop count and at 16 loops, the loops-equals-steps model, and a dense model writing a chain of thought
   (same data as figures 13-14 in synthetic/analyze.py).
Every line is a seed mean; bands are min-max over seeds. The plotted means are printed as a cross-check table.
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

RUNGS = ["10M", "20M", "40M", "80M", "160M"]
MULTS_B = [10, 20, 40, 80, 160]
D_MAX = 24


def load_many(paths: list[str]) -> pd.DataFrame:
    """Concatenate result CSVs, drop diverged runs (as load() in make_figures.py), add the tokens-per-parameter multiple."""
    df = pd.concat([pd.read_csv(p) for p in paths if os.path.exists(p)], ignore_index=True)
    df = df[~df.diverged.astype(str).str.lower().eq("true")].copy()
    df["mult"] = (df.budget_tokens / df.N_rung).round(0).astype(int)
    return df


def same_seed_gap(x: pd.DataFrame, placement: str, r: int, k_bwd: int, mult: int) -> pd.Series:
    """Looped minus dense val_fwe at the same rung (x is one rung), seed and tokens-per-parameter multiple, by seed."""
    loop = x[(x.placement == placement) & (x.r == r) & (x.k_bwd == k_bwd) & (x.mult == mult)].set_index("seed").val_fwe
    dense = x[(x.placement == "dense") & (x.mult == mult)].set_index("seed").val_fwe
    return (loop - dense).dropna()


def panel_a(ax, allr: pd.DataFrame) -> dict:
    """Middle r=4 minus dense against model size at 10N, 20N, 40N."""
    means = {}
    rungs = [r for r in RUNGS if (allr.rung == r).any()]
    for i, m in enumerate([10, 20, 40]):
        xs, ys, lo, hi = [], [], [], []
        for rung in rungs:
            x = allr[allr.rung == rung]
            gap = same_seed_gap(x, "middle", 4, 0, m)
            if len(gap):
                xs.append(float(x.N_rung.iloc[0])); ys.append(gap.mean()); lo.append(gap.min()); hi.append(gap.max())
                means[(m, rung)] = (gap.mean(), len(gap))
        ax.plot(xs, ys, marker="o", ms=4, color=f"C{i}", label=f"{m}N tokens")
        ax.fill_between(xs, lo, hi, color=f"C{i}", alpha=0.15)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log"); ax.minorticks_off()
    ax.set_xticks([float(allr[allr.rung == r].N_rung.iloc[0]) for r in rungs]); ax.set_xticklabels(rungs)
    ax.set_xlabel("model size (rung)"); ax.set_ylabel("looped minus dense loss\n(nats; < 0 = looped better)")
    ax.set_title("A. Equal parameters and tokens:\nthe advantage of looping grows with model size", fontsize=9)
    ax.legend(fontsize=7, loc="upper right", title="middle block looped 4 times", title_fontsize=7)
    ax.text(0.02, 0.03, "3 seeds at 10M–40M, 1 at 80M, 2 at 160M\nband = min–max over seeds", transform=ax.transAxes,
            fontsize=6.5, color="0.35", va="bottom")
    ax.grid(alpha=0.3)
    return means


def panel_b(ax, allr: pd.DataFrame) -> dict:
    """Middle r=4 and r=8 minus dense against tokens per parameter, 10M (solid) and 20M (dashed) rungs."""
    means = {}
    for rung, ls, mk in (("10M", "-", "o"), ("20M", "--", "s")):
        x = allr[allr.rung == rung]
        for r, color in ((4, "C0"), (8, "C1")):
            xs, ys, lo, hi = [], [], [], []
            for m in MULTS_B:
                gap = same_seed_gap(x, "middle", r, 0, m)
                if len(gap):
                    xs.append(m); ys.append(gap.mean()); lo.append(gap.min()); hi.append(gap.max())
                    means[(rung, r, m)] = (gap.mean(), len(gap))
            ax.plot(xs, ys, marker=mk, ms=4, ls=ls, color=color, label=f"{rung}, middle r={r}")
            ax.fill_between(xs, lo, hi, color=color, alpha=0.12)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log", base=2); ax.minorticks_off()
    ax.set_xticks(MULTS_B); ax.set_xticklabels([str(t) for t in MULTS_B])
    ax.set_xlabel("training tokens per parameter (D/N)"); ax.set_ylabel("looped minus dense loss (nats)")
    ax.set_title("B. …and with training tokens per parameter", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.text(0.02, 0.03, "3 seeds to 40 tokens/parameter, 2 beyond\nband = min–max over seeds", transform=ax.transAxes,
            fontsize=6.5, color="0.35", va="bottom")
    ax.grid(alpha=0.3)
    return means


def panel_c(ax, chain_dir: str, cot_glob: str) -> dict:
    """Lookup chains: accuracy against chain length for the six conditions of the overview."""
    reps = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(chain_dir, "*.json")))]
    cot = [json.load(open(p)) for p in sorted(glob.glob(cot_glob))]
    ds = np.arange(1, D_MAX + 1)
    # (name for the printed table, legend label, runs, key in final[d], line style, colour)
    lines = [("dense 8 layers", "dense, 8 layers", [r for r in reps if r["cond"] == "dense8"], "acc", "--", "C0"),
             ("dense 36 layers", "dense, 36 layers", [r for r in reps if r["cond"] == "dense36"], "acc", "--", "C1"),
             ("looped r=8 @ 8 loops", "looped r=8, 8 loops\n(as trained)", [r for r in reps if r["cond"] == "loop_r8"], "r=8", "-", "C2"),
             ("looped r=8 @ 16 loops", "looped r=8, 16 loops\nat inference", [r for r in reps if r["cond"] == "loop_r8"], "r=16", "-", "C3"),
             ("loop_step oracle (loops = d)", "loops = steps model,\nrun with loops = d", [r for r in reps if r["cond"] == "loop_step"], "oracle", "-", "C4"),
             ("dense8 + chain of thought", "dense, 8 layers\n+ chain of thought", cot, "acc", ":", "C5")]
    means, n_seeds = {}, set()
    for name, label, rs, key, ls, color in lines:
        a = np.array([[r["final"][str(d)][key] for d in ds] for r in rs])
        ax.plot(ds, a.mean(0), ls=ls, marker="o", ms=2.5, lw=1.3, color=color, label=label)
        ax.fill_between(ds, a.min(0), a.max(0), color=color, alpha=0.1)
        means[name] = a.mean(0); n_seeds.add(len(rs))
    ax.axvspan(0.5, 8.5, color="grey", alpha=0.08); ax.text(4.5, 1.02, "training range", ha="center", fontsize=7, color="0.35")
    ax.set_xlim(0.5, D_MAX + 0.5); ax.set_ylim(-0.02, 1.08); ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel("steps in the chain (d)"); ax.set_ylabel("answer accuracy")
    ax.set_title("C. Chain task: beyond the training range\nonly a loop run longer extends", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper right", handlelength=1.4, handletextpad=0.5, borderpad=0.35, borderaxespad=0.3,
              labelspacing=0.45)
    seeds = f"{n_seeds.pop()}" if len(n_seeds) == 1 else "/".join(str(n) for n in sorted(n_seeds))
    ax.text(0.02, 0.03, f"mean of {seeds} seeds;\nband = min–max", transform=ax.transAxes, fontsize=6.5, color="0.35", va="bottom")
    ax.grid(alpha=0.3)
    return means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/all_results.csv")
    ap.add_argument("--chain", default="results/synthetic/formal/chain")
    ap.add_argument("--cot", default="results/synthetic/supp2/formal/chain/dense8_cot_*.json")
    ap.add_argument("--out", default="results/figures/fig0_overview")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    scale = load_many([a.results, "results/ext_160m.csv", "results/ext_160m_dense.csv"])
    data = load_many([a.results, "results/ext_data.csv"])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ma = panel_a(axes[0], scale)
    mb = panel_b(axes[1], data)
    mc = panel_c(axes[2], a.chain, a.cot)
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=200); fig.savefig(a.out + ".pdf"); plt.close(fig)
    print("written", a.out + ".png", "and", a.out + ".pdf")

    print("\nCross-check: plotted seed means")
    print("Panel A, middle r=4 minus dense at 40N tokens (gap, seeds):")
    for rung in RUNGS:
        if (40, rung) in ma:
            print(f"  {rung:>5}: {ma[(40, rung)][0]:+.4f}  ({ma[(40, rung)][1]} seeds)")
    print("Panel B, middle r minus dense by tokens per parameter (gap, seeds):")
    for rung in ("10M", "20M"):
        for r in (4, 8):
            cells = [f"{m:>3}: {mb[(rung, r, m)][0]:+.4f} ({mb[(rung, r, m)][1]}s)" for m in MULTS_B if (rung, r, m) in mb]
            print(f"  {rung} r={r}:  " + "   ".join(cells))
    print("Panel C, answer accuracy (seed mean):")
    print(f"  {'line':<30} {'d=12':>6} {'d=16':>6}")
    for name, arr in mc.items():
        print(f"  {name:<30} {arr[11]:6.3f} {arr[15]:6.3f}")


if __name__ == "__main__":
    main()
