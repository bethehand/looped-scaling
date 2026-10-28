"""Figures of the paper, from the committed results in code/results/.

    code/.venv/bin/python paper/scripts/make_figures.py        # -> paper/figures/fig*.pdf (+ .png previews)

Every figure prints the numbers it plots, so that they can be checked against the text. Style: one categorical
palette (validated order; slots follow the entity across figures), an ordinal blue ramp for ordered series (model
size, token budget), hairline solid grid, legends plus selective direct labels, text in ink colours only.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "code" / "results"
OUT = ROOT / "paper" / "figures"

# ---------------------------------------------------------------- style
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4", green="#008300",
         violet="#6250d6", red="#e34948")
RAMP5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]          # blue 250..650, ordinal
RAMP3 = ["#86b6ef", "#2a78d6", "#104281"]
GRAY = {"mid": "#6f6e69", "dark": "#2f2e2b"}                              # dense controls (neutral)
# Entity colours, chosen so that every pair sharing a panel passes the all-pairs CVD and normal-vision checks
# (validate_palette.js --pairs all); orange clashes with red, magenta, yellow and green, so it is kept away from them.
ENT = {"r2": C["violet"], "r4": C["magenta"], "r8": C["blue"], "rand": C["green"], "step": C["aqua"],
       "step_extra": INK, "cot": C["orange"], "pause": C["violet"]}
FULL_W = 6.5

plt.rcParams.update({
    "font.family": "Arial", "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "font.size": 7.5, "axes.titlesize": 7.8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.5, "legend.title_fontsize": 6.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.size": 0, "ytick.minor.size": 0,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.titlelocation": "left", "axes.titlepad": 4,
    "lines.linewidth": 1.3, "lines.markersize": 3.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "lines.dash_capstyle": "round", "legend.frameon": False, "legend.handlelength": 1.8, "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.3, "legend.labelspacing": 0.35, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "figure.dpi": 150,
})
MK = dict(markeredgecolor="white", markeredgewidth=0.6)   # 'surface ring' on markers


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  -> {name}.pdf")


def zero_line(ax, y: float = 0.0) -> None:
    ax.axhline(y, color=INK2, lw=0.7, zorder=1)


def panel_title(ax, letter: str, text: str) -> None:
    ax.set_title(f"({letter}) {text}", fontsize=7.8, color=INK)


def end_label(ax, x, y, text, dx=4, dy=0, ha="left", color=INK2, size=6.5) -> None:
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=size,
                color=color)


def spread(ys: list[float], min_gap: float) -> list[float]:
    """Positions for labels at ys, moved apart to at least min_gap (data units), keeping their order."""
    order = np.argsort(ys)
    pos = np.array(ys, dtype=float)[order]
    for _ in range(50):
        moved = False
        for i in range(1, len(pos)):
            if pos[i] - pos[i - 1] < min_gap:
                c = (pos[i] + pos[i - 1]) / 2
                pos[i - 1], pos[i] = c - min_gap / 2, c + min_gap / 2
                moved = True
        if not moved:
            break
    out = np.empty_like(pos)
    out[order] = pos
    return list(out)


def legend_below(ax, ncol: int = 2, dy: float = -0.30, **kw):
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, dy), ncol=ncol, columnspacing=1.0, **kw)


# ---------------------------------------------------------------- language-model data
def load_lm(paths: list[str]) -> pd.DataFrame:
    """Concatenate result CSVs, drop diverged runs, add the tokens-per-parameter multiple (as make_overview.py)."""
    df = pd.concat([pd.read_csv(RES / p) for p in paths if (RES / p).exists()], ignore_index=True)
    df = df[~df.diverged.astype(str).str.lower().eq("true")].copy()
    df["mult"] = (df.budget_tokens / df.N_rung).round(0).astype(int)
    return df


def same_seed_gap(x: pd.DataFrame, placement: str, r: int, k_bwd: int, mult: int) -> pd.Series:
    """Looped minus dense main-set loss at one rung, by seed, at the same tokens per parameter."""
    loop = x[(x.placement == placement) & (x.r == r) & (x.k_bwd == k_bwd) & (x.mult == mult)]
    loop = loop[loop.get("r_sample", pd.Series("", index=loop.index)).fillna("").astype(str).eq("")]
    loop = loop.groupby("seed").val_fwe.mean()
    dense = x[(x.placement == "dense") & (x.mult == mult)].groupby("seed").val_fwe.mean()
    return (loop - dense).dropna()


RUNGS = ["10M", "20M", "40M", "80M", "160M"]


# ---------------------------------------------------------------- Figure 1: overview
def chain_runs(pattern: str) -> list[dict]:
    return [json.load(open(p)) for p in sorted(glob.glob(str(RES / "synthetic" / pattern)))]


def acc_curve(runs: list[dict], key: str, dmax: int = 24) -> np.ndarray:
    return np.array([[r["final"][str(d)][key] for d in range(1, dmax + 1)] for r in runs])


def fig1_overview() -> None:
    print("Figure 1 (overview)")
    scale = load_lm(["all_results.csv", "ext_160m.csv", "ext_160m_dense.csv"])
    data = load_lm(["all_results.csv", "ext_data.csv"])
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.55), constrained_layout=True)

    ax = axes[0]
    xs_all = {r: float(scale[scale.rung == r].N_rung.iloc[0]) for r in RUNGS}
    for color, m in zip(RAMP3, (10, 20, 40)):
        xs, ys, lo, hi = [], [], [], []
        for rung in RUNGS:
            g = same_seed_gap(scale[scale.rung == rung], "middle", 4, 0, m)
            if len(g):
                xs.append(xs_all[rung]); ys.append(g.mean()); lo.append(g.min()); hi.append(g.max())
        ax.fill_between(xs, lo, hi, color=color, alpha=0.12, lw=0)
        ax.plot(xs, ys, color=color, marker="o", **MK, label=f"{m}")
        end_label(ax, xs[-1], ys[-1], f"{m}N")
        print(f"   A  {m}N: " + "  ".join(f"{r}:{y:+.3f}" for r, y in zip(RUNGS, ys)))
    zero_line(ax)
    ax.set_xscale("log"); ax.set_xticks(list(xs_all.values())); ax.set_xticklabels(RUNGS); ax.minorticks_off()
    ax.set_xlim(xs_all["10M"] / 1.25, xs_all["160M"] * 1.9)
    ax.set_xlabel("model size"); ax.set_ylabel("looped $-$ dense loss (nats)")
    ax.legend(title="tokens per parameter", loc="upper right", ncol=3, columnspacing=0.8, handlelength=1.2)
    panel_title(ax, "a", "Text: advantage grows with size")

    ax = axes[1]
    for rung, ls, mk in (("10M", "-", "o"), ("20M", (0, (3, 1.6)), "s")):
        x = data[data.rung == rung]
        for r, color in ((4, ENT["r4"]), (8, ENT["r8"])):
            xs, ys, lo, hi = [], [], [], []
            for m in (10, 20, 40, 80, 160):
                g = same_seed_gap(x, "middle", r, 0, m)
                if len(g):
                    xs.append(m); ys.append(g.mean()); lo.append(g.min()); hi.append(g.max())
            ax.fill_between(xs, lo, hi, color=color, alpha=0.10, lw=0)
            ax.plot(xs, ys, color=color, ls=ls, marker=mk, **MK, label=f"{rung}, $r$={r}")
            print(f"   B  {rung} r={r}: " + "  ".join(f"{m}N:{y:+.3f}" for m, y in zip(xs, ys)))
    zero_line(ax)
    ax.set_xscale("log", base=2); ax.set_xticks([10, 20, 40, 80, 160]); ax.set_xticklabels(["10", "20", "40", "80", "160"])
    ax.minorticks_off(); ax.set_xlabel("training tokens per parameter")
    legend_below(ax, ncol=2, dy=-0.27, fontsize=6.2, handlelength=2.2)
    panel_title(ax, "b", "Text: and with data")

    ax = axes[2]
    form = chain_runs("formal/chain/*.json")
    by = lambda c: [r for r in form if r["cond"] == c]  # noqa: E731
    cot8 = chain_runs("supp2/formal/chain/dense8_cot_*.json")
    ds = np.arange(1, 25)
    ax.axvspan(0.5, 8.5, color="#f0efec", lw=0, zorder=0)
    ax.text(4.5, 1.045, "training range", ha="center", va="bottom", fontsize=6.3, color=INK2)
    lines = [("dense, 36 layers", by("dense36"), "acc", GRAY["dark"], (0, (3, 1.6)), None),
             ("chain of thought, 8 layers", cot8, "acc", ENT["cot"], "-", None),
             ("$r$=8, 8 loops", by("loop_r8"), "r=8", ENT["r8"], (0, (1, 1.2)), None),
             ("$r$=8, 16 loops", by("loop_r8"), "r=16", ENT["r8"], "-", None),
             ("loops = steps", by("loop_step"), "oracle", ENT["step"], "-", None)]
    for label, runs, key, color, ls, _ in lines:
        a = acc_curve(runs, key)
        ax.fill_between(ds, a.min(0), a.max(0), color=color, alpha=0.08, lw=0)
        ax.plot(ds, a.mean(0), color=color, ls=ls, label=label)
        print(f"   C  {label:<36} d12 {a.mean(0)[11]:.3f}  d16 {a.mean(0)[15]:.3f}  (n={len(runs)})")
    ax.set_xlim(0.5, 24.5); ax.set_ylim(-0.02, 1.12); ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel("steps in the chain ($d$)"); ax.set_ylabel("accuracy")
    legend_below(ax, ncol=2, dy=-0.27, fontsize=6.2, handlelength=2.0)
    panel_title(ax, "c", "Serial task: extension with loops")
    save(fig, "fig1_overview")



# ---------------------------------------------------------------- Figure 2: schematic of the models and the chain task
def fig2_schematic() -> None:
    """(a) the four kinds of language model and the S10 control, as stacks of layers; (b) one chain-task problem and
    how the looped model, a chain of thought, pause tokens and a dense model of equal depth process it."""
    print("Figure 2 (schematic)")
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
    fig = plt.figure(figsize=(FULL_W, 3.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.85])
    LOOP, DENSE = ENT["r4"], GRAY["mid"]
    FILL_LOOP, FILL_DENSE, FILL_OFF = "#fbe4ee", "#e9e8e3", "#f6f5f1"

    # ---- (a) models
    ax = fig.add_subplot(gs[0]); ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(-36, 64)
    unit, base = 3.0, 0.0                                   # one layer = 3 units of height; stacks start at y = 0

    def stack(x, w, parts):
        """parts: list of (n_layers, kind) from bottom to top; kind in dense / loop / off. Returns top y."""
        y = base
        for n, kind in parts:
            fill = {"dense": FILL_DENSE, "loop": FILL_LOOP, "off": FILL_OFF}[kind]
            edge = DENSE if kind == "dense" else LOOP
            ax.add_patch(Rectangle((x - w / 2, y), w, n * unit, facecolor=fill, edgecolor=edge, lw=0.8, zorder=2))
            ax.text(x, y + n * unit / 2, f"{n} layer{'s' if n > 1 else ''}", ha="center", va="center", fontsize=6.3,
                    color=INK, zorder=3)
            y += n * unit
        return y

    def loop_arrow(x, w, y0, y1, label, dy=0):
        """A return arrow on the right of the block from y1 (top) back to y0 (bottom), labelled."""
        xr = x + w / 2
        ax.add_patch(FancyArrowPatch((xr + 0.6, y1 - 0.8), (xr + 0.6, y0 + 0.8), connectionstyle="arc3,rad=-0.9",
                                     arrowstyle="-|>", mutation_scale=7, lw=0.9, color=LOOP, zorder=4))
        ax.text(xr + 4.6, (y0 + y1) / 2 + dy, label, ha="left", va="center", fontsize=6.3, color=LOOP)

    def caption_below(x, title, executes):
        ax.text(x, -6, title, ha="center", va="top", fontsize=6.8, color=INK, linespacing=1.15)
        ax.text(x, -24, executes, ha="center", va="top", fontsize=6.3, color=INK2, linespacing=1.15)

    w = 11
    # dense, 8 layers
    stack(8, w, [(8, "dense")])
    caption_below(8, "Dense", "executes 8 layers")
    # middle block, r loops
    x = 27; y = stack(x, w, [(2, "dense"), (4, "loop"), (2, "dense")])
    loop_arrow(x, w, base + 2 * unit, base + 6 * unit, "$\\times r$,\ninjection")
    caption_below(x, "Middle block\nlooped $r$ times", "executes $2 + 4r + 2$ layers\n(20 at $r = 4$)")
    # whole stack
    x = 49; y = stack(x, w, [(8, "loop")])
    loop_arrow(x, w, base, base + 8 * unit, "$\\times r$,\ninjection")
    caption_below(x, "Whole stack\nlooped $r$ times", "executes $8r$ layers")
    # truncated backprop (middle block)
    x = 72; y = stack(x, w, [(2, "dense"), (4, "loop"), (2, "dense")])
    loop_arrow(x, w, base + 2 * unit, base + 6 * unit, "$\\times r$,\ninjection")
    caption_below(x, "Middle block, $r$ loops,\ntruncated backprop.", "executes $2 + 4r + 2$ layers;\ngradient: last $r/2$ loops")
    # S10 control: dense, 18 layers, narrower
    x = 93; stack(x, 7.5, [(18, "dense")])
    caption_below(x, "Dense, 18 layers (S10)", "same parameters as\nthe looped model, $r = 4$")
    ax.text(0.5, 63, "(a) The language models: the dense reference, three kinds of loop and the dense control of S10",
            ha="left", va="top", fontsize=7.8, color=INK)

    # ---- (b) the chain task
    ax = fig.add_subplot(gs[1]); ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.text(0.5, 99, "(b) The chain task: one problem with $d = 3$ steps, and how each model answers it",
            ha="left", va="top", fontsize=7.8, color=INK)

    def token(x, y, text, w, fill=FILL_DENSE, edge=DENSE, h=11, fs=6.3, color=INK, bold=False):
        ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=1.2",
                                    facecolor=fill, edgecolor=edge, lw=0.7, zorder=2))
        ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=fs, color=color, zorder=3,
                fontweight="bold" if bold else "normal")
        return x + w + 1.2

    def arrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=6, lw=0.8, color=INK2,
                                     zorder=1))

    # the prompt
    y = 82
    ax.text(0.5, y, "prompt", ha="left", va="center", fontsize=6.5, color=INK2)
    x = 9.5
    for t, ww in (("BOS", 5.2), ("25", 4.2), (";", 2.4), ("LUT", 5), ("7", 3.2), (";", 2.4), ("LUT", 5), ("3", 3.2), (";", 2.4),
                  ("LUT", 5), ("19", 3.6), (";", 2.4), ("=>", 4.2)):
        x = token(x, y, t, ww, fill="white", edge=AXIS, fs=5.9)
    ax.text(x + 1, y, "$3d + 4 = 13$ tokens;  $x_1 = T_7(25)$,  $x_2 = T_3(x_1)$,  answer $= T_{19}(x_2)$",
            ha="left", va="center", fontsize=6.1, color=INK2)

    rows = [
        ("Looped, answer only", "loop"),
        ("Chain of thought (dense)", "cot"),
        ("Pause tokens (dense)", "pause"),
        ("Dense, equal depth", "deep"),
    ]
    ys = [60, 42, 25, 8]
    for (label, kind), yy in zip(rows, ys):
        ax.text(0.5, yy, label, ha="left", va="center", fontsize=6.6, color=INK)
        x = 24
        x = token(x, yy, "prompt", 10, fill="white", edge=AXIS)
        if kind == "loop":
            x = token(x, yy, "core run $r$ times, here $r = d$:  loop 1 $\\rightarrow x_1$,  loop 2 $\\rightarrow x_2$,  loop 3 $\\rightarrow$ answer", 50,
                      fill=FILL_LOOP, edge=LOOP, fs=6.0)
            arrow(x - 0.6, x + 2.4, yy); x += 3
            x = token(x, yy, "answer", 9, fill=FILL_LOOP, edge=LOOP, bold=True)
            ax.text(x + 1, yy, "more loops at inference\n$\\rightarrow$ longer chains", ha="left", va="center",
                    fontsize=6.0, color=LOOP, linespacing=1.1)
        elif kind == "cot":
            x = token(x, yy, "8 or 36 layers", 16)
            arrow(x - 0.6, x + 2.4, yy); x += 3
            for t in ("$x_1$", "$x_2$"):
                x = token(x, yy, t, 4.6, fill="white", edge=DENSE)
            x = token(x, yy, "answer", 9, bold=True)
            ax.text(x + 1, yy, "writes the chain out: $d$ tokens,\nthe last of which is the answer", ha="left",
                    va="center", fontsize=6.0, color=INK2, linespacing=1.1)
        elif kind == "pause":
            for _ in range(3):
                x = token(x, yy, "pause", 6.5, fill="white", edge=DENSE, fs=5.8, color=INK2)
            x = token(x, yy, "8 layers", 11)
            arrow(x - 0.6, x + 2.4, yy); x += 3
            x = token(x, yy, "answer", 9, bold=True)
            ax.text(x + 1, yy, "$d$ pause tokens before the answer", ha="left", va="center", fontsize=6.0, color=INK2)
        else:
            x = token(x, yy, "20 or 36 layers, once", 24)
            arrow(x - 0.6, x + 2.4, yy); x += 3
            x = token(x, yy, "answer", 9, bold=True)
            ax.text(x + 1, yy, "the executed depth of 4 or 8 loops,\n2.4--4.4$\\times$ the parameters", ha="left",
                    va="center", fontsize=6.0, color=INK2, linespacing=1.1)
    save(fig, "fig2_schematic")


# ---------------------------------------------------------------- Figure 2: ruler-free multiplier
def fig2_multiplier() -> None:
    print("Figure 2 (ruler-free multiplier, local_phi_s4.csv)")
    lp = pd.read_csv(RES / "supp2" / "local_phi_s4.csv")
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.35), sharey=True, constrained_layout=True)
    for ax, r, letter in zip(axes, (4, 8), "ab"):
        sub = lp[(lp.cell == "middle/full") & (lp.r == r)]
        for color, rung in zip(RAMP5, RUNGS):
            s = sub[(sub.rung == rung) & sub.M.notna()].sort_values("tpp")
            if not len(s):
                continue
            ax.plot(s.tpp, s.M, color=color, lw=1.2, zorder=2, label=rung)
            for _, row in s.iterrows():
                filled = row.status == "interpolated"
                ax.errorbar(row.tpp, row.M, yerr=[[row.M - row.M_lo], [row.M_hi - row.M]], color=color, lw=0.8,
                            capsize=0, zorder=3)
                ax.plot(row.tpp, row.M, marker="o", ms=4.2, color=color, mfc=color if filled else "white",
                        mec=color if not filled else "white", mew=0.9 if not filled else 0.6, zorder=4, ls="none")
            dy = {("160M", 4): 5, ("80M", 4): -5}.get((rung, r), 0)
            end_label(ax, s.tpp.iloc[-1], s.M.iloc[-1], rung, dx=5, dy=dy)
            print(f"   r={r} {rung:>4}: " + "  ".join(f"{int(t)}N:{m:.2f}{'' if st == 'interpolated' else '*'}"
                                                       for t, m, st in zip(s.tpp, s.M, s.status)))
        if r == 4:   # S10: dense models with 18 layers and the looped model's number of parameters
            s10 = pd.read_csv(RES / "supp3" / "s10_table.csv")
            for rung in ("20M", "40M"):
                color, s = RAMP5[RUNGS.index(rung)], s10[s10.rung == rung].sort_values("tpp")
                x = s.tpp * 1.05
                ax.plot(x, s.M_deep, color=color, lw=0.9, ls=(0, (3, 2)), zorder=2)
                ax.errorbar(x, s.M_deep, yerr=[s.M_deep - s.M_deep_lo, s.M_deep_hi - s.M_deep], color=color, lw=0.8,
                            capsize=0, ls="none", zorder=3)
                ax.plot(x, s.M_deep, marker="D", ms=3.6, color=color, ls="none", zorder=4, **MK)
                print(f"   S10 dense 18 layers {rung:>4}: " + "  ".join(f"{int(t)}N:{m:.2f}" for t, m in zip(s.tpp, s.M_deep)))
        ax.axhline(1.0, color=INK2, lw=0.7, zorder=1)
        if r == 8:
            ax.text(200, 0.985, "worth its parameters", fontsize=6.2, color=INK2, va="top", ha="right")
        ax.set_xscale("log", base=2); ax.set_xticks([10, 20, 40, 80, 160])
        ax.set_xticklabels(["10", "20", "40", "80", "160"]); ax.minorticks_off(); ax.set_xlim(8.5, 230)
        ax.set_xlabel("training tokens per parameter")
        panel_title(ax, letter, f"Middle block looped {r} times")
    axes[0].set_ylabel("multiplier $M = N^\\star / N$")
    axes[0].set_ylim(0.5, 1.6)
    h = [plt.Line2D([], [], color=INK2, marker="o", ms=4.2, ls="none", mec="white", mew=0.6, label="interpolated"),
         plt.Line2D([], [], color=INK2, marker="o", ms=4.2, ls="none", mfc="white", mew=0.9, label="beyond the dense sizes"),
         plt.Line2D([], [], color=INK2, marker="D", ms=3.6, lw=0.9, ls=(0, (3, 2)), mec="white", mew=0.6,
                    label="dense, 18 layers (S10)")]
    axes[0].legend(handles=h, loc="lower right")
    save(fig, "fig2_multiplier")


# ---------------------------------------------------------------- Figure 3: equal compute
def fig3_equal_compute() -> None:
    print("Figure 3 (equal training compute, iso_flop_bounds_s4.csv)")
    b = pd.read_csv(RES / "supp2" / "iso_flop_bounds_s4.csv")
    s4 = json.load(open(RES / "supp2" / "s4_readings.json"))["tuned_looped_vs_dense"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.35), sharey=True, constrained_layout=True)
    for ax, cell, letter, rungs in ((axes[0], "middle_r4", "a", RUNGS), (axes[1], "middle_r8", "b", RUNGS[:4])):
        labels = []
        for color, rung in zip(RAMP5, rungs):
            s = b[(b.cell == cell) & (b.rung == rung)]
            if not len(s):
                continue
            g = s.groupby("mult").agg(last=("gap_last_slope", "mean"), pl=("gap_power_law", "mean"),
                                      ext=("extrapolated", "any"), lo=("gap_last_slope", "min"), hi=("gap_last_slope", "max"))
            mid = (g["last"] + g["pl"]) / 2
            ax.plot(g.index, mid, color=color, lw=1.2, zorder=2)
            for m, row in g.iterrows():
                if row.ext:
                    ax.vlines(m, row.pl, row["last"], color=color, lw=3.0, alpha=0.45, zorder=3)
                    ax.plot(m, (row["last"] + row.pl) / 2, marker="o", ms=4.2, mfc="white", mec=color, mew=0.9, ls="none", zorder=4)
                else:
                    ax.plot(m, row["last"], marker="o", ms=4.2, color=color, ls="none", zorder=4, **MK)
            labels.append((rung, mid.iloc[0]))
            print(f"   {cell} {rung:>4}: " + "  ".join(
                f"{int(m)}N:{row['last']:+.3f}" + (f"[{row.pl:+.3f}]" if row.ext else "") for m, row in g.iterrows()))
        for (rung, y0), yl in zip(labels, spread([y for _, y in labels], 0.024)):
            ax.annotate(rung, (10, y0), xytext=(10 / 1.13, yl), textcoords="data", ha="right", va="center",
                        fontsize=6.5, color=INK2)
        zero_line(ax)
        ax.set_xscale("log", base=2); ax.set_xticks([10, 20, 40, 80, 160])
        ax.set_xticklabels(["10", "20", "40", "80", "160"]); ax.minorticks_off(); ax.set_xlim(6.3, 200)
        ax.set_xlabel("looped model's training tokens per parameter")
        panel_title(ax, letter, f"Middle block looped {cell[-1]} times")
    # the looped model at 0.35x the dense rate, the lowest rate S1 tried (S4): 10M, middle r=4
    pts = {80: [], 160: []}
    for k, v in s4.items():
        pts[80 if "80 tokens" in k else 160].append(v["gap"])
    xs = [80, 160]; ys = [np.mean(pts[m]) for m in xs]
    axes[0].errorbar(xs, ys, yerr=[[ys[i] - min(pts[m]) for i, m in enumerate(xs)], [max(pts[m]) - ys[i] for i, m in enumerate(xs)]],
                     color=INK, lw=0.9, capsize=0, ls="none", zorder=5)
    axes[0].plot(xs, ys, marker="D", ms=4.0, color=INK, ls=(0, (1, 1.2)), lw=1.0, zorder=6, **MK)
    print("   tuned 10M r=4 (0.35x): " + "  ".join(f"{m}N:{y:+.3f} (seeds {pts[m]})" for m, y in zip(xs, ys)))
    axes[0].set_ylabel("looped $-$ dense loss at equal FLOPs (nats)")
    axes[0].set_ylim(-0.04, 0.42)
    h = [plt.Line2D([], [], color=INK2, marker="o", ms=4.2, ls="none", mec="white", mew=0.6, label="dense loss measured"),
         plt.Line2D([], [], color=INK2, marker="o", ms=4.2, ls="none", mfc="white", mew=0.9, label="extrapolated (two bounds)"),
         plt.Line2D([], [], color=INK, marker="D", ms=4.0, ls="none", mec="white", mew=0.6, label="10M, looped, 0.35× rate")]
    axes[0].legend(handles=h, loc="upper right")
    save(fig, "fig3_equal_compute")


# ---------------------------------------------------------------- Figure 4: what the block computes on text
def probe(run: str) -> dict:
    return json.load(open(RES / "loop_probe" / f"{run}__branch40.json"))


def fig4_text_mechanism() -> None:
    print("Figure 4 (text: per-loop probes, depth curves)")
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.6), constrained_layout=True)
    loops = np.arange(1, 17)

    ax = axes[0]
    reps = [probe(f"40M_middle_r8_full_s{s}") for s in (42, 43, 44)]
    series = [("run with $k$ loops", "lens", C["blue"], "o"), ("learned map + output layers", "adapter", C["orange"], "s"),
              ("learned linear head", "linear", C["aqua"], "^")]
    for label, key, color, mk in series:
        a = np.array([rp["loss"][key] for rp in reps])
        ax.plot(loops, a.mean(0), color=color, marker=mk, ms=3.0, **MK, label=label)
        print(f"   a  {key:<8} " + " ".join(f"{v:.2f}" for v in a.mean(0)[:9]) + " ...")
    ax.axvline(8, color=INK2, lw=0.7, zorder=1)
    ax.set_ylim(3.1, 6.0); ax.set_xticks([1, 4, 8, 12, 16]); ax.set_xlabel("loop $k$")
    ax.set_ylabel("main-set loss (nats)")
    legend_below(ax, ncol=1, dy=-0.25, fontsize=6.2)
    panel_title(ax, "a", "Fixed $r$=8 (40M)")

    ax = axes[1]
    rp = probe("80M_whole_r8_k4_s42")
    ax.plot(loops, rp["loss"]["lens"], color=C["blue"], marker="o", ms=3.0, **MK, label="run with $k$ loops")
    ax.plot(loops, rp["loss"]["adapter"], color=C["orange"], marker="s", ms=3.0, **MK, label="learned map + output layers")
    for k in (4, 8, 12, 16):
        ax.axvline(k, color=GRID, lw=0.9, zorder=0)
    ax.axvline(8, color=INK2, lw=0.7, zorder=1)
    print("   b  lens " + " ".join(f"{v:.2f}" for v in rp["loss"]["lens"]))
    ax.set_xticks([1, 4, 8, 12, 16]); ax.set_xlabel("loop $k$")
    legend_below(ax, ncol=1, dy=-0.25, fontsize=6.2)
    panel_title(ax, "b", "Truncated backprop (80M)")

    ax = axes[2]
    dc = pd.read_csv(RES / "token_analysis" / "depth_curve_randr.csv")
    ml = pd.read_csv(RES / "token_analysis" / "mean_loss.csv")
    dense = ml[(ml.rung == "20M") & ml.cell.str.startswith("dense") & (ml.valset == "fwe") & (ml.r_eval == 1)].mean_loss
    for run, label, color, ls in (("20M_middle_r8_full_s42", "fixed $r$=8", ENT["r8"], "-"),
                                  ("20M_middle_r8_full_rs1to8_s42", "random $r\\in$[1,8]", ENT["rand"], "-"),
                                  ("20M_middle_r8_k4_rs1to8_s42", "random, truncated", ENT["rand"], (0, (3, 1.6)))):
        rp = probe(run)
        xs, ys = list(loops), list(rp["loss"]["lens"])
        e = dc[(dc.run == run) & (dc.valset == "fwe") & (dc.r_eval > 16)].sort_values("r_eval")
        xs += e.r_eval.tolist(); ys += e.mean_loss.tolist()
        every = [i for i, x in enumerate(xs) if x in (1, 2, 4, 8, 16, 24, 32)]
        ax.plot(xs, ys, color=color, ls=ls, marker="o", ms=3.0, markevery=every, **MK, label=label)
        print(f"   c  {label:<18} " + " ".join(f"{x}:{y:.3f}" for x, y in zip(xs, ys) if x in (1, 2, 4, 8, 16, 24, 32)))
    ax.axhline(dense.mean(), color=GRAY["mid"], lw=0.9, ls=(0, (3, 1.6)), label="dense 20M")
    print(f"   c  dense 20M (seeds {len(dense)}) {dense.mean():.3f}")
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16, 32]); ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax.minorticks_off(); ax.set_ylim(3.35, 4.6); ax.set_xlabel("loops at inference")
    legend_below(ax, ncol=2, dy=-0.25, fontsize=6.2)
    panel_title(ax, "c", "Random loop count (20M)")
    save(fig, "fig4_text_mechanism")


# ---------------------------------------------------------------- Figure 5: the chain task
def fig5_chain() -> None:
    print("Figure 5 (chain task: extension)")
    form = chain_runs("formal/chain/*.json")
    by = lambda c: [r for r in form if r["cond"] == c]  # noqa: E731
    s2 = chain_runs("supp2/formal/chain/*.json")
    fmt = lambda f, c: [r for r in s2 if r.get("format") == f and r["cond"] == c]  # noqa: E731
    ds = np.arange(1, 25)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.75), sharey=True, constrained_layout=True)

    def draw(ax, lines, letter, title):
        ax.axvspan(0.5, 8.5, color="#f0efec", lw=0, zorder=0)
        for label, runs, key, color, ls in lines:
            a = acc_curve(runs, key)
            ax.fill_between(ds, a.min(0), a.max(0), color=color, alpha=0.08, lw=0)
            ax.plot(ds, a.mean(0), color=color, ls=ls, label=label)
            print(f"   {letter}  {label:<30} d8 {a.mean(0)[7]:.3f} d12 {a.mean(0)[11]:.3f} d16 {a.mean(0)[15]:.3f} (n={len(runs)})")
        ax.set_xlim(0.5, 24.5); ax.set_ylim(-0.02, 1.08); ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
        ax.set_xlabel("steps in the chain ($d$)")
        legend_below(ax, ncol=1, dy=-0.25, fontsize=6.2, handlelength=2.2)
        panel_title(ax, letter, title)

    dash, dot = (0, (3, 1.6)), (0, (1, 1.2))
    draw(axes[0], [("dense, 8 layers (6.5M)", by("dense8"), "acc", GRAY["mid"], dot),
                   ("dense, 20 layers (16.1M)", by("dense20"), "acc", GRAY["mid"], dash),
                   ("dense, 36 layers (29.0M)", by("dense36"), "acc", GRAY["dark"], dash),
                   ("looped $r$=4, 20 layers", by("loop_r4"), "r=4", ENT["r4"], "-"),
                   ("looped $r$=8, 36 layers", by("loop_r8"), "r=8", ENT["r8"], "-")],
         "a", "Training loop count")
    draw(axes[1], [("$r$=8, 8 loops", by("loop_r8"), "r=8", ENT["r8"], dot),
                   ("$r$=8, 16 loops", by("loop_r8"), "r=16", ENT["r8"], "-"),
                   ("random, 8 loops", by("loop_rand"), "r=8", ENT["rand"], dot),
                   ("random, 16 loops", by("loop_rand"), "r=16", ENT["rand"], "-"),
                   ("loops = steps, $d$ loops", by("loop_step"), "oracle", ENT["step"], "-")],
         "b", "More loops at inference")
    draw(axes[2], [("chain of thought, 8 layers", fmt("cot", "dense8"), "acc", ENT["cot"], "-"),
                   ("chain of thought, 36 layers", fmt("cot", "dense36"), "acc", ENT["cot"], dash),
                   ("pause tokens, 8 layers", fmt("pause", "dense8"), "acc", ENT["pause"], "-"),
                   ("loops = steps, $d$ loops", by("loop_step"), "oracle", ENT["step"], "-")],
         "c", "Writing the steps out")
    axes[0].set_ylabel("accuracy")
    save(fig, "fig5_chain")


# ---------------------------------------------------------------- Figure 6: probes on the chain task
def parse_first(s: str) -> list[float]:
    return [float("nan") if t.strip() == "nan" else float(t) for t in s.strip("[]").split(",")]


def fig6_chain_probes() -> None:
    print("Figure 6 (chain task: probes, formal_probe_frontier.csv)")
    fr = pd.read_csv(RES / "synthetic" / "formal_probe_frontier.csv")
    steps = np.arange(2, 9)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.3), constrained_layout=True)

    def looped(ax, specs):
        ax.plot(steps, steps, color=INK2, lw=0.8, ls=(0, (1, 1.2)), label="one step per loop", zorder=1)
        for cond, label, color, mk, open_ in specs:
            rows = fr[fr.cond == cond]
            a = np.array([parse_first(v) for v in rows["first"]])
            mean = np.nanmean(a, 0)
            ax.fill_between(steps, np.nanmin(a, 0), np.nanmax(a, 0), color=color, alpha=0.10, lw=0)
            kw = dict(mfc="none", mec=color, mew=1.0, ms=6.0, ls=(0, (3, 1.6)), lw=1.5) if open_ else dict(ms=3.6, **MK)
            ax.plot(steps, mean, color=color, marker=mk, label=label, **kw)
            print(f"   {cond:<10} first loop: " + " ".join(f"{v:.1f}" for v in mean)
                  + f" | persistence own {rows.persistence_own.mean():.2f}, diag share {rows.diag_share.mean():.2f}")
        ax.set_xlabel("intermediate value $x_j$"); ax.set_xticks(steps); ax.set_ylim(0.6, 8.6); ax.set_yticks(range(1, 9))
        ax.legend(loc="upper left", fontsize=6.2)

    looped(axes[0], [("loop_step", "loops = steps", ENT["step"], "s", True), ("loop_r8", "$r$=8", ENT["r8"], "o", False)])
    axes[0].set_ylabel("first loop where $x_j$ is decodable")
    panel_title(axes[0], "a", "One step per loop")
    looped(axes[1], [("loop_r2", "$r$=2", ENT["r2"], "o", False), ("loop_r4", "$r$=4", ENT["r4"], "o", False),
                     ("loop_rand", "random", ENT["rand"], "o", False)])
    panel_title(axes[1], "b", "Several steps per loop")
    ax = axes[2]
    for cond, label, color in (("dense20", "dense, 20 layers", GRAY["mid"]), ("dense36", "dense, 36 layers", GRAY["dark"])):
        rows = fr[fr.cond == cond]
        first = True
        for v in rows["first"]:
            v = np.array(parse_first(v))
            if np.isfinite(v).sum() == 0:
                continue
            ax.plot(steps, v, color=color, marker="o", ms=2.8, lw=1.0, alpha=0.9, label=label if first else None, **MK)
            first = False
        n_none = sum(np.isfinite(np.array(parse_first(v))).sum() == 0 for v in rows["first"])
        print(f"   {cond}: {len(rows)} seeds, {n_none} with no decodable value")
    ax.set_xlabel("intermediate value $x_j$"); ax.set_ylabel("first layer where $x_j$ is decodable")
    ax.set_xticks(steps)
    ax.legend(loc="upper left", fontsize=6.2)
    panel_title(ax, "c", "Dense models (per seed)")
    save(fig, "fig6_chain_probes")


# ---------------------------------------------------------------- Figure 7: stopping by confidence
def halting(cond: str, supp: bool = False) -> list[dict]:
    d = "halting_supp2" if supp else "halting"
    return [json.load(open(p)) for p in sorted(glob.glob(str(RES / "synthetic" / d / f"{cond}_lr*.json")))]


def fig7_stopping() -> None:
    print("Figure 7 (stopping: confidence >= 0.99, reporting set)")
    ds = np.arange(1, 25)
    models = [("$r$=8", halting("loop_r8"), ENT["r8"], "o"), ("random", halting("loop_rand"), ENT["rand"], "o"),
              ("loops = steps", halting("loop_step"), ENT["step"], "s"),
              ("loops = steps + 0–4 extra", halting("loop_step_extra", supp=True), ENT["step_extra"], "D")]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.55), layout="constrained")
    for label, runs, color, mk in models:
        acc = np.array([[r["results"]["report"][str(d)]["conf_0.99"]["acc"] for d in ds] for r in runs])
        loops = np.array([[r["results"]["report"][str(d)]["conf_0.99"]["loops"] for d in ds] for r in runs])
        own = np.array([[r["results"]["report"][str(d)]["own"]["acc"] for d in ds] for r in runs])
        axes[0].plot(ds, acc.mean(0), color=color, marker=mk, ms=2.8, **MK, label=label)
        axes[1].plot(ds, loops.mean(0), color=color, marker=mk, ms=2.8, **MK, label=label)
        print(f"   {label:<26} n={len(runs)}  conf acc d1-8 {acc.mean(0)[:8].mean():.3f} d12 {acc.mean(0)[11]:.3f} "
              f"d16 {acc.mean(0)[15]:.3f} | loops d4 {loops.mean(0)[3]:.1f} d12 {loops.mean(0)[11]:.1f} | own d12 {own.mean(0)[11]:.3f}")
    for ax in axes:
        ax.axvspan(0.5, 8.5, color="#f0efec", lw=0, zorder=0)
        ax.set_xlim(0.5, 24.5); ax.set_xticks([1, 4, 8, 12, 16, 20, 24]); ax.set_xlabel("steps in the chain ($d$)")
    axes[1].plot(ds, ds, color=INK2, lw=0.8, ls=(0, (1, 1.2)), zorder=1)
    axes[1].text(23.6, 20.6, "loops = $d$", ha="right", va="top", fontsize=6.3, color=INK2)
    axes[0].set_ylim(-0.02, 1.05); axes[0].set_ylabel("accuracy")
    axes[1].set_ylim(0, 33); axes[1].set_ylabel("loops used (mean)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4, fontsize=6.4, columnspacing=1.4)
    panel_title(axes[0], "a", "Accuracy when stopping at confidence 0.99")
    panel_title(axes[1], "b", "Loops the stop uses")
    save(fig, "fig7_stopping")


# ---------------------------------------------------------------- Figure 8: deployment cost
def fig8_deployment() -> None:
    print("Figure 8 (deployment: S3 equal loss, S6 KV sharing)")
    eq = pd.read_csv(RES / "supp" / "s3_equal_loss.csv")
    s = eq[(eq.placement == "middle") & (eq.r == 4) & (eq.width >= 448)].sort_values("width")
    sizes = {448: "20M", 640: "40M", 896: "80M", 1280: "160M"}
    x = np.arange(len(s))
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    for label, col, color, mk in (("decoding time per token", "decode_ms_graph_b1_ratio", C["blue"], "o"),
                                  ("KV cache per token", "kv_kb_per_token_ratio", C["orange"], "s"),
                                  ("peak memory, batch 32", "peak_mb_b32_ratio", C["aqua"], "^"),
                                  ("weight memory", "weight_mb_ratio", INK, "D")):
        y = s[col].to_numpy()
        ax.plot(x, y, color=color, marker=mk, ms=3.6, **MK, label=label)
        ext = s.extrapolated.to_numpy()
        ax.plot(x[ext], y[ext], marker=mk, ms=3.6, ls="none", mfc="white", mec=color, mew=0.9)
        print(f"   a  {label:<26} " + " ".join(f"{sizes[w]}:{v:.2f}" for w, v in zip(s.width, y)))
    ax.axhline(1.0, color=INK2, lw=0.7, zorder=1)
    ax.text(-0.15, 1.04, "dense model of equal loss", fontsize=6.2, color=INK2, va="bottom")
    ax.set_xticks(x); ax.set_xticklabels([sizes[w] for w in s.width]); ax.set_xlim(-0.3, len(s) - 0.7)
    ax.set_ylim(0.6, 2.7); ax.set_ylabel("looped / dense"); ax.set_xlabel("model size")
    legend_below(ax, ncol=2, dy=-0.27, fontsize=6.2)
    panel_title(ax, "a", "Equal loss, middle block $r$=4")

    ax = axes[1]
    kv = json.load(open(RES / "supp2" / "kv_share.json"))
    rows = [("20M_middle_r8_full_s42", "$r$=8, full backprop"), ("20M_middle_r8_k4_s42", "$r$=8, truncated"),
            ("20M_middle_r4_full_s42", "$r$=4"), ("20M_middle_r4_full_lrs0.5_s42", "$r$=4, half the rate"),
            ("20M_middle_r8_full_rs1to8_s42", "random loop count"), ("20M_middle_r8_k4_rs1to8_s42", "random, truncated")]
    y = np.arange(len(rows))[::-1]
    ax.axvspan(0.03, 0.05, color="#f0efec", lw=0, zorder=0)
    ax.text(0.04, len(rows) - 0.35, "gain of fixed-loop\nmodels over dense", ha="center", va="bottom", fontsize=6.0, color=INK2)
    for yi, (run, label) in zip(y, rows):
        d = kv[run]["delta_shared_minus_full"]
        ax.barh(yi, d, height=0.42, color=C["blue"], zorder=2)
        txt = "0.000" if abs(d) < 5e-4 else f"{d:+.3f}"
        ax.text(max(d, 0) + 0.004, yi, txt, va="center", fontsize=6.3, color=INK2)
        print(f"   b  {label:<22} delta {d:+.4f}")
    ax.set_yticks(y); ax.set_yticklabels([lab for _, lab in rows]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 0.2); ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20]); ax.set_xticklabels(["0", "0.05", "0.10", "0.15", "0.20"])
    ax.set_ylim(-0.6, len(rows) + 0.35); ax.grid(axis="y", visible=False)
    ax.set_xlabel("loss increase when loops share the KV cache (nats)")
    panel_title(ax, "b", "Sharing the cache across loops (20M)")
    save(fig, "fig8_deployment")


if __name__ == "__main__":
    fig1_overview()
    fig2_schematic()
    fig2_multiplier()
    fig3_equal_compute()
    fig4_text_mechanism()
    fig5_chain()
    fig6_chain_probes()
    fig7_stopping()
    fig8_deployment()
