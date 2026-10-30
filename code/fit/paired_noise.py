"""How much the looped-minus-dense gap of runs with the same seed varies across seeds, against how much single runs
vary. Runs with the same seed and width start from the same weights in the layers they share, and every run reads the
same data in the same order; the question is whether this pairing makes small gaps resolvable. Grid runs at
10M--40M (three seeds), equal-token accounting, end-of-cooldown loss on the main validation set, diverged rows dropped
as in the fits. Written after the results, on 2026-10-30 (deviation log): the paper had said that a difference of
0.01 nats between runs that share a seed is resolvable, which this check does not support at 10M--20M.

    python -m fit.paired_noise   -> results/fit/paired_noise.json
"""
from __future__ import annotations

import json

import pandas as pd

SEEDS = (42, 43, 44)
BUDGETS = (10, 20, 40)


def spread(x):
    return float(max(x) - min(x))


def main():
    df = pd.read_csv("results/all_results.csv")
    df = df[(df.accounting == "iso_token") & ~df.diverged.astype(str).str.lower().eq("true")].copy()
    df["budget"] = df.budget_mult.round().astype(int)
    loss = {(r.rung, r.placement, r.backprop, int(r.r), int(r.budget), int(r.seed)): float(r.val_fwe)
            for r in df.itertuples()}
    cells = sorted({(p, b, int(r)) for p, b, r in zip(df.placement, df.backprop, df.r) if p != "dense"})
    out = {}
    for rung in ("10M", "20M", "40M"):
        per_budget = {}
        for budget in BUDGETS:
            dense = [loss.get((rung, "dense", "full", 1, budget, s)) for s in SEEDS]
            if None in dense:
                continue
            rows = {}
            for p, b, r in cells:
                looped = [loss.get((rung, p, b, r, budget, s)) for s in SEEDS]
                if None in looped:
                    continue
                gaps = [l - d for l, d in zip(looped, dense)]
                rows[f"{p}/{b}/r{r}"] = dict(looped_spread=spread(looped), gap_spread=spread(gaps),
                                            gaps_per_seed=[round(g, 4) for g in gaps])
            per_budget[f"{budget}N"] = dict(dense_spread=spread(dense), cells=rows)
        dense_spreads = [v["dense_spread"] for v in per_budget.values()]
        gap_all = sorted(c["gap_spread"] for v in per_budget.values() for c in v["cells"].values())
        gap_r4 = [v["cells"]["middle/full/r4"]["gap_spread"] for v in per_budget.values()]
        out[rung] = dict(per_budget=per_budget,
                         dense_spread_range=[min(dense_spreads), max(dense_spreads)],
                         gap_spread_middle_full_r4_range=[min(gap_r4), max(gap_r4)],
                         gap_spread_all_cells=dict(n=len(gap_all), min=gap_all[0], median=gap_all[len(gap_all) // 2],
                                                   max=gap_all[-1]))
    json.dump(out, open("results/fit/paired_noise.json", "w"), indent=1)
    for rung, v in out.items():
        print(rung, "dense spread %.4f-%.4f" % tuple(v["dense_spread_range"]),
              "| gap spread, middle r=4 full %.4f-%.4f" % tuple(v["gap_spread_middle_full_r4_range"]),
              "| all cells min %.4f median %.4f max %.4f" % (v["gap_spread_all_cells"]["min"],
                                                             v["gap_spread_all_cells"]["median"],
                                                             v["gap_spread_all_cells"]["max"]))


if __name__ == "__main__":
    main()
