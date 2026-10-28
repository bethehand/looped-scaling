"""S10 readings at the best learning rate tried (docs/supplementary-experiments-3.md, "Readings, fixed in advance":
if 0.71 or 1.41 times the rate beats the rule's rate by more than 0.01 nats at a budget, the readings are also reported
at the best rate tried, seed 42, as in S1). fit/supp3_analysis.py, written before any S10 result, flags the case
(lr_check.report_at_best) but does not compute these readings; this script, written on 2026-10-28 after the results,
does. The primary reading stays at the rule's rate.

    python fit/supp3_best_lr.py   -> results/supp3/s10_best_lr.json

As in S1, each side takes its own best tested rate at every budget, seed 42: the 18-layer dense model from S10
(0.71, 1 and 1.41 times the rule's rate), the looped middle block (r = 4) from S1 (0.5, 0.71, 1 and 1.41 times;
results/supp/s1_points.csv). Both are read off the same 8-layer ladder as in fit/supp3_analysis.py (frozen rates,
seed means). Also given: the same readings for seed 42 at the rule's rate, and the per-seed paired gaps at the
rule's rate (the plan asks for seed means of the gap; the gap's interval columns in s10_table.csv are not resampled).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import fit.local_phi as lp
from fit.supp3_analysis import LADDER_FILES, TPPS, load_deep

S1_POINTS = "results/supp/s1_points.csv"


def main():
    ladder = lp.load(LADDER_FILES)
    curves = lp.dense_curves(ladder)
    deep = load_deep("results/supp3/supp3_deep.csv")
    s1 = pd.read_csv(S1_POINTS)
    loop = ladder[(ladder.cell == "middle/full") & (ladder.r == 4)]
    out = {"best_rate_seed42": [], "paired_gaps_rule_rate": []}
    for rung in ("20M", "40M"):
        N_deep = float(deep[deep.rung == rung].N.iloc[0])
        N_loop = float(loop[loop.rung == rung].N.iloc[0])
        for tpp in TPPS:
            g = deep[(deep.rung == rung) & (deep.tpp == tpp)]
            gl = loop[(loop.rung == rung) & (loop.tpp == tpp)]
            D = float(g.budget_tokens.iloc[0])
            paired = (g[g.lrs == 1.0].set_index("seed").L - gl.set_index("seed").L).dropna()
            out["paired_gaps_rule_rate"].append(dict(rung=rung, tpp=tpp, by_seed={int(s): round(v, 4) for s, v in
                                                                                   paired.items()},
                                                     mean=round(paired.mean(), 4)))
            d42 = g[g.seed == 42].set_index("lrs").L
            l42 = s1[(s1.rung == rung) & (s1.placement == "middle") & (s1.mult == tpp) & (s1.seed == 42)].set_index("f").loss
            if l42.empty or 1.0 not in d42.index:
                continue
            row = dict(rung=rung, tpp=tpp)
            for tag, Ld, Ll in (("rule", d42[1.0], l42[1.0]), ("best", d42.min(), l42.min())):
                Md = lp.dense_size_at(Ld, D, curves)[0] / N_deep
                Ml = lp.dense_size_at(Ll, D, curves)[0] / N_loop
                row.update({f"{tag}_loss_deep": round(Ld, 4), f"{tag}_loss_loop": round(Ll, 4),
                            f"{tag}_gap": round(Ld - Ll, 4), f"{tag}_M_deep": round(Md, 4),
                            f"{tag}_M_loop": round(Ml, 4), f"{tag}_R": round(np.log(Md) / np.log(Ml), 4)})
            row.update(best_f_deep=float(d42.idxmin()), best_f_loop=float(l42.idxmin()),
                       deep_best_at_edge=bool(d42.idxmin() in (min(d42.index), max(d42.index))))
            out["best_rate_seed42"].append(row)
    b40 = [r for r in out["best_rate_seed42"] if r["rung"] == "40M"]
    if len(b40) == len(TPPS):
        out["mean_R_40M_seed42"] = dict(rule=round(float(np.mean([r["rule_R"] for r in b40])), 4),
                                        best=round(float(np.mean([r["best_R"] for r in b40])), 4))
    os.makedirs("results/supp3", exist_ok=True)
    json.dump(out, open("results/supp3/s10_best_lr.json", "w"), indent=1)
    print(pd.DataFrame(out["best_rate_seed42"]).to_string(index=False))
    print(json.dumps(out.get("mean_R_40M_seed42"), indent=1))
    print(pd.DataFrame(out["paired_gaps_rule_rate"]).to_string(index=False))


if __name__ == "__main__":
    main()
