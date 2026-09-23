"""Week 3: choose lr0 per width from the sweep and write configs/lr_table.json.

    python fit/collect_results.py --manifest configs/manifest_sweep.csv --out results/sweep.csv
    python scripts/pick_lr.py --results results/sweep.csv
Prints, per width: loss for each lr (dense r=1, budget 10N), the beta2=0.99 comparison at 448,
and the +-2x cross-check on the 80M looped cell. Picks the lr with the lowest main validation loss;
warns if the optimum sits on the edge of the grid (then extend the grid before freezing).
"""
from __future__ import annotations

import argparse
import json
import os
import re

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/sweep.csv")
    ap.add_argument("--loss-col", default="valavg_fwe")
    ap.add_argument("--out", default="configs/lr_table.json")
    a = ap.parse_args()
    df = pd.read_csv(a.results)
    df["lr_tag"] = df.name.str.extract(r"_lr([0-9.e+-]+)")[0].astype(float)
    df["b2"] = df.name.str.contains("_b2_099")
    df["xcheck"] = df.name.str.extract(r"_x([0-9.]+)_s")[0].astype(float)
    table = {}
    for w, g in df[(df.placement == "dense") & (~df.b2) & df.lr_tag.notna()].groupby("d_model"):
        g = g.sort_values("lr_tag")
        print(f"width {w}:")
        for _, r in g.iterrows():
            print(f"   lr {r.lr_tag:.2e}  {a.loss_col} {r[a.loss_col]:.4f}")
        best = g.loc[g[a.loss_col].idxmin()]
        edge = best.lr_tag in (g.lr_tag.min(), g.lr_tag.max())
        print(f"   -> best lr {best.lr_tag:.2e}" + ("   WARNING: on grid edge, extend the sweep" if edge else ""))
        table[str(int(w))] = float(best.lr_tag)
    b2 = df[df.b2]
    if len(b2):
        print("beta2=0.99 at width 448:")
        for _, r in b2.sort_values("lr_tag").iterrows():
            print(f"   lr {r.lr_tag:.2e}  {a.loss_col} {r[a.loss_col]:.4f}")
    xc = df[df.xcheck.notna()]
    if len(xc):
        print("80M middle r=4 cross-check (factor on the dense-tuned lr):")
        for _, r in xc.sort_values("xcheck").iterrows():
            print(f"   x{r.xcheck}  lr {r.lr0:.2e}  {a.loss_col} {r[a.loss_col]:.4f}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(table, open(a.out, "w"), indent=1)
    print("wrote", a.out, table)


if __name__ == "__main__":
    main()
