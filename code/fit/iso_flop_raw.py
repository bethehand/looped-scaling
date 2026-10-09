"""Raw equal-training-FLOP comparison, no fitted formula (check of the H4 verdict; exploratory, 2026-10-09).

For every looped cell, rung, seed and budget D: the dense model of the same rung and seed trained with the same
training FLOPs sees D x rho tokens (rho = looped / dense FLOPs per token); its loss is read off the dense ruler by
interpolating in log D (beyond the ruler's largest budget: extrapolated with the last segment's slope, flagged).
gap_flop = looped loss - dense loss at equal FLOPs (positive = looped worse); gap_token = at equal tokens.

    python fit/iso_flop_raw.py --results results/all_results.csv --ext results/ext_data.csv --out results/fit/iso_flop_raw.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def dense_at(ruler: pd.Series, mult: float) -> tuple[float, bool]:
    """ruler: loss indexed by budget multiple (sorted). Returns (loss, extrapolated)."""
    xs, ys = np.log(ruler.index.values.astype(float)), ruler.values.astype(float)
    x = np.log(mult)
    if x <= xs[-1]:
        return float(np.interp(x, xs, ys)), False
    slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    return float(ys[-1] + slope * (x - xs[-1])), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/all_results.csv")
    ap.add_argument("--ext", default="results/ext_data.csv")
    ap.add_argument("--out", default="results/fit/iso_flop_raw.csv")
    a = ap.parse_args()
    df = pd.read_csv(a.results)
    df = df[~df.diverged.astype(str).str.lower().eq("true")]
    if os.path.exists(a.ext):
        df = pd.concat([df, pd.read_csv(a.ext)], ignore_index=True)
    df["mult"] = (df.budget_tokens / df.N_rung).round(1)
    df["cell"] = np.where(df.placement == "dense", "dense",
                          df.placement + "_r" + df.r.astype(str) + np.where(df.k_bwd > 0, "_k" + df.k_bwd.astype(str), ""))
    rows = []
    for rung, dr in df.groupby("rung"):
        dense_fpt = dr[dr.cell == "dense"].train_flops_per_token.iloc[0]
        for seed, ds in dr.groupby("seed"):
            ruler = ds[ds.cell == "dense"].set_index("mult").val_fwe.sort_index()
            if len(ruler) < 3:
                continue
            for cell, dc in ds[ds.cell != "dense"].groupby("cell"):
                rho = dc.train_flops_per_token.iloc[0] / dense_fpt
                for _, r in dc.iterrows():
                    if r.mult not in (10.0, 20.0, 40.0, 80.0, 160.0):   # iso-token budgets only (the truncated cells' rho budgets skipped)
                        continue
                    d_tok, _ = dense_at(ruler, r.mult)
                    d_flop, extra = dense_at(ruler, r.mult * rho)
                    rows.append(dict(rung=rung, cell=cell, seed=seed, mult=r.mult, rho=round(rho, 2), looped=r.val_fwe,
                                     dense_equal_tokens=round(d_tok, 4), dense_equal_flops=round(d_flop, 4),
                                     dense_equiv_mult=round(r.mult * rho, 1), extrapolated=extra,
                                     gap_token=round(r.val_fwe - d_tok, 4), gap_flop=round(r.val_fwe - d_flop, 4)))
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False)
    order = {"10M": 0, "20M": 1, "40M": 2, "80M": 3}
    summ = (out.groupby(["rung", "cell", "mult"]).agg(rho=("rho", "first"), gap_token=("gap_token", "mean"),
                                                        gap_flop=("gap_flop", "mean"), seeds=("seed", "count"),
                                                        extrapolated=("extrapolated", "max")).reset_index())
    summ["o"] = summ.rung.map(order)
    summ = summ.sort_values(["o", "cell", "mult"]).drop(columns="o")
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))
    print(f"\n{len(out)} rows -> {a.out}; gap_flop > 0 everywhere means dense wins at equal training FLOPs "
          f"(extrapolated = dense budget beyond the ruler)")


if __name__ == "__main__":
    main()
