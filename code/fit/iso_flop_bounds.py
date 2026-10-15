"""Bounds on the equal-compute comparison where the dense loss is extrapolated (exploratory; step 2 of 2026-10-15).

fit/iso_flop_raw.py reads the dense loss at equal training FLOPs off each dense curve (rung, seed); beyond the
curve's largest budget it extends the last segment in log D. This script recomputes those comparisons with three
ways to extend a dense curve and checks each on the measured curves (backtest: drop the last one or two budgets,
predict the last one):
  last_slope : the last segment extended linearly in log D (what iso_flop_raw.py uses)
  power_law  : L = E + B (D/N)^-beta fitted to all budgets of that curve (beta on a grid, E and B by least squares)
  flat       : no further improvement after the largest budget. Dense loss does not get worse with more data, so this
               can only overstate the dense loss: a bound in the looped model's favour.
gap = looped - dense at equal training FLOPs (positive: dense better).

    python fit/iso_flop_bounds.py   -> results/fit/iso_flop_bounds.csv, results/fit/iso_flop_bounds.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

BETAS = np.round(np.arange(0.05, 2.0001, 0.01), 2)
METHODS = ("last_slope", "power_law", "flat")


def fit_power_law(mult: np.ndarray, L: np.ndarray) -> tuple[float, float, float]:
    """Least-squares L = E + B mult^-beta with E >= 0, B > 0; beta on a grid. Returns (E, B, beta)."""
    best = None
    for b in BETAS:
        X = np.column_stack([np.ones_like(mult), mult ** -b])
        coef, *_ = np.linalg.lstsq(X, L, rcond=None)
        E, B = coef
        if E < 0 or B <= 0:
            continue
        sse = float(((X @ coef - L) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, E, B, b)
    if best is None:
        raise ValueError("no admissible power law")
    return best[1], best[2], best[3]


def extend(mult: np.ndarray, L: np.ndarray, x: float, method: str) -> tuple[float, bool]:
    """Dense loss at budget multiple x from the measured points (sorted by mult). Interpolates in log D inside."""
    lx, lm = np.log(x), np.log(mult)
    if lx <= lm[-1] + 1e-9:
        return float(np.interp(lx, lm, L)), False
    if method == "last_slope":
        s = (L[-1] - L[-2]) / (lm[-1] - lm[-2])
        return float(L[-1] + s * (lx - lm[-1])), True
    if method == "flat":
        return float(L[-1]), True
    E, B, b = fit_power_law(mult, L)
    return float(E + B * x ** -b), True


def backtest(rulers: dict) -> pd.DataFrame:
    rows = []
    for (rung, seed), (mult, L) in rulers.items():
        for drop in (1, 2):
            if len(mult) - drop < 3:
                continue
            m_tr, L_tr = mult[:-drop], L[:-drop]
            for method in METHODS:
                pred, _ = extend(m_tr, L_tr, mult[-1], method)
                rows.append(dict(rung=rung, seed=seed, held_out_mult=mult[-1], doublings=drop, method=method,
                                 error=pred - L[-1]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/all_results.csv")
    ap.add_argument("--ext", nargs="+", default=["results/ext_data.csv", "results/ext_160m.csv", "results/ext_160m_dense.csv"])
    ap.add_argument("--out", default="results/fit/iso_flop_bounds.csv")
    ap.add_argument("--summary", default="results/fit/iso_flop_bounds.json")
    a = ap.parse_args()
    df = pd.read_csv(a.results)
    df = df[~df.diverged.astype(str).str.lower().eq("true")]
    for p in a.ext:
        if os.path.exists(p):
            df = pd.concat([df, pd.read_csv(p)], ignore_index=True)
    df["mult"] = (df.budget_tokens / df.N_rung).round(1)
    df["cellr"] = np.where(df.placement == "dense", "dense",
                           df.placement + "_r" + df.r.astype(str) + np.where(df.k_bwd > 0, "_k" + df.k_bwd.astype(str), ""))

    rulers, rows = {}, []
    for rung, dr in df.groupby("rung"):
        dense_fpt = dr[dr.cellr == "dense"].train_flops_per_token.iloc[0]
        for seed, ds in dr.groupby("seed"):
            ruler = ds[ds.cellr == "dense"].set_index("mult").val_fwe.sort_index()
            if len(ruler) < 3:
                continue
            mult, L = ruler.index.values.astype(float), ruler.values.astype(float)
            rulers[(rung, seed)] = (mult, L)
            for cell, dc in ds[ds.cellr != "dense"].groupby("cellr"):
                rho = dc.train_flops_per_token.iloc[0] / dense_fpt
                for _, r in dc.iterrows():
                    if r.mult not in (10.0, 20.0, 40.0, 80.0, 160.0):     # the same rows as iso_flop_raw.py
                        continue
                    row = dict(rung=rung, cell=cell, seed=seed, mult=r.mult, rho=round(rho, 2), looped=r.val_fwe,
                               dense_equiv_mult=round(r.mult * rho, 1))
                    for method in METHODS:
                        d, extra = extend(mult, L, r.mult * rho, method)
                        row[f"gap_{method}"] = round(r.val_fwe - d, 4)
                    row["extrapolated"] = extra
                    row["doublings_beyond"] = round(max(0.0, np.log2(r.mult * rho / mult[-1])), 2)
                    rows.append(row)
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False)

    bt = backtest(rulers)
    bt_sum = (bt.groupby(["method", "doublings", "held_out_mult"]).error.agg(["min", "max", "mean", "count"])
              .round(4).reset_index())
    ex = out[out.extrapolated]
    summary = {
        "rows": len(out), "measured": int((~out.extrapolated).sum()), "extrapolated": int(len(ex)),
        "measured_min_gap": float(out[~out.extrapolated].gap_last_slope.min()),
        "extrapolated_doublings_beyond": [float(ex.doublings_beyond.min()), float(ex.doublings_beyond.max())],
        "looped_better_at_equal_flops": {m: int((ex[f"gap_{m}"] < 0).sum()) for m in METHODS},
        "extrapolated_gap_range": {m: [float(ex[f"gap_{m}"].min()), float(ex[f"gap_{m}"].max())] for m in METHODS},
        "looped_better_rows_power_law": ex[ex.gap_power_law < 0][["rung", "cell", "seed", "mult", "gap_last_slope",
                                                                   "gap_power_law", "gap_flat"]].to_dict("records"),
        "looped_better_rows_flat": ex[ex.gap_flat < 0][["rung", "cell", "seed", "mult", "gap_last_slope",
                                                         "gap_power_law", "gap_flat"]].to_dict("records"),
        "backtest": bt_sum.to_dict("records"),
    }
    with open(a.summary, "w") as f:
        json.dump(summary, f, indent=1, default=float)

    pd.set_option("display.width", 200)
    print("backtest (predicted - actual dense loss; negative = too optimistic for the dense model):")
    print(bt_sum.to_string(index=False))
    print(f"\n{len(out)} comparisons, {len(ex)} with an extrapolated dense loss "
          f"({summary['extrapolated_doublings_beyond'][0]}-{summary['extrapolated_doublings_beyond'][1]} doublings beyond the curve)")
    for m in METHODS:
        print(f"  {m:10s}: looped better in {summary['looped_better_at_equal_flops'][m]:3d} of {len(ex)}; "
              f"gap range {summary['extrapolated_gap_range'][m][0]:+.3f} to {summary['extrapolated_gap_range'][m][1]:+.3f}")
    print(f"-> {a.out}, {a.summary}")


if __name__ == "__main__":
    main()
