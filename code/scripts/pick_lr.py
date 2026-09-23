"""Choose lr0 (and beta2) per width from the learning-rate sweep; rules declared in 03_偏离记录.md before any result.

    python fit/collect_results.py --manifest configs/manifest_sweep.csv --out results/sweep.csv
    python scripts/pick_lr.py --results results/sweep.csv            # -> configs/lr_table.json

Rules
  metric   end-of-cooldown loss on the main validation set (val_fwe)
  lr0      if the lowest of the 5 points is interior: vertex of a parabola in log2(lr) through it and its two
           neighbours (clipped to that bracket); if it is on the grid edge: WARN, extend the grid by one 2x point
  beta2    if at width 448 the best beta2=0.99 loss beats the best beta2=0.95 loss by > 0.005 nats,
           widths 320/448/640 use 0.99, else 0.95 everywhere
  loop     80M middle r=4 cross-check: take the cross-check lr closest (in log) to the 80M dense lr0; the rule
           "looped cells reuse the dense lr" holds if its loss is within 0.01 nats of the looped best
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

BETA2_MARGIN = 0.005
LOOP_TOL = 0.01
SMALL_BATCH_WIDTHS = (320, 448, 640)


def vertex(lrs: np.ndarray, losses: np.ndarray) -> tuple[float, str]:
    i = int(np.argmin(losses))
    if i == 0 or i == len(lrs) - 1:
        return float(lrs[i]), "EDGE"
    x = np.log2(lrs[i - 1:i + 2]); y = losses[i - 1:i + 2]
    a, b, _ = np.polyfit(x, y, 2)
    if a <= 0:
        return float(lrs[i]), "grid"
    xv = float(np.clip(-b / (2 * a), x[0], x[2]))
    return float(2 ** xv), "parabola"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/sweep.csv")
    ap.add_argument("--loss-col", default="val_fwe")
    ap.add_argument("--out", default="configs/lr_table.json")
    a = ap.parse_args()
    df = pd.read_csv(a.results)
    df["lr_tag"] = df.name.str.extract(r"_lr([0-9.e+-]+?)(?:_b2|_s)")[0].astype(float)
    df["b2"] = df.name.str.contains("_b2_099")
    df["xcheck"] = df.name.str.extract(r"_x([0-9.]+)_s")[0].astype(float)
    lr0, beta2, best_loss = {}, {}, {}
    dense = df[(df.placement == "dense") & (~df.b2) & df.lr_tag.notna()]
    for w, g in dense.groupby("d_model"):
        g = g.sort_values("lr_tag")
        lrs, losses = g.lr_tag.to_numpy(), g[a.loss_col].to_numpy()
        v, how = vertex(lrs, losses)
        print(f"width {int(w)} ({len(g)} of 5 points):")
        for lr, l in zip(lrs, losses):
            print(f"   lr {lr:.2e}   loss {l:.4f}" + ("   <- lowest" if l == losses.min() else ""))
        print(f"   -> lr0 {v:.3e} ({how})" + ("   WARNING: lowest point on the grid edge, extend the sweep" if how == "EDGE" else ""))
        lr0[str(int(w))] = float(f"{v:.3g}")
        best_loss[int(w)] = float(losses.min())
    b2 = df[df.b2 & df.lr_tag.notna()]
    use_099 = False
    if len(b2) and 448 in best_loss:
        g = b2.sort_values("lr_tag")
        print("width 448 with beta2 = 0.99:")
        for lr, l in zip(g.lr_tag, g[a.loss_col]):
            print(f"   lr {lr:.2e}   loss {l:.4f}")
        gain = best_loss[448] - float(g[a.loss_col].min())
        use_099 = gain > BETA2_MARGIN
        print(f"   best(0.95) - best(0.99) = {gain:+.4f} nats -> beta2 = {'0.99 for widths 320/448/640' if use_099 else '0.95 everywhere'}")
    for w in lr0:
        beta2[w] = 0.99 if (use_099 and int(w) in SMALL_BATCH_WIDTHS) else 0.95
    xc = df[df.xcheck.notna()].sort_values("lr0")
    if len(xc):
        print("80M middle r=4 cross-check:")
        for lr, l in zip(xc.lr0, xc[a.loss_col]):
            print(f"   lr {lr:.2e}   loss {l:.4f}")
        if "896" in lr0:
            near = xc.iloc[int(np.argmin(np.abs(np.log2(xc.lr0.to_numpy()) - math.log2(lr0["896"]))))]
            gap = float(near[a.loss_col]) - float(xc[a.loss_col].min())
            print(f"   dense lr0 {lr0['896']:.3e}; nearest cross-check point {near.lr0:.2e} is {gap:.4f} nats above the looped best"
                  f" -> {'rule holds' if gap <= LOOP_TOL else 'rule FAILS: run the extra r=4/r=8 sweep at 20M'}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"lr0": lr0, "beta2": beta2}, open(a.out, "w"), indent=1)
    print("wrote", a.out, {"lr0": lr0, "beta2": beta2})


if __name__ == "__main__":
    main()
