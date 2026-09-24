"""Decide whether looped cells need a different learning rate than the dense ruler (rules declared in
03_偏离记录.md before the check was run).

    python fit/collect_results.py --manifest configs/manifest_looplr.csv --out results/looplr.csv
    python scripts/pick_loop_lr.py --results results/looplr.csv          # -> configs/loop_lr_multiplier.json

Per cell (placement, r, k) at width 448: parabola vertex in log2(lr) through the three factors 0.5 / 0.71 / 1.0 of the
dense lr0; if the lowest point is an end point, the vertex is that end point (at 0.5: WARN, the optimum may be lower).
shift_c = log2(vertex_c / lr0_dense). Pooled shift s = mean over cells, 95% interval by bootstrap over cells.
Slope of shift on log2(r) with a bootstrap interval.
Decision:  s >= -0.25                         -> mode "none"   (looped cells keep the dense lr0)
           s <  -0.25, slope interval spans 0 -> mode "shared" (all looped cells x 2**s)
           s <  -0.25, slope interval excl. 0 -> mode "per_r"  (x 2**(a + b*log2 r), fitted)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

THRESH = -0.25
N_BOOT = 2000


def vertex_shift(f: np.ndarray, loss: np.ndarray) -> tuple[float, str]:
    order = np.argsort(f); f, loss = f[order], loss[order]
    i = int(np.argmin(loss)); x = np.log2(f)
    if i == 0:
        return float(x[0]), "EDGE_LOW"
    if i == len(f) - 1:
        return float(x[-1]), "edge_high"
    a, b, _ = np.polyfit(x, loss, 2)
    if a <= 0:
        return float(x[i]), "grid"
    return float(np.clip(-b / (2 * a), x[0], x[-1])), "parabola"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/looplr.csv")
    ap.add_argument("--loss-col", default="val_fwe")
    ap.add_argument("--out", default="configs/loop_lr_multiplier.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df = pd.read_csv(a.results)
    df["factor"] = df.name.str.extract(r"_lrx([0-9.]+)_s")[0].astype(float)
    cells = []
    for (pl, r, k), g in df.groupby(["placement", "r", "k_bwd"]):
        if len(g) < 3:
            print(f"{pl} r={r} k={k}: only {len(g)} of 3 points, skipped"); continue
        sh, how = vertex_shift(g.factor.to_numpy(), g[a.loss_col].to_numpy())
        pts = "  ".join(f"x{fx:<4} {l:.4f}" for fx, l in sorted(zip(g.factor, g[a.loss_col])))
        print(f"{pl:6s} r={r} {'full' if k == 0 else 'k' + str(k):4s}  {pts}  -> optimum x{2 ** sh:.2f} ({how})"
              + ("   WARNING: lowest at 0.5x" if how == "EDGE_LOW" else ""))
        cells.append(dict(placement=pl, r=r, k=k, shift=sh))
    c = pd.DataFrame(cells)
    rng = np.random.default_rng(a.seed)
    s = float(c["shift"].mean())
    boots = [float(c["shift"].sample(len(c), replace=True, random_state=int(rng.integers(1e9))).mean()) for _ in range(N_BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    X = np.log2(c.r.to_numpy()); Y = c["shift"].to_numpy()
    slope, icpt = np.polyfit(X, Y, 1)
    sl = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(c), len(c))
        if len(set(X[idx])) < 2:
            continue
        sl.append(np.polyfit(X[idx], Y[idx], 1)[0])
    slo, shi = np.percentile(sl, [2.5, 97.5])
    print(f"pooled shift s = {s:+.2f} log2 (x{2 ** s:.2f}), 95% interval [{lo:+.2f}, {hi:+.2f}]")
    print(f"slope on log2 r = {slope:+.2f}, 95% interval [{slo:+.2f}, {shi:+.2f}]")
    if s >= THRESH:
        out = {"mode": "none"}
        print("decision: looped cells keep the dense lr0")
    elif slo <= 0 <= shi:
        out = {"mode": "shared", "shared": round(2 ** s, 3)}
        print(f"decision: all looped cells use lr0 x {2 ** s:.3f}")
    else:
        out = {"mode": "per_r", "per_r": {str(r): round(2 ** (icpt + slope * np.log2(r)), 3) for r in (2, 4, 8)}}
        print(f"decision: per-r multipliers {out['per_r']}")
    out.update(pooled_shift=round(s, 3), interval=[round(lo, 3), round(hi, 3)], slope=round(slope, 3),
               slope_interval=[round(slo, 3), round(shi, 3)], cells=cells)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
