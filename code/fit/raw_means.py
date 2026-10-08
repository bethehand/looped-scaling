"""Mean loss per (run, validation set, inference loop count) from the raw per-token arrays of token_analysis.py.

    python fit/raw_means.py --pattern rs1to8 --out results/token_analysis/depth_curve_randr.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/token_analysis/raw")
    ap.add_argument("--pattern", default="", help="substring the run name must contain")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(os.path.join(a.raw, "*__r*.npy"))):
        name = os.path.basename(p)[:-4]
        if name.startswith("targets__"):
            continue
        parts = name.split("__")                 # run__ckpt__valset__rN, or run__valset__rN for the oldest arrays
        if len(parts) not in (3, 4):
            continue
        run, vs, r = parts[0], parts[-2], parts[-1]
        ckpt = parts[1] if len(parts) == 4 else ""
        if a.pattern not in run:
            continue
        x = np.load(p).astype(np.float32)
        rows.append(dict(run=run, ckpt=ckpt, valset=vs, r_eval=int(r[1:]), mean_loss=round(float(x.mean()), 5), n_tokens=len(x)))
    rows.sort(key=lambda d: (d["run"], d["valset"], d["r_eval"]))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"{a.out}: {len(rows)} rows")
    for d in rows:
        if d["valset"] == "fwe":
            print(f"  {d['run']} r={d['r_eval']:2d}: {d['mean_loss']:.4f}")


if __name__ == "__main__":
    main()
