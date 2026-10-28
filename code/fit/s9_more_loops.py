"""S9 on the chain table's "more loops" definition (added 2026-10-28 after a reader's review; exploratory).

For each S9 run (loops = steps + 0-4 random extra loops), the largest solvable length and the 12- and 16-step accuracies
over d loops and every evaluated larger count (12, 16, 24, 32), as synthetic/analyze.py's more_loops does for the other
looped conditions of the chain table.

    python fit/s9_more_loops.py --out results/supp2/s9_more_loops.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from synthetic.analyze import acc, max_solvable  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="results/synthetic/supp2/formal/chain")
    ap.add_argument("--out", default="results/supp2/s9_more_loops.json")
    a = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(os.path.join(a.path, "*.json"))):
        r = json.load(open(p))
        if r["cond"] != "loop_step_extra":
            continue
        f = r["final"]
        keys = [k for k in f["1"] if k == "oracle" or (k.startswith("r=") and int(k[2:]) > 8)]
        rows.append(dict(seed=r["seed"], keys=keys, maxd_oracle=max_solvable(f, "oracle"),
                         maxd_more=max(max_solvable(f, k) for k in keys),
                         acc12_oracle=f["12"]["oracle"], acc12_more=acc(f, 12, keys),
                         acc16_oracle=f["16"]["oracle"], acc16_more=acc(f, 16, keys)))
    out = dict(runs=rows, maxd_more_mean=float(np.mean([r["maxd_more"] for r in rows])),
               maxd_more_min=min(r["maxd_more"] for r in rows), maxd_more_max=max(r["maxd_more"] for r in rows),
               acc12_more_mean=round(float(np.mean([r["acc12_more"] for r in rows])), 4),
               acc16_more_mean=round(float(np.mean([r["acc16_more"] for r in rows])), 4))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"S9 more loops: maxd {out['maxd_more_mean']:.1f} ({out['maxd_more_min']}-{out['maxd_more_max']}), "
          f"acc12 {out['acc12_more_mean']:.3f}, acc16 {out['acc16_more_mean']:.3f} -> {a.out}")


if __name__ == "__main__":
    main()
