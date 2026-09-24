"""Copy each run's small result files from runs/ (a symlink to the NVMe, outside git) into results/runs/ (tracked),
so they can be committed and pushed from the GPU machine.

    python scripts/export_results.py            # then: git add results && git commit && git push
Copies runs/<name>/results.jsonl and runs/<name>/run_info.json; checkpoints and logs are never copied.
"""
from __future__ import annotations

import argparse
import filecmp
import glob
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/runs")
    a = ap.parse_args()
    new = same = 0
    for d in sorted(glob.glob(os.path.join(a.runs, "*", ""))):
        name = os.path.basename(os.path.normpath(d))
        for f in ("results.jsonl", "run_info.json"):
            src = os.path.join(d, f)
            if not os.path.exists(src):
                continue
            dst = os.path.join(a.out, name, f)
            if os.path.exists(dst) and filecmp.cmp(src, dst, shallow=False):
                same += 1
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            new += 1
    n_done = len(glob.glob(os.path.join(a.out, "*", "results.jsonl")))
    print(f"copied {new} new or changed files ({same} unchanged); {n_done} runs with results in {a.out}")


if __name__ == "__main__":
    main()
