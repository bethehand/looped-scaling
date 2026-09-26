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
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.spike_stats import write_spike_stats  # noqa: E402


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
    # loss-spike counts from the training logs, which stay on the GPU machine (03_偏离记录.md, 2026-09-26)
    spikes_out = os.path.join(os.path.dirname(os.path.normpath(a.out)), "spikes.csv")
    print(f"spike statistics for {write_spike_stats(a.runs, spikes_out)} runs -> {spikes_out}")


if __name__ == "__main__":
    main()
