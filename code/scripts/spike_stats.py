"""Count loss spikes in every run's training log (rule declared in 03_偏离记录.md on 2026-09-26, before the 20M, 40M
and 80M whole-stack truncated results existed).

A spike is a rise of more than 1.0 nats in the logged training loss between consecutive log lines of the same phase
(the trunk or one cooldown branch), counted after step 400; a minor spike is a rise of more than 0.5. After a resume
the re-logged steps replace the earlier ones. The first line of a phase and the first line after a resume are not
compared: before commit bad884c those lines were averaged over too few steps and read far too low. One row per run
goes to results/spikes.csv.

    python scripts/spike_stats.py            # also run at the end of scripts/export_results.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

MAJOR, MINOR, AFTER_STEP = 1.0, 0.5, 400


def spike_row(log_path: str) -> dict:
    phases: dict[str, dict[int, float]] = {}
    last: dict[str, int] = {}
    skip: set[tuple[str, int]] = set()
    for r in csv.DictReader(open(log_path)):
        ph, s = r["phase"], int(r["step"])
        if ph not in last or s <= last[ph]:
            skip.add((ph, s))            # first line of the phase, or first line after a resume
        last[ph] = s
        phases.setdefault(ph, {})[s] = float(r["loss"])   # later lines (after a resume) win
    jumps: list[tuple[float, int]] = []
    for ph, by_step in phases.items():
        pts = sorted(by_step.items())
        jumps += [(b - a, s) for (p, a), (s, b) in zip(pts, pts[1:])
                  if s > AFTER_STEP and (ph, p) not in skip and (ph, s) not in skip]
    major = [s for d, s in jumps if d > MAJOR]
    return dict(n_spikes=len(major), n_minor=sum(d > MINOR for d, _ in jumps),
                max_rise=round(max((d for d, _ in jumps), default=0.0), 3),
                first_spike_step=min(major) if major else "",
                last_step=max((s for p in phases.values() for s in p), default=0))


def write_spike_stats(runs: str = "runs", out: str = "results/spikes.csv") -> int:
    rows = []
    for p in sorted(glob.glob(os.path.join(runs, "*", "train_log.csv"))):
        name = os.path.basename(os.path.dirname(p))
        rows.append(dict(name=name, **spike_row(p)))
    if not rows:
        return 0
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/spikes.csv")
    a = ap.parse_args()
    n = write_spike_stats(a.runs, a.out)
    print(f"spike statistics for {n} runs -> {a.out}")


if __name__ == "__main__":
    main()
