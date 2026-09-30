"""Loss spikes and collapses in every run's training log (rules declared in 03_偏离记录.md: spikes on 2026-09-26,
collapses on 2026-09-30, each before the results it governs existed).

Spike: a rise of more than 1.0 nats in the logged training loss between consecutive log lines of the same phase (the
trunk or one cooldown branch), counted after step 400; a minor spike is a rise of more than 0.5. After a resume the
re-logged steps replace the earlier ones. The first line of a phase and the first line after a resume are not
compared: before commit bad884c those lines were averaged over too few steps and read far too low.

Collapse: a quick validation of the trunk more than 1.0 nats above the best trunk quick validation so far that never
comes back within 0.5 of that best before the trunk ends. Its onset is the last quick validation before that rise.
A cooldown result is marked diverged when (a) its branch starts after a trunk collapse onset, or (b) the last quick
validation of its branch is more than 1.0 above the best trunk quick validation before the branch start. Diverged
results enter no fit.

One row per run goes to results/spikes.csv.

    python scripts/spike_stats.py            # also run at the end of scripts/export_results.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os

MAJOR, MINOR, AFTER_STEP = 1.0, 0.5, 400
COLLAPSE_RISE, RECOVERED = 1.0, 0.5
COOLDOWN_FRAC = 0.2          # pre-registered, the same for every run


def _read(log_path: str):
    phases: dict[str, dict[int, tuple[int, float, float]]] = {}   # phase -> step -> (tokens, loss, quick val)
    last: dict[str, int] = {}
    skip: set[tuple[str, int]] = set()
    for r in csv.DictReader(open(log_path)):
        ph, s = r["phase"], int(r["step"])
        if ph not in last or s <= last[ph]:
            skip.add((ph, s))            # first line of the phase, or first line after a resume
        last[ph] = s
        phases.setdefault(ph, {})[s] = (int(r["tokens"]), float(r["loss"]), float(r["quick_val"]))   # later wins
    return phases, skip


def _trunk_collapse(qv: list[tuple[int, int, float]]) -> tuple[int, int] | None:
    """qv = [(step, tokens, quick val)] of the trunk in step order -> (step, tokens) of the collapse onset, or None."""
    best = math.inf
    for i, (_, _, q) in enumerate(qv):
        if q > best + COLLAPSE_RISE and all(w > best + RECOVERED for _, _, w in qv[i + 1:]):
            return qv[i - 1][0], qv[i - 1][1]
        best = min(best, q)
    return None


def spike_row(log_path: str) -> dict:
    phases, skip = _read(log_path)
    jumps: list[tuple[float, int]] = []
    for ph, by_step in phases.items():
        pts = sorted((s, v[1]) for s, v in by_step.items())
        jumps += [(b - a, s) for (p, a), (s, b) in zip(pts, pts[1:])
                  if s > AFTER_STEP and (ph, p) not in skip and (ph, s) not in skip]
    major = [s for d, s in jumps if d > MAJOR]
    trunk = sorted((s, tok, q) for s, (tok, _, q) in phases.get("trunk", {}).items() if not math.isnan(q))
    onset = _trunk_collapse(trunk)
    batch = next((tok // s for s, tok, _ in reversed(trunk) if s > 0), 0)
    diverged = []
    for ph, by_step in phases.items():
        if not ph.startswith("cool_") or not batch:
            continue
        end = int(ph[len("cool_"):])
        start = int(round(end * (1 - COOLDOWN_FRAC)))
        start -= start % batch
        before = [q for s, tok, q in trunk if tok <= start]
        own = [v[2] for s, v in sorted(by_step.items()) if not math.isnan(v[2])]
        if (onset and start > onset[1]) or (before and own and own[-1] > min(before) + COLLAPSE_RISE):
            diverged.append(end)
    return dict(n_spikes=len(major), n_minor=sum(d > MINOR for d, _ in jumps),
                max_rise=round(max((d for d, _ in jumps), default=0.0), 3),
                first_spike_step=min(major) if major else "",
                last_step=max((s for p in phases.values() for s in p), default=0),
                collapse_step=onset[0] if onset else "",
                diverged_budgets=" ".join(str(e) for e in sorted(diverged)))


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
