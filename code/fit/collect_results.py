"""Merge runs/*/results.jsonl into one flat CSV for fitting.

    python fit/collect_results.py --manifest configs/manifest.csv --out results/all_results.csv
Adds: rung, accounting (iso_token | iso_flop), cell = placement/backprop, budget_mult = D / N_rung, and from
results/spikes.csv (scripts/spike_stats.py) n_spikes of the run and diverged (the result comes from a collapsed
training segment; such rows enter no fit, 03_偏离记录.md 2026-09-30).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

RULER_MULTS = (5, 10, 20, 40, 80)
LOOP_MULTS = (10, 20, 40)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="configs/manifest.csv")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/all_results.csv")
    ap.add_argument("--spikes", default="results/spikes.csv", help="per-run spike / collapse table (optional)")
    a = ap.parse_args()
    spk = {r["name"]: r for r in csv.DictReader(open(a.spikes))} if os.path.exists(a.spikes) else {}
    man = {r["name"]: r for r in csv.DictReader(open(a.manifest))} if os.path.exists(a.manifest) else {}
    rows, other = [], set()
    for path in sorted(glob.glob(os.path.join(a.runs, "*", "results.jsonl"))):
        for line in open(path):
            d = json.loads(line)
            if man and d["name"] not in man:   # results/runs holds every experiment; keep only this manifest's runs
                other.add(d["name"])
                continue
            m = man.get(d["name"], {})
            N_rung = int(m.get("N_rung", 0)) or None
            mult = d["budget_tokens"] / N_rung if N_rung else float("nan")
            mults = RULER_MULTS if d["placement"] == "dense" else LOOP_MULTS
            iso_token = any(abs(mult - k) / k < 0.02 for k in mults)
            flat = {k: v for k, v in d.items() if k not in ("val", "val_avg")}
            for k, v in d["val"].items():
                flat[f"val_{k}"] = v
            for k, v in d["val_avg"].items():
                flat[f"valavg_{k}"] = v
            flat.update(rung=m.get("rung", ""), N_rung=N_rung, budget_mult=round(mult, 3),
                        accounting="iso_token" if iso_token else "iso_flop",
                        backprop="full" if d["k_bwd"] == 0 else "trunc",
                        cell=f"{d['placement']}/{'full' if d['k_bwd'] == 0 else 'trunc'}")
            sp = spk.get(d["name"])
            flat.update(n_spikes=int(sp["n_spikes"]) if sp else "",
                        diverged=bool(sp) and str(d["budget_tokens"]) in sp.get("diverged_budgets", "").split())
            rows.append(flat)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} rows -> {a.out}" + (f" ({len(other)} runs not in {a.manifest} skipped)" if other else "")
          + (f"; {sum(r['diverged'] for r in rows)} marked diverged" if spk else f"; no {a.spikes}, nothing marked"))


if __name__ == "__main__":
    main()
