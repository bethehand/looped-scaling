"""Supplementary experiments S1-S3 (docs/supplementary-experiments.md, frozen before any of these runs started).

    python scripts/make_supp.py             # S1 configs + configs/manifest_supp_lr.csv, and the one queue for S1/S2/S3
    python -u scripts/run_queue.py --manifest configs/queue_supp.csv --follow --gpus 0,1,2,3
    python scripts/make_supp.py --edges     # run by the queue once S1 round 1 is done: edge rule, then the loop-count curves

S1  text: dense and middle r=4 (full backprop) at 10M/20M/40M, lr = 0.5 / 0.71 / 1.41 x the frozen lr0 (1x is the main
    grid run), seed 42, and also 43 at 20M; budgets: 10M both cells 10-160N, 20M/40M dense 10-80N, looped 10-40N.
    Edge rule, once per size and cell: lowest loss at 40N (10M: 40N or 160N) at 0.5x -> add 0.35x; at 1.41x -> add 2x.
    Then (Q5) loss vs inference loop count on the 32N trunk checkpoints of every middle r=4 run.
S2  synthetic chain task: loop_r4 / loop_r8 / loop_rand / loop_step at lr 5e-4 and 1e-3, seeds 2-4, 32000 steps and all
    other settings as in the formal runs, each followed by the intermediate-value probe of the formal runs.
S3  inference cost: scripts/bench_inference.py after the training runs.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts import make_configs as mc  # noqa: E402

SUPP_DIR = "configs/supp"
MANIFEST = "configs/manifest_supp_lr.csv"
QUEUE = "configs/queue_supp.csv"
QUEUE_COLS = ["name", "kind", "priority", "status", "config", "hours_est", "cmd", "after"]
LR_TABLE = "configs/lr_table_frozen.json"
EDGE_DECISION = "results/supp/edge_decision.json"

S1_RUNGS = ("10M", "20M", "40M")
S1_CELLS = (("dense", 1, 0), ("middle", 4, 0))
S1_FACTORS = (0.5, 0.71, 1.41)
S1_EDGE = {0.5: 0.35, 1.41: 2.0}
S1_SEEDS = {"10M": (42,), "20M": (42, 43), "40M": (42,)}
S1_BUDGETS = {("10M", "dense"): [10, 20, 40, 80, 160], ("10M", "middle"): [10, 20, 40, 80, 160],
              ("20M", "dense"): [10, 20, 40, 80], ("20M", "middle"): [10, 20, 40],
              ("40M", "dense"): [10, 20, 40, 80], ("40M", "middle"): [10, 20, 40]}
S1_EDGE_BUDGETS = {"10M": (40, 160), "20M": (40,), "40M": (40,)}
Q5_R_EVAL = "1,2,3,4,5,6,8,12,16"

S2_CONDS = ("loop_r4", "loop_r8", "loop_rand", "loop_step")
S2_LRS = (5e-4, 1e-3)
S2_SEEDS = (2, 3, 4)
S2_STEPS = 32000
S2_HOURS = {"loop_r4": 1.34, "loop_r8": 2.32, "loop_rand": 1.46, "loop_step": 1.25}   # measured in the formal runs
S2_OUT, S2_CKPT = "results/synthetic/lr_seeds", "runs/synthetic/lr_seeds"


def _lr_tables() -> tuple[dict, dict]:
    t = json.load(open(LR_TABLE))
    return t["lr0"], t["beta2"]


def n_rung(rung: str) -> int:
    return mc.analytic_N(mc.model_cfg(mc.RUNGS[rung], "dense", 1, 0))["N"]


def s1_name(rung: str, placement: str, r: int, k: int, seed: int, f: float) -> str:
    """f = 1 is the main-grid run itself."""
    return mc.run_name(rung, placement, r, k, seed, "" if f == 1 else f"_lrs{f:g}")


def s1_run(rung: str, placement: str, r: int, k: int, seed: int, f: float, cfg_dir: str = SUPP_DIR) -> dict:
    """Write one S1 config; return its manifest row."""
    lr0, beta2 = _lr_tables()
    width = mc.RUNGS[rung]
    cfg, row = mc.make_run(rung, width, placement, r, k, seed, n_rung(rung), lr0, S1_BUDGETS[(rung, placement)],
                           iso_flop=False, tag=f"_lrs{f:g}", lr_override=float(lr0[str(width)]) * f,
                           betas=(0.9, float(beta2.get(str(width), 0.95))), cfg_dir=cfg_dir)
    os.makedirs(cfg_dir, exist_ok=True)
    yaml.safe_dump(cfg, open(os.path.join(cfg_dir, cfg["name"] + ".yaml"), "w"), sort_keys=False)
    row["lr_factor"] = f
    return row


def s2_jobs() -> list[dict]:
    jobs = []
    for cond in S2_CONDS:
        for lr in S2_LRS:
            for seed in S2_SEEDS:
                res = f"{cond}_lr{lr:g}_s{seed}_lut32"
                train = f"synth_{cond}_lr{lr:g}_s{seed}"
                jobs.append(dict(name=train, kind="s2_train", hours_est=S2_HOURS[cond], after="",
                                 cmd=(f"synthetic/train_synth.py --task chain --cond {cond} --seed {seed} --lr {lr:g} "
                                      f"--steps {S2_STEPS} --curriculum-frac 0.5 --family lookup --n-ops 32 "
                                      f"--out {S2_OUT} --ckpt-dir {S2_CKPT} --device cuda")))
                jobs.append(dict(name=f"probe_{cond}_lr{lr:g}_s{seed}", kind="s2_probe", hours_est=0.1, after=train,
                                 cmd=(f"synthetic/probe_synth.py --ckpt {S2_CKPT}/chain/{res}.pt --d 8 --r-max 16 "
                                      f"--out {S2_OUT}/probes --device cuda")))
    return jobs


def _queue_row(name: str, kind: str, priority: int, hours: float, config: str = "", cmd: str = "", after: str = "") -> dict:
    return dict(name=name, kind=kind, priority=priority, status="pending", config=config, hours_est=hours, cmd=cmd,
                after=after)


def _write_csv(path: str, rows: list[dict], cols: list[str] | None = None) -> None:
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def build() -> tuple[list[dict], list[dict]]:
    """S1 manifest rows and the queue (S1 round 1, S2 with probes, the edge-rule job, the S3 benchmark)."""
    s1 = [s1_run(rung, p, r, k, seed, f) for rung in S1_RUNGS for p, r, k in S1_CELLS
          for seed in S1_SEEDS[rung] for f in S1_FACTORS]
    s1.sort(key=lambda x: -x["hours_est"])
    for i, row in enumerate(s1):
        row["priority"] = i
    s2 = s2_jobs()
    probes = [j for j in s2 if j["kind"] == "s2_probe"]
    train = sorted([j for j in s2 if j["kind"] == "s2_train"], key=lambda j: -j["hours_est"])
    q = [_queue_row("s1_edges", "s1_edges", 0, 0.0, cmd="scripts/make_supp.py --edges",
                    after=" ".join(r["name"] for r in s1))]
    q += [_queue_row(j["name"], j["kind"], 1 + i, j["hours_est"], cmd=j["cmd"], after=j["after"]) for i, j in enumerate(probes)]
    base = len(q)
    q += [_queue_row(r["name"], "s1", base + r["priority"], r["hours_est"], config=r["config"]) for r in s1]   # S1 first
    base = len(q)
    q += [_queue_row(j["name"], j["kind"], base + i, j["hours_est"], cmd=j["cmd"]) for i, j in enumerate(train)]
    q.append(_queue_row("s3_bench", "s3", len(q), 1.0, cmd="scripts/bench_inference.py --out results/bench/inference.json",
                        after=" ".join(r["name"] for r in q)))
    return s1, q


# ---- edge rule (S1) ----
def _results_rows(name: str) -> list[dict]:
    for d in ("runs", "results/runs"):
        p = os.path.join(d, name, "results.jsonl")
        if os.path.exists(p):
            return [json.loads(line) for line in open(p)]
    return []


def s1_loss(rung: str, placement: str, f: float, seed: int, mult: int) -> float | None:
    """End-of-cooldown main-validation loss of an S1 point; 1x comes from the main grid (80N/160N at 10M from the data
    extension, which continues the same trunk)."""
    r, k = (1, 0) if placement == "dense" else (4, 0)
    names = [s1_name(rung, placement, r, k, seed, f)]
    if f == 1 and rung == "10M":
        names.append(mc.run_name(rung, placement, r, k, seed, "_d160"))
    N = n_rung(rung)
    for name in names:
        for d in _results_rows(name):
            if abs(d["budget_tokens"] / N - mult) / mult < 0.02:
                return float(d["val"]["fwe"])
    return None


def edge_decision() -> dict:
    """{(rung, placement): [extra factors]} by the edge rule, with the losses it used. Raises if a point is missing."""
    out = {}
    for rung in S1_RUNGS:
        for placement, _, _ in S1_CELLS:
            extra, seen = [], {}
            for mult in S1_EDGE_BUDGETS[rung]:
                curve = {}
                for f in (*S1_FACTORS, 1.0):
                    vals = [s1_loss(rung, placement, f, seed, mult) for seed in S1_SEEDS[rung]]
                    if any(v is None for v in vals):
                        raise FileNotFoundError(f"missing S1 result: {rung} {placement} x{f} {mult}N")
                    curve[f] = sum(vals) / len(vals)
                best = min(curve, key=curve.get)
                seen[f"{mult}N"] = dict(curve={f"{k:g}": round(v, 5) for k, v in sorted(curve.items())}, best=best)
                if best in S1_EDGE and S1_EDGE[best] not in extra:
                    extra.append(S1_EDGE[best])
            out[f"{rung}/{placement}"] = dict(budgets=seen, extra=extra)
    return out


def edges() -> None:
    if os.path.exists(EDGE_DECISION):
        print(f"{EDGE_DECISION} exists; the edge rule is applied once")
        return
    dec = edge_decision()
    s1 = list(csv.DictReader(open(MANIFEST)))
    queue = list(csv.DictReader(open(QUEUE)))
    new_runs = []
    for key, d in dec.items():
        rung, placement = key.split("/")
        r, k = (1, 0) if placement == "dense" else (4, 0)
        for f in d["extra"]:
            for seed in S1_SEEDS[rung]:
                new_runs.append(s1_run(rung, placement, r, k, seed, f))
    r4 = [s1_name(rung, "middle", 4, 0, seed, f) for rung in S1_RUNGS for seed in S1_SEEDS[rung] for f in (*S1_FACTORS, 1.0)]
    r4 += [row["name"] for row in new_runs if row["placement"] == "middle"]
    queue_new = [_queue_row(row["name"], "s1_edge", -1, row["hours_est"], config=row["config"]) for row in new_runs]
    queue_new.append(_queue_row("s1_depth", "s1_q5", -1, 0.5, after=" ".join(row["name"] for row in new_runs
                                                                            if row["placement"] == "middle"),
                                cmd=(f"fit/token_analysis.py --stage raw --runs {' '.join(r4)} --valsets fwe "
                                     f"--tokens 1000000 --r-eval {Q5_R_EVAL} --ckpt branch40 --budget-mult 40 "
                                     f"--device cuda --out results/supp/token_analysis")))
    queue_new.append(_queue_row("s1_depth_means", "s1_q5", -1, 0.0, after="s1_depth",
                                cmd="fit/raw_means.py --raw results/supp/token_analysis/raw --out results/supp/depth_curve.csv"))
    if new_runs:
        _write_csv(MANIFEST, s1 + new_runs)
    _write_csv(QUEUE, queue + queue_new, QUEUE_COLS)
    os.makedirs(os.path.dirname(EDGE_DECISION), exist_ok=True)
    json.dump(dict(rule="lowest loss at 40N (10M: 40N or 160N) at 0.5x -> add 0.35x; at 1.41x -> add 2x; once",
                   decision=dec, added=[row["name"] for row in new_runs]), open(EDGE_DECISION, "w"), indent=1)
    print("edge rule: " + ("; ".join(f"{k} +{d['extra']}" for k, d in dec.items() if d["extra"]) or "no extra points"))
    print(f"appended {len(queue_new)} queue rows: " + " ".join(q["name"] for q in queue_new))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", action="store_true", help="apply the S1 edge rule (needs every round-1 result)")
    a = ap.parse_args()
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    mc.COMPILE = True     # as in the main grid
    if a.edges:
        edges()
        return
    s1, q = build()
    _write_csv(MANIFEST, s1)
    _write_csv(QUEUE, q, QUEUE_COLS)
    h1 = sum(r["hours_est"] for r in s1)
    h2 = sum(float(r["hours_est"]) for r in q if r["kind"].startswith("s2"))
    print(f"S1: {len(s1)} runs, {h1:.1f} GPU-h -> {MANIFEST}, configs in {SUPP_DIR}/")
    print(f"S2: {sum(r['kind'] == 's2_train' for r in q)} runs + probes, {h2:.1f} GPU-h")
    print(f"queue {QUEUE}: {len(q)} jobs, about {mc.makespan_days(sorted([float(r['hours_est']) for r in q], reverse=True)) * 24:.0f} h on 4 GPUs before edge points")


if __name__ == "__main__":
    main()
