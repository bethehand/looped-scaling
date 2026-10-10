"""Run the synthetic-task suite of extension C on several GPUs, one job per free GPU; resumable.

    python -u synthetic/suite.py --stage all --gpus 0,1,2,3       # sweep -> choose -> formal -> probe

Protocol (03_偏离记录.md, 2026-10-11, written before any sweep result):
  sweep  : chain task (32 fixed lookup tables), seed 1, 16000 steps, curriculum 0.5; every condition at lr 2.5e-4,
           5e-4 and 1e-3 (pilot runs with identical settings are reused). Edge rule, applied once per condition: best
           at 2.5e-4 -> also 1.25e-4; best at 1e-3 -> also 2e-3.
  choose : per condition, the lr with the highest mean accuracy over d = 1..8 at the condition's own loop count
           (r for loop_rN and loop_rand, loops = d for loop_step); in-distribution only, so the choice cannot favour
           the extrapolation tested by P3 -> results/synthetic/lr_choice.json
  formal : chain and pointer chasing, seeds 2 and 3, 32000 steps, the chosen lr (pointer chasing uses the chain
           choice; its longer sequences run as 2 micro-batches)
  probe  : intermediate-value probes on the formal chain checkpoints (d = 8, up to 16 loops)
A GPU takes a job only when none of this runner's jobs runs on it and nvidia-smi reports under 1 GB in use, so other
jobs on the machine are left alone and GPUs join as they free up. Jobs whose result json exists are skipped; a failed
job is retried once. Output of each job: logs/synth/<stage>/<task>/<name>.log
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONDS = ["dense8", "dense20", "dense36", "loop_r2", "loop_r4", "loop_r8", "loop_rand", "loop_step"]
EXEC_LAYERS = {"dense8": 8, "dense20": 20, "dense36": 36, "loop_r2": 12, "loop_r4": 20, "loop_r8": 36,
               "loop_rand": 22, "loop_step": 22}          # mean executed depth during training (random/step: r ~ 4.5)
SWEEP_LRS = [2.5e-4, 5e-4, 1e-3]
EDGE = {2.5e-4: 1.25e-4, 1e-3: 2e-3}
SWEEP = dict(seed=1, steps=16000, out="results/synthetic", ckpt="runs/synthetic")
FORMAL = dict(seeds=(2, 3), steps=32000, out="results/synthetic/formal", ckpt="runs/synthetic/formal")
CHOICE = "results/synthetic/lr_choice.json"
D_IN = range(1, 9)


def own_key(cond: str) -> str:
    if cond.startswith("dense"):
        return "acc"
    return "oracle" if cond == "loop_step" else "r=8" if cond in ("loop_r8", "loop_rand") else f"r={cond[-1]}"


def result_name(task: str, cond: str, lr: float, seed: int) -> str:
    return f"{cond}_lr{lr:g}_s{seed}" + ("_lut32" if task == "chain" else "")


def train_job(stage: str, task: str, cond: str, lr: float, seed: int, steps: int, out: str, ckpt: str, py: str) -> dict:
    name = result_name(task, cond, lr, seed)
    cmd = [py, "-u", "synthetic/train_synth.py", "--task", task, "--cond", cond, "--seed", str(seed), "--lr", f"{lr:g}",
           "--steps", str(steps), "--curriculum-frac", "0.5", "--out", out, "--ckpt-dir", ckpt, "--device", "cuda"]
    if task == "chain":
        cmd += ["--family", "lookup", "--n-ops", "32"]
    else:
        cmd += ["--micro", "2"]
    cost = EXEC_LAYERS[cond] * steps * (1.8 if task == "hops" else 1.0)
    return dict(stage=stage, name=f"{task}/{name}", done=os.path.join(out, task, name + ".json"), cmd=cmd, cost=cost)


def score(path: str, cond: str) -> float | None:
    if not os.path.exists(path):
        return None
    f = json.load(open(path))["final"]
    k = own_key(cond)
    return sum(f[str(d)][k] for d in D_IN) / len(D_IN)


def sweep_jobs(py: str) -> list[dict]:
    jobs = []
    for cond in CONDS:
        lrs = list(SWEEP_LRS)
        scores = {lr: score(train_job("sweep", "chain", cond, lr, SWEEP["seed"], SWEEP["steps"], SWEEP["out"],
                                      SWEEP["ckpt"], py)["done"], cond) for lr in lrs}
        if all(v is not None for v in scores.values()):
            best = max(scores, key=scores.get)
            if best in EDGE:
                lrs.append(EDGE[best])
        jobs += [train_job("sweep", "chain", cond, lr, SWEEP["seed"], SWEEP["steps"], SWEEP["out"], SWEEP["ckpt"], py)
                 for lr in lrs]
    return jobs


def choose(py: str) -> dict:
    out = {}
    for cond in CONDS:
        lrs = sorted({j["cmd"][j["cmd"].index("--lr") + 1] for j in sweep_jobs(py) if f"/{cond}_lr" in j["name"]}, key=float)
        sc = {lr: score(train_job("sweep", "chain", cond, float(lr), SWEEP["seed"], SWEEP["steps"], SWEEP["out"],
                                  SWEEP["ckpt"], py)["done"], cond) for lr in lrs}
        if any(v is None for v in sc.values()):
            raise RuntimeError(f"sweep incomplete for {cond}: {sc}")
        best = max(sc, key=sc.get)
        out[cond] = dict(lr=float(best), scores={k: round(v, 4) for k, v in sc.items()}, metric=own_key(cond))
    os.makedirs(os.path.dirname(CHOICE), exist_ok=True)
    json.dump(dict(rule="mean accuracy over d=1..8 at the condition's own loop count, chain sweep seed 1", choice=out),
              open(CHOICE, "w"), indent=1)
    print("[suite] learning rates: " + "  ".join(f"{c} {v['lr']:g}" for c, v in out.items()), flush=True)
    return out


def formal_jobs(py: str) -> list[dict]:
    choice = json.load(open(CHOICE))["choice"]
    jobs = []
    for task in ("chain", "hops"):
        tj = [train_job("formal", task, cond, choice[cond]["lr"], seed, FORMAL["steps"], FORMAL["out"], FORMAL["ckpt"], py)
              for seed in FORMAL["seeds"] for cond in CONDS]
        jobs += sorted(tj, key=lambda j: -j["cost"])              # chain first, longest first within a task
    return jobs


def probe_jobs(py: str) -> list[dict]:
    choice = json.load(open(CHOICE))["choice"]
    jobs = []
    for seed in FORMAL["seeds"]:
        for cond in CONDS:
            name = result_name("chain", cond, choice[cond]["lr"], seed)
            ck = os.path.join(FORMAL["ckpt"], "chain", name + ".pt")
            out = os.path.join(FORMAL["out"], "probes")
            jobs.append(dict(stage="probe", name=f"probe/{name}", done=os.path.join(out, f"{name}__d8.json"),
                             cmd=[py, "-u", "synthetic/probe_synth.py", "--ckpt", ck, "--d", "8", "--r-max", "16",
                                  "--out", out, "--device", "cuda"], cost=1.0))
    return jobs


def gpu_memory() -> dict[int, int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return {int(a): int(b) for a, b in (line.split(",") for line in out.strip().splitlines())}
    except Exception:
        return {}


def run(jobs_fn, gpus: list[int], poll: float, max_retries: int = 1) -> None:
    """Run the jobs returned by jobs_fn() (re-evaluated every round, so jobs can appear as results come in)."""
    running: dict[int, tuple[dict, subprocess.Popen]] = {}
    tries: dict[str, int] = {}
    while True:
        for g, (job, proc) in list(running.items()):
            if proc.poll() is not None:
                ok = proc.returncode == 0 and os.path.exists(job["done"])
                print(f"[suite] gpu{g} {job['name']} {'done' if ok else f'FAILED rc={proc.returncode}'}", flush=True)
                del running[g]
        busy = {job["name"] for job, _ in running.values()}
        todo = [j for j in jobs_fn() if not os.path.exists(j["done"]) and j["name"] not in busy
                and tries.get(j["name"], 0) <= max_retries]
        if not todo and not running:
            return
        mem = gpu_memory()
        for g in gpus:
            if not todo:
                break
            if g in running or mem.get(g, 10 ** 6) >= 1000:
                continue
            job = todo.pop(0)
            tries[job["name"]] = tries.get(job["name"], 0) + 1
            log = os.path.join("logs", "synth", job["stage"], job["name"] + ".log")
            os.makedirs(os.path.dirname(log), exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            proc = subprocess.Popen(job["cmd"], stdout=open(log, "a"), stderr=subprocess.STDOUT, env=env)
            running[g] = (job, proc)
            print(f"[suite] gpu{g} <- {job['name']} (try {tries[job['name']]})", flush=True)
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["sweep", "choose", "formal", "probe", "all"])
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true", help="print the jobs of the requested stage and exit")
    a = ap.parse_args()
    os.chdir(ROOT)
    gpus = [int(g) for g in a.gpus.split(",") if g.strip()]
    py = a.python
    if a.dry_run:
        fn = {"sweep": sweep_jobs, "formal": formal_jobs, "probe": probe_jobs}.get(a.stage, sweep_jobs)
        for j in fn(py):
            print(("done " if os.path.exists(j["done"]) else "todo ") + j["name"])
        return
    if a.stage in ("sweep", "all"):
        run(lambda: sweep_jobs(py), gpus, a.poll)
        print("[suite] sweep finished", flush=True)
    if a.stage in ("choose", "all"):
        choose(py)
    if a.stage in ("formal", "all"):
        run(lambda: formal_jobs(py), gpus, a.poll)
        print("[suite] formal runs finished", flush=True)
    if a.stage in ("probe", "all"):
        run(lambda: probe_jobs(py), gpus, a.poll)
        print("[suite] probes finished", flush=True)


if __name__ == "__main__":
    main()
