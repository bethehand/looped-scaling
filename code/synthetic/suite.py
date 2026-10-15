"""Run the synthetic-task suite of extension C on several GPUs, one job per free GPU; resumable.

    python -u synthetic/suite.py --stage all --gpus 0,1,2,3       # sweep -> choose -> formal -> probe

Protocol (docs/deviation-log.md, 2026-10-11, written before any sweep result):
  sweep  : chain task (32 fixed lookup tables), seed 1, 16000 steps, curriculum 0.5; every condition at lr 2.5e-4,
           5e-4 and 1e-3 (pilot runs with identical settings are reused). Edge rule, applied once per condition: best
           at 2.5e-4 -> also 1.25e-4; best at 1e-3 -> also 2e-3.
  choose : per condition, the lr with the highest mean accuracy over d = 1..8 at the condition's own loop count
           (r for loop_rN and loop_rand, loops = d for loop_step); in-distribution only, so the choice cannot favour
           the extrapolation tested by P3 -> results/synthetic/lr_choice.json
  formal : chain, seeds 2-6 (4-6 added 2026-10-11), 32000 steps, the chosen lr; pointer chasing (2 micro-batches)
           was dropped on 2026-10-11 after no condition learned it under the chain-calibrated recipe
  probe  : intermediate-value probes on the formal chain checkpoints (d = 8, up to 16 loops)
Supplementary experiments S5 and S9 (docs/supplementary-experiments-2.md, frozen 2026-10-15; --stage supp2): the same
protocol (sweep -> choice -> formal -> probes) for the chain-of-thought and pause-token baselines (dense 8 and 36 layers
with --format cot, dense 8 with --format pause) and the new condition loop_step_extra; results in
results/synthetic/supp2/{sweep,formal,probes}, the learning-rate choice in results/synthetic/lr_choice_supp2.json.
A GPU takes a job only when none of this runner's jobs runs on it and nvidia-smi reports less than --gpu-free-mb MiB
in use (default 1000, i.e. an idle card), so other jobs on the machine are left alone and GPUs join as they free up;
a higher value lets these small jobs share cards with the text runs. Jobs whose result json exists are skipped; a failed
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
               "loop_rand": 22, "loop_step": 22, "loop_step_extra": 30}   # mean executed depth during training (random/step: r ~ 4.5)
SUPP2_CONDS = [("dense36", "cot"), ("dense8", "cot"), ("dense8", "pause"), ("loop_step_extra", "single")]   # S5, S5, S5, S9
SUPP2 = dict(sweep_out="results/synthetic/supp2/sweep", sweep_ckpt="runs/synthetic/supp2/sweep",
             out="results/synthetic/supp2/formal", ckpt="runs/synthetic/supp2/formal", probes="results/synthetic/supp2/probes")
SUPP2_CHOICE = "results/synthetic/lr_choice_supp2.json"
SWEEP_LRS = [2.5e-4, 5e-4, 1e-3]
EDGE = {2.5e-4: 1.25e-4, 1e-3: 2e-3}
SWEEP = dict(seed=1, steps=16000, out="results/synthetic", ckpt="runs/synthetic")
# 2026-10-11 (deviation log): pointer chasing dropped by the author after no condition learned it; seeds 4-6 added for the
# chain task with the same settings, declared before their results
FORMAL = dict(seeds=(2, 3, 4, 5, 6), steps=32000, out="results/synthetic/formal", ckpt="runs/synthetic/formal")
FORMAL_TASKS = ("chain",)
CHOICE = "results/synthetic/lr_choice.json"
D_IN = range(1, 9)


def own_key(cond: str) -> str:
    if cond.startswith("dense"):
        return "acc"
    return "oracle" if cond in ("loop_step", "loop_step_extra") else "r=8" if cond in ("loop_r8", "loop_rand") else f"r={cond[-1]}"


def result_name(task: str, cond: str, lr: float, seed: int, fmt: str = "single") -> str:
    return f"{cond}{'' if fmt == 'single' else '_' + fmt}_lr{lr:g}_s{seed}" + ("_lut32" if task == "chain" else "")


def train_job(stage: str, task: str, cond: str, lr: float, seed: int, steps: int, out: str, ckpt: str, py: str,
              fmt: str = "single") -> dict:
    name = result_name(task, cond, lr, seed, fmt)
    cmd = [py, "-u", "synthetic/train_synth.py", "--task", task, "--cond", cond, "--seed", str(seed), "--lr", f"{lr:g}",
           "--steps", str(steps), "--curriculum-frac", "0.5", "--out", out, "--ckpt-dir", ckpt, "--device", "cuda"]
    if task == "chain":
        cmd += ["--family", "lookup", "--n-ops", "32"]
    else:
        cmd += ["--micro", "2"]
    if fmt != "single":
        cmd += ["--format", fmt]
    cost = EXEC_LAYERS[cond] * steps * (1.8 if task == "hops" else 1.0) * (1.3 if fmt != "single" else 1.0)
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
    for task in FORMAL_TASKS:
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


# ---- supplementary experiments S5 and S9 (the same protocol as the formal suite, separate output folders) ----
def supp2_key(cond: str, fmt: str) -> str:
    return cond if fmt == "single" else f"{cond}_{fmt}"


def supp2_sweep_jobs(py: str) -> list[dict]:
    jobs = []
    for cond, fmt in SUPP2_CONDS:
        lrs = list(SWEEP_LRS)
        scores = {lr: score(train_job("supp2_sweep", "chain", cond, lr, SWEEP["seed"], SWEEP["steps"], SUPP2["sweep_out"],
                                      SUPP2["sweep_ckpt"], py, fmt)["done"], cond) for lr in lrs}
        if all(v is not None for v in scores.values()):
            best = max(scores, key=scores.get)
            if best in EDGE:
                lrs.append(EDGE[best])
        jobs += [train_job("supp2_sweep", "chain", cond, lr, SWEEP["seed"], SWEEP["steps"], SUPP2["sweep_out"],
                           SUPP2["sweep_ckpt"], py, fmt) for lr in lrs]
    return sorted(jobs, key=lambda j: -j["cost"])


def supp2_choose(py: str) -> dict:
    out = {}
    for cond, fmt in SUPP2_CONDS:
        key = supp2_key(cond, fmt)
        lrs = sorted({j["cmd"][j["cmd"].index("--lr") + 1] for j in supp2_sweep_jobs(py) if f"/{key}_lr" in j["name"]}, key=float)
        sc = {lr: score(train_job("supp2_sweep", "chain", cond, float(lr), SWEEP["seed"], SWEEP["steps"], SUPP2["sweep_out"],
                                  SUPP2["sweep_ckpt"], py, fmt)["done"], cond) for lr in lrs}
        if any(v is None for v in sc.values()):
            raise RuntimeError(f"sweep incomplete for {key}: {sc}")
        best = max(sc, key=sc.get)
        out[key] = dict(lr=float(best), scores={k: round(v, 4) for k, v in sc.items()}, metric=own_key(cond), cond=cond, format=fmt)
    os.makedirs(os.path.dirname(SUPP2_CHOICE), exist_ok=True)
    json.dump(dict(rule="mean final-answer accuracy over d=1..8 at the condition's own loop count, chain sweep seed 1",
                   choice=out), open(SUPP2_CHOICE, "w"), indent=1)
    print("[suite] supp2 learning rates: " + "  ".join(f"{k} {v['lr']:g}" for k, v in out.items()), flush=True)
    return out


def supp2_formal_jobs(py: str) -> list[dict]:
    choice = json.load(open(SUPP2_CHOICE))["choice"]
    jobs = [train_job("supp2_formal", "chain", cond, choice[supp2_key(cond, fmt)]["lr"], seed, FORMAL["steps"],
                      SUPP2["out"], SUPP2["ckpt"], py, fmt)
            for seed in FORMAL["seeds"] for cond, fmt in SUPP2_CONDS]
    return sorted(jobs, key=lambda j: -j["cost"])


def supp2_probe_jobs(py: str) -> list[dict]:
    choice = json.load(open(SUPP2_CHOICE))["choice"]
    jobs = []
    for cond, fmt in SUPP2_CONDS:
        if not cond.startswith("loop"):
            continue
        for seed in FORMAL["seeds"]:
            name = result_name("chain", cond, choice[supp2_key(cond, fmt)]["lr"], seed, fmt)
            ck = os.path.join(SUPP2["ckpt"], "chain", name + ".pt")
            jobs.append(dict(stage="supp2_probe", name=f"probe/{name}", done=os.path.join(SUPP2["probes"], f"{name}__d8.json"),
                             cmd=[py, "-u", "synthetic/probe_synth.py", "--ckpt", ck, "--d", "8", "--r-max", "16",
                                  "--out", SUPP2["probes"], "--device", "cuda"], cost=1.0))
    return jobs


def gpu_memory() -> dict[int, int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return {int(a): int(b) for a, b in (line.split(",") for line in out.strip().splitlines())}
    except Exception:
        return {}


def run(jobs_fn, gpus: list[int], poll: float, max_retries: int = 1, gpu_free_mb: int = 1000) -> None:
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
            if g in running or mem.get(g, 10 ** 6) >= gpu_free_mb:
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
    ap.add_argument("--stage", default="all", choices=["sweep", "choose", "formal", "probe", "all", "supp2"])
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true", help="print the jobs of the requested stage and exit")
    ap.add_argument("--gpu-free-mb", type=int, default=1000,
                    help="a GPU takes a job when nvidia-smi reports less than this much memory in use (MiB)")
    a = ap.parse_args()
    os.chdir(ROOT)
    gpus = [int(g) for g in a.gpus.split(",") if g.strip()]
    py = a.python
    if a.dry_run:
        fn = {"sweep": sweep_jobs, "formal": formal_jobs, "probe": probe_jobs, "supp2": supp2_sweep_jobs}.get(a.stage, sweep_jobs)
        for j in fn(py):
            print(("done " if os.path.exists(j["done"]) else "todo ") + j["name"])
        return
    if a.stage in ("sweep", "all"):
        run(lambda: sweep_jobs(py), gpus, a.poll, gpu_free_mb=a.gpu_free_mb)
        print("[suite] sweep finished", flush=True)
    if a.stage in ("choose", "all"):
        choose(py)
    if a.stage in ("formal", "all"):
        run(lambda: formal_jobs(py), gpus, a.poll, gpu_free_mb=a.gpu_free_mb)
        print("[suite] formal runs finished", flush=True)
    if a.stage in ("probe", "all"):
        run(lambda: probe_jobs(py), gpus, a.poll, gpu_free_mb=a.gpu_free_mb)
        print("[suite] probes finished", flush=True)
    if a.stage == "supp2":
        run(lambda: supp2_sweep_jobs(py), gpus, a.poll, gpu_free_mb=a.gpu_free_mb)
        print("[suite] supp2 sweep finished", flush=True)
        supp2_choose(py)
        run(lambda: supp2_formal_jobs(py), gpus, a.poll, gpu_free_mb=a.gpu_free_mb)
        print("[suite] supp2 formal runs finished", flush=True)
        run(lambda: supp2_probe_jobs(py), gpus, a.poll, gpu_free_mb=a.gpu_free_mb)
        print("[suite] supp2 probes finished", flush=True)


if __name__ == "__main__":
    main()
