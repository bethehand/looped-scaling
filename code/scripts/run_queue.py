"""One job per GPU from a manifest. Re-runnable; jobs resume from their own checkpoints.

    python scripts/run_queue.py --manifest configs/manifest.csv --gpus 0,1,2,3 [--max-retries 2]
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time


def read_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--state", default=None, help="status file; default runs/queue_<manifest file name>")
    args = ap.parse_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    # the tracked manifest is never modified; progress lives in an untracked state file under runs/
    state = args.state or os.path.join("runs", "queue_" + os.path.basename(args.manifest))
    rows = read_manifest(args.manifest)
    if os.path.exists(state):
        prev = {r["name"]: r for r in read_manifest(state)}
        for r in rows:
            if r["name"] in prev:
                r["status"], r["retries"] = prev[r["name"]]["status"], prev[r["name"]].get("retries", "0")
    for r in rows:
        r.setdefault("retries", "0")
        if r["status"] == "running":   # stale from a previous queue process
            r["status"] = "pending"
    os.makedirs(os.path.dirname(state) or ".", exist_ok=True)
    write_manifest(state, rows)
    print(f"[queue] {len(rows)} jobs, state file {state}", flush=True)
    running: dict[str, tuple[subprocess.Popen, dict]] = {}
    while True:
        # reap
        for gpu, (proc, row) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            row["status"] = "done" if rc == 0 else "failed"
            if rc != 0:
                row["retries"] = str(int(row["retries"]) + 1)
                if int(row["retries"]) <= args.max_retries:
                    row["status"] = "pending"
            print(f"[queue] gpu{gpu} {row['name']} exited rc={rc} -> {row['status']}", flush=True)
            del running[gpu]
            write_manifest(state, rows)
        # launch
        pending = sorted([r for r in rows if r["status"] == "pending"], key=lambda r: int(r["priority"]))
        for gpu in gpus:
            if gpu in running or not pending:
                continue
            row = pending.pop(0)
            os.makedirs(f"runs/{row['name']}", exist_ok=True)
            log = open(f"runs/{row['name']}/stdout.log", "a")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            proc = subprocess.Popen([args.python, "-m", "looped.train", "--config", row["config"], "--device", "cuda"],
                                    stdout=log, stderr=subprocess.STDOUT, env=env)
            row["status"] = "running"
            running[gpu] = (proc, row)
            print(f"[queue] gpu{gpu} <- {row['name']} ({row['card_days']} card-days est.)", flush=True)
            write_manifest(state, rows)
        if not running and not pending:
            print("[queue] all jobs finished", flush=True)
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
