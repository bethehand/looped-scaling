"""One job per GPU from a manifest. Re-runnable; jobs resume from their own checkpoints.

    python -u scripts/run_queue.py --manifest configs/manifest.csv --gpus 0,1,2,3 2>&1 | tee -a logs/queue_main.log
Each job's output is echoed live with a [gpuN] prefix (use --quiet to turn that off) and always written to
runs/<job>/stdout.log.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import threading
import time


def _pump(proc, log_path, prefix, echo):
    """Copy a job's output line by line into its own log file and, unless --quiet, onto the queue's stdout."""
    with open(log_path, "ab") as f:
        for line in iter(proc.stdout.readline, b""):
            f.write(line)
            f.flush()
            if echo:
                sys.stdout.write(prefix + line.decode("utf-8", "replace"))
                sys.stdout.flush()


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
    ap.add_argument("--quiet", action="store_true", help="do not echo job output (it is always written to runs/<job>/stdout.log)")
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
        for gpu, (proc, row, pump) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            pump.join(timeout=10)   # let the last lines reach the log and the screen
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
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
            proc = subprocess.Popen([args.python, "-m", "looped.train", "--config", row["config"], "--device", "cuda"],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
            pump = threading.Thread(target=_pump, daemon=True,
                                    args=(proc, f"runs/{row['name']}/stdout.log", f"[gpu{gpu}] ", not args.quiet))
            pump.start()
            row["status"] = "running"
            running[gpu] = (proc, row, pump)
            print(f"[queue] gpu{gpu} <- {row['name']} ({row['card_days']} card-days est.)", flush=True)
            write_manifest(state, rows)
        if not running and not pending:
            print("[queue] all jobs finished", flush=True)
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
