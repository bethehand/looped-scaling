"""collect_results keeps only the runs of the manifest it is given (results/runs holds every experiment)."""
import csv
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(tmp_path, name, placement="dense", k=0):
    d = tmp_path / "runs" / name
    d.mkdir(parents=True)
    row = dict(name=name, placement=placement, r=1, k_bwd=k, budget_tokens=1000,
               val={"fwe": 3.5}, val_avg={"fwe": 3.5})
    (d / "results.jsonl").write_text(json.dumps(row) + "\n")


def test_only_manifest_runs_are_collected(tmp_path):
    _run(tmp_path, "20M_whole_r8_full_lrx1.0_s42", placement="whole")
    _run(tmp_path, "20M_dense_r1_lr1.50e-03_s42")          # a sweep run, not in this manifest
    man = tmp_path / "manifest.csv"
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "rung", "N_rung"])
        w.writeheader()
        w.writerow(dict(name="20M_whole_r8_full_lrx1.0_s42", rung="20M", N_rung=100))
    out = tmp_path / "out.csv"
    subprocess.run([sys.executable, os.path.join(ROOT, "fit", "collect_results.py"), "--manifest", str(man),
                    "--runs", str(tmp_path / "runs"), "--out", str(out)], check=True)
    rows = list(csv.DictReader(open(out)))
    assert [r["name"] for r in rows] == ["20M_whole_r8_full_lrx1.0_s42"]
    assert rows[0]["budget_mult"] == "10.0"
