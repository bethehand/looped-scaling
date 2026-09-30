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


def test_diverged_and_spikes_from_spike_table(tmp_path):
    run = tmp_path / "runs" / "40M_whole_r8_k4_s43"
    run.mkdir(parents=True)
    rows = [dict(name=run.name, placement="whole", r=8, k_bwd=4, budget_tokens=b, val={"fwe": v}, val_avg={"fwe": v})
            for b, v in ((1000, 3.3), (4000, 7.4))]
    (run / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    spikes = tmp_path / "spikes.csv"
    spikes.write_text("name,n_spikes,diverged_budgets\n40M_whole_r8_k4_s43,3,4000\n")
    out = tmp_path / "out.csv"
    subprocess.run([sys.executable, os.path.join(ROOT, "fit", "collect_results.py"), "--manifest", str(tmp_path / "none.csv"),
                    "--runs", str(tmp_path / "runs"), "--spikes", str(spikes), "--out", str(out)], check=True)
    got = {int(r["budget_tokens"]): r for r in csv.DictReader(open(out))}
    assert got[1000]["diverged"] == "False" and got[4000]["diverged"] == "True"
    assert got[1000]["n_spikes"] == "3"
