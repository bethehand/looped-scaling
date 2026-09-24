"""Looped-cell learning-rate check: decision rule and the multiplier applied by make_configs."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
CELLS = [("middle", 2, 0), ("middle", 4, 0), ("middle", 4, 2), ("middle", 8, 0), ("middle", 8, 4),
         ("whole", 2, 0), ("whole", 4, 0), ("whole", 4, 2), ("whole", 8, 0), ("whole", 8, 4)]


def _synthetic(tmp_path, shift_of_r):
    rows = []
    rng = np.random.default_rng(0)
    for pl, r, k in CELLS:
        opt = shift_of_r(r)
        for f in (0.5, 0.71, 1.0):
            rows.append(dict(name=f"20M_{pl}_r{r}_{'full' if k == 0 else 'k' + str(k)}_lrx{f}_s42", placement=pl, r=r,
                             k_bwd=k, val_fwe=3.6 + 0.04 * (np.log2(f) - opt) ** 2 + rng.normal(0, 0.0005)))
    path = tmp_path / "looplr.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _run(tmp_path, csv):
    out = tmp_path / "mult.json"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "pick_loop_lr.py"), "--results", str(csv),
                    "--out", str(out)], check=True, capture_output=True)
    return json.load(open(out))


def test_no_shift(tmp_path):
    assert _run(tmp_path, _synthetic(tmp_path, lambda r: 0.0))["mode"] == "none"


def test_shared_shift(tmp_path):
    d = _run(tmp_path, _synthetic(tmp_path, lambda r: -0.5))
    assert d["mode"] == "shared" and abs(d["shared"] - 2 ** -0.5) < 0.06


def test_per_r_shift(tmp_path):
    d = _run(tmp_path, _synthetic(tmp_path, lambda r: -0.15 * np.log2(r) - 0.3))
    assert d["mode"] == "per_r"
    assert d["per_r"]["8"] < d["per_r"]["4"] < d["per_r"]["2"] < 1


def test_make_configs_applies_multiplier(tmp_path):
    mult = tmp_path / "m.json"
    json.dump({"mode": "shared", "shared": 0.7}, open(mult, "w"))
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_configs.py"), "--out", str(tmp_path / "c"),
                    "--lr-table", os.path.join(ROOT, "configs", "lr_table_frozen.json"), "--loop-lr-mult", str(mult)],
                   check=True, capture_output=True, cwd=ROOT)
    dense = yaml.safe_load(open(tmp_path / "c" / "runs" / "40M_dense_r1_s42.yaml"))["train"]["lr0"]
    loop = yaml.safe_load(open(tmp_path / "c" / "runs" / "40M_whole_r8_k4_s42.yaml"))["train"]["lr0"]
    assert abs(dense - 0.00183) < 1e-9 and abs(loop - 0.00183 * 0.7) < 1e-9
