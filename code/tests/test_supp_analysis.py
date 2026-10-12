"""fit/supp_analysis.py: S1 gaps and readings, Q5, S2 and S3 on synthetic inputs with known answers."""
import json
import math
import os
import shutil

import pandas as pd
import pytest

from fit import supp_analysis as sa
from scripts import make_supp as ms

ROOT = os.path.join(os.path.dirname(__file__), "..")
BASE_DIFF = {"10M": 0.0, "20M": -0.03, "40M": -0.06}      # looped minus dense at each side's best learning rate
LOOP_OPT = {"10M": 0.5, "20M": 1.0, "40M": 1.0}           # the 10M looped model prefers half the shared rate


def _loss(rung, placement, f):
    if placement == "dense":
        return 3.5 + 0.05 * math.log2(f) ** 2
    return 3.5 + BASE_DIFF[rung] + 0.05 * (math.log2(f) - math.log2(LOOP_OPT[rung])) ** 2


def _write_s1():
    for rung in ms.S1_RUNGS:
        N = ms.n_rung(rung)
        for placement, r, k in ms.S1_CELLS:
            for seed in ms.S1_SEEDS[rung]:
                for f in (*ms.S1_FACTORS, 1.0):
                    for mult in ms.S1_BUDGETS[(rung, placement)]:
                        name = ms.s1_name(rung, placement, r, k, seed, f)
                        if f == 1 and rung == "10M" and mult >= (80 if placement == "middle" else 160):
                            name = ms.mc.run_name(rung, placement, r, k, seed, "_d160")
                        os.makedirs(os.path.join("results", "runs", name), exist_ok=True)
                        with open(os.path.join("results", "runs", name, "results.jsonl"), "a") as fh:
                            fh.write(json.dumps(dict(name=name, budget_tokens=int(mult * N) - 999,
                                                     val=dict(fwe=_loss(rung, placement, f)))) + "\n")


@pytest.fixture
def tmp_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("results/supp")
    return tmp_path


def test_reading_categories():
    assert sa.reading(0.005, 0.06, -0.1) == "does not hold"
    assert sa.reading(0.04, 0.06, -0.01) == "holds"
    assert sa.reading(0.04, 0.06, 0.01) == "weakened"
    assert sa.reading(0.02, 0.06, -0.1) == "weakened"


def test_s1_gaps_and_readings(tmp_cwd):
    _write_s1()
    sa.s1("results/supp")
    g = pd.read_csv("results/supp/s1_gaps.csv")
    row = g[(g.rung == "10M") & (g.mult == 40)].iloc[0]
    assert row.delta_1 == pytest.approx(0.05) and row.delta_star == pytest.approx(0.0, abs=1e-12)
    assert row.h == pytest.approx(0.05) and row.loop_best_f == 0.5 and row.loop_vertex_how == "EDGE"
    row = g[(g.rung == "40M") & (g.mult == 40)].iloc[0]
    assert row.delta_star == pytest.approx(-0.06) and row.dense_vertex_f == pytest.approx(1.0)
    assert row.dense_vertex_how == "parabola" and row.dense_vertex_loss == pytest.approx(3.5)
    assert set(g[g.rung == "10M"].mult) == {10, 20, 40, 80, 160}
    rd = json.load(open("results/supp/s1_readings.json"))
    q1 = rd["Q1_scale"]["40N"]
    assert q1["I_1"] == pytest.approx(0.11) and q1["I_star"] == pytest.approx(0.06)
    assert q1["reading"] == "holds" and q1["monotone"]
    assert rd["Q2_data"]["10M"]["reading"] == "does not hold"          # the fake gap does not change with data
    assert rd["Q3_equal_compute"]["n_loop_better_retuned"] == 6        # flat dense curve: 20M and 40M, 3 budgets each
    assert rd["Q4_bias"]["max_abs_h"] == pytest.approx(0.05) and not rd["Q4_bias"]["within_tol"]


def test_s1_incomplete_has_no_readings(tmp_cwd):
    _write_s1()
    shutil.rmtree("results/runs/40M_middle_r4_full_lrs0.5_s42")
    sa.s1("results/supp")
    assert os.path.exists("results/supp/s1_gaps.csv") and not os.path.exists("results/supp/s1_readings.json")


def test_q5_readings(tmp_cwd):
    rows = []
    for rung in ms.S1_RUNGS:
        for f in (0.5, 0.71, 1.0, 1.41):
            for seed in ms.S1_SEEDS[rung]:
                run = ms.s1_name(rung, "middle", 4, 0, seed, f)
                for r in (1, 2, 3, 4, 5, 6, 8, 12, 16):
                    bump = 0.0 if r == 4 else (0.002 if f == 0.5 else 0.1 * f)
                    rows.append(dict(run=run, ckpt="branch40", valset="fwe", r_eval=r, mean_loss=3.5 + bump, n_tokens=1))
    pd.DataFrame(rows).to_csv("results/supp/depth_curve.csv", index=False)
    sa.q5("results/supp")
    out = json.load(open("results/supp/q5_readings.json"))
    assert out["replicates_5_14"] and out["more_loops_help"] == []
    assert out["spread_4_to_16"]["20M x0.5"] == pytest.approx(0.002)


def test_s2_on_the_formal_runs(tmp_cwd):
    for d in ("formal",):
        shutil.copytree(os.path.join(ROOT, "results/synthetic", d), os.path.join("results/synthetic", d))
    sa.s2("results/supp", new_dir="results/synthetic/formal", formal_dir="results/synthetic/formal")
    out = json.load(open("results/supp/s2_readings.json"))
    assert set(out["cells"]) == {"loop_r4 lr0.000125", "loop_r8 lr0.00025", "loop_rand lr0.000125", "loop_step lr0.000125"}
    assert out["formal_reference"]["loop_r8"] == pytest.approx(3.2) and out["formal_reference"]["loop_r4"] == pytest.approx(3.8)
    for c in out["cells"].values():   # identical runs: all learned, same E -> "does not support"
        assert c["n_learned"] == 5 and c["R2"] == "does not support"
    runs = pd.read_csv("results/supp/s2_runs.csv")
    assert {"slope", "persistence", "E", "learned"} <= set(runs.columns)


def test_s3_equal_parameters_and_equal_loss(tmp_cwd):
    for f in ("all_results.csv", "ext_160m.csv", "ext_160m_dense.csv"):
        shutil.copy(os.path.join(ROOT, "results", f), os.path.join("results", f))
    rows = []
    for w in (320, 448, 640, 896, 1280):
        for placement, r in (("dense", 1), ("middle", 4)):
            cfg = ms.mc.model_cfg(w, placement, r, 0)
            N = ms.mc.analytic_N(cfg)["N"]
            rows.append(dict(width=w, placement=placement, r=r, executed_layers=cfg.executed_layers, N=N,
                             weight_mb=N / 5e5, decode_ms_graph_b1=0.01 * cfg.executed_layers * w / 320,
                             kv_kb_per_token=2 * cfg.executed_layers * w * 2 / 1024))
    os.makedirs("results/bench")
    json.dump(dict(meta={}, rows=rows), open("results/bench/inference.json", "w"))
    sa.s3("results/supp")
    cells = pd.read_csv("results/supp/s3_cells.csv")
    m = cells[(cells.placement == "middle") & (cells.width == 640)].iloc[0]
    assert m.kv_kb_per_token_vs_dense == pytest.approx(20 / 8) and m.decode_ms_graph_b1_vs_dense == pytest.approx(20 / 8)
    eq = pd.read_csv("results/supp/s3_equal_loss.csv").set_index("width")
    assert set(eq.index) == {320, 448, 640, 896, 1280}
    # the looped model is worse than the dense one at 10M and better from 40M on (results 5.12)
    assert eq.loc[320, "N_dense_equal"] < eq.loc[320, "N"] and eq.loc[640, "N_dense_equal"] > eq.loc[640, "N"]
    assert eq.loc[1280, "extrapolated"]               # better than the largest dense model: extrapolated
