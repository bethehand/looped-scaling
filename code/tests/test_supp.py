"""Supplementary experiments (scripts/make_supp.py): S1 grid, S2 jobs, the queue, and the S1 edge rule."""
import csv
import json
import math
import os
import shutil

import pytest

from scripts import make_supp as ms

ROOT = os.path.join(os.path.dirname(__file__), "..")


@pytest.fixture
def tmp_root(tmp_path, monkeypatch):
    os.makedirs(tmp_path / "configs")
    shutil.copy(os.path.join(ROOT, ms.LR_TABLE), tmp_path / ms.LR_TABLE)
    monkeypatch.chdir(tmp_path)
    ms.mc.COMPILE = True
    s1, q = ms.build()
    ms._write_csv(ms.MANIFEST, s1)
    ms._write_csv(ms.QUEUE, q, ms.QUEUE_COLS)
    return tmp_path


def _fake_results(f_opt):
    """Write results.jsonl for every S1 point and the 1x references: loss = base + 0.01 (log2 f - log2 f_opt)^2."""
    for rung in ms.S1_RUNGS:
        N = ms.n_rung(rung)
        for placement, r, k in ms.S1_CELLS:
            for seed in ms.S1_SEEDS[rung]:
                for f in (*ms.S1_FACTORS, 1.0):
                    rows = {}
                    for mult in ms.S1_BUDGETS[(rung, placement)]:
                        opt = f_opt.get((rung, placement, mult), 0.8)
                        loss = 3.5 + 0.01 * (math.log2(f) - math.log2(opt)) ** 2
                        name = ms.s1_name(rung, placement, r, k, seed, f)
                        if f == 1 and rung == "10M" and mult >= (80 if placement == "middle" else 160):
                            name = ms.mc.run_name(rung, placement, r, k, seed, "_d160")    # the data extension
                        rows.setdefault(name, []).append(dict(name=name, budget_tokens=int(mult * N) - 1000,
                                                              val=dict(fwe=loss)))
                    for name, rr in rows.items():
                        os.makedirs(os.path.join("runs", name), exist_ok=True)
                        with open(os.path.join("runs", name, "results.jsonl"), "a") as fh:
                            for d in rr:
                                fh.write(json.dumps(d) + "\n")


def test_grid_queue_and_jobs(tmp_root):
    s1 = list(csv.DictReader(open(ms.MANIFEST)))
    q = {r["name"]: r for r in csv.DictReader(open(ms.QUEUE))}
    assert len(s1) == 24 and len([r for r in s1 if r["rung"] == "20M"]) == 12
    lr0 = json.load(open(ms.LR_TABLE))["lr0"]
    for r in s1:
        cfg = ms.yaml.safe_load(open(r["config"]))
        assert cfg["train"]["lr0"] == pytest.approx(lr0[str(r["width"])] * float(r["lr_factor"]))
        assert cfg["train"]["compile"] is True and cfg["train"]["betas"] == [0.9, 0.95]
        assert len(cfg["train"]["budgets"]) == len(ms.S1_BUDGETS[(r["rung"], r["placement"])])
    kinds = [r["kind"] for r in q.values()]
    assert kinds.count("s2_train") == 24 and kinds.count("s2_probe") == 24 and kinds.count("s1") == 24
    p = q["probe_loop_r8_lr0.001_s3"]
    assert p["after"] == "synth_loop_r8_lr0.001_s3" and "lr_seeds/chain/loop_r8_lr0.001_s3_lut32.pt" in p["cmd"]
    t = q["synth_loop_step_lr0.0005_s4"]["cmd"]
    for flag in ("--steps 32000", "--curriculum-frac 0.5", "--family lookup", "--n-ops 32", "--seed 4", "--lr 0.0005"):
        assert flag in t
    assert set(q["s1_edges"]["after"].split()) == {r["name"] for r in s1}
    assert set(q["s3_bench"]["after"].split()) == set(q) - {"s3_bench"}
    # S1 goes before S2, longest first within each
    s1p = sorted((int(r["priority"]), float(r["hours_est"])) for r in q.values() if r["kind"] == "s1")
    s2p = [int(r["priority"]) for r in q.values() if r["kind"] == "s2_train"]
    assert max(p for p, _ in s1p) < min(s2p) and [h for _, h in s1p] == sorted([h for _, h in s1p], reverse=True)


def test_edge_rule_appends_points_once(tmp_root):
    _fake_results({("10M", "dense", 160): 0.3,        # best tested 0.5x at 160N only -> 0.35x
                   ("40M", "middle", 40): 2.5,        # best tested 1.41x -> 2x
                   ("20M", "dense", 80): 0.2})        # 80N is not an edge budget at 20M -> nothing
    ms.edges()
    dec = json.load(open(ms.EDGE_DECISION))
    extra = {k: d["extra"] for k, d in dec["decision"].items()}
    assert extra == {"10M/dense": [0.35], "10M/middle": [], "20M/dense": [], "20M/middle": [], "40M/dense": [],
                     "40M/middle": [2.0]}
    assert sorted(dec["added"]) == ["10M_dense_r1_lrs0.35_s42", "40M_middle_r4_full_lrs2_s42"]
    q = {r["name"]: r for r in csv.DictReader(open(ms.QUEUE))}
    assert q["40M_middle_r4_full_lrs2_s42"]["priority"] == "-1" and q["40M_middle_r4_full_lrs2_s42"]["config"]
    depth = q["s1_depth"]
    assert depth["after"] == "40M_middle_r4_full_lrs2_s42"
    runs = depth["cmd"].split("--runs ")[1].split(" --")[0].split()
    assert "40M_middle_r4_full_lrs2_s42" in runs and "20M_middle_r4_full_s43" in runs and "10M_middle_r4_full_s42" in runs
    assert len(runs) == 4 * 4 + 1 and not any("dense" in r for r in runs)
    assert q["s1_depth_means"]["after"] == "s1_depth"
    assert len(list(csv.DictReader(open(ms.MANIFEST)))) == 26
    n = len(q)
    ms.edges()                                         # applied once
    assert len(list(csv.DictReader(open(ms.QUEUE)))) == n


def test_edge_rule_needs_every_point(tmp_root):
    with pytest.raises(FileNotFoundError):
        ms.edge_decision()
