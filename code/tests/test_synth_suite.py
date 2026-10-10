"""Synthetic suite runner: sweep grid with the edge rule, learning-rate choice, formal and probe job lists, scheduling."""
import json
import os
import sys

import pytest

from synthetic import suite


@pytest.fixture
def tmp_suite(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "SWEEP", dict(seed=1, steps=16000, out=str(tmp_path / "sweep"), ckpt=str(tmp_path / "ck")))
    monkeypatch.setattr(suite, "FORMAL", dict(seeds=(2, 3), steps=32000, out=str(tmp_path / "formal"), ckpt=str(tmp_path / "fck")))
    monkeypatch.setattr(suite, "CHOICE", str(tmp_path / "lr_choice.json"))
    return tmp_path


def _write(job, acc):
    os.makedirs(os.path.dirname(job["done"]), exist_ok=True)
    cond = job["name"].split("/")[1].split("_lr")[0]
    key = suite.own_key(cond)
    json.dump(dict(final={str(d): {key: acc} for d in range(1, 25)}), open(job["done"], "w"))


def test_sweep_edge_rule_choice_and_formal_jobs(tmp_suite):
    best = {"dense8": 1e-3, "loop_r8": 2.5e-4}                  # edge cases; all others peak at 5e-4
    for j in suite.sweep_jobs("py"):
        cond = j["name"].split("/")[1].split("_lr")[0]
        lr = float(j["cmd"][j["cmd"].index("--lr") + 1])
        _write(j, 0.9 if lr == best.get(cond, 5e-4) else 0.5)
    jobs = suite.sweep_jobs("py")
    todo = [j["name"] for j in jobs if not os.path.exists(j["done"])]
    assert sorted(todo) == ["chain/dense8_lr0.002_s1_lut32", "chain/loop_r8_lr0.000125_s1_lut32"]
    with pytest.raises(RuntimeError):
        suite.choose("py")                                     # extension runs missing
    for j in jobs:
        if j["name"] in todo:
            _write(j, 0.95 if "loop_r8" in j["name"] else 0.1)  # loop_r8 improves again at 1.25e-4; dense8 does not
    assert len(suite.sweep_jobs("py")) == 26                   # the edge rule is applied once, not recursively
    ch = suite.choose("py")
    assert ch["dense8"]["lr"] == 1e-3 and ch["loop_r8"]["lr"] == 1.25e-4 and ch["loop_step"]["lr"] == 5e-4
    assert ch["loop_step"]["metric"] == "oracle" and ch["loop_r2"]["metric"] == "r=2" and ch["dense36"]["metric"] == "acc"
    formal = suite.formal_jobs("py")
    assert len(formal) == 32 and all(j["name"].startswith("chain/") for j in formal[:16])
    assert all("--micro" in j["cmd"] for j in formal[16:]) and all("lookup" in j["cmd"] for j in formal[:16])
    assert all(j["cmd"][j["cmd"].index("--steps") + 1] == "32000" for j in formal)
    assert "chain/loop_r8_lr0.000125_s2_lut32" in {j["name"] for j in formal}
    assert len(suite.probe_jobs("py")) == 16


def test_runner_launches_on_free_gpus_and_finishes(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "gpu_memory", lambda: {0: 0, 1: 5000})   # gpu 1 busy with someone else's job
    monkeypatch.chdir(tmp_path)
    jobs = [dict(stage="t", name=f"x/{i}", done=str(tmp_path / f"done{i}.json"),
                 cmd=[sys.executable, "-c", f"open(r'{tmp_path / f'done{i}.json'}', 'w').write('{{}}')"], cost=1)
            for i in range(3)]
    suite.run(lambda: jobs, [0, 1], poll=0.05)
    assert all(os.path.exists(j["done"]) for j in jobs)
    logs = sorted(os.listdir(tmp_path / "logs" / "synth" / "t" / "x"))
    assert logs == ["0.log", "1.log", "2.log"]
