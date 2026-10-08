"""Every manifest row points to a config file that exists and loads into a valid run."""
import csv
import os

import pytest

from looped.train import load_run_config

ROOT = os.path.join(os.path.dirname(__file__), "..")


@pytest.mark.parametrize("manifest", ["configs/manifest.csv", "configs/manifest_sweep.csv", "configs/manifest_looplr.csv",
                                      "configs/manifest_ext_160m.csv", "configs/manifest_ext_160m_dense.csv",
                                      "configs/manifest_ext_data.csv", "configs/manifest_ext_randr.csv"])
def test_manifest_configs_exist_and_load(manifest):
    path = os.path.join(ROOT, manifest)
    if not os.path.exists(path):
        pytest.skip(f"{manifest} not generated")
    rows = list(csv.DictReader(open(path)))
    assert rows
    names = set()
    for r in rows:
        cfg = os.path.join(ROOT, r["config"])
        assert os.path.exists(cfg), f"{manifest}: {r['name']} -> missing {r['config']}"
        name, out_dir, m, t, d = load_run_config(cfg)
        assert name == r["name"] and name not in names
        names.add(name)
        assert out_dir == f"runs/{name}"
        assert t.budgets == [int(b) for b in r["budgets"].split()]


def test_extension_runs_continue_the_finished_trunks(tmp_path):
    import json
    from scripts import make_configs as mc
    lr = json.load(open(os.path.join(ROOT, "configs/lr_table_frozen.json")))
    man = mc.ext_runs(lr["lr0"], lr["beta2"], cfg_dir=str(tmp_path))
    rows = {r["name"]: r for rows in man.values() for r in rows}
    # branch checkpoints that exist in the finished seed-42 runs (file names from GS01, 2026-10-08)
    assert rows["20M_middle_r4_full_d80_s42"]["init_from"] == "runs/20M_middle_r4_full_s42/branch_858914816.pt"
    assert rows["10M_middle_r8_full_d160_s42"]["init_from"] == "runs/10M_middle_r8_full_s42/branch_492912640.pt"
    assert rows["10M_dense_r1_d160_s42"]["init_from"] == "runs/10M_dense_r1_s42/branch_985989120.pt"
    for name, r in rows.items():
        cfg = load_run_config(os.path.join(str(tmp_path), name + ".yaml"))
        t = cfg[3]
        assert t.budgets == [int(b) for b in r["budgets"].split()] and t.init_from == r["init_from"]
        assert t.r_sample == r["r_sample"]
        if r["init_from"]:
            start = int(r["init_from"].rsplit("_", 1)[1][:-3])
            assert start < min(t.budgets) * (1 - t.cooldown_frac)     # the new trunk continues past the inherited point
            assert r["tokens_processed"] < mc.tokens_processed(t.budgets, t.cooldown_frac)
    assert rows["20M_middle_r8_k4_rs1to8_s42"]["r_sample"] == "uniform:1:8"
    assert rows["160M_middle_r4_full_s43"]["budgets"].split()[-1] == rows["160M_dense_r1_s43"]["budgets"].split()[-1]
