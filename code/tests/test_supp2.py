"""Supplementary experiments S4-S9 (scripts/make_supp2.py): the S4 configs continue the right checkpoints."""
import os
import shutil

import yaml

from scripts import make_supp as ms
from scripts import make_supp2 as m2

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_s4_configs_continue_the_finished_dense_trunks(tmp_path, monkeypatch):
    os.makedirs(tmp_path / "configs")
    shutil.copy(os.path.join(ROOT, ms.LR_TABLE), tmp_path / ms.LR_TABLE)
    monkeypatch.chdir(tmp_path)
    rows = m2.s4_rows()
    by_name = {r["name"]: r for r in rows}
    assert list(by_name) == ["10M_dense_r1_d640_s42", "10M_dense_r1_d640_s43", "20M_dense_r1_d320_s42",
                             "20M_dense_r1_d320_s43", "10M_middle_r4_full_lrs0.35_s43"]
    # the checkpoints that exist on the training machine: the d160 runs' trunk at 128N, the main-grid 20M trunk at 64N
    assert by_name["10M_dense_r1_d640_s42"]["init_from"] == "runs/10M_dense_r1_d160_s42/branch_1971978240.pt"
    assert by_name["20M_dense_r1_d320_s43"]["init_from"] == "runs/20M_dense_r1_s43/branch_1717960704.pt"
    for name, row in by_name.items():
        cfg = yaml.safe_load(open(row["config"]))
        assert cfg["train"].get("init_from", "") == row["init_from"] and "data_offset" not in cfg["train"]
        assert max(cfg["train"]["budgets"]) <= m2.STREAM_TOKENS
    c10 = yaml.safe_load(open(by_name["10M_dense_r1_d640_s42"]["config"]))
    N10 = ms.n_rung("10M")
    assert [round(b / N10) for b in c10["train"]["budgets"]] == [320, 640] and c10["train"]["seed"] == 42
    assert c10["model"]["placement"] == "dense" and c10["train"]["lr0"] == ms._lr_tables()[0]["320"]
    c20 = yaml.safe_load(open(by_name["20M_dense_r1_d320_s43"]["config"]))
    assert [round(b / ms.n_rung("20M")) for b in c20["train"]["budgets"]] == [160, 320] and c20["train"]["seed"] == 43
    lp = yaml.safe_load(open(by_name["10M_middle_r4_full_lrs0.35_s43"]["config"]))
    assert lp["model"]["placement"] == "middle" and lp["model"]["r"] == 4 and lp["train"]["seed"] == 43
    assert abs(lp["train"]["lr0"] - 0.35 * ms._lr_tables()[0]["320"]) < 1e-9
    assert [round(b / N10) for b in lp["train"]["budgets"]] == [10, 20, 40, 80, 160]
    # the inherited trunk is not counted in the time estimate
    assert 0 < by_name["10M_dense_r1_d640_s42"]["hours_est"] < by_name["10M_middle_r4_full_lrs0.35_s43"]["hours_est"] * 4
