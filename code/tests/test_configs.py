"""Every manifest row points to a config file that exists and loads into a valid run."""
import csv
import os

import pytest

from looped.train import load_run_config

ROOT = os.path.join(os.path.dirname(__file__), "..")


@pytest.mark.parametrize("manifest", ["configs/manifest.csv", "configs/manifest_sweep.csv", "configs/manifest_looplr.csv"])
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
