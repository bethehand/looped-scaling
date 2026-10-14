"""The trainer refuses to overwrite the record of a started run with a different configuration."""
import json
import os

import pytest

from looped.train import check_run_record, run_has_progress


def _info(lr0=1e-3, compile_=False):
    return dict(name="x", model=dict(d_model=320, r=4), train=dict(lr0=lr0, seed=42, betas=(0.9, 0.95), compile=compile_),
                data=dict(train_shards="data/*.bin"), git_commit="abc")


def test_mismatch_on_started_run_raises(tmp_path):
    rec = tmp_path / "run_info.json"
    rec.write_text(json.dumps(_info(lr0=1e-3)))
    (tmp_path / "TRUNK_DONE").write_text("1")
    assert run_has_progress(str(tmp_path))
    with pytest.raises(RuntimeError, match="train.lr0"):
        check_run_record(str(rec), _info(lr0=2e-3))


def test_same_config_and_resume_free_settings_pass(tmp_path):
    rec = tmp_path / "run_info.json"
    rec.write_text(json.dumps(_info()))
    check_run_record(str(rec), _info())                    # tuples and lists compare equal after the JSON round trip
    check_run_record(str(rec), _info(compile_=True))       # compile does not change what the run computes


def test_no_progress_means_no_check(tmp_path):
    assert not run_has_progress(str(tmp_path))
    open(os.path.join(tmp_path, "results.jsonl"), "w").close()
    assert not run_has_progress(str(tmp_path))             # an empty results file is not progress
