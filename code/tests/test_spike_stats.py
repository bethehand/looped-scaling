"""Loss-spike counting (03_偏离记录.md, 2026-09-26): rises within one phase only, after step 400, resumes deduplicated."""
import csv

from scripts.spike_stats import spike_row, write_spike_stats


def _log(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "step", "tokens", "lr", "loss", "tok_per_s", "mfu", "quick_val"])
        for phase, step, loss in rows:
            w.writerow([phase, step, 0, 0, loss, 0, 0, "nan"])


def test_spikes_counted_per_phase(tmp_path):
    p = tmp_path / "train_log.csv"
    _log(p, [("trunk", 390, 9.0), ("trunk", 400, 4.0),     # before step 400: a large drop and no counted rise
             ("trunk", 410, 5.2),                           # +1.2 at 410: spike
             ("trunk", 420, 4.0), ("trunk", 430, 4.6),      # +0.6: minor only
             ("trunk", 440, 3.5),
             ("trunk", 430, 3.6), ("trunk", 440, 3.5),      # resumed from 420: these replace the earlier lines
             ("cool_1", 420, 0.8), ("cool_1", 430, 3.8),    # old logs: a branch's first line read far too low
             ("cool_1", 440, 3.7)])                         # (and no rise is taken across phases)
    r = spike_row(str(p))
    assert r["n_spikes"] == 1 and r["first_spike_step"] == 410
    assert r["n_minor"] == 1                                 # 430 now reads 3.6 (after the resume), so only 410 counts
    assert abs(r["max_rise"] - 1.2) < 1e-9 and r["last_step"] == 440


def test_write_spike_stats(tmp_path):
    for name, rows in (("a", [("trunk", 10, 9.0), ("trunk", 410, 3.0), ("trunk", 420, 4.5)]),
                       ("b", [("trunk", 10, 9.0), ("trunk", 410, 3.0)])):
        (tmp_path / "runs" / name).mkdir(parents=True)
        _log(tmp_path / "runs" / name / "train_log.csv", rows)
    out = tmp_path / "spikes.csv"
    assert write_spike_stats(str(tmp_path / "runs"), str(out)) == 2
    got = {r["name"]: r for r in csv.DictReader(open(out))}
    assert got["a"]["n_spikes"] == "1" and got["b"]["n_spikes"] == "0"
