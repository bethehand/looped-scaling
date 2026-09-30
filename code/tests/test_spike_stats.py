"""Loss spikes (03_偏离记录.md, 2026-09-26) and collapses (2026-09-30): rises within one phase only, after step 400,
resumes deduplicated; a trunk that never recovers marks every later branch diverged."""
import csv

from scripts.spike_stats import spike_row, write_spike_stats


def _log(path, rows):
    """rows: (phase, step, loss) or (phase, step, loss, quick val); 100 tokens per step."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "step", "tokens", "lr", "loss", "tok_per_s", "mfu", "quick_val"])
        for phase, step, loss, *q in rows:
            w.writerow([phase, step, 100 * step, 0, loss, 0, 0, f"{q[0]:.5f}" if q else "nan"])


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


def test_collapse_marks_later_branches(tmp_path):
    p = tmp_path / "train_log.csv"
    trunk = [(200, 5.0), (400, 4.5), (600, 4.2), (800, 5.5),   # +1.3 at 800 but back to 4.1 at 1000: a spike only
             (1000, 4.1), (1200, 4.0), (1400, 7.4), (1600, 7.4), (1800, 7.3)]   # from 1400 on it never comes back
    _log(p, [("trunk", s, q, q) for s, q in trunk]
         + [("cool_50000", 450, 6.1, 6.0),     # starts at step 400 (best trunk quick val so far 4.5), ends at 6.0
            ("cool_100000", 1000, 3.9, 3.9),   # starts at step 800, before the collapse onset (step 1200): kept
            ("cool_200000", 1900, 7.3, 7.3)])  # starts at step 1600, after the onset
    r = spike_row(str(p))
    assert r["collapse_step"] == 1200
    assert r["diverged_budgets"] == "50000 200000"
    assert r["n_spikes"] == 2                 # the recovered rise at 800 and the collapse at 1400
