"""run_queue: arbitrary commands, dependencies between jobs, rows appended while the queue runs."""
import csv
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
QUEUE = os.path.join(ROOT, "scripts", "run_queue.py")
COLS = ["name", "priority", "status", "config", "hours_est", "cmd", "after"]


def _manifest(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow(dict(dict(priority=i, status="pending", config="", hours_est=0, cmd="", after=""), **r))


def _write_cmd(tmp_path, name, need=None, fail=False):
    """A job that checks its prerequisite file, then writes <name>.done (or exits 1)."""
    code = ""
    if need:
        code += f"import os, sys; sys.exit(3) if not os.path.exists(r'{tmp_path / (need + '.done')}') else None; "
    code += "import sys; sys.exit(1)" if fail else f"open(r'{tmp_path / (name + '.done')}', 'w').write('x')"
    return f'-c "{code}"'


def _run(tmp_path, manifest, *extra):
    return subprocess.run([sys.executable, QUEUE, "--manifest", str(manifest), "--gpus", "0,1", "--poll", "0.05",
                           "--quiet", *extra], cwd=tmp_path, capture_output=True, text=True, timeout=120)


def _state(tmp_path, manifest):
    return {r["name"]: r for r in csv.DictReader(open(tmp_path / "runs" / ("queue_" + manifest.name)))}


def test_commands_run_in_dependency_order(tmp_path):
    m = tmp_path / "q.csv"
    # c is listed first, but must wait for b, which waits for a
    _manifest(m, [dict(name="c", cmd=_write_cmd(tmp_path, "c", need="b"), after="b"),
                  dict(name="b", cmd=_write_cmd(tmp_path, "b", need="a"), after="a"),
                  dict(name="a", cmd=_write_cmd(tmp_path, "a"))])
    out = _run(tmp_path, m)
    assert out.returncode == 0, out.stdout + out.stderr
    assert all((tmp_path / f"{n}.done").exists() for n in "abc")
    assert {r["status"] for r in _state(tmp_path, m).values()} == {"done"}
    assert "all jobs finished" in out.stdout


def test_follow_picks_up_appended_rows(tmp_path):
    m = tmp_path / "q.csv"
    # job a appends job z to the manifest it is running from (as the edge-rule job of the supplementary runs does)
    script = tmp_path / "append.py"
    script.write_text("import csv\n"
                      f"w = csv.writer(open(r'{m}', 'a', newline=''))\n"
                      f"w.writerow(['z', 5, 'pending', '', 0, {_write_cmd(tmp_path, 'z')!r}, 'a'])\n")
    _manifest(m, [dict(name="a", cmd=str(script))])
    out = _run(tmp_path, m, "--follow")
    assert out.returncode == 0, out.stdout + out.stderr
    assert (tmp_path / "z.done").exists()
    assert set(_state(tmp_path, m)) == {"a", "z"} and "appended: z" in out.stdout


def test_failed_prerequisite_stops_its_dependents(tmp_path):
    m = tmp_path / "q.csv"
    _manifest(m, [dict(name="a", cmd=_write_cmd(tmp_path, "a", fail=True)),
                  dict(name="b", cmd=_write_cmd(tmp_path, "b"), after="a"),
                  dict(name="c", cmd=_write_cmd(tmp_path, "c"))])
    out = _run(tmp_path, m, "--max-retries", "0")
    assert out.returncode == 0, out.stdout + out.stderr
    st = _state(tmp_path, m)
    assert st["a"]["status"] == "failed" and st["b"]["status"] == "pending" and st["c"]["status"] == "done"
    assert not (tmp_path / "b.done").exists() and "prerequisites never finished for: b" in out.stdout
