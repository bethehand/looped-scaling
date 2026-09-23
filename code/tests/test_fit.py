"""Fitting pipeline recovers known parameters from synthetic F1 data."""
import numpy as np
import pandas as pd

from fit.fit_laws import fit, predict_log, leave_one_rung_out


def _synthetic(phi_by_cell, seed=0, noise=0.003):
    rng = np.random.default_rng(seed)
    E, A, alpha, B, beta = 1.8, 400.0, 0.30, 800.0, 0.30
    rows = []
    rungs = {"10M": 10e6, "20M": 20e6, "40M": 40e6, "80M": 80e6}
    for rung, N in rungs.items():
        for m in (5, 10, 20, 40, 80):
            rows.append(dict(rung=rung, placement="dense", cell="dense", r=1, N_once=N, N_rec=0.0, D=m * N))
        for cell, phi in phi_by_cell.items():
            for r in (2, 4, 8):
                for m in (10, 20, 40):
                    rows.append(dict(rung=rung, placement=cell.split("/")[0], cell=cell, r=r,
                                     N_once=0.5 * N, N_rec=0.5 * N, D=m * N))
    df = pd.DataFrame(rows)
    theta = np.array([phi_by_cell.get(c, 0.0) for c in df.cell])
    shared = [np.log(E), np.log(A), alpha, np.log(B), beta]
    df["L"] = np.exp(predict_log(shared, df.N_once.values, df.N_rec.values, df.r.values, df.D.values, "F1", theta)
                     + rng.normal(0, noise, len(df)))
    return df, shared


def test_two_stage_recovers_phi():
    truth = {"middle/full": 0.6, "middle/trunc": 0.4}
    df, shared = _synthetic(truth)
    ruler = df[df.placement == "dense"]
    s1 = fit(ruler, "F1", n_starts=60)
    # E/A/alpha trade off on a 4-rung ruler (loosely identified, as iso-depth also reports); phi must still recover
    assert abs(s1["shared"]["alpha"] - 0.30) < 0.10
    assert abs(s1["shared"]["beta"] - 0.30) < 0.05
    s2 = fit(df[df.placement != "dense"], "F1", n_starts=60, shared_fixed=np.array(s1["shared_raw"]))
    for c, v in truth.items():
        assert abs(s2["theta"][c] - v) < 0.05, (c, s2["theta"][c])
    j = fit(df, "F1", n_starts=100)
    assert abs(j["shared"]["alpha"] - 0.30) < 0.05
    assert "dense" not in j["theta"]


def test_joint_and_loro_run():
    df, _ = _synthetic({"whole/full": 0.5})
    j = fit(df, "F1", n_starts=40)
    assert abs(j["theta"]["whole/full"] - 0.5) < 0.08
    loro = leave_one_rung_out(df, "F1", n_starts=20)
    assert set(loro) == {"10M", "20M", "40M", "80M"}
    assert all(v["rmse_log"] < 0.02 for v in loro.values())
