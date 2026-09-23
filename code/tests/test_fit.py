"""Fitting pipeline recovers known parameters from synthetic F1 data."""
import numpy as np
import pandas as pd

from fit.fit_laws import fit, predict_log, leave_one_rung_out


def _synthetic(phi_by_cell, seed=0, noise=0.003):
    rng = np.random.default_rng(seed)
    E, A, alpha, B, beta = 1.8, 400.0, 0.30, 800.0, 0.30
    rows = []
    rungs = {"10M": 10e6, "20M": 20e6, "40M": 40e6, "80M": 80e6, "160M": 160e6}
    for rung, N in rungs.items():
        for m in (5, 10, 20, 40, 80):
            rows.append(dict(rung=rung, placement="dense", cell="dense", r=1, N_once=N, N_rec=0.0, D=m * N))
        if rung == "160M":   # the real design: 160M is a ruler-only point
            continue
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
    # E/A/alpha are only loosely identified on a short ruler (iso-depth reports the same); phi must still recover.
    # The 16x ruler span (10M..160M) keeps alpha within tolerance; the in-sample fit must be at noise level.
    assert abs(s1["shared"]["alpha"] - 0.30) < 0.10
    assert abs(s1["shared"]["beta"] - 0.30) < 0.06
    assert s1["loss"] < 5e-4
    s2 = fit(df[df.placement != "dense"], "F1", n_starts=60, shared_fixed=np.array(s1["shared_raw"]))
    for c, v in truth.items():
        assert abs(s2["theta"][c] - v) < 0.05, (c, s2["theta"][c])
    j = fit(df, "F1", n_starts=100)
    assert abs(j["shared"]["alpha"] - 0.30) < 0.06
    assert "dense" not in j["theta"]


def test_joint_and_loro_run():
    df, _ = _synthetic({"whole/full": 0.5})
    j = fit(df, "F1", n_starts=40)
    assert abs(j["theta"]["whole/full"] - 0.5) < 0.08
    loro = leave_one_rung_out(df, "F1", n_starts=20)
    assert set(loro) == {"10M", "20M", "40M", "80M", "160M"}
    assert all(v["rmse_log"] < 0.03 for v in loro.values()), loro


def test_load_parameter_definitions(tmp_path):
    from fit.fit_laws import load
    rows = pd.DataFrame([
        dict(name="d", rung="10M", placement="dense", backprop="full", cell="dense/full", accounting="iso_token", r=1,
             k_bwd=0, seed=42, N_once=15_406_400, N_rec=0, N=15_406_400, emb_in=5_242_880, budget_tokens=1e8, valavg_fwe=3.5),
        dict(name="m", rung="10M", placement="middle", backprop="full", cell="middle/full", accounting="iso_token", r=4,
             k_bwd=0, seed=42, N_once=10_325_000, N_rec=5_286_000, N=15_611_000, emb_in=5_242_880, budget_tokens=1e8,
             valavg_fwe=3.4),
    ])
    path = tmp_path / "r.csv"; rows.to_csv(path, index=False)
    a = load(str(path), "valavg_fwe", None, "with_head")
    b = load(str(path), "valavg_fwe", None, "no_head")
    assert (a.N_once - b.N_once == 5_242_880).all() and (a.N_rec == b.N_rec).all()
    assert b.loc[b.placement == "dense", "N"].iloc[0] == 15_406_400 - 5_242_880
