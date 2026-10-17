"""Step 2 of the plan of 2026-10-15 (exploratory re-analyses, no new runs): the ruler-free multiplier and local phi
(fit/local_phi.py), the bounds on the extrapolated equal-compute comparisons (fit/iso_flop_bounds.py) and the published
results restated as multipliers (fit/external_multipliers.py)."""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fit"))
import external_multipliers as em  # noqa: E402
import iso_flop_bounds as ifb  # noqa: E402
import iso_flop_raw as ifr  # noqa: E402
import local_phi as lp  # noqa: E402

E, A, ALPHA, B, BETA = 1.8, 60.0, 0.25, 400.0, 0.3


def law(N, D):
    return E + A * N ** -ALPHA + B * D ** -BETA


def curves(sizes=(16e6, 32e6, 64e6, 128e6), tpps=(5, 10, 20, 40, 80)):
    return {f"{int(N / 1e6)}M": (N, np.log([t * N for t in tpps]), np.array([law(N, t * N) for t in tpps]))
            for N in sizes}


def test_local_phi_recovers_a_known_effective_size():
    cur = curves()
    N_once, N_rec, r, phi = 14e6, 18e6, 4, 0.45
    N = N_once + N_rec
    D = 20 * N
    n_eff = N_once + r ** phi * N_rec
    Ns, status = lp.dense_size_at(law(n_eff, D), D, cur)
    assert status == "interpolated"
    assert abs(Ns / n_eff - 1) < 0.02                       # linear interpolation in log N between rungs a factor 2 apart
    g = [dict(rung="32M", cell="middle/full", r=r, tpp=20, D=D, N=N, N_once=N_once, N_rec=N_rec, seeds={42: law(n_eff, D)})]
    res = lp.evaluate(g, cur)
    assert abs(res.M.iloc[0] - n_eff / N) < 0.03 and abs(res.phi.iloc[0] - phi) < 0.03


def test_local_phi_flags_extrapolation_and_censoring():
    cur = curves()
    D = 20 * 32e6
    L64, L128 = law(64e6, D), law(128e6, D)
    Ns, status = lp.dense_size_at(L128 - 0.3 * (L64 - L128), D, cur)   # a little better than the largest dense model
    assert status == "extrapolated" and 128e6 < Ns <= 256e6
    _, status = lp.dense_size_at(L128 - 5 * (L64 - L128), D, cur)     # far beyond a factor 2: only a bound
    assert status == "above"
    _, status = lp.dense_size_at(law(16e6, D) + 1.0, D, cur)
    assert status == "below"
    flat = {"a": (10e6, np.log([1e8, 1e9]), np.array([4.0, 3.5])), "b": (20e6, np.log([1e8, 1e9]), np.array([4.1, 3.6]))}
    assert lp.dense_size_at(3.7, 5e8, flat)[1] == "none"              # bigger is not better at this D: undefined


def test_local_phi_trend_regression_recovers_a_plane():
    rows = []
    for rung, N in (("10M", 10e6), ("20M", 20e6), ("40M", 40e6), ("80M", 80e6)):
        for tpp in (10, 20, 40):
            x, y = np.log2(N / lp.N_REF), np.log2(tpp / lp.TPP_REF)
            rows.append(dict(rung=rung, cell="c", r=4, tpp=tpp, N_ref=N, status="interpolated",
                             M=2 ** (-0.2 + 0.3 * x + 0.2 * y - 0.05 * x * y)))
    t = lp.trends(pd.DataFrame(rows))["c r=4"]
    assert np.allclose([t["b0"], t["b_size"], t["b_data"], t["b_interaction"]], [-0.2, 0.3, 0.2, -0.05])
    be = lp.break_even(t)
    assert be["20"] == pytest.approx(lp.N_REF * 2 ** (0.2 / 0.3) / 1e6)


def test_iso_flop_bounds_extensions():
    mult = np.array([5.0, 10, 20, 40, 80])
    L = 2.0 + 3.0 * mult ** -0.4
    Ef, Bf, bf = ifb.fit_power_law(mult, L)
    assert (Ef, Bf, bf) == pytest.approx((2.0, 3.0, 0.4), abs=1e-6)
    ruler = pd.Series(L, index=mult)
    for x in (30.0, 160.0):                                            # last_slope is iso_flop_raw.py's rule
        assert ifb.extend(mult, L, x, "last_slope")[0] == pytest.approx(ifr.dense_at(ruler, x)[0])
    assert ifb.extend(mult, L, 160, "flat") == (L[-1], True)
    assert ifb.extend(mult, L, 160, "power_law")[0] == pytest.approx(2.0 + 3.0 * 160 ** -0.4)


def test_iso_flop_bounds_on_the_committed_results(tmp_path, monkeypatch):
    if not os.path.exists("results/all_results.csv"):
        pytest.skip("results not present")
    out = str(tmp_path / "bounds.csv")
    monkeypatch.setattr(sys, "argv", ["iso_flop_bounds.py", "--out", out, "--summary", out + ".json"])
    ifb.main()
    b = pd.read_csv(out)
    raw = pd.read_csv("results/fit/iso_flop_raw.csv")
    m = raw.merge(b, on=["rung", "cell", "seed", "mult"])
    assert len(m) == len(raw) == 320
    assert np.allclose(m.gap_flop, m.gap_last_slope, atol=1e-4)
    ex = b[b.extrapolated]
    assert len(ex) == 99 and (b[~b.extrapolated].gap_last_slope > 0).all()
    assert (ex.gap_last_slope >= ex.gap_power_law - 1e-9).all() and (ex.gap_power_law >= ex.gap_flat - 0.02).all()
    import json
    bt = pd.DataFrame(json.load(open(out + ".json"))["backtest"])
    # on the measured curves the last-slope rule always flatters the dense model and the power law never does
    assert (bt[bt.method == "last_slope"]["max"] < 0).all() and (bt[bt.method == "power_law"]["min"] > 0).all()


def test_external_multipliers_arithmetic():
    rows = {(r["study"], r["setting"]): r for r in em.rows()}
    assert rows[("Schwethelm et al. 2026", "iso-depth, r=4")]["M"] == pytest.approx(1.455, abs=0.002)
    frac = (math.log(8.16) - math.log(7.90)) / (math.log(8.16) - math.log(7.40))     # 12 -> 24 layers is a factor 2
    assert rows[("Saunshi et al. 2025", "Loop 12x2")]["M"] == pytest.approx(2 ** frac, abs=1e-9)
    frac = (math.log(9.25) - math.log(8.79)) / (math.log(9.25) - math.log(8.75))     # 4x6 lies between 6 and 8 layers
    assert rows[("Saunshi et al. 2025", "Loop 4x6")]["M"] == pytest.approx(6 * (8 / 6) ** frac / 4, abs=1e-9)
    assert rows[("Zhu et al. 2025 (Ouro)", "53M, T=4")]["relation"] == "<"
    assert rows[("Prairie et al. 2026 (Parcae)", "1.3B, T=8")]["relation"] == ">"
    ns, rel = em.from_ladder(2.5, [(1.0, 2.0), (4.0, 3.0)], higher_is_better=True)
    assert rel == "=" and ns == pytest.approx(2.0)
