"""Ruler-free effective-parameter multiplier and local phi (exploratory; step 2 of the plan of 2026-10-15).

For every looped cell, rung, r and iso-token budget D, find the size N* of a dense model that reaches the same loss
with the same number of training tokens, using only the measured dense runs; no scaling law is fitted.
  1. Each dense rung's seed-mean loss is interpolated linearly in log D at D (never extrapolated in D).
  2. Across the dense rungs available at D, log N is interpolated linearly in the loss. Beyond the smallest or the
     largest available rung, the end segment is extended by at most a factor of 2 in N (flagged `extrapolated`);
     further out the value is censored (only a bound is reported).
Then
    multiplier M = N* / N                                 N: the looped model's own parameters (with head, as in
                                                          the primary fit; M = 1 means "as good as a dense model
                                                          of the same size")
    local phi    = log((N* - N_once) / N_rec) / log r     the pre-registered form F1, N_eff = N_once + r^phi N_rec
Intervals: seeds are resampled (each looped group and each dense rung independently), 95% percentile intervals.
Trends: per cell and r, least squares of log2 M on x = log2(N / 20M), y = log2(D/N / 20) and x*y on the group
means, refitted on every resample. The 80M rung has one seed, so its points carry no seed resampling of their own.

    python fit/local_phi.py      -> results/fit/local_phi.csv, results/fit/local_phi_trends.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

FILES = ["results/all_results.csv", "results/ext_data.csv", "results/ext_160m.csv", "results/ext_160m_dense.csv"]
ISO_TOKEN_TPP = (10, 20, 40, 80, 160)
MAX_EXTRAP = 2.0          # at most one rung (a factor of 2 in N) beyond the dense rungs available at D
N_REF, TPP_REF = 20e6, 20.0


def load(files=FILES, loss_col="val_fwe", n_def="with_head", exclude_spiking=False) -> pd.DataFrame:
    """n_def as in fit/fit_laws.py: with_head (primary) or no_head (N without the output head, the iso-depth
    convention). exclude_spiking drops every run with a loss spike (as the no_spikes robustness fit does)."""
    df = pd.concat([pd.read_csv(f) for f in files if os.path.exists(f)], ignore_index=True)
    df = df[~df.diverged.astype(str).str.lower().eq("true")].copy()
    if exclude_spiking:
        df = df[pd.to_numeric(df.n_spikes, errors="coerce").fillna(0) == 0]
    if n_def == "no_head":
        df["N_once"] = df.N_once - df.emb_in
        df["N"] = df.N - df.emb_in
    elif n_def != "with_head":
        raise ValueError(n_def)
    df["tpp"] = (df.budget_tokens / df.N_rung).round(0)
    df["L"] = df[loss_col]
    looped = (df.placement != "dense") & df.tpp.isin(ISO_TOKEN_TPP)   # iso-token budgets; rho-scaled budgets dropped
    return df[(df.placement == "dense") | looped].reset_index(drop=True)


def _mean_by_seed(values: dict, rng) -> float:
    """values: seed -> loss. With rng, the seeds are resampled with replacement."""
    v = np.array(list(values.values()), dtype=float)
    if rng is not None and len(v) > 1:
        v = v[rng.integers(0, len(v), len(v))]
    return float(v.mean())


def dense_curves(df: pd.DataFrame, rng=None) -> dict:
    """rung -> (N, sorted log D array, loss array); seed means per budget (resampled with rng)."""
    out = {}
    for rung, g in df[df.placement == "dense"].groupby("rung"):
        pts = sorted((float(D), _mean_by_seed(dict(zip(gg.seed, gg.L)), rng)) for D, gg in g.groupby("budget_tokens"))
        out[rung] = (float(g.N.iloc[0]), np.log([p[0] for p in pts]), np.array([p[1] for p in pts]))
    return out


def dense_size_at(L: float, D: float, curves: dict) -> tuple[float, str]:
    """N* with L_dense(N*, D) = L. Returns (N*, status); status: interpolated | extrapolated | below | above | none."""
    x = np.log(D)
    avail = []
    for N, xs, ys in curves.values():
        if xs[0] - 1e-3 <= x <= xs[-1] + 1e-3:
            avail.append((N, float(np.interp(x, xs, ys))))
    avail.sort()
    if len(avail) < 2:
        return np.nan, "none"
    logN = np.log([a[0] for a in avail])
    Ls = np.array([a[1] for a in avail])
    for i in range(len(avail) - 1):
        hi, lo = Ls[i], Ls[i + 1]
        if lo <= L <= hi and hi > lo:
            return float(np.exp(logN[i] + (hi - L) / (hi - lo) * (logN[i + 1] - logN[i]))), "interpolated"
    if L > Ls[0]:                     # worse than the smallest dense model available at D
        i, edge = 0, logN[0]
    else:                             # better than the largest
        i, edge = len(avail) - 2, logN[-1]
    hi, lo = Ls[i], Ls[i + 1]
    if hi <= lo:
        return np.nan, "none"
    logNs = logN[i] + (hi - L) / (hi - lo) * (logN[i + 1] - logN[i])
    if abs(logNs - edge) <= np.log(MAX_EXTRAP):
        return float(np.exp(logNs)), "extrapolated"
    return float(np.exp(edge + np.sign(logNs - edge) * np.log(MAX_EXTRAP))), ("below" if logNs < edge else "above")


def groups(df: pd.DataFrame) -> list[dict]:
    out = []
    looped = df[df.placement != "dense"]
    for (rung, cell, r, tpp), g in looped.groupby(["rung", "cell", "r", "tpp"]):
        first = g.iloc[0]
        out.append(dict(rung=rung, cell=cell, r=int(r), tpp=int(tpp), D=float(g.budget_tokens.mean()),
                        N=float(first.N), N_once=float(first.N_once), N_rec=float(first.N_rec),
                        seeds=dict(zip(g.seed, g.L))))
    return out


def evaluate(grps: list[dict], curves: dict, rng=None) -> pd.DataFrame:
    rows = []
    for g in grps:
        L = _mean_by_seed(g["seeds"], rng)
        Ns, status = dense_size_at(L, g["D"], curves)
        M = Ns / g["N"]
        arg = (Ns - g["N_once"]) / g["N_rec"]
        phi = np.log(arg) / np.log(g["r"]) if arg > 0 else -np.inf
        rows.append(dict(rung=g["rung"], cell=g["cell"], r=g["r"], tpp=g["tpp"], loss=L, N_star=Ns, M=M, phi=phi,
                         status=status))
    return pd.DataFrame(rows)


COEFS = ("b0", "b_size", "b_data", "b_interaction")


def trends(res: pd.DataFrame, n_min: int = 6) -> dict:
    """Per (cell, r): log2 M = b0 + b1 x + b2 y + b3 x y, x = log2(N/20M), y = log2(tpp/20); usable points only.
    Also the residual standard deviation and classical least-squares 95% intervals (which, unlike the seed
    resampling, include the misfit of this linear form)."""
    from scipy import stats
    out = {}
    use = res[res.status.isin(["interpolated", "extrapolated"])]
    for (cell, r), g in use.groupby(["cell", "r"]):
        if len(g) < n_min or g.rung.nunique() < 2 or g.tpp.nunique() < 2:
            continue
        x = np.log2(g.N_ref.values / N_REF)
        y = np.log2(g.tpp.values / TPP_REF)
        X = np.column_stack([np.ones_like(x), x, y, x * y])
        z = np.log2(g.M.values)
        b, *_ = np.linalg.lstsq(X, z, rcond=None)
        dof = len(z) - 4
        s = float(np.sqrt(((z - X @ b) ** 2).sum() / dof)) if dof > 0 else np.nan
        se = s * np.sqrt(np.diag(np.linalg.pinv(X.T @ X))) if dof > 0 else np.full(4, np.nan)
        tq = stats.t.ppf(0.975, dof) if dof > 0 else np.nan
        out[f"{cell} r={r}"] = dict(zip(COEFS, b), n=len(g), resid_sd=s,
                                    ols_ci95={c: [b[i] - tq * se[i], b[i] + tq * se[i]] for i, c in enumerate(COEFS)})
    return out


def break_even(c: dict, tpps=(10, 20, 40, 80, 160)) -> dict:
    """Dense-equivalent break-even: the N (in M params) at which log2 M = 0, for each D/N."""
    out = {}
    for t in tpps:
        y = np.log2(t / TPP_REF)
        slope = c["b_size"] + c["b_interaction"] * y
        out[str(t)] = float(N_REF * 2 ** (-(c["b0"] + c["b_data"] * y) / slope) / 1e6) if slope > 0 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=FILES)
    ap.add_argument("--loss-col", default="val_fwe")
    ap.add_argument("--n-def", default="with_head", choices=["with_head", "no_head"])
    ap.add_argument("--exclude-spiking", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/fit/local_phi.csv")
    ap.add_argument("--trends-out", default="results/fit/local_phi_trends.json")
    a = ap.parse_args()

    df = load(a.files, a.loss_col, a.n_def, a.exclude_spiking)
    rung_N = df[df.placement == "dense"].groupby("rung").N.first().to_dict()
    grps = groups(df)
    point = evaluate(grps, dense_curves(df))
    point["N_ref"] = point.rung.map(rung_N)
    point["n_seeds"] = [len(g["seeds"]) for g in grps]
    tr_point = trends(point)

    rng = np.random.default_rng(a.seed)
    boot_M, boot_phi, boot_tr = [], [], {k: [] for k in tr_point}
    for _ in range(a.n_boot):
        res = evaluate(grps, dense_curves(df, rng), rng)
        res["N_ref"] = point.N_ref.values
        res.loc[~res.status.isin(["interpolated", "extrapolated"]), "M"] = np.nan
        boot_M.append(res.M.values)
        boot_phi.append(res.phi.values)
        t = trends(res)
        for k in boot_tr:
            if k in t:
                boot_tr[k].append([t[k][c] for c in COEFS])
    boot_M, boot_phi = np.array(boot_M), np.array(boot_phi)
    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # groups that are censored in every resample stay NaN
        point["M_lo"], point["M_hi"] = np.nanpercentile(boot_M, 2.5, axis=0), np.nanpercentile(boot_M, 97.5, axis=0)
        point["phi_lo"], point["phi_hi"] = (np.nanpercentile(np.where(np.isfinite(boot_phi), boot_phi, np.nan), q, axis=0)
                                            for q in (2.5, 97.5))
    order = {"10M": 0, "20M": 1, "40M": 2, "80M": 3, "160M": 4}
    point = point.sort_values(["cell", "r", "rung", "tpp"], key=lambda s: s.map(order) if s.name == "rung" else s)
    cols = ["cell", "r", "rung", "tpp", "n_seeds", "loss", "N_star", "M", "M_lo", "M_hi", "phi", "phi_lo", "phi_hi", "status"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    point[cols].round(4).to_csv(a.out, index=False)

    out = {"definition": __doc__.split("\n\n")[0], "n_boot": a.n_boot, "loss_col": a.loss_col, "n_def": a.n_def,
           "exclude_spiking": a.exclude_spiking, "regressor_x": "log2(N_rung / 20M)", "regressor_y": "log2((D/N) / 20)",
           "response": "log2 M", "ci95": "seed resampling (percentile)",
           "ols_ci95": "classical least squares, includes the misfit of the linear form",
           "break_even_N_M": "N (millions) at which M = 1, by D/N, from the fitted plane", "cells": {}}
    for k, v in tr_point.items():
        bt = np.array(boot_tr[k]) if boot_tr[k] else np.full((1, 4), np.nan)
        ci = {c: [float(np.nanpercentile(bt[:, i], 2.5)), float(np.nanpercentile(bt[:, i], 97.5))]
              for i, c in enumerate(COEFS)}
        out["cells"][k] = {**{c: float(v[c]) for c in COEFS}, "n_points": v["n"], "resid_sd": v["resid_sd"],
                           "ci95": ci, "ols_ci95": {c: [float(x) for x in v["ols_ci95"][c]] for c in COEFS},
                           "n_boot_used": len(boot_tr[k]), "break_even_N_M": break_even(v)}
    with open(a.trends_out, "w") as f:
        json.dump(out, f, indent=1)

    pd.set_option("display.width", 220)
    show = point[cols].copy()
    print(show.round(3).to_string(index=False))
    print("\nlog2 M = b0 + b_size*log2(N/20M) + b_data*log2((D/N)/20) + b_interaction*(product); 95% intervals "
          "(seeds | classical):")
    for k, v in out["cells"].items():
        print(f"  {k:18s} n={v['n_points']:2d} sd={v['resid_sd']:.3f}  " + "  ".join(
            f"{c}={v[c]:+.3f} [{v['ci95'][c][0]:+.3f},{v['ci95'][c][1]:+.3f} | {v['ols_ci95'][c][0]:+.3f},{v['ols_ci95'][c][1]:+.3f}]"
            for c in COEFS))
        print(f"  {'':18s} break-even N (M params) by D/N: {v['break_even_N_M']}")
    print(f"\n-> {a.out}, {a.trends_out}")


if __name__ == "__main__":
    main()
