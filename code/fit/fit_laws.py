"""Scaling-law fitting (pre-registration §4).

Forms
  F1 : L = E + A * (N_once + r^phi * N_rec)^-alpha + B * D^-beta                 (iso-depth [01])
  F2 : L = E + A * (N_once + N_rec * (1 - rho^r) / (1 - rho))^-alpha + B * D^-beta  (geometric saturation)
  F3 : F1 or F2 with a cell-specific phi / rho and shared E, A, alpha, B, beta

Protocol
  stage 1 : fit (E, A, alpha, B, beta) on the dense ruler only (N_rec = 0)
  stage 2 : hold them fixed, fit phi (or rho) per cell on the looped rows            <- primary
  joint   : fit everything at once                                                     <- secondary
  loss    : Huber (delta = 1e-3) on log L, summed over rows, LSE parameterisation (Chinchilla [08])
  optimiser: L-BFGS-B, multi-start (default 500) from random inits inside box bounds
  CI      : block bootstrap over (rung, r, budget) cells, 1000 resamples
  model selection: leave-one-rung-out prediction RMSE on log L, plus AIC (H3)

    python fit/fit_laws.py --results results/all_results.csv --loss-col valavg_fwe --out results/fit
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

DELTA = 1e-3
BOUNDS_SHARED = dict(logE=(-3, 3), logA=(-5, 20), alpha=(0.01, 1.5), logB=(-5, 20), beta=(0.01, 1.5))
BOUNDS_LOOP = dict(phi=(0.0, 1.5), rho=(1e-3, 0.999))


def n_eff(N_once, N_rec, r, form, theta):
    if form == "F1":
        return N_once + (r ** theta) * N_rec
    g = np.where(np.abs(1 - theta) < 1e-9, r, (1 - theta ** r) / (1 - theta))
    return N_once + g * N_rec


def predict_log(params, N_once, N_rec, r, D, form, theta):
    """log L via LSE for numerical stability: L = exp(logE) + exp(logA - alpha*log Neff) + exp(logB - beta*log D)."""
    logE, logA, alpha, logB, beta = params
    ne = n_eff(N_once, N_rec, r, form, theta)
    terms = np.stack([np.full_like(D, logE, dtype=float), logA - alpha * np.log(ne), logB - beta * np.log(D)])
    return logsumexp(terms, axis=0)


def huber(resid, delta=DELTA):
    a = np.abs(resid)
    return np.where(a <= delta, 0.5 * resid ** 2, delta * (a - 0.5 * delta)).sum()


def _objective(x, df, form, cell_of_row, theta_index, shared_fixed):
    if shared_fixed is None:
        shared, thetas = x[:5], x[5:]
    else:
        shared, thetas = shared_fixed, x
    theta = thetas[theta_index] if len(thetas) else np.zeros(len(df))
    lp = predict_log(shared, df.N_once.values, df.N_rec.values, df.r.values, df.D.values, form, theta)
    return huber(lp - np.log(df.L.values))


def fit(df: pd.DataFrame, form: str, n_starts: int = 500, seed: int = 0, shared_fixed=None,
        per_cell: bool = True) -> dict:
    """Fit shared (unless fixed) + one theta per cell (or one global theta if per_cell=False)."""
    rng = np.random.default_rng(seed)
    looped_rows = df.N_rec.values > 0
    cells = sorted(df.cell[looped_rows].unique()) if per_cell else ["all"]
    cell_of_row = df.cell.values if per_cell else np.array(["all"] * len(df))
    # rows without a recurrent block (the dense ruler) get index 0; theta has no effect on them
    theta_index = np.array([cells.index(c) if c in cells else 0 for c in cell_of_row])
    looped = bool(looped_rows.any())
    n_theta = len(cells) if looped else 0
    tb = BOUNDS_LOOP["phi" if form == "F1" else "rho"]
    bounds = ([] if shared_fixed is not None else list(BOUNDS_SHARED.values())) + [tb] * n_theta
    # deterministic Chinchilla-style grid of starting points, then random restarts (platform-stable optimum)
    starts = []
    if shared_fixed is None:
        import itertools
        theta0 = 0.5 if form == "F1" else 0.7
        for logE, logA, alpha, logB, beta in itertools.product((-0.5, 0.5, 1.0), (3.0, 7.0, 11.0), (0.2, 0.4, 0.8),
                                                               (3.0, 7.0, 11.0), (0.2, 0.4, 0.8)):
            starts.append(np.array([logE, logA, alpha, logB, beta] + [theta0] * n_theta))
    else:
        for t in np.linspace(tb[0] + 0.05, tb[1] - 0.05, 7):
            starts.append(np.full(n_theta, t))
    for _ in range(n_starts):
        starts.append(np.array([rng.uniform(lo, hi) for lo, hi in bounds]))
    best = None
    for x0 in starts:
        res = minimize(_objective, x0, args=(df, form, cell_of_row, theta_index, shared_fixed),
                       method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    x = best.x
    shared = shared_fixed if shared_fixed is not None else x[:5]
    thetas = x if shared_fixed is not None else x[5:]
    out = dict(form=form, loss=float(best.fun), n_rows=int(len(df)),
               shared=dict(E=float(np.exp(shared[0])), A=float(np.exp(shared[1])), alpha=float(shared[2]),
                           B=float(np.exp(shared[3])), beta=float(shared[4])),
               theta={c: float(t) for c, t in zip(cells, thetas)} if n_theta else {},
               shared_raw=[float(v) for v in shared])
    k = (0 if shared_fixed is not None else 5) + n_theta
    out["aic"] = float(2 * k + 2 * len(df) * np.log(max(best.fun / len(df), 1e-12)))  # Huber-loss AIC proxy
    return out


def bootstrap(df, form, base, n_boot=1000, seed=0, n_starts=30):
    rng = np.random.default_rng(seed)
    keys = df[["rung", "r", "D"]].astype(str).agg("|".join, axis=1).values
    uniq = np.unique(keys)
    shared_fixed = np.array(base["shared_raw"])
    samples = {c: [] for c in base["theta"]}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(keys == u)[0] for u in pick])
        sub = df.iloc[idx]
        if (sub.N_rec.values > 0).sum() < 3:
            continue
        f = fit(sub[sub.N_rec > 0], form, n_starts=n_starts, seed=int(rng.integers(1e9)), shared_fixed=shared_fixed)
        for c, v in f["theta"].items():
            samples[c].append(v)
    return {c: dict(lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)), n=len(v))
            for c, v in samples.items() if v}


def leave_one_rung_out(df, form, n_starts=100):
    out = {}
    for rung in sorted(df.rung.unique()):
        tr, te = df[df.rung != rung], df[df.rung == rung]
        if len(te) == 0 or len(tr) == 0:
            continue
        f = fit(tr, form, n_starts=n_starts, per_cell=True)
        cells = list(f["theta"])
        theta = np.array([f["theta"].get(c, 0.0) for c in te.cell.values]) if cells else np.zeros(len(te))
        lp = predict_log(f["shared_raw"], te.N_once.values, te.N_rec.values, te.r.values, te.D.values, form, theta)
        out[rung] = dict(rmse_log=float(np.sqrt(np.mean((lp - np.log(te.L.values)) ** 2))), n=int(len(te)))
    return out


def load(results_csv: str, loss_col: str, accounting: str | None, n_def: str = "with_head") -> pd.DataFrame:
    """n_def = with_head : N = non-embedding + output head (pre-registered primary; Porian / Parcae convention)
       n_def = no_head   : N excludes the output head as well (iso-depth / Kaplan convention; secondary, for
                           direct comparison with iso-depth's phi). Embeddings are untied, so head = V*d = emb_in."""
    df = pd.read_csv(results_csv)
    df = df.rename(columns={"budget_tokens": "D", loss_col: "L"})
    if accounting:
        df = df[(df.accounting == accounting) | (df.placement == "dense")]
    df = df[["name", "rung", "placement", "backprop", "cell", "accounting", "r", "k_bwd", "seed",
             "N_once", "N_rec", "N", "emb_in", "D", "L"]].copy()
    if n_def == "no_head":
        df["N_once"] = df["N_once"] - df["emb_in"]
        df["N"] = df["N"] - df["emb_in"]
    elif n_def != "with_head":
        raise ValueError(n_def)
    df["cell"] = np.where(df.placement == "dense", "dense", df.cell)
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/all_results.csv")
    ap.add_argument("--loss-col", default="valavg_fwe")
    ap.add_argument("--accounting", default="iso_token", help="iso_token | iso_flop | all")
    ap.add_argument("--out", default="results/fit")
    ap.add_argument("--n-starts", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-def", default="with_head", choices=["with_head", "no_head"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load(a.results, a.loss_col, None if a.accounting == "all" else a.accounting, a.n_def)
    ruler = df[df.placement == "dense"]
    looped = df[df.placement != "dense"]
    report = {}
    for form in ("F1", "F2"):
        stage1 = fit(ruler, form, n_starts=a.n_starts)
        stage2 = fit(looped, form, n_starts=a.n_starts, shared_fixed=np.array(stage1["shared_raw"]))
        joint = fit(df, form, n_starts=a.n_starts)
        glob_theta = fit(looped, form, n_starts=a.n_starts, shared_fixed=np.array(stage1["shared_raw"]), per_cell=False)
        ci = bootstrap(looped, form, stage2, n_boot=a.n_boot)
        loro = leave_one_rung_out(df, form)
        restricted = {}
        if form == "F1":
            for fixed in (0.0, 1.0):
                sub = looped.copy()
                lp = predict_log(stage1["shared_raw"], sub.N_once.values, sub.N_rec.values, sub.r.values,
                                 sub.D.values, "F1", np.full(len(sub), fixed))
                restricted[f"phi={fixed}"] = float(huber(lp - np.log(sub.L.values)))
        report[form] = dict(stage1_ruler=stage1, stage2_per_cell=stage2, stage2_global=glob_theta, joint=joint,
                            bootstrap_ci=ci, leave_one_rung_out=loro, restricted=restricted)
        print(f"== {form}: ruler alpha={stage1['shared']['alpha']:.3f} beta={stage1['shared']['beta']:.3f} "
              f"E={stage1['shared']['E']:.3f}; per-cell theta={ {k: round(v, 3) for k, v in stage2['theta'].items()} }")
        print(f"   CI={ {k: (round(v['lo'], 3), round(v['hi'], 3)) for k, v in ci.items()} }; LORO={loro}")
    out = os.path.join(a.out, f"fit_{a.accounting}_{a.n_def}.json")
    json.dump(report, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
