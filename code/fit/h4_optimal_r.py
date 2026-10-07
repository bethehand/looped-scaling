"""H4 (pre-registration §2): the compute-optimal loop count r, computed from the fitted formulas, no test.

Two accountings:
  train : fixed training FLOPs C. For every cell and every (rung, r) on the design grid, D = C / train_flops_per_token
          and L = E + A * N_eff^-alpha + B * D^-beta with N_eff = N_once + r^phi * N_rec. The best (rung, r) per C gives
          the optimal r; r = 1 is the dense ruler at the same rungs.
  deploy: fixed deployment (forward) FLOPs per token. Model sizes are interpolated continuously along each cell's rung
          geometry (N_once and N_rec scale together; FLOPs per token are linear in executed parameters, fitted per cell
          from the grid), and the loss is compared at a fixed number of training tokens per executed parameter
          (default 20, Chinchilla-like) and in the data limit D -> infinity.
Uses the primary two-stage parameters by default; --params joint reports the secondary joint fit as well.

    python fit/h4_optimal_r.py --results results/all_results.csv --fit results/fit/primary/fit_iso_token_with_head.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fit.fit_laws as fl  # noqa: E402

RUNGS = ["10M", "20M", "40M", "80M"]


def loss(sh: dict, n_eff: np.ndarray, d: np.ndarray) -> np.ndarray:
    return sh["E"] + sh["A"] * np.asarray(n_eff, float) ** (-sh["alpha"]) + sh["B"] * np.asarray(d, float) ** (-sh["beta"])


def grid(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (cell, rung, r): parameter split and FLOPs per token (identical across seeds and budgets)."""
    g = df.groupby(["cell", "placement", "k_bwd", "rung", "r"], as_index=False).agg(
        N_once=("N_once", "first"), N_rec=("N_rec", "first"), N=("N", "first"),
        train_fpt=("train_flops_per_token", "first"), deploy_fpt=("deploy_flops_per_token", "first"))
    return g[g.rung.isin(RUNGS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/all_results.csv")
    ap.add_argument("--fit", default="results/fit/primary/fit_iso_token_with_head.json")
    ap.add_argument("--form", default="F1", choices=["F1", "F2"])
    ap.add_argument("--stage", default="stage2_per_cell", choices=["stage2_per_cell", "joint"])
    ap.add_argument("--tokens-per-param", type=float, default=20.0)
    ap.add_argument("--out", default="results/fit/h4_optimal_r.json")
    a = ap.parse_args()
    rep = json.load(open(a.fit))[a.form][a.stage]
    sh, theta = rep["shared"], rep["theta"]
    raw = pd.read_csv(a.results)
    raw = raw[~raw.diverged.astype(str).str.lower().eq("true")]
    raw["cell"] = np.where(raw.placement == "dense", "dense", raw.cell)
    g = grid(raw)

    def n_eff(row) -> float:
        if row.cell == "dense":
            return row.N
        return float(fl.n_eff(row.N_once, row.N_rec, row.r, a.form, theta[row.cell]))

    g["N_eff"] = g.apply(n_eff, axis=1)
    out = dict(form=a.form, stage=a.stage, shared=sh, theta=theta, train={}, deploy={})
    # ---- training-FLOP matched: discrete design grid, D free ----
    print(f"[{a.stage}] training-FLOP matched (best rung and r on the design grid; dense = r 1):")
    for C in [1e17, 3e17, 1e18, 3e18, 1e19, 3e19, 1e20]:
        gg = g.copy()
        gg["D"] = C / gg.train_fpt
        gg["L"] = loss(sh, gg.N_eff, gg.D)
        best = gg.sort_values("L").groupby("cell").head(1).set_index("cell")
        dense_L = float(best.loc["dense", "L"]) if "dense" in best.index else float("nan")
        row = {c: dict(r=int(best.loc[c, "r"]), rung=best.loc[c, "rung"], L=round(float(best.loc[c, "L"]), 4),
                       gain_vs_dense=round(dense_L - float(best.loc[c, "L"]), 4)) for c in best.index}
        out["train"][f"{C:.0e}"] = row
        print(f"  C={C:.0e}: " + "  ".join(f"{c}: r={v['r']} ({v['rung']}, L={v['L']:.3f}, vs dense {v['gain_vs_dense']:+.3f})"
                                           for c, v in row.items()))
    # ---- deployment-FLOP matched: continuous size along each cell's geometry ----
    print(f"[{a.stage}] deployment-FLOP matched (continuous size; D = {a.tokens_per_param:g} x executed params, and D -> inf):")
    for cell, gc in g.groupby("cell"):
        if cell == "dense":
            continue
        res = {}
        for r, gr in gc.groupby("r"):
            # geometry at this r: N_once, N_rec proportional to the executed parameter count; FLOPs linear in it
            n_exec = gr.N_once + r * gr.N_rec
            k_dep = float(np.polyfit(n_exec, gr.deploy_fpt, 1)[0])
            frac_once, frac_rec = float((gr.N_once / n_exec).mean()), float((gr.N_rec / n_exec).mean())
            res[int(r)] = dict(k_dep=k_dep, frac_once=frac_once, frac_rec=frac_rec)
        dense_rows = g[g.cell == "dense"]
        k_dep_dense = float(np.polyfit(dense_rows.N, dense_rows.deploy_fpt, 1)[0])
        table = {}
        for F in [1e8, 3e8, 1e9, 3e9]:
            entry = {}
            for r, geo in res.items():
                n_exec = F / geo["k_dep"]
                n_once, n_rec = geo["frac_once"] * n_exec, geo["frac_rec"] * n_exec
                ne = float(fl.n_eff(n_once, n_rec, r, a.form, theta[cell]))
                entry[f"r={r}"] = dict(L_chinchilla=round(float(loss(sh, ne, a.tokens_per_param * n_exec)), 4),
                                       L_inf=round(float(sh["E"] + sh["A"] * ne ** (-sh["alpha"])), 4))
            n_dense = F / k_dep_dense
            entry["dense"] = dict(L_chinchilla=round(float(loss(sh, n_dense, a.tokens_per_param * n_dense)), 4),
                                  L_inf=round(float(sh["E"] + sh["A"] * n_dense ** (-sh["alpha"])), 4))
            best_c = min(entry, key=lambda k: entry[k]["L_chinchilla"]); best_i = min(entry, key=lambda k: entry[k]["L_inf"])
            entry["best"] = dict(chinchilla=best_c, data_limit=best_i)
            table[f"{F:.0e}"] = entry
        out["deploy"][cell] = table
        print(f"  {cell}: " + "  ".join(f"F={F}: best {v['best']['chinchilla']} / {v['best']['data_limit']}" for F, v in table.items()))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
