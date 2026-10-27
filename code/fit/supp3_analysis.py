"""S10 readings (docs/supplementary-experiments-3.md, frozen 2026-10-27 before any run): dense models with 18 layers and
the looped model's number of parameters, against the looped middle block (r = 4) and the 8-layer dense ladder.

    python fit/supp3_analysis.py   -> results/supp3/s10_table.csv, results/supp3/s10_readings.json

Inputs: results/supp3/supp3_deep.csv (fit/collect_results.py with configs/manifest_supp3_deep.csv); the ladder and the
looped runs as in results/supp2/local_phi_s4.csv (all_results + extensions + the S4 dense points).
Quantities, as the plan fixes them:
  gap      dense-18 minus looped main-set loss at the same tokens and seed, per budget (seed means of paired gaps);
  M_deep   N*/N_deep, N* read off the 8-layer ladder with fit/local_phi.dense_size_at, next to M_loop of the same size;
  R        log M_deep / log M_loop; primary reading at 40M, averaged over the three budgets;
  lr       seed 42 at 0.71 and 1.41 times the rule's rate: the best rate and its margin over the rule's rate;
  equal compute (exploratory): the looped loss at budget B against the dense-18 loss at the tokens the looped run's
           training FLOPs buy it (interpolated in log D between the dense-18 budgets; never extrapolated).
Intervals: seeds resampled (dense-18, looped and every ladder rung independently), 95% percentile intervals.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import fit.local_phi as lp

LADDER_FILES = ["results/all_results.csv", "results/ext_data.csv", "results/ext_160m.csv", "results/ext_160m_dense.csv",
                "results/supp2/s4_dense_points.csv"]
TPPS = (10, 20, 40)


def load_deep(path):
    d = pd.read_csv(path)
    d = d[~d.diverged.astype(str).str.lower().eq("true")].copy()
    d["tpp"] = (d.budget_tokens / d.N_rung).round(0).astype(int)
    d["L"] = d["val_fwe"]
    d["lrs"] = d.name.str.extract(r"_lrs([0-9.]+)_")[0].astype(float).fillna(1.0)
    return d[(d.accounting == "iso_token") & d.tpp.isin(TPPS)]


def read_out(ladder, deep, rng=None):
    curves = lp.dense_curves(ladder, rng)
    loop = ladder[(ladder.placement == "middle") & (ladder.cell == "middle/full") & (ladder.r == 4)]
    rows = []
    for (rung, tpp), g in deep[deep.lrs == 1.0].groupby(["rung", "tpp"]):
        gl = loop[(loop.rung == rung) & (loop.tpp == tpp)]
        if gl.empty:
            continue
        Ld = lp._mean_by_seed(dict(zip(g.seed, g.L)), rng)
        Ll = lp._mean_by_seed(dict(zip(gl.seed, gl.L)), rng)
        D = float(g.budget_tokens.mean())
        Nd, sd = lp.dense_size_at(Ld, D, curves)
        Nl, sl = lp.dense_size_at(Ll, float(gl.budget_tokens.mean()), curves)
        Md, Ml = Nd / float(g.N.iloc[0]), Nl / float(gl.N.iloc[0])
        R = np.log(Md) / np.log(Ml) if Ml > 0 and abs(np.log(Ml)) > 1e-9 else np.nan
        paired = [g[g.seed == s].L.iloc[0] - gl[gl.seed == s].L.iloc[0] for s in sorted(set(g.seed) & set(gl.seed))]
        rows.append(dict(rung=rung, tpp=int(tpp), loss_deep=Ld, loss_loop=Ll, gap_paired=float(np.mean(paired)),
                         M_deep=Md, status_deep=sd, M_loop=Ml, status_loop=sl, R=R))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", default="results/supp3/supp3_deep.csv")
    ap.add_argument("--out-dir", default="results/supp3")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ladder = lp.load(LADDER_FILES)
    deep = load_deep(a.deep)
    point = read_out(ladder, deep)
    rng = np.random.default_rng(a.seed)
    boots = [read_out(ladder, deep, rng) for _ in range(a.n_boot)]
    for col in ("M_deep", "M_loop", "R", "gap_paired"):
        arr = np.array([b[col].values for b in boots])
        point[col + "_lo"] = np.nanpercentile(arr, 2.5, axis=0)
        point[col + "_hi"] = np.nanpercentile(arr, 97.5, axis=0)
    os.makedirs(a.out_dir, exist_ok=True)
    point.to_csv(os.path.join(a.out_dir, "s10_table.csv"), index=False)

    out = {"table": point.round(4).to_dict(orient="records")}
    p40 = point[point.rung == "40M"]
    if len(p40) == len(TPPS):
        R_avg = float(p40.R.mean())
        R_boot = np.array([b[b.rung == "40M"].R.mean() for b in boots])
        reading = ("most of the advantage is a depth effect" if R_avg >= 0.75 else
                   "partly depth, partly sharing" if R_avg >= 0.25 else "looping buys more than depth")
        out["primary_40M"] = dict(R_mean=R_avg, R_mean_ci95=[float(np.nanpercentile(R_boot, 2.5)),
                                                             float(np.nanpercentile(R_boot, 97.5))], reading=reading,
                                  on_par=bool((p40.gap_paired.abs() <= 0.01).all()))
    # learning-rate check (seed 42)
    lr = {}
    for (rung, tpp), g in deep[deep.seed == 42].groupby(["rung", "tpp"]):
        by = dict(zip(g.lrs, g.L))
        if 1.0 in by and len(by) > 1:
            best = min(by, key=by.get)
            lr[f"{rung} {tpp}N"] = dict(losses={str(k): round(v, 4) for k, v in sorted(by.items())}, best=best,
                                        margin_over_rule=round(by[1.0] - by[best], 4),
                                        report_at_best=bool(by[1.0] - by[best] > 0.01))
    out["lr_check"] = lr
    # equal compute (exploratory)
    from looped.flops import flops_per_token
    from looped.model import ModelConfig
    eqc = []
    loop = ladder[(ladder.cell == "middle/full") & (ladder.r == 4)]
    for rung, g in deep[deep.lrs == 1.0].groupby("rung"):
        gl = loop[loop.rung == rung]
        if gl.empty:
            continue
        fd = flops_per_token(ModelConfig(d_model=int(g.d_model.iloc[0]), n_layers=int(g.n_layers.iloc[0])))["train"]
        fl = flops_per_token(ModelConfig(d_model=int(gl.d_model.iloc[0]), placement="middle", r=4))["train"]
        xs = np.log(sorted(g.budget_tokens.unique()))
        ys = np.array([g[g.budget_tokens == D].L.mean() for D in sorted(g.budget_tokens.unique())])
        for tpp, gg in gl.groupby("tpp"):
            D = float(gg.budget_tokens.mean())
            x = np.log(D * fl / fd)
            if xs[0] <= x <= xs[-1]:
                eqc.append(dict(rung=rung, tpp=int(tpp), flops_ratio=round(fl / fd, 3), loss_loop=round(gg.L.mean(), 4),
                                loss_deep_equal_flops=round(float(np.interp(x, xs, ys)), 4),
                                gap_loop_minus_deep=round(float(gg.L.mean() - np.interp(x, xs, ys)), 4)))
    out["equal_compute_exploratory"] = eqc
    json.dump(out, open(os.path.join(a.out_dir, "s10_readings.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "table"}, indent=1, default=float))
    print(point[["rung", "tpp", "gap_paired", "M_deep", "M_loop", "R", "status_deep"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
