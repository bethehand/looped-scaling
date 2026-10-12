"""Analysis of the supplementary experiments S1-S3 with the readings fixed in docs/supplementary-experiments.md
(this script was written and committed before any of these results existed).

    python fit/supp_analysis.py            # every part whose inputs exist
Inputs : results/runs/<run>/results.jsonl (S1 and its 1x references from the main grid and the data extension),
         results/supp/depth_curve.csv (Q5), results/synthetic/lr_seeds/ and results/synthetic/formal/ (S2),
         results/bench/inference.json (S3), results/all_results.csv and results/ext_160m*.csv (S3 at equal loss)
Outputs: results/supp/s1_points.csv, s1_gaps.csv, s1_equal_compute.csv, s1_readings.json, q5_readings.json,
         s2_runs.csv, s2_readings.json, s3_cells.csv, s3_equal_loss.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from looped.flops import flops_per_token  # noqa: E402
from scripts import make_supp as ms  # noqa: E402
from scripts.make_configs import RULER_ONLY, RUNGS, model_cfg  # noqa: E402
from scripts.pick_lr import vertex  # noqa: E402

OUT = "results/supp"
FACTORS_ALL = (0.35, 0.5, 0.71, 1.0, 1.41, 2.0)
TOL = 0.01                    # Q1/Q2 "does not hold" below this improvement; Q4 bias tolerance (nats)
LEARNED = 0.99                # S2: mean accuracy over d = 1..8 at the run's own loop count
D_TRAIN = 8


# ------------------------------------------------------------------------------------------------------------- S1
def s1_points() -> pd.DataFrame:
    rows = []
    for rung in ms.S1_RUNGS:
        for placement, _, _ in ms.S1_CELLS:
            for f in FACTORS_ALL:
                for seed in ms.S1_SEEDS[rung]:
                    for mult in ms.S1_BUDGETS[(rung, placement)]:
                        loss = ms.s1_loss(rung, placement, f, seed, mult)
                        if loss is not None:
                            rows.append(dict(rung=rung, placement=placement, f=f, seed=seed, mult=mult, loss=loss))
    return pd.DataFrame(rows, columns=["rung", "placement", "f", "seed", "mult", "loss"])


def seed_means(pts: pd.DataFrame) -> pd.DataFrame:
    """Mean over the rung's seeds; a point missing a seed is dropped rather than averaged over fewer seeds."""
    g = pts.groupby(["rung", "placement", "mult", "f"]).agg(loss=("loss", "mean"), n=("loss", "size")).reset_index()
    return g[g.n == g.rung.map({r: len(s) for r, s in ms.S1_SEEDS.items()})].drop(columns="n")


def _vertex_loss(s: pd.Series) -> tuple[float, str, float]:
    """Pre-registered parabola rule (scripts/pick_lr.py) on log2(factor): vertex factor, how, fitted loss there."""
    s = s.sort_index()
    v, how = vertex(s.index.to_numpy(dtype=float), s.to_numpy())
    if how != "parabola":
        return v, how, float(s.min())
    i = int(np.argmin(s.to_numpy()))
    x, y = np.log2(s.index.to_numpy(dtype=float)[i - 1:i + 2]), s.to_numpy()[i - 1:i + 2]
    return v, how, float(np.polyval(np.polyfit(x, y, 2), np.log2(v)))


def s1_gaps(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rung in ms.S1_RUNGS:
        for mult in ms.S1_BUDGETS[(rung, "middle")]:
            d = c[(c.rung == rung) & (c.mult == mult)]
            dn = d[d.placement == "dense"].set_index("f").loss
            lp = d[d.placement == "middle"].set_index("f").loss
            if 1.0 not in dn.index or 1.0 not in lp.index:
                continue
            row = dict(rung=rung, mult=mult, dense_1x=dn[1.0], loop_1x=lp[1.0], delta_1=lp[1.0] - dn[1.0],
                       dense_best=dn.min(), dense_best_f=dn.idxmin(), loop_best=lp.min(), loop_best_f=lp.idxmin(),
                       dense_factors=" ".join(f"{x:g}" for x in sorted(dn.index)),
                       loop_factors=" ".join(f"{x:g}" for x in sorted(lp.index)))
            row["delta_star"] = row["loop_best"] - row["dense_best"]
            row["h"] = row["delta_1"] - row["delta_star"]
            for name, s in (("dense", dn), ("loop", lp)):
                row[f"{name}_vertex_f"], row[f"{name}_vertex_how"], row[f"{name}_vertex_loss"] = _vertex_loss(s)
            row["delta_vertex"] = row["loop_vertex_loss"] - row["dense_vertex_loss"]
            rows.append(row)
    return pd.DataFrame(rows)


def reading(improvement_star: float, improvement_1: float, final_star: float) -> str:
    """Q1/Q2: holds / weakened / does not hold, as fixed in the plan."""
    if improvement_star < TOL:
        return "does not hold"
    if improvement_star >= 0.5 * improvement_1 and final_star < 0:
        return "holds"
    return "weakened"


def _interp(x_new: float, xs: np.ndarray, ys: np.ndarray) -> tuple[float, bool]:
    """Linear interpolation (xs ascending); beyond the range, the slope of the end segment (flagged)."""
    if xs[0] <= x_new <= xs[-1]:
        return float(np.interp(x_new, xs, ys)), False
    i = (-2, -1) if x_new > xs[-1] else (0, 1)
    slope = (ys[i[1]] - ys[i[0]]) / (xs[i[1]] - xs[i[0]])
    return float(ys[i[1]] + slope * (x_new - xs[i[1]])), True


def _interp_log(x_new: float, xs: np.ndarray, ys: np.ndarray) -> tuple[float, bool]:
    """Linear in log x."""
    return _interp(float(np.log(x_new)), np.log(np.asarray(xs, float)), np.asarray(ys, float))


def s1_equal_compute(c: pd.DataFrame) -> pd.DataFrame:
    """Results 5.11 method on the S1 points: looped at budget D against the dense curve at D x rho, at the shared
    learning rate (1x) and with each side at its best tested learning rate for every budget."""
    rows = []
    for rung in ms.S1_RUNGS:
        w = RUNGS[rung]
        rho = flops_per_token(model_cfg(w, "middle", 4, 0))["train"] / flops_per_token(model_cfg(w, "dense", 1, 0))["train"]
        d = c[c.rung == rung]
        dense = d[d.placement == "dense"]
        shared = dense[dense.f == 1.0].set_index("mult").loss.sort_index()
        best = dense.groupby("mult").loss.min().sort_index()
        loop = d[d.placement == "middle"]
        for mult in sorted(loop.mult.unique()):
            lp = loop[loop.mult == mult].set_index("f").loss
            if 1.0 not in lp.index:
                continue
            e1, x1 = _interp_log(mult * rho, shared.index.to_numpy(float), shared.to_numpy())
            eb, xb = _interp_log(mult * rho, best.index.to_numpy(float), best.to_numpy())
            rows.append(dict(rung=rung, mult=mult, rho=round(rho, 3), dense_mult=round(mult * rho, 1),
                             extrapolated=x1 or xb, gap_1x=lp[1.0] - e1, gap_best=lp.min() - eb))
    return pd.DataFrame(rows)


def s1_readings(g: pd.DataFrame, eq: pd.DataFrame) -> dict:
    def at(rung, mult, col):
        x = g[(g.rung == rung) & (g.mult == mult)]
        return float(x[col].iloc[0]) if len(x) else float("nan")
    out = {}
    q1 = {}
    for mult in (40, 10, 20):
        i1 = at("10M", mult, "delta_1") - at("40M", mult, "delta_1")
        ist = at("10M", mult, "delta_star") - at("40M", mult, "delta_star")
        ds = [at(r, mult, "delta_star") for r in ms.S1_RUNGS]
        q1[f"{mult}N"] = dict(I_1=i1, I_star=ist, delta_star_40M=at("40M", mult, "delta_star"),
                              reading=reading(ist, i1, at("40M", mult, "delta_star")),
                              delta_star_by_size=dict(zip(ms.S1_RUNGS, ds)),
                              monotone=bool(ds[0] > ds[1] > ds[2]))
    out["Q1_scale"] = dict(primary="40N", **q1)
    q2 = {}
    for rung, lo, hi in (("10M", 10, 160), ("20M", 10, 40), ("40M", 10, 40)):
        i1 = at(rung, lo, "delta_1") - at(rung, hi, "delta_1")
        ist = at(rung, lo, "delta_star") - at(rung, hi, "delta_star")
        q2[rung] = dict(span=f"{lo}N-{hi}N", I_1=i1, I_star=ist, delta_star_end=at(rung, hi, "delta_star"),
                        reading=reading(ist, i1, at(rung, hi, "delta_star")))
    out["Q2_data"] = dict(primary="10M", **q2)
    flips = eq[eq.gap_best < 0]
    out["Q3_equal_compute"] = dict(n=int(len(eq)), n_loop_better_retuned=int(len(flips)),
                                   n_loop_better_shared=int((eq.gap_1x < 0).sum()),
                                   loop_better=[f"{r.rung} {r.mult}N" for r in flips.itertuples()],
                                   min_gap_retuned=float(eq.gap_best.min()) if len(eq) else float("nan"))
    out["Q4_bias"] = dict(max_abs_h=float(g.h.abs().max()), within_tol=bool((g.h.abs() <= TOL).all()),
                          h={f"{r.rung} {r.mult}N": float(r.h) for r in g.itertuples()})
    return out


def s1(out_dir: str) -> None:
    pts = s1_points()
    if pts.empty:
        print("S1: no results yet")
        return
    c = seed_means(pts)
    g = s1_gaps(c)
    pts.to_csv(os.path.join(out_dir, "s1_points.csv"), index=False)
    g.to_csv(os.path.join(out_dir, "s1_gaps.csv"), index=False, float_format="%.5f")
    have = set(zip(c.rung, c.placement, c.f, c.mult))
    missing = [(r, p, f, m) for r in ms.S1_RUNGS for p, _, _ in ms.S1_CELLS for f in (*ms.S1_FACTORS, 1.0)
               for m in ms.S1_BUDGETS[(r, p)] if (r, p, f, m) not in have]
    complete = not missing
    pd.set_option("display.width", 200)
    print(g[["rung", "mult", "delta_1", "dense_best_f", "loop_best_f", "delta_star", "h", "delta_vertex"]].round(4).to_string(index=False))
    if not complete:
        print(f"S1: {len(missing)} round-1 points missing, readings not computed")
        return
    eq = s1_equal_compute(c)
    eq.to_csv(os.path.join(out_dir, "s1_equal_compute.csv"), index=False, float_format="%.5f")
    rd = s1_readings(g, eq)
    json.dump(rd, open(os.path.join(out_dir, "s1_readings.json"), "w"), indent=1)
    print(json.dumps({k: (v if k != "Q4_bias" else {kk: vv for kk, vv in v.items() if kk != "h"}) for k, v in rd.items()},
                     indent=1, default=float))


# ------------------------------------------------------------------------------------------------------------- Q5
def _run_key(run: str) -> tuple[str, float, int]:
    rung, rest = run.split("_", 1)
    f = float(rest.split("_lrs")[1].split("_s")[0]) if "_lrs" in rest else 1.0
    return rung, f, int(run.rsplit("_s", 1)[1])


def q5(out_dir: str, path: str = "results/supp/depth_curve.csv") -> None:
    if not os.path.exists(path):
        print("Q5: no depth curves yet")
        return
    d = pd.read_csv(path)
    d = d[d.valset == "fwe"].copy()
    d[["rung", "f", "seed"]] = pd.DataFrame([_run_key(r) for r in d.run], index=d.index)
    m = d.groupby(["rung", "f", "r_eval"]).mean_loss.mean().reset_index()
    spread, gain = {}, {}
    for (rung, f), x in m.groupby(["rung", "f"]):
        s = x.set_index("r_eval").mean_loss
        spread[(rung, f)] = float(s[s.index >= 4].max() - s[s.index >= 4].min())
        gain[(rung, f)] = float(s[4] - min(s[8], s[16]))
    rep = all(spread.get((r, 0.5), np.inf) < spread.get((r, 1.41), -np.inf) for r in ms.S1_RUNGS)
    out = dict(replicates_5_14=bool(rep), more_loops_help=[f"{r} x{f:g}" for (r, f), v in gain.items() if v > 0.005],
               spread_4_to_16={f"{r} x{f:g}": v for (r, f), v in sorted(spread.items())},
               gain_8_or_16_over_4={f"{r} x{f:g}": v for (r, f), v in sorted(gain.items())})
    json.dump(out, open(os.path.join(out_dir, "q5_readings.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------------------------- S2
def s2(out_dir: str, new_dir: str = "results/synthetic/lr_seeds", formal_dir: str = "results/synthetic/formal") -> None:
    from synthetic import analyze as sa
    if not os.path.isdir(os.path.join(new_dir, "chain")):
        print("S2: no results yet")
        return
    new, formal = sa.summarise(os.path.join(new_dir, "chain")), sa.summarise(os.path.join(formal_dir, "chain"))
    new = new[new.cond.isin(ms.S2_CONDS)]
    for df in (new, formal):
        df["E"] = df.maxd_best - D_TRAIN
        df["learned"] = df.in_dist >= LEARNED
    if os.path.isdir(os.path.join(new_dir, "probes")):
        pr = sa.probe_frontier(os.path.join(new_dir, "probes"))[["cond", "seed", "lr", "slope", "persistence", "reached"]]
        new = new.merge(pr, on=["cond", "seed", "lr"], how="left")
    new.to_csv(os.path.join(out_dir, "s2_runs.csv"), index=False, float_format="%.4f")
    ref = formal[formal.cond.isin(ms.S2_CONDS)].groupby("cond").E.mean()
    out = {}
    for (cond, lr), x in new.groupby(["cond", "lr"]):
        lx = x[x.learned]
        e = float(lx.E.mean()) if len(lx) else float("nan")
        if len(lx) >= 2 and e <= ref[cond] - 2:
            r2 = "supports"
        elif len(lx) >= 2 and e >= ref[cond] - 1:
            r2 = "does not support"
        else:
            r2 = "inconclusive"
        out[f"{cond} lr{lr:g}"] = dict(n_seeds=int(len(x)), n_learned=int(len(lx)), in_dist=[round(v, 4) for v in x.in_dist],
                                      E_learned_mean=e, E_formal_mean=float(ref[cond]), E_all=[int(v) for v in x.E],
                                      R2=r2)
    json.dump(dict(formal_reference=ref.to_dict(), cells=out), open(os.path.join(out_dir, "s2_readings.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------------------------- S3
def losses_40n() -> pd.DataFrame:
    """Loss at 40N per (width, cell): the median over seeds (40M whole r=8 has one seed that never recovered from a spike)."""
    a = pd.concat([pd.read_csv("results/all_results.csv")] +
                  [pd.read_csv(p) for p in ("results/ext_160m.csv", "results/ext_160m_dense.csv") if os.path.exists(p)])
    diverged = a.diverged.astype(str).str.lower().eq("true")
    a = a[(a.accounting == "iso_token") & ((a.budget_mult - 40).abs() < 1) & (a.k_bwd == 0) & ~diverged]
    return a.groupby(["d_model", "placement", "r"]).val_fwe.median().rename("loss").reset_index()


def s3(out_dir: str, path: str = "results/bench/inference.json") -> None:
    if not os.path.exists(path):
        print("S3: no benchmark yet")
        return
    b = pd.DataFrame(json.load(open(path))["rows"])
    metrics = [c for c in b.columns if c.startswith(("decode_ms", "prefill_ms", "peak_mb", "kv_", "weight_mb", "tokens_per_s"))]
    dense = b[b.placement == "dense"].set_index("width")
    cells = b.copy()
    for c in metrics:
        cells[f"{c}_vs_dense"] = cells.apply(lambda r: r[c] / dense.loc[r.width, c] if r.width in dense.index else np.nan, axis=1)
    cells.to_csv(os.path.join(out_dir, "s3_cells.csv"), index=False, float_format="%.4f")
    lo = losses_40n()
    dl = lo[lo.placement == "dense"].merge(dense.reset_index()[["width", "N"]], left_on="d_model", right_on="width")
    dl = dl.sort_values("loss")                       # ascending loss = descending N
    dsorted = dense.sort_values("N")
    rows = []
    for r in lo[lo.placement != "dense"].itertuples():
        me = b[(b.width == r.d_model) & (b.placement == r.placement) & (b.r == r.r)]
        if me.empty:
            continue
        me = me.iloc[0]
        # the dense size with the same 40N loss (loss against log N), then the dense model's metrics there (log-log)
        logn, extra = _interp(float(r.loss), dl.loss.to_numpy(float), np.log(dl.N.to_numpy(float)))
        row = dict(width=r.d_model, placement=r.placement, r=r.r, loss=r.loss, N=me.N, N_dense_equal=float(np.exp(logn)),
                   extrapolated=extra)
        for c in metrics:
            if dsorted[c].notna().all() and me[c] == me[c]:
                v, _ = _interp_log(float(np.exp(logn)), dsorted.N.to_numpy(float), np.log(dsorted[c].to_numpy(float)))
                row[f"{c}_ratio"] = float(me[c] / np.exp(v))
        rows.append(row)
    eq = pd.DataFrame(rows)
    eq.to_csv(os.path.join(out_dir, "s3_equal_loss.csv"), index=False, float_format="%.4f")
    print(eq.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="s1,q5,s2,s3")
    a = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs(OUT, exist_ok=True)
    for p in a.parts.split(","):
        print(f"==== {p}")
        dict(s1=s1, q5=q5, s2=s2, s3=s3)[p](OUT)


if __name__ == "__main__":
    main()
