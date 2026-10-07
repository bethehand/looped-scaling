"""EXPLORATORY, not pre-registered: phi per rung and per r, with the pre-registered ruler held fixed.

The pre-registered model gives one phi per cell for all rungs and all r. This script refits theta separately for
each (cell, rung), each (cell, r) and each (cell, rung, r), using the stage-1 ruler parameters of a finished fit, and
lets phi go negative (bounds (-1, 1.5)) so that "worse than the dense model of the same N" shows up as phi < 0 instead
of being clipped at 0. Results are reported next to the pre-registered fit and labelled exploratory.

    python fit/phi_by_rung.py --results results/all_results.csv --fit results/fit/primary/fit_iso_token_with_head.json
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


def refit(df: pd.DataFrame, label: pd.Series, form: str, shared: np.ndarray, n_starts: int) -> dict:
    g = df.copy()
    g["cell"] = label.values
    res = fl.fit(g, form, n_starts=n_starts, shared_fixed=shared, per_cell=True)
    return {k: round(float(v), 4) for k, v in res["theta"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/all_results.csv")
    ap.add_argument("--fit", default="results/fit/primary/fit_iso_token_with_head.json")
    ap.add_argument("--form", default="F1", choices=["F1", "F2"])
    ap.add_argument("--accounting", default="iso_token")
    ap.add_argument("--n-starts", type=int, default=200)
    ap.add_argument("--out", default="results/fit/exploratory_phi_by_rung.json")
    a = ap.parse_args()
    shared = np.array(json.load(open(a.fit))[a.form]["stage1_ruler"]["shared_raw"])
    fl.BOUNDS_LOOP["phi"] = (-1.0, 1.5)           # exploratory: allow "worth less than its parameters"
    df = fl.load(a.results, "val_fwe", a.accounting)
    looped = df[df.placement != "dense"].reset_index(drop=True)
    out = dict(form=a.form, ruler_from=a.fit, note="exploratory; phi bounds (-1, 1.5); ruler fixed from the primary fit",
               by_rung=refit(looped, looped.cell + "@" + looped.rung, a.form, shared, a.n_starts),
               by_r=refit(looped, looped.cell + "@r" + looped.r.astype(str), a.form, shared, a.n_starts),
               by_rung_r=refit(looped, looped.cell + "@" + looped.rung + "@r" + looped.r.astype(str), a.form, shared,
                               a.n_starts))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    cells = sorted({k.split("@")[0] for k in out["by_rung"]})
    rungs = ["10M", "20M", "40M", "80M"]
    print(f"phi by rung ({a.form}, ruler fixed, phi may be negative):")
    print("%-14s" % "cell" + "".join("%8s" % r for r in rungs))
    for c in cells:
        print("%-14s" % c + "".join("%8.2f" % out["by_rung"].get(f"{c}@{r}", float("nan")) for r in rungs))
    print("phi by r:")
    print("%-14s" % "cell" + "".join("%8s" % f"r={r}" for r in (2, 4, 8)))
    for c in cells:
        print("%-14s" % c + "".join("%8.2f" % out["by_r"].get(f"{c}@r{r}", float("nan")) for r in (2, 4, 8)))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
