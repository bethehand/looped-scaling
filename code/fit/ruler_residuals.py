"""The pre-registration's rule for the 10M rung (docs/preregistration.md, section 4): if the 10M rung's residual in the
dense-ruler fit exceeds twice that of the other rungs, fits with and without 10M are reported, the latter as primary.
This script evaluates the rule on the primary ruler (results/fit/primary/fit_iso_token_with_head.json, stage 1, fitted
to the 54 dense results): the residual of each dense result in log loss, summarised per rung, and the ratios of the
10M rung's residual to the other rungs'. First run on 2026-10-28 (deviation log); the rule did not apply.

    python -m fit.ruler_residuals   -> results/fit/ruler_residuals.json
"""
from __future__ import annotations

import json

import numpy as np

from fit.fit_laws import load, predict_log


def main():
    fitj = json.load(open("results/fit/primary/fit_iso_token_with_head.json"))["F1"]["stage1_ruler"]
    df = load("results/all_results.csv", "val_fwe", "iso_token")
    d = df[df.placement == "dense"].copy()
    lp = predict_log(fitj["shared_raw"], d.N_once.values, d.N_rec.values, d.r.values, d.D.values, "F1", np.zeros(len(d)))
    d["resid"] = lp - np.log(d.L.values)
    per_rung = {r: dict(n=int(len(g)), rms=float(np.sqrt((g.resid ** 2).mean())), mean_abs=float(g.resid.abs().mean()),
                        max_abs=float(g.resid.abs().max())) for r, g in d.groupby("rung")}
    r10, others = d[d.rung == "10M"].resid, d[d.rung != "10M"].resid
    out = dict(n_dense_results=int(len(d)), per_rung=per_rung,
               ratio_rms_10M_to_other_rungs_pooled=float(np.sqrt((r10 ** 2).mean()) / np.sqrt((others ** 2).mean())),
               ratio_mean_abs_10M_to_other_rungs_pooled=float(r10.abs().mean() / others.abs().mean()),
               ratio_rms_10M_to_largest_other_rung=float(np.sqrt((r10 ** 2).mean()) / max(v["rms"] for k, v in per_rung.items() if k != "10M")),
               rule_threshold=2.0)
    out["rule_applies"] = bool(max(out["ratio_rms_10M_to_other_rungs_pooled"], out["ratio_mean_abs_10M_to_other_rungs_pooled"]) > 2.0)
    json.dump(out, open("results/fit/ruler_residuals.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
