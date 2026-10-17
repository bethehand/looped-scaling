"""Published looped-model results restated as effective-parameter multipliers (exploratory; step 2 of 2026-10-15).

M = N* / N: the size N* of a non-looped model that would match the looped model, divided by the looped model's
own size N, the same quantity as fit/local_phi.py. Where a paper reports a ladder of non-looped models, log N is
interpolated linearly in the reported metric between the two neighbouring sizes (beyond the ladder only a bound is
given); where it reports phi, M follows from N_eff = N_once + r^phi N_rec. Every input below is a number printed in
the paper (or read off a plot where marked), with its location, checked on 2026-10-15 against the arXiv versions
cited in docs/report.md. Caveats per row: the metric (loss or benchmark), whether the non-looped ladder had the
same tokens as the looped model (equal D) or the same tokens per parameter (equal D/N; the bigger comparison models
then also saw more data, so M is understated), and multi-epoch data.

    python fit/external_multipliers.py   -> results/fit/external_multipliers.csv
"""
from __future__ import annotations

import argparse
import math
import os

import pandas as pd


def from_ladder(value: float, ladder: list[tuple[float, float]], higher_is_better: bool) -> tuple[float, str]:
    """N* by interpolating log N linearly in the metric between neighbouring ladder sizes. Returns (N*, relation)."""
    pts = sorted(ladder)
    vals = [v if higher_is_better else -v for _, v in pts]
    x = value if higher_is_better else -value
    for (n0, _), (n1, _), v0, v1 in zip(pts, pts[1:], vals, vals[1:]):
        if v0 <= x <= v1 and v1 > v0:
            return math.exp(math.log(n0) + (x - v0) / (v1 - v0) * (math.log(n1) - math.log(n0))), "="
    if x > vals[-1]:
        return pts[-1][0], ">"
    return pts[0][0], "<"


def rows() -> list[dict]:
    out = []

    # Saunshi et al. 2025 (arXiv 2502.17416), Table 3: Pile validation perplexity after 250B tokens, k-layer blocks.
    # Non-looped ladder at the same tokens: 4, 6, 8, 12 and 24 layers (10.12, 9.25, 8.75, 8.16, 7.40); N taken
    # proportional to the number of layers (the 24-layer model has 1B parameters). Until 2026-10-17 only the two
    # 12-layer looped models were restated, against the 12- and 24-layer points; the three smaller blocks looped 3, 4
    # and 6 times were added after a citation check of the paper (deviation log 2026-10-17). The 12-layer values do
    # not change, since they lie between the same two ladder points.
    ladder = [(4, math.log(10.12)), (6, math.log(9.25)), (8, math.log(8.75)), (12, math.log(8.16)), (24, math.log(7.40))]
    for name, k, ppl, r, place in (("Loop 12x2", 12, 7.90, 2, "whole"), ("Middle loop (4x1,4,1)", 12, 7.81, 4, "middle"),
                                   ("Loop 8x3", 8, 8.19, 3, "whole"), ("Loop 6x4", 6, 8.42, 4, "whole"),
                                   ("Loop 4x6", 4, 8.79, 6, "whole")):
        ns, rel = from_ladder(math.log(ppl), ladder, higher_is_better=False)
        out.append(dict(study="Saunshi et al. 2025", source="Table 3", setting=name, placement=place, r=r,
                        N=f"{k} layers (24-layer model: 1B)", tokens="250B", tpp=f"~{round(500 * 12 / k, -1):.0f}",
                        metric="Pile log-perplexity", comparison="equal params, equal tokens", M=ns / k, relation=rel,
                        note="N taken proportional to the number of layers"))

    # Schwethelm et al. 2026 (arXiv 2604.21106), Table 2: phi = 0.459 [0.41, 0.53]; layout 2+2 prelude/coda,
    # 16/r recurrent layers; N_once = 4 n_b, N_rec = (16/r) n_b + n_i with n_i = 2d^2 = n_b/6 (Section 3.2).
    for r in (2, 4, 8):
        n_once, n_rec = 4.0, 16 / r + 1 / 6
        M = (n_once + r ** 0.459 * n_rec) / (n_once + n_rec)
        out.append(dict(study="Schwethelm et al. 2026", source="Table 2, Section 3.2", setting=f"iso-depth, r={r}",
                        placement="middle", r=r, N="11M-1.14B", tokens="0.34-7.78B", tpp="compute-optimal grid",
                        metric="loss (fitted law)", comparison="equal params, equal tokens (from phi)", M=M,
                        relation="=", note="phi = 0.46 [0.41, 0.53] from a fitted law; abstract: r=4, 410M ~ 580M "
                                           "non-looped (M = 1.41)"))

    # Prairie et al. 2026, Parcae (arXiv 2604.12946), Table 5: validation perplexity, mu_rec = 8, backprop depth 4.
    # Transformer ladder 140M/370M/770M/1.3B on 11.2B/29.6B/61.6B/104B tokens (about 80 tokens per parameter each).
    ladder = [(140e6, math.log(21.48)), (370e6, math.log(15.79)), (770e6, math.log(13.08)), (1.3e9, math.log(11.95))]
    for size, label, ppl, tok in ((140e6, "140M", 19.06, "11.2B"), (370e6, "370M", 14.49, "29.6B"),
                                  (770e6, "770M", 12.49, "61.6B"), (1.3e9, "1.3B", 11.42, "104B")):
        ns, rel = from_ladder(math.log(ppl), ladder, higher_is_better=False)
        out.append(dict(study="Prairie et al. 2026 (Parcae)", source="Table 5; Table 21 (tokens)",
                        setting=f"{label}, T=8", placement="middle", r=8, N=label, tokens=tok, tpp="~80",
                        metric="validation log-perplexity", comparison="equal params, equal tokens per parameter",
                        M=ns / size, relation=rel,
                        note="ladder at equal D/N: the bigger Transformers also saw more tokens, so M is understated"))

    # Zhu et al. 2025, Ouro (arXiv 2510.25741 v6), App. D.1, Figs. 13-14: 20B FineWeb-Edu tokens for every model,
    # benchmark average (6 tasks); values READ OFF THE PLOTS (no numbers in the text). Non-looped ladder = T=1.
    t1 = [(53e6, 0.339), (134e6, 0.365), (374e6, 0.409), (778e6, 0.440), (1.36e9, 0.461)]
    t4 = {53e6: 0.330, 134e6: 0.374, 374e6: 0.422, 778e6: 0.463, 1.36e9: 0.480}
    for size, label in ((53e6, "53M"), (134e6, "134M"), (374e6, "374M"), (778e6, "778M"), (1.36e9, "1.36B")):
        ns, rel = from_ladder(t4[size], t1, higher_is_better=True)
        out.append(dict(study="Zhu et al. 2025 (Ouro)", source="App. D.1, Figs. 13-14 (plot readings)",
                        setting=f"{label}, T=4", placement="whole", r=4, N=label, tokens="20B",
                        tpp=f"~{20e9 / size:.0f}", metric="benchmark average", comparison="equal params, equal tokens",
                        M=ns / size, relation=rel, note="values read off plots, about +-0.003 each"))

    # Marchenko et al. 2026, Closing the Loop (arXiv 2610.00673), Section 5.2.1: the 1.4B LoopLM (4 steps) beats the
    # parameter-matched dense 1.4B on all 12 benchmarks; the 3.9B dense model (inference-compute matched at 3 steps)
    # is ahead or level on every benchmark. 310B tokens: 210B of pretraining (about 13 epochs of a 16B-token corpus)
    # plus 100B of mid-training.
    out.append(dict(study="Marchenko et al. 2026", source="Section 5.2.1, Table 2", setting="1.4B, 4 steps",
                    placement="whole", r=4, N="1.4B", tokens="310B (multi-epoch)", tpp="~220",
                    metric="12 benchmarks", comparison="equal params, equal tokens (lower bound); 3.9B dense (upper)",
                    M=float("nan"), relation="1 < M < 2.8", note="bounds only: 1 < M < 3.9/1.4"))

    # Kaplan et al. 2020 (arXiv 2001.08361), App. D.2, Fig. 17: "slightly better when comparing models with equal
    # parameter count" for 2x, 4x and 8x reuse. No number is given.
    out.append(dict(study="Kaplan et al. 2020", source="App. D.2, Fig. 17", setting="2x/4x/8x reuse",
                    placement="whole", r="2-8", N="non-embedding", tokens="-", tpp="-", metric="loss",
                    comparison="equal params", M=float("nan"), relation="slightly above 1", note="qualitative only"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/fit/external_multipliers.csv")
    a = ap.parse_args()
    df = pd.DataFrame(rows())
    df["M"] = df.M.round(2)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 220)
    print(df[["study", "setting", "r", "tpp", "metric", "comparison", "relation", "M"]].to_string(index=False))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
