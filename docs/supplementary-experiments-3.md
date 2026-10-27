# Supplementary experiment S10: is the looped model's parameter efficiency on text a depth effect?

Plan written and frozen on 2026-10-27, before any of its runs, at the author's request ("按你说的补第一个实验 用我的gs01
现在做"). Its SHA-256 is recorded in `supplementary-experiments-3.sha256` and in the deviation log. The labels follow
the paper: results of this plan are pre-specified.

## Question

Every dense reference on text has eight layers and differs only in width, while a middle block looped 4 times executes
2 + 4·4 + 2 = 20 layers. The multiplier M = N*/N (what each parameter of the looped model is worth in parameters of the
8-layer dense ladder) may therefore measure what depth buys at a fixed number of parameters, not what sharing buys. A
review of 2026-10-27 named this the referee's most likely question. S10 trains dense models with the same number of
parameters and nearly the same executed depth as the looped models, but no shared layers.

## Design

| Size | Looped model of the grid (middle block, r = 4) | Dense, 8 layers (the ladder) | New: dense, 18 layers |
|---|---|---|---|
| 20M | 8 × 448, executes 20 layers, N = 27.25M | 8 × 448, N = 26.84M | 18 × 320, N = 28.11M |
| 40M | 8 × 640, executes 20 layers, N = 50.97M | 8 × 640, N = 50.15M | 18 × 448, N = 51.22M |

- N counts the output head, as everywhere in the study; the head dimension stays 64, so widths are multiples of 64 and
  18 layers is the depth that matches the looped model's N within about 3% (20M) and 0.5% (40M). Executed depth 18 against the looped model's 20.
- Training is the recipe of the grid's 20M and 40M runs, unchanged: the same data in the same order, the same batch
  (131,072 and 204,800 tokens per step), warm-up (2 N_rung tokens), AdamW settings, one constant-rate trunk to 32 N_rung
  with cooldown branches of 20% that end at 10, 20 and 40 N_rung (the looped runs' budgets, in tokens identical to
  theirs), the same evaluation sets. Seeds 42, 43 and 44 (initialisation only), as in the grid.
- Learning rate: the frozen per-width rate (`code/configs/lr_table_frozen.json`: width 320 → 2.19e-3, width 448 →
  1.53e-3) for the embedding and output layers; the hidden layers use (L/12)^−1/2 times it with L = 18, as the code does
  for any depth.
- Learning-rate check: seed 42 at 0.71 and 1.41 times that rate, at both sizes and all three budgets.
- Training FLOPs per token (code accounting, forward + backward): 20M: dense-8 205M, looped 456M, dense-18 239M;
  40M: dense-8 364M, looped 835M, dense-18 406M.
- Runs: 6 main runs + 4 learning-rate runs. Configs in `code/configs/supp3/`, manifest
  `code/configs/manifest_supp3_deep.csv` (rung = the size the run is compared with, so budget multiples are relative to
  that rung's N). No change to the training code. Results are exported to `code/results/supp3/` and kept out of
  `results/all_results.csv` (the collector keeps only the main manifest's runs), so no pre-registered fit changes.
- Cost and time: about 15 GPU-hours for the main runs and 10 for the check, about 8 hours on GS01's four GPUs;
  about 23 GB of checkpoints on /mnt/nvme0 (315 GB free on 2026-10-27).

## Quantities

1. Gap: dense-18 minus looped loss on the main validation set at the same tokens and seed, per budget; seed means.
2. M_deep = N*/N_deep, with N* read off the 8-layer dense ladder (including the dense points of S4) exactly as for the
   looped models (`code/fit/local_phi.py`), with seed-resampling intervals; next to it M_loop of the same size and budget.
3. R = log M_deep / log M_loop at 40M, per budget and on average over the three budgets: the share of the looped
   model's advantage over the 8-layer ladder that a dense model of the same size and depth recovers.

## Readings, fixed in advance

- Primary, at 40M on average over the three budgets (M_loop is 1.15–1.34 there; at 20M it is near 1 and R is unstable,
  so 20M is reported descriptively):
  - R ≥ 0.75: "at equal parameters, most of the looped model's advantage on text is a depth effect; a dense model of
    the same size and depth recovers it with about half the training compute";
  - 0.25 ≤ R < 0.75: "partly depth, partly sharing";
  - R < 0.25: "looping buys more than depth: a deeper dense model of the same size recovers little of the advantage".
- Gap: if |dense-18 − looped| ≤ 0.01 nats at all three 40M budgets, also report "on par at 40M".
- Learning rate: if 0.71 or 1.41 times the rate beats the rule's rate by more than 0.01 nats at a budget, the readings
  are also reported at the best rate tried (seed 42), as in S1; the primary reading stays at the rule's rate.
- Equal compute: the dense-18 runs are compared with the looped runs at equal training FLOPs as in Section 4.3
  (exploratory).

## Prediction

Before any run: R at 40M between 0.25 and 0.75 ("partly depth"); the 18-layer dense model better than the 8-layer one
of the same size by 0.01–0.03 nats at 40M; the rule's learning rate within 0.01 nats of the best of the three. Reasons:
reports that sub-billion models gain from depth at a fixed size (for example MobileLLM, arXiv 2402.14905) against the
weak dependence on shape at larger sizes (Kaplan et al. 2020).

## Use

A paragraph in the paper's Section 4.2 and the limitations, the report and the README; the outcome is reported
whichever way it falls.
