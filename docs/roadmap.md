# Roadmap

> English translation (2026-10-11) of the project roadmap kept in Chinese since 2026-09-24.

Updated 2026-09-26. When an item is done, change `[ ]` to `[x]`.

## Goal

Write a paper: run controlled experiments on small models from 10M to 160M to measure **how many parameters one loop is worth**, i.e. the loop-equivalence exponent φ, and answer how two design choices change it.

- **H1 Truncated backprop**: if backpropagation passes only through the second half of the loops, does φ become smaller? It counts only if it holds under both iso-token and iso-FLOP (equal compute) accounting.
- **H2 Loop placement**: whether looping only the middle layers gives a higher φ than looping the whole stack.
- **H3 Functional form**: whether φ saturates with the loop count, and whether the "geometric saturation" formula predicts more accurately than the r^φ formula.
- **H4 Optimal loop count**: the optimal number of loops for a given training compute, and for a given inference compute.

Scale and conditions: looped models at four rungs, 10M, 20M, 40M and 80M, and the dense ruler additionally at 160M; 1, 2, 4 and 8 loops; FineWeb-Edu 10BT, single epoch; 4 RTX 4090 GPUs. The settings in `docs/preregistration.md` are authoritative; all later changes are recorded in `docs/deviation-log.md`.

## Phase 0: Survey and topic selection

- [x] Survey the recurrent-depth direction; write a survey report and 15 candidate topics
- [x] Choose a small-scale version of topic #3
- [x] Close reading of the 9 papers the study builds on; write reading notes
- [x] Freeze the pre-registration v1.0 and record its checksum

## Phase 1: Code and environment

- [x] Models: three architectures (dense, middle-block looped, whole-stack looped); input injection; truncated backprop; residual scaling
- [x] Parameter and FLOP counting, training script (WSD schedule, cooldown branches, resumable training), job queue, fitting scripts
- [x] GitHub repository; the Mac and GS01 sync through it
- [x] GS01 environment: PyTorch 2.6, package mirrors inside China, setuptools and Triton
- [x] Fix two PyTorch 2.6 issues: with truncation, the looped layers received no gradient; the compiler mistook the residual scaling for a quantization coefficient
- [x] Per-layer compilation for speed: measured 1.28 to 1.42× faster
- [x] Tests: all pass on the Mac; on GS01 the core tests and the 14 compile tests pass

## Phase 2: Data

- [x] Training data: FineWeb-Edu 10BT, files 000 to 012, 10.41B tokens in total
- [x] Self-trained 16k vocabulary, numbers split into chunks of at most 3 digits, checksum frozen
- [x] Four validation sets: main validation set 50M tokens; second validation set, FineMath set and code set 20M each

## Phase 3: The 5 pre-run checks

- [x] Check 1: truncated-backprop gradients are correct; the looped model at r=1 matches the dense model bit for bit
- [x] Check 2: dense-model loss decreases monotonically with width; looped r=4 beats r=1 at equal parameters (80M had already passed; 40M passed on 2026-10-07: middle-block r=4, three-seed mean at 10N 3.394 vs. 3.404 for the dense model, better by 0.033 and 0.052 at 20N and 40N respectively)
- [x] Check 3: FLOP counts agree with measurements
- [x] Check 4: measure throughput and GPU memory for all 11 configurations; rearrange the schedule
- [x] Check 5: learning-rate sweep completed and frozen

## Phase 4: Learning-rate sweep (33 runs, about 16 hours)

- [x] Write the selection rule into the deviation log before the results come in
- [x] Launch the sweep (2026-09-24)
- [x] After about 2 hours, check whether the best learning rate for 10M to 40M lies inside the tested range: all three rungs are in the interior, no extra runs needed; β2 set to 0.95
- [x] After everything has finished, run pick_lr; the minima of all five rungs are in the interior, no extra runs needed
- [x] Freeze the learning rate and β2 for each width; write them into Appendix C of the deviation log (2026-09-25)
- [x] The author chose to do the loop learning-rate check first (2026-09-25)
- [x] Loop learning-rate check: 20M, 10 loop configurations × 3 learning rates, 30 runs in total, no failures (finished 2026-09-25; faster after the GS01 power supply was restored at 11:02)
- [x] Decide the learning-rate multiplier for the looped models by the decision rule: no adjustment, the dense-model learning rates are kept (pooled shift 0.97×, threshold 0.84×)

## Phase 5: Main grid (111 runs, about 12 days)

- [x] Seed diagnostic: the deficit of whole-stack r=4 with full backprop reproduces on seeds 43 and 44 (gap 0.39, threshold 0.3); the main grid goes ahead as planned (2026-09-25)
- [x] After the power supply was restored, re-measure the speed of the cells with gradient checkpointing: with compilation they are 1.39 to 1.45× faster, so by the rule all 6 now use compilation (2026-09-26)
- [x] Regenerate the configs with the frozen learning rates and push them: 111 configs, estimated about 12 days on 4 GPUs (2026-09-26)
- [x] Launch the main grid: started 2026-09-26 01:12, code version 35f42ee; 10M runs first, all 33 expected to finish around 19:30 the same day
- [x] At the end of the first day, check whether the 10M results are reasonable: all 33 finished normally; at 10M most loop configurations are worse than the dense model, the gap narrows as data increases, and whole-stack r=8 is the best (2026-09-26, details in the deviation log)
- [x] After the 40M results are in, do the deferred check 2: passed (2026-10-07)
- [x] Check progress every day and handle failures until everything is done: on the morning of 2026-10-07 all 111 runs had finished, 0 failures; 1 run collapsed and 30 runs had loss spikes, all handled according to the rules
- [x] Exploratory experiment after the main grid: 10M whole-stack r=4 truncated, learning rate halved, 3 seeds — all blow-ups disappeared, and at large budgets the results were actually better (finished 2026-10-08, see docs/results.md 5.4)

## Phase 6: Fitting and analysis (about 2 weeks)

- [x] Collect all results: 474 result rows, 4 voided under the collapse rule (2026-10-07, `code/results/all_results.csv`)
- [x] Fit the ruler on the dense models: E sits at the lower bound, α = 0.054, weakly identified; joint fit α = 0.267 (2026-10-07, see docs/results.md Section 3)
- [x] Fit φ per cell with bootstrap confidence intervals: middle block with full backprop 0.13, whole stack truncated 0.17, the rest pushed to 0; the intervals are wide (2026-10-07)
- [x] Compare functional forms with leave-one-rung-out extrapolation: r^φ beats geometric saturation, per-cell φ beats a shared φ (2026-10-07)
- [x] Decide H1 to H4 by the pre-registered rules: H1 not supported (iso-token and iso-FLOP accounting agree), H2 depends on the accounting, H3 not supported and in the opposite direction, H4 the dense model is optimal under both kinds of accounting (2026-10-07, see docs/results.md Section 4)
- [x] Test "with whole-stack looping at small r, truncation is actually better": it holds at 10N for 10M and 20M, and the two are level at 40N; for 80M r=8 it holds at 10N and is reversed at 40N; see docs/results.md 5.3 (2026-10-07)
- [x] Robustness: second validation set (higher φ), mean of the last three evaluation points (consistent), dropping r=8 (φ nearly zero), seed-to-seed spread, excluding the output head (slightly lower), half budget, excluding runs with spikes (2026-10-07)
- [x] Report the number of "blow-ups" (loss spikes) per run, and do a robustness fit without the runs that blew up (rule in the deviation log, 2026-09-26); results whose cooldown started only after a collapse enter no fit (rule of 2026-09-30; only 40M whole-stack r=8 truncated, seed 43) — `code/results/spikes.csv`, `results/fit/no_spikes/`, docs/results.md 3.3 and 5.4 (2026-10-07)
- [x] Figures: the four main figures have been generated in `code/results/figures/` (2026-10-07, `fit/make_figures.py`)
- [x] Per-token analysis: which tokens the loops help; dense models from three seeds compared with each other as the null baseline; changing the loop count at inference (2026-10-08, docs/results.md 5.7, Figures 6 and 7)

## Phase 6.5: Extension experiments after the main grid (from 2026-10-08; outside the pre-registration, reported as exploratory only)

- [x] Add two switches, off by default, to the training code: continue the trunk from the checkpoint of a finished run (`init_from`), and a random loop count at every step (`r_sample`); 62 tests pass; config generation with `make_configs.py --ext` (2026-10-08)
- [x] 160M middle-block r=4, two seeds + 160M dense model, seed 43 (done 2026-10-11; docs/results.md 5.12, Figure 12: the gap keeps improving with scale up to 160M)
- [x] Data extension: 20M looped to 80N, 10M looped to 80N/160N, 10M dense to 160N (seed 42 finished 2026-10-09 05:00, docs/results.md 5.10, Figure 11: the advantage keeps growing with D/N, and 10M r=8 at 160N is equivalent to 1.8× the data); the five runs for seed 43 started on GPU 3 at 2026-10-09 11:30 (about 8 hours)
- [x] Random loop-count training: 20M middle-block r=8, one run each with full and truncated backprop; inference loop-count sweep from 1 to 32 and per-loop probes (finished 2026-10-09, docs/results.md 5.9, Figure 10): fully robust to the loop count (contraction to a fixed point), but more loops bring no gain, and it ties with the dense model
- [x] Per-loop linear probes: 40N checkpoints of 80M/40M/20M middle-block r=8 and 80M whole-stack r=8 truncated (2026-10-08, docs/results.md 5.8, Figures 8 and 9); the probes and the loop-count sweep for the random-loop-count models are queued to run automatically after their training
- [x] Design of the synthetic "controllable step count" reasoning tasks: v2 frozen on 2026-10-09 (docs/synthetic-reasoning-design.md), code in `code/synthetic/`
- [ ] Synthetic-task pilot (T1 × 5 conditions × 1 seed + learning-rate sweep, GPU 3) → formal suite (2 tasks × 8 conditions × 2 seeds) → intermediate-value probes → write into Section 5 of docs/results.md; whether to include this in the paper will be discussed after the pilot
- [ ] Write the results into the "Extensions" subsection of Section 5 of docs/results.md; update the figures and tables

## Phase 7: Paper and open-source release (about 3 weeks)

- [x] Decide the framing: measurement as the backbone, mechanism as the highlight (agreed with the author on 2026-10-09); the outline and claims ledger are in our internal notes
- [ ] Write the paper (Sections 2–4 can be written first; Sections 5 and 6 wait for the additional seeds; Section 7 waits for the synthetic tasks)
- [ ] Open-source the code, configs and all training logs
- [ ] Post on arXiv
- [ ] Submit: ICML 2027, COLM 2027 or TMLR

## Approximate timeline (updated 2026-09-26)

| Phase | Estimate |
|---|---|
| Learning-rate sweep | Done (September 25) |
| Loop learning-rate check | Done (September 25) |
| Seed diagnostic and speed measurement | Done (September 26) |
| Main grid | Started September 26, about 12 days, finishing around October 8 |
| Fitting and analysis | Pre-registered part done (October 7); exploratory analysis done October 8 |
| Extension experiments | Started October 8; 160M finishes around October 11, the rest in 1 to 2 days |
| Paper writing | In parallel with the extension experiments; draft around mid-November |

## Possible later work

- Looped cells at 160M (already started as an extension experiment, 2026-10-08; r=8 and the whole stack left for later)
- Multi-epoch, data-constrained settings, left for the next paper
