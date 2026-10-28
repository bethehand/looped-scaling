# Roadmap

> English translation (2026-10-11) of the project roadmap kept in Chinese since 2026-09-24.

Updated 2026-10-28 (first written 2026-09-24). When an item is done, change `[ ]` to `[x]`.

## Goal

Write a paper: run controlled experiments on small models from 10M to 160M to measure **how many parameters one loop is worth**, i.e. the loop-equivalence exponent φ, and answer how two design choices change it.

- **H1 Truncated backprop**: if backpropagation passes only through the second half of the loops, does φ become smaller? It counts only if it holds under both iso-token and iso-FLOP (equal compute) accounting.
- **H2 Loop placement**: whether looping only the middle layers gives a higher φ than looping the whole stack.
- **H3 Functional form**: whether φ saturates with the loop count, and whether the "geometric saturation" formula predicts more accurately than the r^φ formula.
- **H4 Optimal loop count**: the optimal number of loops for a given training compute, and for a given inference compute.

Scale and conditions: looped models at four rungs, 10M, 20M, 40M and 80M, and the dense ruler additionally at 160M; 1, 2, 4 and 8 loops; FineWeb-Edu 10BT, each run reading a prefix of one fixed-order stream (no document seen twice); 4 RTX 4090 GPUs. The settings in `docs/preregistration.md` are authoritative; all later changes are recorded in `docs/deviation-log.md`.

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
- [x] Check 2: dense-model loss decreases monotonically with width; looped r=4 beats r=1 at equal parameters (80M had already passed; 40M passed on 2026-10-07: middle-block r=4, three-seed mean at 10N 3.394 vs. 3.404 for the dense model, better by 0.031 and 0.050 at 20N and 40N respectively)
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
- [x] Compare functional forms with leave-one-rung-out extrapolation: r^φ beats geometric saturation, per-cell φ beats a shared φ (2026-10-07; the first reading was withdrawn on 2026-10-14: in the pre-registered split, with 160M in neither training set, the data do not separate the two forms)
- [x] Decide H1 to H4 by the pre-registered rules: H1 not supported (iso-token and iso-FLOP accounting agree), H2 depends on the measure, H3 not supported (the data do not separate the two forms; the first reading, "in the opposite direction", was withdrawn on 2026-10-14), H4 the dense model is optimal under both kinds of accounting (2026-10-07, see docs/results.md Section 4)
- [x] Test "with whole-stack looping at small r, truncation is actually better": it holds at 10N for 10M and 20M, and the two are level at 40N; for 80M r=8 it holds at 10N and is reversed at 40N; see docs/results.md 5.3 (2026-10-07)
- [x] Robustness: second validation set (higher φ), mean of the last three evaluation points (consistent), dropping r=8 (φ nearly zero), seed-to-seed spread, excluding the output head (slightly lower), half budget, excluding runs with spikes (2026-10-07)
- [x] Report the number of "blow-ups" (loss spikes) per run, and do a robustness fit without the runs that blew up (rule in the deviation log, 2026-09-26); results whose cooldown branch starts after a collapse, or ends more than 1.0 nats above the lowest trunk loss before it, enter no fit (rule of 2026-09-30; four results of two runs: three of 40M whole-stack r=8 truncated, seed 43, which collapsed, and the 10N result of 40M whole-stack r=8 with full backprop, seed 44, after a severe spike) — `code/results/spikes.csv`, `results/fit/no_spikes/`, docs/results.md 3.3 and 5.4 (2026-10-07)
- [x] Figures: the four main figures have been generated in `code/results/figures/` (2026-10-07, `fit/make_figures.py`)
- [x] Per-token analysis: which tokens the loops help; dense models from three seeds compared with each other as the null baseline; changing the loop count at inference (2026-10-08, docs/results.md 5.7, Figures 6 and 7)

## Phase 6.5: Extension experiments after the main grid (from 2026-10-08; outside the pre-registration, reported as exploratory only)

- [x] Add two switches, off by default, to the training code: continue the trunk from the checkpoint of a finished run (`init_from`), and a random loop count at every step (`r_sample`); 62 tests pass; config generation with `make_configs.py --ext` (2026-10-08)
- [x] 160M middle-block r=4, two seeds + 160M dense model, seed 43 (done 2026-10-11; docs/results.md 5.12, Figure 12: the gap keeps improving with scale up to 160M)
- [x] Data extension: 20M looped to 80N, 10M looped to 80N/160N, 10M dense to 160N (seed 42 finished 2026-10-09 05:00, docs/results.md 5.10, Figure 11: the advantage keeps growing with D/N, and 10M r=8 at 160N is equivalent to about 2.2× the data on the dense curves extended in S4, two seeds; first given as 1.8×, from seed 42 with the dense curve extrapolated beyond 160N); the five runs for seed 43 started on GPU 3 at 2026-10-09 11:30 (about 8 hours)
- [x] Random loop-count training: 20M middle-block r=8, one run each with full and truncated backprop; inference loop-count sweep from 1 to 32 and per-loop probes (finished 2026-10-09, docs/results.md 5.9, Figure 10): fully robust to the loop count (contraction to a fixed point), but more loops bring no gain, and it ties with the dense model
- [x] Per-loop linear probes: 40N checkpoints of 80M/40M/20M middle-block r=8 and 80M whole-stack r=8 truncated (2026-10-08, docs/results.md 5.8, Figures 8 and 9); the probes and the loop-count sweep for the random-loop-count models are queued to run automatically after their training
- [x] Design of the synthetic "controllable step count" reasoning tasks: v2 frozen on 2026-10-09 (docs/synthetic-reasoning-design.md), code in `code/synthetic/`
- [x] Synthetic-task pilot and learning-rate sweep (done 2026-10-11)
- [x] Formal chain runs, seeds 2 and 3 (done 2026-10-11); pointer chasing (T2) was not learned by the four conditions tried on it (two run to the end) and is dropped from this paper by the author's decision
- [x] Chain seeds 4–6 → intermediate-value probes (three positions per step) → verdicts on P1–P5 → Section 5.15 of docs/results.md (done 2026-10-12; P1 direction holds, P2 reversed, P3 partly, P4 half, P5 holds; P3 revised on 2026-10-14: none of its three parts holds as stated; revised again on 2026-10-15: P3's third part holds, P2 restated as not supported as stated with small gaps; revised on 2026-10-27: P3's first part is half right, since fewer loops than trained do hurt, and the readings of P3's third part and of P4 were fixed after the results)
- [x] Write the results into Section 5 of docs/results.md (5.9–5.15) and update the figures and tables (done 2026-10-12; the supplementary experiments were added later as 5.16 and 5.18)

## Phase 6.6: Supplementary experiments S1–S3 (2026-10-13 to 2026-10-14; outside the pre-registration, exploratory)

- [x] Plan frozen before any run ([supplementary-experiments.md](supplementary-experiments.md)); code, configs and the queue in commit 97385b6
- [x] S1: learning rate re-tuned at the comparison budgets (dense and middle-block r=4 at 10M/20M/40M; 24 runs plus 1 edge point) → Q1 and Q2 hold, Q3 unchanged, Q4 bias above 0.01 at 10M (up to 0.024, always against the looped model), Q5 replicates 5.14
- [x] S2: synthetic task at learning rates 5e-4 and 1e-3 (four looped conditions, seeds 2–4; 24 runs and probes) → R1: 8/12 learned at 5e-4, 2/12 at 1e-3; R2: 3 cells do not support, 1 supports, 4 inconclusive
- [x] S3: inference cost on one RTX 4090 (KV-cached decoding): decoding time and KV cache follow the executed layers; at equal loss, at most 19% less weight memory; higher peak memory at batch 32, about the same at batch 1 for middle blocks at 80M–160M (re-measured 2026-10-14)
- [x] Update the README, the report and docs/results.md as fixed in the plan (results §5.16, 2026-10-14)
- [x] External review of the published results: five errors corrected and one code safeguard added (2026-10-14; deviation log); the two remaining robustness variants of the H3 test recomputed and one more wording error corrected (2026-10-15)
- [x] Second review (2026-10-15): about 35 errors and overstatements corrected (synthetic P3's third part holds; P4 judged by the design's measure; the equal-compute claim limited to the measured range; numbers and wording), related work and 30 references added to the report, every table and figure mapped to its command in code/README.md (deviation log 2026-10-15)

## Phase 6.7: Supplementary experiment S10 (2026-10-27 to 2026-10-28; outside the pre-registration, exploratory)

- [x] Plan frozen before any run ([supplementary-experiments-3.md](supplementary-experiments-3.md)); configs and manifest in commit 941a71b; analysis script written before any result (5893604)
- [x] S10: dense models with 18 layers and the looped model's number of parameters at 20M and 40M (6 runs, plus 4 learning-rate runs) → R = 0.50 at 40M, "partly depth, partly sharing", as predicted; at the best tested learning rate the deep model is ahead at 10N; the part of the advantage that grows with data is not depth ([results §5.19](results.md); deviation log 2026-10-28)

## Phase 7: Paper and open-source release (about 3 weeks)

- [x] Decide the framing: measurement as the backbone, mechanism as the highlight (agreed with the author on 2026-10-09); the outline and claims ledger are in [paper-plan.md](paper-plan.md)
- [ ] Write the paper: first draft and a revision after a referee-style read, 2026-10-17, and corrections after a third review, 2026-10-27, and a reorganisation along the argument with S10 added, 2026-10-28, in [paper/](../paper/) ([PDF](../paper/paper.pdf)); the plan is in [paper-plan.md](paper-plan.md). Next: the author's read, then submission to TMLR
- [x] Open-source the code, configs and all training logs (public repository; the training logs were added on 2026-10-15)
- [ ] Post on arXiv
- [ ] Submit: ICML 2027, COLM 2027 or TMLR

## Approximate timeline (updated 2026-10-28)

| Phase | Estimate |
|---|---|
| Learning-rate sweep | Done (September 25) |
| Loop learning-rate check | Done (September 25) |
| Seed diagnostic and speed measurement | Done (September 26) |
| Main grid | September 26 to October 7 (111 runs, no failures) |
| Fitting and analysis | Pre-registered part done (October 7); exploratory analysis done October 8 |
| Extension experiments | October 8 to 12; supplementary experiments S1–S3 October 13 to 14, S4–S9 October 15 to 17 and S10 October 27 to 28 |
| Paper writing | Plan and first draft October 17 (positioning and title decided that day); corrections after a third review October 27; reorganised along the argument, with S10, October 28 |

## Proposed next steps (2026-10-15; the author's decision on each step is noted below)

Step 1 (corrections, references, reproduction map and training logs) and step 2 (re-analyses) were completed on 2026-10-15.

- **Step 2, re-analyses, no GPU (done 2026-10-15; results §5.15 and §5.17):** ruler-free φ as a function of size and tokens per parameter, with bootstrap intervals and an interaction term; bounds on the equal-compute gap under two extrapolations; external results (Ouro, Parcae, iso-depth, Kaplan) restated as effective-parameter multipliers; report the adaptive-stopping evaluation already in the formal synthetic results (pre-specified in the frozen design for the step-matched condition and not yet reported; the other conditions' values marked exploratory)
- **Step 3, cheap experiments (S4–S9, about 60–70 GPU-hours; plan `supplementary-experiments-2.md` frozen on 2026-10-15, run on GS01 from 2026-10-15 22:07 to 2026-10-17 09:37; done, [results §5.18](results.md)):** dense models at high tokens per parameter (10M to 640N, 20M to 320N, plus a 10M looped model at 0.35× the dense rate, the lowest rate S1 tried; replaces the extrapolated equal-compute comparisons), chain-of-thought and pause-token baselines on the chain task, with the inference compute of each counted (is the same inference compute better spent on more loops or on writing out the steps?), KV-cache sharing tested on the existing 20M models, seeds that change the data order. Test-time compute: better stopping rules (stop when the prediction no longer changes between loops) and voting over loop counts on the existing synthetic checkpoints (minutes of GPU time); a new training condition, loops = steps plus a random number of extra loops, to see whether a model can both extrapolate far and stop by itself (five seeds, about 7 GPU-hours; a design change). **Outcome:** the dense model wins at equal compute throughout the measured range, now to 640 tokens per parameter, with the two level on average at 10M/160N under the lowest rate tried for the looped model (−0.011 and +0.005; S4); a chain of thought is perfect in range at under a third of the compute and does not extend at all, pause tokens add nothing (S5); sharing the KV cache costs the fixed-loop models 1.8 to 3.7 times their gain (S6); in distribution the data order matters 0.1–0.5× as much as the initialization (S7); the model's own confidence is the best stopping rule in every condition except r = 4, where a state rule running far more loops ties it (S8); loops = steps plus random extra loops both extends (11.6 steps) and stops by its confidence (0.93) (S9). Deviation log 2026-10-17 for the two out-of-memory stops and the smaller departures.
- **Step 4, optional (not run; the author decided on 2026-10-17 to skip all four items and go to the paper; the limitation each one leaves is listed in [paper-plan.md](paper-plan.md) §8):** a dense control of equal executed depth on text (about 60 GPU-hours); a second synthetic task family; two more seeds at 80M; dense and looped models at 20M–40M trained on reasoning-heavy text (math and code) to redo the equal-parameter and equal-compute comparisons (about 40 GPU-hours; tests whether the larger gains on math and code text, results 5.6, reach parity or better at equal compute; added by the author on 2026-10-15)
- **Step 5, paper (from 2026-10-17; plan, claims ledger and outline in [paper-plan.md](paper-plan.md); positioning and title decided on 2026-10-17; draft in [paper/](../paper/)):** position as a pre-registered, multi-seed replication and extension of the iso-depth study, or around the question of when recurrent depth works as test-time compute; a new title; author information

## Possible later work

- Looped cells at 160M (already started as an extension experiment, 2026-10-08; r=8 and the whole stack left for later)
- Multi-epoch, data-constrained settings, left for the next paper
