# Supplementary experiments S4–S9 (step 3 of the plan of 2026-10-15)

Status: **frozen** on 2026-10-15 with the author's confirmation ("按你说的来 做第3步"). This file and its Chinese
original (kept on the author's machine) were committed before any of these runs started; their SHA-256 hashes are in
the [deviation log](deviation-log.md). Any later change is logged there under the usual rules.

Date: 2026-10-15

## 0. Why, and how this relates to the pre-registration

All six experiments are outside the pre-registration and **do not change the H1–H4 verdicts**. They follow the second
review of 2026-10-15 and the re-analyses of step 2 ([results §5.17](results.md)), which left four questions that the
existing runs cannot answer:

1. At equal training compute, the dense model wins everywhere its loss was measured; at the largest data per parameter
   the two are at parity within the error of an extrapolation. Is it parity, or does looping overtake? (S4)
2. Reasoning inside the architecture (more loops) against reasoning written out as tokens (chain of thought): which is
   better at equal parameters, and at equal inference compute? This is the author's core question. (S5)
3. The inference-time use of loops: can the model tell when to stop, and can a model be trained that both extrapolates
   far and stops by itself? (S8, S9)
4. Two stated gaps of the report: KV-cache sharing across loops was not measured (S6), and the seeds vary the
   initialization only (S7).

Everything here is exploratory. Each experiment states its decision quantities and its readings before it runs.

## S4. Text: dense models at high tokens per parameter

**Question:** in the equal-compute comparison ([results §5.11](results.md)), the dense model's loss at the looped
model's compute is extrapolated beyond the dense curve for 99 of 320 comparisons. Two extrapolations that bracket
the truth on the measured curves put the two models at parity, within 0.02 nats, at 10M with 160 tokens per
parameter and 20M with 80 (§5.17(b)). Measured, is it parity, a dense win, or a looped win?

### Design

- Continue the finished dense trunks with the training code and learning rates as they are (dense rate 1×, the best
  rate at every budget tested in S1):
  - 10M dense, seeds 42 and 43: from `runs/10M_dense_r1_d160_s<seed>/branch_1971978240.pt` (the trunk at 128
    tokens per parameter) to budgets of 320 and 640 tokens per parameter (branch points at 256 and 512);
  - 20M dense, seeds 42 and 43: from `runs/20M_dense_r1_s<seed>/branch_1717960704.pt` (64 tokens per parameter) to
    160 and 320 (branch points at 128 and 256).
- One more looped run: 10M middle block r=4 (full backprop) at 0.35× the dense rate, seed 43, budgets 10–160 tokens
  per parameter, the S1 recipe; S1 has this run for seed 42 only, and it is the looped model's best rate at 80–160.
- Two seeds, not three: the looped runs at 80 and 160 tokens per parameter exist for seeds 42 and 43 only, and the
  comparison is within a seed.
- Data: 640 tokens per parameter at 10M reads 9.86B of the 10.41B tokens of the training stream; 320 at 20M reads
  8.59B. No run repeats data. 160M cannot be extended this way (it would need more than 14B tokens).
- Cost: about 6 GPU-hours per dense seed at 10M, 7 at 20M, 3.5 for the looped run: about 30 GPU-hours.

### Quantities

- `fit/iso_flop_raw.py` and `fit/iso_flop_bounds.py` rerun with the new dense points: the dense curves now reach
  640 (10M) and 320 (20M) tokens per parameter, so the comparisons at 10M/80–160 and 20M/40–80 that were
  extrapolated become interpolated (all but the whole-stack r=8 truncated cell at 10M/160, whose compute multiple
  4.3 × 160 = 688 is 0.1 doubling beyond 640).
- S1's Q3 with the tuned looped model (0.35×, seeds 42 and 43) against the measured dense curve.
- `fit/local_phi.py` rerun: the multipliers at 10M/80–160 and 20M/80 gain dense rungs at those budgets.

### Readings, fixed in advance

For each (cell, seed) at the top budget (10M at 160, 20M at 80 tokens per parameter), gap = looped loss − dense loss
at equal training FLOPs, now measured (interpolated in log D on the extended curve).

- **Prediction.** The backtests of §5.17(b) bracket the measured gap between the last-segment rule (+0.03 to +0.08
  here) and the power law (−0.02 to +0.01). I predict gaps between −0.01 and +0.04, the dense model still ahead in
  most of these comparisons, and no gap below −0.02.
- **Reading rule.** If every measured gap at the top budget is above +0.01: "the dense model wins at equal training
  compute throughout the measured range, now to 640 tokens per parameter". If the gaps lie within ±0.01: "parity at
  the largest data per parameter". If any cell is below −0.01 in both seeds: "looping overtakes at <cell, tokens per
  parameter>", reported with the size of the margin. Mixed cases are reported cell by cell.
- The new dense points do not enter the pre-registered fits, which are frozen; they enter the exploratory analyses
  only (§5.11, §5.17).

## S5. Synthetic task: chain of thought and pause tokens against loops

**Question:** the looped models solve longer chains than they were trained on when given more loops (§5.15). A dense
model that writes the intermediate values out as tokens also computes step by step, in tokens instead of in depth.
At equal parameters, and at equal inference compute, which is better? And is it the serial computation that matters,
or just the extra tokens?

### Design

The chain task of the formal suite (32 lookup tables, trained on 1–8 steps, tested on 1–24; `synthetic/train_synth.py`
with `--family lookup --n-ops 32`, 32000 steps, batch 512, curriculum 0.5), with a new `--format` option:

- `cot`: the sequence is `BOS x₀ SEP (LUT c SEP)×d ARROW x₁ … x_d`. Training: loss at the ARROW position and at
  x₁ … x_{d−1}, predicting x₁ … x_d (teacher forcing). Evaluation: greedy generation of d tokens after ARROW; correct
  if the d-th generated token is x_d; also recorded: the share of chains whose d generated tokens all match.
- `pause`: `BOS x₀ SEP (LUT c SEP)×d PAUSE×d ARROW x_d`, with d pause tokens (one per step, so the sequence has as
  many tokens as the chain-of-thought one); loss and answer at ARROW, one forward pass (Goyal et al. 2024).
- Sequence length 112 (4d + 5 for d ≤ 24; the single-answer format uses 80).

Conditions, 5 seeds (2–6) each, learning rate by the formal suite's protocol (seed 1, 16000 steps, 2.5e-4 / 5e-4 /
1e-3 with the edge rule; chosen by the final-answer accuracy on d = 1..8):

| Condition | Parameters | Why |
|---|---|---|
| dense 8 layers, chain of thought | 6.5M | equal parameters to the looped model |
| dense 36 layers, chain of thought | 29.0M | equal executed depth to looped r=8 |
| dense 8 layers, pause tokens | 6.5M | the same extra tokens without serial computation |

Optional, after everything else and only if GPU time is left: looped r=8 with chain of thought (are the two
complementary?). Its result, if it exists, is reported as a fourth row.

**Inference compute** (reported with every comparison, not a decision quantity), in layer-passes per problem, where
L = 3d + 4 is the prompt length: looped r=8 at its own loop count 36·L, at 16 loops 68·L; dense 8 with chain of
thought 8·(L + d) with a KV cache (8·Σ(L + i) without); dense 36 with chain of thought 36·(L + d); pause 8·(L + d).
At equal parameters the chain-of-thought model uses about a third to a sixth of the looped model's compute, so if
the two tie, chain of thought is the cheaper one.

Caveat stated in advance: chain-of-thought training supervises every intermediate value; the looped models are
supervised on the answer only. A looped model with intermediate supervision is outside this plan.

### Quantities

Final-answer accuracy per step count (seed mean, min–max), the largest solvable step count (every length up to d at
≥ 0.9), at 12 and 16 steps; for chain of thought also the all-steps-correct share.

### Readings, fixed in advance

- **Q1, chain of thought against loops at equal parameters.** Dense 8 with chain of thought against looped r=8 with
  more loops (§5.15: 0.86 at 12 steps, 0.37 at 16), the random-loop model (0.83, 0.49) and the step-matched model
  (0.96, 0.37). Prediction: chain of thought extrapolates at least as far: ≥ 0.7 at 12 steps and ≥ 0.4 at 16. If it
  is below 0.4 at 12 steps, the likely cause is position extrapolation (sequences of up to 101 tokens against ≤ 36 in
  training), and the report says so.
- **Q2, pause tokens.** Decision quantity: accuracy at 12 steps minus that of the plain dense 8-layer model (0.06).
  Prediction: below 0.1 (no extrapolation: the extra tokens add parallel computation, not serial depth).
- **Q3, equal depth.** Dense 36 with chain of thought against dense 36 (0.02 at 12 steps). Prediction: ≥ 0.8 at 12.
- **Q4, compute.** The report presents Q1 at equal parameters and at equal inference compute separately, using the
  layer-pass counts above.
- **How the results are used.** A new subsection of the report ("loops against chain of thought") and
  [results §5.15](results.md); the summary's sentence that loops were not compared with chain of thought is replaced.
  P1–P5 are untouched.

## S6. Text: sharing the KV cache across loops at inference

**Question:** at deployment every executed layer keeps its own KV cache (S3: 2.5× the dense cache for middle r=4,
4.5× for r=8). Ouro reuses the last loop's cache for all loops at decoding with almost no loss; Huang et al. (2026)
find that this works for models trained with truncated backprop or sampled depths and fails for fixed-depth training
with full backprop. Does it hold for our models?

### Design

- `looped/infer.py` gains a `shared` option: one cache slot per unique layer instead of per executed layer. Each
  loop reads the previous tokens' keys and values from the shared slot (written at their last loop) and the current
  token's from the current loop; the slot keeps the last loop's values ("terminal KV sharing"). The default is
  unchanged, and the full-cache path is tested to equal the training forward pass as before.
- Evaluation (`scripts/kv_share_eval.py`): token-by-token decoding over the first 262,144 tokens (256 windows of
  1024) of the main validation set, loss per token, full cache against shared cache, on the 40N branch-start
  checkpoints (the 32N trunk, seed 42, as in §5.7–5.9):
  20M middle r=8 full backprop; 20M middle r=8 truncated (k=4); 20M middle r=8 random loop count, full and truncated
  (`*_rs1to8_*`); 20M middle r=4 full backprop at the dense rate (1×) and at 0.5× (the S1 run, a more convergent
  iteration by Q5). The random-loop model also at 16 loops.
- Cost: minutes per model on one GPU.

### Quantities

Δ = loss(shared) − loss(full) on the same tokens, per model.

### Readings, fixed in advance

- Prediction: fixed r=8, full backprop: Δ > 0.3 (breaks); truncated k=4: Δ < 0.1; random loop count: Δ < 0.05;
  r=4 at 0.5×: smaller Δ than at 1×.
- Rule: Δ ≤ 0.02 (the seed-noise scale of the main set): "sharing is usable"; 0.02 < Δ ≤ 0.1: "degrades";
  Δ > 0.1: "breaks". The report's S3 section gains one paragraph and the limitation "not measured" is removed.

## S7. Text: seeds that change the data order

**Question:** the three seeds change the initialization only; every run reads the same data in the same order. The
seed intervals in the report (and the "two 160M seeds agree within 0.001" statement) may understate the variation a
different data order would produce.

### Design

- `looped/train.py` gains a `data_offset` field (tokens; default 0 = unchanged): the sampler starts that far into the
  stream. A test checks that offset 0 produces the previous batches exactly. The offset is part of the run record.
- Runs, seed 42, budgets 10/20/40 tokens per parameter, offsets 3B and 6B tokens (rounded down to whole batches; the
  main grid at 20M reads tokens 0–2.1B, so the three orders share no data): 20M dense and 20M middle r=4 full
  backprop, 4 runs, about 7 GPU-hours.

### Quantities

Per cell and budget, on all four validation sets: the losses under the three data orders (offset 0 = the main-grid
run) and the data-order spread (max − min); the init-seed spread at offset 0 (seeds 42–44); the looped − dense gap
under each data order (both models on the same order) and its spread.

### Readings, fixed in advance

- Prediction: on the main set the data-order spread of the loss is 1–3× the init-seed spread (0.005–0.02 nats at
  40N); the paired gap varies less than the losses; on the code set the data-order spread is ≥ 0.1.
- Rule: if the data-order spread exceeds the init-seed spread by more than 2× on the main set, every seed interval in
  the report gets the caveat "initialization only; a different data order adds about X nats", and the 160M statement
  the same. No verdict changes.

## S8. Synthetic task: stopping rules and voting over loop counts

**Question:** the state-change stop of §5.15 recovers about 80% of the gain of the best fixed loop count for the
fixed-loop models but runs 2–3× the needed loops and fails for the step-matched model. Can a label-free rule do
better?

### Design

`synthetic/halting.py` runs every looped formal checkpoint (5 seeds of r=2, r=4, r=8, random, step-matched) for 32
loops on fresh problems (2000 per step count, d = 1..24) and applies, per problem:

1. the state rule (relative change of the answer-position state below tol) with tol ∈ {0.05, 0.02, 0.01, 0.005};
2. the prediction rule: stop when the predicted answer has not changed for m consecutive loops, m ∈ {1, 2, 3};
3. the confidence rule: stop when the top softmax probability is ≥ τ, τ ∈ {0.9, 0.99};
4. voting: the majority over the predictions at loops 12, 16, 24 and 32, and over every loop from 8 to 32 (ties go
   to the later loop);
5. rule 2 with m = 2 and a minimum of the model's own loop count.

Two problem sets: selection (generator seed 777) and reporting (seed 888).

### Quantities

Per rule and condition: accuracy on d = 1–8 (mean), 12, 16 and 20; mean loops used on d ≤ 8 and at 12 and 16;
alongside the own loop count, the best fixed loop count (chosen with the answers; an upper bound) and the existing
state rule (tol 0.02).

### Readings, fixed in advance

- Selection: per condition, the rule with the highest mean accuracy over d = 9–16 on the selection set, among rules
  whose d ≤ 8 accuracy is within 0.01 of the own-loop-count accuracy. It is reported on the reporting set.
- Predictions: the prediction rule (m = 2) beats the state rule, recovering ≥ 85% of the best-fixed gain at 12 steps
  for r=4 and r=8, with ≤ 1.5× the own loop count on d ≤ 8; voting does not beat the best fixed loop count; no rule
  brings the step-matched model above 0.2 at 12 steps.
- Use: [results §5.15](results.md) and the report's §5 paragraph on stopping.

## S9. Synthetic task: a model trained with loops = steps plus random extra loops

**Question:** the step-matched model extrapolates furthest but cannot stop (§5.15): it was never run past step d in
training, so its state keeps changing. If it is trained to keep looping after the last step, does it keep its
extrapolation and gain the ability to stop?

### Design

- New condition `loop_step_extra`: the r=8 architecture with r = d + U{0, 1, 2, 3, 4} at every training step
  (`regime="step_extra"`); everything else as `loop_step`.
- Learning rate by the suite's protocol (seed 1, 16000 steps, 2.5e-4 / 5e-4 / 1e-3 with the edge rule, chosen by
  in-distribution accuracy at loops = d); formal runs seeds 2–6, 32000 steps; the standard probes (d = 8, 16 loops).
- About 9 GPU-hours.

### Quantities

As §5.15 for a looped condition: accuracy at loops = d, with more loops and with the adaptive stop (tol 0.02), the
gains over 8 loops at d = 9..24; from the probes, persistence within 16 loops and the diagonal share; the S8 rule
selected for it.

### Readings, fixed in advance

Predictions: (i) in-range accuracy at loops = d ≥ 0.99; (ii) persistence over 16 loops ≥ 0.8 (step-matched: 0.04);
(iii) adaptive stop at 12 steps ≥ 0.6 (step-matched: 0.04); (iv) largest solvable step count at loops = d ≥ 11
(step-matched: 12.8, i.e. a loss of at most about 2 steps).
Reading: (ii)–(iv) hold: "a model can both extrapolate and stop"; (ii) and (iii) hold but not (iv): "a trade-off:
holding the answer costs extrapolation"; (ii) fails: "random extra loops do not teach persistence".

## Execution

- **Order.** Batch 1, started first: S4 (5 training runs, no code change) and S8 (one script, minutes). While S4
  runs: the code for S9 and S5, then batch 2: S9 (sweep → choice → formal → probes) and S5 (sweep → choice →
  formal). S6 and S7 fill free GPUs. Every run goes through `scripts/run_queue.py` or `synthetic/suite.py`; each
  job's output is in `runs/<job>/stdout.log` or `logs/synth/`.
- **Time.** About 60–70 GPU-hours (S4 30, S5 20–25, S6 1, S7 7, S8 0.2, S9 9), 1–1.5 days on four GPUs; the
  optional S5 row adds about 12.
- **Code changes**, each committed before the runs that need it, each with tests:
  - config generation for S4 (`scripts/make_configs.py`, `scripts/make_supp.py`; no training code);
  - `synthetic/train_synth.py`: the `--format cot|pause` option (S5) and the `step_extra` regime (S9), the
    existing conditions unchanged; `synthetic/suite.py`: stages for S5 and S9; `synthetic/halting.py` (S8);
  - `looped/infer.py`: the `shared` option (S6) and `scripts/kv_share_eval.py`;
  - `looped/train.py` and `looped/data.py`: the `data_offset` field (S7), default 0, tested to leave the previous
    behaviour unchanged. S4 is launched with the training code as it is before this change.
  - Analysis: the readings above are computed by `fit/supp2_analysis.py`, written before any result exists.
- **Results** are exported, committed and pushed when the queue has finished; the analysis is run on the Mac.
