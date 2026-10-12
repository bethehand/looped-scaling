# Supplementary experiments S1–S3

Status: **frozen** on 2026-10-13 with the author's confirmation. This file and its Chinese original (kept on the
author's machine) were committed before any supplementary run started; their SHA-256 hashes are in the
[deviation log](deviation-log.md). Any later change is logged there under the usual rules.

Date: 2026-10-12

## 0. Why, and how this relates to the pre-registration

All three experiments are outside the pre-registration. They are exploratory supplements and **do not change the
H1–H4 verdicts**. Their purpose is to test whether the main findings hold up in two respects: the learning rate, and
the cost at deployment.

**Relation to the earlier learning-rate checks.** The looped text models used the dense model's learning rate, but
this was a tested decision. The pre-registration phase ran two checks:

- On 2026-09-25, all 10 looped cells at 20M, each at 0.5, 0.71 and 1 times the dense rate, with a 10N budget. By the
  rule written in advance, no adjustment was needed.
- A cross-check of middle-block r=4 at 80M, which passed by the criterion written in advance.

Four things were not covered by those checks:

1. **They used the 10N budget only.** The main comparisons are at 40N, and the data extension goes to 160N. Longer
   training usually favors a lower learning rate. The small 10M half-learning-rate experiment was 0.064 better at 40N
   (results 5.4).
2. **The dense model's learning rate was also tuned at 10N only.** So with long training neither side may be at its
   optimum, and the direction of the bias is unknown.
3. **They did not test any point above 1×.** For 9 of the 10 cells the lowest point was at 1×, so the optimum may be
   higher.
4. **They used 20M only.** The trend "a loop is worth more at larger scale" starts at 10M, and 10M has the highest
   learning rate of all rungs.

**One claim about the synthetic task needs testing.** Item 4 of the README recipe says that "at the dense-optimal rate
the same architecture learned a fixed-length program". The sweep table (seed 1, 16000 steps) shows that at 1e-3 the
looped models reached only 0.30–0.60 accuracy within the training range and solved at most 3 steps, i.e. they did not
learn the task. So "learned a fixed program" is not a clean description: the models may simply have learned more slowly.
S2 is designed to separate these two explanations.

## S1. Text: re-tuning the learning rate at the budgets used for comparison

**Question:** do "a loop is worth more at larger scale", "a loop is worth more with more data" and "to save
compute, don't loop" still hold when each side uses its own best learning rate?

### Design

| Item | Setting |
|---|---|
| Sizes | 10M, 20M, 40M |
| Cells | dense; middle block r=4 with full backprop (the cell of Figure 12, and the recommended recipe) |
| Learning rates | 0.5, 0.71 and 1.41 times the frozen η₀; the 1× point is the existing main-grid run |
| Budgets | 10M, both cells: 10/20/40/80/160N. 20M and 40M: dense 10/20/40/80N (the equal-compute comparison needs 80N), looped 10/20/40N (as in the main grid) |
| Seeds | 42, as in all earlier learning-rate checks. At 20M, also seed 43: the seed-to-seed spread at 20M is about 0.01, 3–5 times that at 10M and 40M (computed from the main grid before any S1 result existed) |
| Everything else | identical to the main grid: batch, warm-up, cooldown fraction, evaluation, compilation |
| Edge rule | If the lowest point of a (size, cell) at 40N (at 10M, also at 160N) is at 0.5×, add 0.35×; if it is at 1.41×, add 2×. Applied once. At 20M the decision uses the mean of the two seeds, and an added point is run with both seeds |
| Runs | 24 (6 each at 10M and 40M; 12 at 20M with two seeds), plus edge points |
| GPU-hours | about 82, plus at most about 28 for edge points |

### Quantities

For each (size, budget):

- Gap at the shared learning rate: Δ₁ = looped(1×) − dense(1×). Negative means the looped model is better.
- Gap at each side's own best: Δ* = lowest looped loss over learning rates − lowest dense loss over learning rates.
  "Lowest" is the lowest measured point (primary); the vertex given by the pre-registered parabola rule is secondary.
- Bias from the shared learning rate: h = Δ₁ − Δ*. Positive means the shared rate made the looped model look worse
  than it is.

At 20M every point, including 1×, is the mean of seeds 42 and 43.

### Readings, fixed in advance

**Q1. Trend with scale (primary budget 40N).** Let I₁ = Δ₁(10M) − Δ₁(40M), which is 0.065 for seed 42 (at 20M,
the two-seed mean), and I* = Δ*(10M) − Δ*(40M).

- Holds: I* ≥ 0.5 × I₁ and Δ*(40M) < 0.
- Weakened: 0.01 ≤ I* < 0.5 × I₁, or Δ*(40M) ≥ 0.
- Does not hold: I* < 0.01.

10N and 20N are computed the same way and reported as secondary results, together with whether the three points
10M, 20M and 40M are monotone.

**Q2. Trend with data (10M, from 10N to 160N).** Let I₁ᴰ = Δ₁(10N) − Δ₁(160N), which is 0.119 for seed 42, and
I*ᴰ likewise. The three readings are as in Q1, with "Δ*(40M) < 0" replaced by "Δ*(160N) < 0". 20M and 40M, from
10N to 40N, are secondary results.

**Q3. Equal compute.** Redo the equal-compute comparison at 10M–40M with each side's best losses, by the method of
results 5.11 (interpolate along the dense curve in log D; beyond its range, extrapolate with the last segment and
mark it). If any comparison turns in favor of the looped model, it is reported as is, and the README row "Save
training compute: don't loop" is qualified.

**Q4. Size of the bias.** Report every h. If all |h| ≤ 0.01, the report says "the comparisons at the shared learning
rate are accurate to within 0.01 nats".

**Q5 (exploratory). Does the learning rate decide the kind of iteration on text?** On the 32N trunk checkpoints
(the branch point of the 40N budget, the same kind of checkpoint as in results 5.8) of every S1 middle-block r=4 run,
measure the loss as a function of the inference loop count (1, 2, 3, 4, 5, 6, 8, 12, 16 loops; 1M tokens of the main
validation set).

- "Results 5.14 replicates" if, at all three sizes, the range of the loss over 4–16 loops is smaller at 0.5× than at
  1.41×.
- Also reported: whether, at any learning rate, more loops (8 or 16) beat the 4 training loops by more than 0.005.

### How the results are used (fixed in advance)

- Q1 holds: keep "a loop is worth more at larger scale" and add "this still holds at 10M–40M when both sides are
  re-tuned". Weakened or does not hold: rewrite the README and the report, and add the re-tuned points to Figure 12.
- Q2 likewise, for "a loop is worth more with more data".
- The pre-registered H1–H4 verdicts are not changed by S1. S1 is reported in the report and the paper as a robustness
  analysis.

## S2. Synthetic task: does the learning rate change how fast the model learns, or what it learns?

**Question:** at high learning rates the looped models do not extrapolate. Is that because they learn a different
solution, or because they have not finished learning?

### Design

| Item | Setting |
|---|---|
| Conditions | looped r=4, looped r=8, random loop count, loops = steps (the four conditions that learned to extrapolate in the formal runs) |
| Learning rates | 5e-4 and 1e-3. The formal runs used the swept choices of 1.25e-4 to 2.5e-4; 5e-4 is the best rate of the 36-layer dense model, 1e-3 that of the 8- and 20-layer dense models |
| Seeds | 2, 3, 4, paired with the formal runs |
| Steps and everything else | 32000 steps; all other settings identical to the formal runs (lookup-table chain task, curriculum 0.5, batch 512) |
| Probes | the intermediate-value probe on every checkpoint (d=8, up to 16 loops), as in the formal runs |
| Runs | 24 training runs, 24 probes |
| GPU-hours | about 40 |

### Quantities

For each run:

- Accuracy within the training range: mean accuracy over 1–8 steps at the run's own loop count. At least 0.99
  counts as "learned the training range". All 20 formal runs of these four conditions are at least 0.996.
- Extrapolation E: the longest solvable chain minus 8, with the longest solvable chain as defined in the formal
  analysis (results 5.15, `synthetic/analyze.py`): the largest d such that every length from 1 to d reaches 90%
  accuracy under the best of the run's own loop count, 12/16/24/32 loops, and loops = steps.

The reference is the mean E of the formal runs (chosen learning rate, seeds 2–6): r=4 3.8, r=8 3.2, random loop
count 4.2, loops = steps 4.8.

### Readings, fixed in advance

**R1. Is the task learned?** For each (condition, learning rate), the number of the 3 seeds that learned the
training range.

**R2. What is learned?** Using only the runs that learned the training range:

- Supports "the learning rate changes the solution": at that learning rate at least 2 runs learned the range, and
  their mean E ≤ the formal mean E − 2.
- Does not support it: at least 2 runs learned the range, and their mean E ≥ the formal mean E − 1.
- Otherwise inconclusive, including when fewer than 2 runs learned the range.

The probe slope and persistence are reported as well, without a reading.

### How the results are used (fixed in advance)

- R2 supports: item 4 of the README recipe keeps "learned a fixed program instead of an algorithm that extends", with
  the number of seeds.
- R2 does not support: drop the "fixed program" wording and say that the learning rate affects how fast the task is
  learned, not what is learned.
- Most high-learning-rate runs do not learn the range: say that "at the dense-optimal learning rate the looped models
  did not learn the task in the same number of steps".

## S3. Cost at deployment: memory and generation latency

**Question:** what do the parameters saved by looping cost in latency and memory at deployment? This is a
measurement only, with no reading.

- **Implementation.** KV-cached prefill and token-by-token decoding for all three placements, in new files; the model
  code used for training is not changed. A test requires the cached decoding to reproduce the full forward pass.
- **Conditions.** One idle RTX 4090 on GS01, bf16, the decode step in a CUDA graph, with the eager numbers reported as
  well. Random initialization, since latency and memory do not depend on the weight values. Widths 320–1280 ×
  {dense, middle block r=2/4/8, whole stack r=4/8}.
- **Metrics:**
  - parameters and weight memory;
  - KV cache per token;
  - prefill latency for 512 tokens (batch 1);
  - decode latency per token (batch 1 and 32, context from 512 to 1024);
  - throughput at batch 32;
  - peak memory.
- **Analysis:**
  - At equal parameters: looped-to-dense ratios.
  - At equal loss: for each looped cell (40N), interpolate along the dense models at 40N (10M–160M) in log N to find
    the dense size with the same loss, then interpolate its latency and memory. Report how much weight memory the
    looped model saves and how much extra latency and KV cache it costs. Losses are the median over seeds, because
    one seed of 40M whole-stack r=8 never recovered from a loss spike; 160M has middle-block r=4 only.
- **Where the results go.** The README row "Save parameters or memory" and recipe item 6.
- About 1 GPU-hour.

## Execution

- **Queue.** One queue for all four GPUs (`code/configs/queue_supp.csv`, generated by `code/scripts/make_supp.py`),
  one job per GPU at a time:
  1. S1 round 1, longest first; then the S2 training runs, longest first;
  2. each S2 probe as soon as its training run has finished;
  3. once all of S1 round 1 has finished, the queue runs the edge rule (`scripts/make_supp.py --edges`), which appends
     any added points and, after them, the Q5 loop-count curves;
  4. S3 after all round-1 jobs have finished (no other job on the same GPU; edge points may still run on other GPUs).
- **Time.** About 125–150 GPU-hours in total, about 1.5–2 days on four GPUs.
- **Code changes (completed and committed before the runs start):**
  - new script `scripts/make_supp.py`: the S1 configs and manifest (`configs/manifest_supp_lr.csv`), the whole queue,
    and the edge rule;
  - `scripts/run_queue.py` can run arbitrary commands, wait for prerequisite jobs, and pick up jobs appended while it
    runs;
  - new files `looped/infer.py` (KV-cached generation) and `scripts/bench_inference.py` (S3 measurement);
  - the analysis script `fit/supp_analysis.py`, written before any result exists;
  - new tests.
  - The training code (`looped/model.py`, `looped/train.py`, `synthetic/train_synth.py`, `synthetic/probe_synth.py`)
    is not changed.
- **Pushing the results.** When the queue has finished, the results are exported, committed and pushed to GitHub. The
  analysis and the readings are done on the Mac with `fit/supp_analysis.py`.
