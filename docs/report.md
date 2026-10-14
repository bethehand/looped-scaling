# How much is a loop worth? Technical report

**A pre-registered scaling study of looped transformers from 10M to 160M parameters, and a synthetic test of
step-by-step reasoning inside the loop**

Qirun Li · October 2026 · draft, results as of 2026-10-14 · code, configs, logs and every number:
[github.com/bethehand/looped-scaling](https://github.com/bethehand/looped-scaling) · [中文版](report.zh.md)

---

## Summary

A looped (recurrent-depth) transformer runs one block of layers *r* times. It costs *r* times the compute of that
block and adds no parameters. We measured what that buys, with a pre-registered grid of 111 training runs (10M–80M
looped models, a dense reference up to 160M, 3 seeds at the small sizes), three extension studies, and three
supplementary experiments whose plan was frozen before they ran.

1. **On web text, loops buy parameter and data efficiency, not compute efficiency.** At equal parameters and tokens, a
   middle block looped 4 times goes from 0.10 nats *worse* than the dense model at 10M to 0.03–0.06 nats *better* at
   160M, and the advantage keeps growing with the number of training tokens per parameter: at 160 tokens per
   parameter a 10M looped model matches a dense model trained on 1.7× as many tokens. At equal training compute the
   dense model wins every one of 320 comparisons. Re-tuning the learning rates of both models at 10M–40M
   (supplementary experiment S1) leaves all three results intact; the shared rate had, if anything, understated the
   looped model, by about 0.02 nats at 10M.
2. **The pre-registered number, φ, is not identified in absolute terms**, because the dense ruler that converts loss
   into "effective parameters" is weakly identified: the primary fit gives φ = 0–0.17 per cell, a joint fit 0.29–0.41.
   Its trend is robust: per-size fits rise from below zero at 10M to about 0.65 at 160M.
3. **None of the four pre-registered hypotheses held as stated.** Truncated backpropagation did not lower φ (it helped
   whole-stack loops), a form in which φ saturates in *r* did not extrapolate better than r^φ, and the dense model
   was compute-optimal.
4. **On text, the loop does not iterate towards an answer.** Trained with a fixed loop count it runs a fixed-length
   program whose last loop is a large "write-out" step; with truncated backprop it learns a periodic cycle; with a
   random loop count (as in Huginn) it contracts to a fixed point. None gains anything from extra loops at inference.
5. **On a task that needs serial steps, loops do learn step-by-step reasoning.** On chains of table lookups, looped
   models learn to compute one step per loop and solve longer chains than they were trained on when given more loops.
   Dense models of the same depth, with up to 4.4× the parameters, fail right after the training range. The learning
   rate matters: at the rate that is optimal for a dense model, 10 of 12 looped runs did not learn the task in the
   same number of steps; runs that learned it at a rate above the swept one mostly learned the same extendable algorithm
   (supplementary experiment S2).
6. **At deployment, loops cost latency and memory.** With one KV cache per executed layer, decoding time and cache
   grow with the executed layers. At equal loss, a middle block looped 4 times needs 14–19% less weight memory at
   40M–160M, but 2.2–2.3× the KV cache and 2.4–2.5× the decoding time per token, and its peak memory at batch 32 is
   about 1.6× that of the dense model (supplementary experiment S3).

The short version: putting reasoning into the architecture works when the task needs it and the loop is trained to
iterate; on ordinary text the task rarely needs it, and what remains is a parameter-efficiency gain that grows with
scale and data, paid for in decoding time.

---

## 1. Why this study

Looped transformers re-use a block of layers several times: Universal Transformers, and recently Huginn, Ouro and
others. Two promises are made for them. First, **parameter efficiency**: a looped model should behave like a larger
dense model, which matters when memory or bandwidth, not arithmetic, limits deployment. Second, **latent reasoning**:
if the loop iterates on a problem, running more loops at test time should solve harder problems without writing out
a chain of thought.

Both claims are hard to check from the literature. Studies differ in data, tokenizers, training recipes and model
families; seeds are rarely reported; and the quantity of interest, how much one more loop is worth, is usually read
off a single fit. We therefore ran a controlled ladder of sizes with one recipe, wrote the hypotheses and the analysis
down before training ([pre-registration](preregistration.md), frozen 2026-09-23), and logged every later change
([deviation log](deviation-log.md)).

## 2. Setup

| | |
|---|---|
| Model sizes | looped models at 10M, 20M, 40M and 80M (widths 320 / 448 / 640 / 896, 8 unique layers); dense reference ("ruler") also at 160M (width 1280). N = non-embedding parameters + output head of the dense model: 15.4M / 26.8M / 50.2M / 92.7M / 179.6M |
| Looped cells | **middle block** (2 layers once, 4 layers looped, 2 layers once) or **whole stack** (all 8 looped); r ∈ {2, 4, 8}; full backprop, or truncated backprop through the last r/2 loops at r = 4, 8 |
| Data | FineWeb-Edu sample-10BT (10.4B tokens, one epoch, fixed order), 16k BPE. Validation: held-out FineWeb-Edu (main), SlimPajama, FineMath, code |
| Training | AdamW, WSD schedule with cooldown branches at 10, 20 and 40 tokens per parameter (dense: 5–80); learning rates from a 33-run sweep on the dense models; 3 seeds at 10M–40M, 1 at 80M |
| Measurement | N_eff = N_once + r^φ · N_rec, with a ruler L(N, D) = E + A·N^−α + B·D^−β fitted on the dense models; Huber loss in log space, 500 random starts, 1000 block-bootstrap resamples |
| Compute | 4× RTX 4090; the main grid took 11 days; the extensions about 5 more; the supplementary experiments S1–S3 ([plan](supplementary-experiments.md), frozen before they ran) 1.3 more |

The four pre-registered hypotheses: **H1** truncated backprop lowers φ (under both iso-token and iso-FLOP
accounting); **H2** a looped middle block has a higher φ than a looped whole stack; **H3** φ saturates in *r*; **H4**
the compute-optimal *r* is small under training compute and larger under deployment compute.

## 3. What a loop is worth on text

### 3.1 The pre-registered verdicts

| Hypothesis | Verdict |
|---|---|
| H1 truncation lowers φ | Not supported. Middle block 0.00 vs 0.13 (predicted direction, overlapping intervals); whole stack 0.17 vs 0.00 (opposite direction). Same under iso-FLOP accounting. |
| H2 middle block > whole stack | Depends on the view. φ is mixed; raw losses at matched executed depth favour the middle block in 24 of 24 comparisons. |
| H3 saturation in r | Not supported. In the pre-registered extrapolation test the saturating form predicts 80M from 10M–40M better than r^φ (0.0085 vs 0.0096) but 10M from 20M–80M slightly worse (0.0203 vs 0.0200); the data do not separate the two forms. |
| H4 compute-optimal r | The dense model (r = 1) is optimal under both training and deployment compute. |

A surprise along the way: **truncated backprop helped whole-stack loops**. At 20M, a whole stack looped 4 times
reached 3.645 with truncation and 3.789 with full backprop at 10 tokens per parameter, and at 10M the only looped
cells better than the dense model were whole-stack loops with r = 8. Truncation also brought loss spikes: 30 of 111 runs had at least one, almost all in
truncated or whole-stack cells, and one run collapsed. Halving the learning rate removed the spikes in a follow-up.

### 3.2 Why the number φ is not settled

φ needs a ruler: the dense-model fit that says how much loss a given number of parameters buys. Fitted on our five
dense sizes, the ruler puts its irreducible loss at the lower bound and its parameter exponent at α = 0.054, far
below the usual 0.3. Five sizes cannot pin down E, A and α separately. A joint fit over all runs gives α = 0.267 and
roughly doubles φ (0.29–0.41 instead of 0–0.17). The pre-registered φ values are reported as such
([results §3](results.md)), but the conclusions below rest on quantities that need no ruler.

### 3.3 The value grows with scale and with data

<p align="center"><img src="../code/results/figures/fig12_scale_middle_r4.png" width="60%"></p>

**Scale.** The same cell, a middle block looped 4 times, compared with the dense model of the same size and seed
(Figure 12). At 40 tokens per parameter the loss difference is +0.018, −0.025, −0.050, −0.055 and −0.063 nats from 10M
to 160M (negative = looped better), and the curve is monotone at every budget. The two 160M seeds agree within 0.001
nats. Expressed as φ with the ruler held fixed (exploratory), the same cell gives −0.32, 0.19, 0.39, 0.52, 0.65. The
increments shrink in loss terms (0.043, 0.025, 0.005, 0.008 per doubling), and φ cannot exceed 1, so the curve must
bend somewhere; we do not see where yet.

<p align="center"><img src="../code/results/figures/fig11_data_extension.png" width="90%"></p>

**Data.** Continuing finished 10M and 20M runs to 80 and 160 tokens per parameter (Figure 11), each doubling of the
data adds 0.02–0.03 nats of advantage for the middle-block cells, with no sign of saturation. Expressed as the amount
of data a dense model needs to match the looped one, a 10M middle block looped 8 times goes from 0.85× at 10 tokens per
parameter to 1.68× at 160 (two seeds); a 20M one reaches 1.65× at 80. Whole-stack loops show the same direction.

**Compute.** Converting each looped run's training compute into the number of tokens a dense model could have seen,
and reading the dense loss off the measured dense curve, the dense model is better in all 320 comparisons. The gap
shrinks with the budget (80M middle r = 4: +0.135 nats at 10 tokens per parameter, +0.034 at 40) but never closes.
Loops are a way to trade compute for parameters and data, not a way to save compute.

**Learning rate.** All looped language models used the dense models' learning rates, after a pre-registered check at
20M and 10 tokens per parameter found no better one. Supplementary experiment S1 re-tuned both the dense model and the
middle block looped 4 times at 10M, 20M and 40M (0.5–1.41× the dense rate, plus 0.35× where the pre-set edge rule
required it) at every budget used above. The dense model's best rate stayed at 1× (at 40M, 0.71× was better by 0.001 nats at 40 tokens per
parameter and by 0.004 at 80). For the looped model the shared rate cost at most 0.006 nats at 20M–40M, but at 10M
it preferred 0.5× and below from 20 tokens per parameter on, worth about 0.02 nats. With each side at its best rate,
the gap at 40 tokens per parameter is −0.005, −0.031 and −0.055 nats from 10M to 40M (shared rate: +0.015, −0.031,
−0.050): the scale trend is somewhat smaller but holds by the rule fixed in advance; the data trend at 10M is larger
(+0.082 at 10 tokens per parameter to −0.059 at 160); and the dense model still wins all 11 equal-compute comparisons
([results §5.16](results.md)).

### 3.4 Where the gain lands

Per token, the gain on web text is a small, broad shift: about half of the tokens improve, and the gain is not
concentrated on hard tokens beyond what two dense seeds differ by. On math and code the average gain is 3–7 times
larger, but the seed-to-seed noise on those sets is 10–100 times larger too.

## 4. What the looped block computes on text

We probed trained models loop by loop: reading the state after each loop with the model's own output layers (a "logit
lens"), with a trained linear map in front of them, and with a trained linear head.

<p align="center"><img src="../code/results/figures/fig8_loop_probes.png" width="95%"></p>

- **Fixed loop count (r = 8).** With a trained linear map, the state after loop 4 already reads out within 0.1–0.2
  nats of the final loss, and loops 5–7 add little. The last loop is a large non-linear move that rewrites the state
  into what the output layers expect; 37–44% of the tokens settle their prediction only there. Running fewer or more
  loops at inference breaks the model (Figure 6). This replicated across seeds at 20M and 40M.
- **Truncated backprop (whole stack, k = 4)** learns a period-4 cycle: the state returns to almost the same point every
  4 loops, so the model works at 4, 8, 12 or 16 loops and gains nothing from more.
- **Random loop count during training** (r drawn from 1–8 at every step, as in Huginn) learns a contraction to a fixed
  point: the loss at 1 to 32 inference loops is flat after about 4 loops. The model is robust to the loop count, but
  at its nominal 8 loops it is 0.05 nats worse than fixed-r training and ties the dense model.
- **Learning rate.** Lower learning rates give flatter curves over the inference loop count. This showed first in
  the 20M models of the pre-registered learning-rate check, and S1 confirmed it at 10M, 20M and 40M (middle block
  looped 4 times, checkpoints at 32 tokens per parameter): over 4–16 loops the loss varies by 0.36–0.77 nats at 0.5×
  the dense rate and by 3.4–4.6 nats at 1.41×. Lower rates make the iteration closer to convergent, but on text more
  loops than trained never helped, at any rate.

<p align="center"><img src="../code/results/figures/fig10_random_r_depth.png" width="55%"></p>

## 5. Can loops learn to reason step by step?

Text gives the loop little reason to iterate: predicting the next word rarely needs many serial steps. We therefore
built a task that does ([design](synthetic-reasoning-design.md), frozen 2026-10-09). A problem lists a start value
and a sequence of table lookups, for example `25 ; LUT 7 ; LUT 3 ; LUT 19 ; … =>`, where each LUT is one of 32 fixed
random functions on 0–96; the model must output the final value in a single forward pass. Two lookups compose into one
of 1024 arbitrary functions, so the only way to handle long chains is to compute step by step. Models are trained on
1–8 steps and tested up to 24. Problems are drawn at random during training, so short ones repeat (there are only
3,104 one-step problems), but chains longer than 8 steps never occur in training. We compare dense models of 8, 20 and 36 layers with an 8-layer looped model (2 + 4×r +
2 executed layers) trained with r = 2, 4 or 8 loops, with a random loop count, or with as many loops as the chain has
steps. Every condition uses the learning rate picked by a sweep on in-distribution accuracy only, and has 5 seeds.

| Model | Params | Largest solvable step count* | With more loops at inference | Accuracy at 12 steps** | at 16 steps** |
|---|---|---|---|---|---|
| Dense, 8 layers | 6.5M | 4.2 (4–5) | — | 0.06 | 0.05 |
| Dense, 20 layers | 16.1M | 6.4 (6–7) | — | 0.04 | 0.03 |
| Dense, 36 layers | 29.0M | 8.0 (8–8) | — | 0.02 | 0.02 |
| Looped, r = 2 | 6.6M | 6.4 (6–7) | 6.4 (6–7) | 0.57 | 0.23 |
| Looped, r = 4 | 6.6M | 8.0 (8–8) | 11.8 (10–15) | 0.82 | 0.28 |
| Looped, r = 8 | 6.6M | 9.0 (9–9) | 11.2 (11–12) | 0.86 | 0.37 |
| Looped, random loop count | 6.6M | 11.4 (10–13) | 12.2 (10–15) | 0.83 | 0.49 |
| Looped, loops = steps | 6.6M | 12.8 (11–16) | — | 0.96 | 0.37 |

\* mean (min–max) over 5 seeds of the longest chain length up to which every length is solved at ≥ 90%, at the
model's own loop count. \*\* best available loop count, mean over seeds.

<p align="center"><img src="../code/results/figures/fig14_synthetic_inference_loops.png" width="95%"></p>

**Loops buy steps, and here weight sharing beats extra depth.** At equal parameters, more loops solve longer chains
(4.2 → 6.4 → 8.0 → 9.0 steps). The dense models of equal executed depth, with 2.4–4.4× the parameters, are worse and
fail right after the training range; at 12 steps they are at chance level while looped models reach 0.57–0.96.

**More loops at inference solve longer chains** (Figure 14). A model trained with 8 loops solves 9 steps at 8 loops
and about 11 with 16 or 32. The gain is real but bounded: at 16 steps it is at 0.37. The model trained with a random
loop count gains as well: at 16 steps its accuracy rises from 0.08 with 8 loops to 0.49 with 16.

<p align="center"><img src="../code/results/figures/fig15_synthetic_probes.png" width="95%"></p>

**The probes show one step per loop** (Figure 15). We train a linear probe for each intermediate value x_j at its own
position in the input, after every loop. In the model trained with 8 loops, x_j appears at about loop j (x_8 at loop 8; 1.0–1.1
loops per step in all 5 seeds), and each value stays decodable afterwards (persistence 0.96). The model
trained with loops = steps also computes one step per loop, but overwrites each value in the next loop (persistence
0.04), so it only works with exactly d loops. Models trained with fewer loops pack more steps into each loop (about
1.6 per loop at r = 4, 2.4 at r = 2), just enough to cover the training range. The dense models compute late, from
about layer 21 of 36, and in several seeds the intermediate values never appear at the step positions at all.

**The learning rate decides whether the loop learns the task.** All looped models above were trained at
1.25e-4–2.5e-4, 2–8 times lower than the learning rates chosen for the dense models (5e-4–1e-3). In the learning-rate
sweep (seed 1) the same r = 8 architecture trained at 1e-3 worked only at exactly 8 loops and had no step-by-step
frontier, like the language models of section 4; but it had also learned the training range much worse (accuracy 0.54
against 0.999 at 2.5e-4, after the same 16000 steps). Probes on all sweep checkpoints show the same split: for looped
r = 8 the step-by-step frontier is present at learning rates up to 5e-4 (about one step per loop) and absent at 1e-3
and 2e-3; looped r = 4 and the random-loop-count model behave the same way. Supplementary experiment S2 then trained
the four extrapolating looped conditions at 5e-4 and 1e-3 for the full 32000 steps with three seeds. At 1e-3, 10 of
12 runs still did not learn the training range (accuracy 0.49–0.98). At 5e-4, 8 of 12 did, and those mostly extended
about as far as the formal runs (looped r = 8: 2.7 steps beyond the training range against 3.2; loops = steps: 4.0
against 4.8); the exception was the random loop count, whose two learned seeds extended 2 steps against 4.2. So a
high learning rate mainly keeps the looped model from learning the task in the given steps; when it does learn the
task, it mostly learns the same extendable algorithm ([results §5.16](results.md)).

**Against the pre-registered predictions** ([results §5.15](results.md)): P1 (more loops, more steps at equal
parameters) holds in direction, but r = 8 did not reach 16 steps; P2 (iso-depth dense models at least as good) is
reversed; P3 (inference-time scaling) fails in the direction that favours looping — fixed-r models were *not* hurt by
extra loops, and the random-loop-count model, predicted not to gain, gained on long chains; P4 (only the
step-matched model computes one step per loop) is half right — the fixed-r model does too, and persistence is the
real difference; P5 (much larger gaps than on text) holds.

## 6. Practical takeaways

- **Loop when parameters or data are the constraint, not compute or latency.** The advantage at equal parameters
  grows with model size and with tokens per parameter; at equal compute a dense model is better. A middle block looped
  r = 2, 4, 8 times costs 1.4–1.5×, 2.2–2.4×, 3.7–4.1× the training FLOPs per token of the same-size dense model, and
  1.5×, 2.4–2.6×, 4.5–4.6× its decoding time per token.
- **Loop a middle block, with full backprop.** Middle-block loops beat whole-stack loops at matched depth in every
  comparison. Truncated backprop accounted for 25 of the 30 runs with loss spikes (16 of them whole-stack loops);
  halving the learning rate removed the spikes in a 10M follow-up.
- **Start with r = 4 on text, and judge at your target scale.** At 40M–80M, r = 8 did about as well as r = 4 while
  costing about 1.7× more. A loop is worth more at larger sizes and with more tokens per parameter, so small-scale
  ablations undersell it.
- **Do not expect test-time scaling on text.** Fixed-loop training gives a model that only works at its training loop
  count; random-loop training makes it robust to the loop count but not better with more loops.
- **Tune the looped model's learning rate separately**, especially at small scale and for step-by-step tasks. On text
  the dense rate was close to the looped model's best at 20M–40M but 0.02 nats off at 10M; on the reasoning task, at
  the dense-optimal rate most looped runs did not learn the task in the same number of steps. Check which algorithm
  the model learned with an inference-loop sweep (or probes): a model that only works at its training loop count has
  learned a fixed program; one whose accuracy holds or improves with more loops is iterating.
- **Count the KV cache and the latency, not just the parameters.** With one cache per executed layer, a loop
  multiplies both. At equal loss the looped models saved at most 19% of the weight memory; at batch 1 their peak
  memory was about the same as the dense model's (at 80M–160M, 1–2% lower), and with more sequences it was higher.
  The saving in weights pays off only where weights dominate memory (one sequence at a time, short contexts, larger
  models). Schemes that share the cache across loops may change
  this; we did not test them.

## 7. What went wrong, and what we changed

Every item below is dated in the [deviation log](deviation-log.md).

- **Two power failures** of the training machine (2026-09-24 and 2026-10-07) cut CPU speed by about 30×; runs resumed
  from checkpoints and no result was affected.
- **Hyper-parameters are those of the dense models.** A pre-registered learning-rate check for looped cells (20M,
  0.5–1× the dense rate, 10N) found no better rate, but 9 of 10 cells were best at the upper edge of the tested range,
  and the check covered neither longer budgets nor other sizes. Supplementary experiment S1
  ([plan](supplementary-experiments.md), frozen before it ran) re-tuned both the dense and the looped model at
  10M–40M at every budget up to 40N (160N at 10M): the scale, data and compute results hold, and the shared rate had
  understated the looped model by up to 0.024 nats at 10M (at most 0.005 at 20M–40M). On the synthetic task the best
  looped learning rate was 2–8× lower than the dense one (S2).
- **The synthetic task changed twice before its formal runs.** Modular-arithmetic chains were not learnable from a
  flat mixture of lengths (a curriculum was added) and remained very slow to learn, so the operations were replaced
  by fixed lookup tables. Both changes were made before any comparison between conditions.
- **The learning-rate criterion saturated.** Several looped conditions scored about 0.999 in-distribution at two or
  three learning rates; ties went to the lowest rate by implementation, which the protocol had not specified, and
  extrapolation turned out to be sensitive to this choice.
- **A pointer-chasing control was dropped.** It was never piloted; under the chain-calibrated recipe no condition
  learned even a single hop.
- **The first probe read only one position** per step and missed intermediate values that one seed stores elsewhere;
  it now reads three positions and keeps the best.
- **An external review found five errors after these results were first published (corrected 2026-10-14).** (1) The
  verdict that the random-loop-count model does not gain from extra loops used the longest solvable chain instead of
  the accuracy difference the design specified; by the latter it gains (16 steps: 0.08 → 0.49). (2) The batch-1 peak
  memory in S3 counted two KV caches; it was re-measured. (3) The AIC used for H3 is a proxy from the Huber loss, not a
  likelihood-based AIC, and is no longer cited as evidence. (4) The extrapolation test of H3 kept the 160M rung in its
  training set, unlike the pre-registered split; redone, it no longer favours r^φ in both directions. (5) The design
  said synthetic problems never repeat; short ones do. A sixth finding, that relaunching a finished run could
  overwrite its configuration record, affected no published run (all 219 records match their results); the trainer
  now refuses such a relaunch.
- **One unexplained anomaly:** at 160M, one looped seed is 0.2–0.6 nats worse than the other on the
  out-of-distribution sets at 40 tokens per parameter, while the main-set losses are identical.

## 8. Limitations

- Scale: 10M–160M parameters and a 16k vocabulary, far below the 1B+ models where looped LMs are deployed. The
  scaling trends may bend; we have five sizes, one seed at 80M and one looped cell at 160M.
- One data source (FineWeb-Edu), one epoch; multi-epoch regimes, where loops may matter more, are not covered.
- Deployment cost was measured on one GPU type, for models of at most 160M parameters, with one KV cache per executed
  layer; larger models, where weights dominate memory, and schemes that share the cache across loops were not tested.
- The learning rate was re-tuned for one looped cell at 10M–40M (S1), with one seed (two at 20M); the other cells,
  80M and 160M use the dense rates.
- The synthetic evidence comes from one task family and 6.6M-parameter models; extrapolation length varies widely
  across seeds.

## 9. Reproducing

```
cd code
uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e .
python -m pytest tests -q
```

[code/README.md](../code/README.md) lists the full sequence: data preparation, learning-rate sweep, main grid, fits,
extensions and figures. Every run's results and training log are in `code/results/`, the fits in
`code/results/fit/`, the synthetic results in `code/results/synthetic/`, and all figures are produced by
`code/fit/make_figures.py` and `code/synthetic/analyze.py`.
