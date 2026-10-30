# What a loop buys: parameter efficiency on text, extendable depth on a serial task

**A pre-registered study of looped (recurrent-depth) transformers from 10M to 160M parameters, on web text and on a
synthetic task that needs serial steps.**

Qirun Li · October 2026 · technical report, results as of 2026-10-28 · code, configs, training logs and every number:
[github.com/bethehand/looped-scaling](https://github.com/bethehand/looped-scaling) · [中文版](report.zh.md)

---

## Summary

A looped (recurrent-depth) transformer runs one block of layers *r* times. This costs *r* times the compute of that
block but adds no parameters. Two benefits are expected in return: the looped model should behave like a larger one
(parameter efficiency), and running more loops at inference should solve harder problems without writing out a chain of
thought (latent reasoning). We tested both with one model family and one training recipe. A dense (not looped) model in
our grid runs eight layers once; the looped model we report most often runs its middle four layers four times, with two
layers before and two after, so it runs 20 layers in total. The pre-registered grid has 111 language models (looped
models of 10M–80M parameters, dense models up to 160M). Later runs added a 160M looped
model, more data, a synthetic task and supplementary experiments (S1–S10), each planned before it ran.

Losses below are in nats per token on held-out web text (FineWeb-Edu). Between initialization seeds, the loss of the
20M dense model varies by 0.014–0.024 nats. Pairing runs by seed does not remove this noise, so a difference of 0.01
nats in one seed is within it at 10M–20M and resolvable only at 40M (section 2). "Tokens per parameter" is training tokens divided by the parameter count N. Each result is labelled
pre-registered, pre-specified (fixed in a plan before its runs) or exploratory (chosen after seeing data); section 2
defines the remaining terms.

<p align="center"><img src="../code/results/figures/fig0_overview.png" width="100%"><br><em>Figure 1. (A) Looped minus dense loss at equal parameters and tokens; lower is better for the loop. The looped model is worse than the dense model at 10M, mixed at 20M, and better at every budget from 40M on. (B) The advantage grows with training tokens per parameter: the curves fall further below zero. (C) On chains of table lookups (trained on 1–8 steps), looped models run with more loops solve longer chains. No dense model gets past the training range, and neither does a written chain of thought, whose prompt grows with the chain.</em></p>

### On web text

1. **At equal parameters, looping helps from about 40M on, and more with more data.** At equal parameters and training
   tokens, a middle block looped 4 times is 0.02–0.10 nats *worse* than the dense model at 10M and 0.03–0.06 nats
   *better* at 160M (at 10–40 tokens per parameter). The advantage keeps growing with data: at 160 tokens per parameter,
   a 10M model looped 8 times matches a dense model trained on about 2.2× as many tokens.
   Put as a multiplier, each parameter of the 4-loop model is worth 0.6–1.0 dense parameters at 10–20M with 10–20 tokens
   per parameter, and 1.26–1.46 at 40M–160M with 20–40.

2. **At equal training compute, the dense model wins.** Given the same training FLOPs, and so more tokens, the dense model
   wins all 258 comparisons in which its loss was measured, when both models use the
   learning rate tuned for the dense model. The margin shrinks with data, to 0.01–0.05 nats at the largest looped
   budgets. With the looped model at the lowest learning rate tried, the two are level at 10M and 160 tokens per
   parameter, the one measured comparison the dense model does not win. Re-tuning the learning rates of both models at 10M–40M (S1) leaves the size trend,
   the data trend and the equal-compute result in place; the shared rate had, if anything, understated the looped model,
   by about 0.02 nats at 10M.

3. **At 40M, about half of the loop's advantage comes from depth** (S10, section 3.7). To separate depth from weight
   sharing, we trained dense models with 18 layers and about the looped model's number of parameters. The share of the
   loop's advantage that this deeper model recovers (R, the ratio of the two log-multipliers averaged over three
   budgets) is 0.50 (95% interval 0.41–0.62), in the band the plan had set for "partly depth, partly sharing", as it had
   predicted. The loop's lead over the deeper model grows with data. With each model at its best tested learning rate (one
   seed), depth accounts for all of the advantage at the shortest budget.

4. **None of the four pre-registered hypotheses held as stated, and the pre-registered law could not pin down the
   value of a loop.** The pre-registration measured that value by one exponent per loop type, φ in N_eff = N_once + r^φ · N_rec: with φ = 0 the looped parameters count once, with φ = 1 they count
   *r* times. With five model sizes, the dense "ruler" that turns loss into effective parameters is weakly identified,
   and φ came out at 0–0.17 in the primary fit and 0–0.41 in a joint fit. Truncated backpropagation did not lower φ. A looped middle block beat a
   looped whole stack on raw losses but not consistently on φ. A saturating form did not extrapolate better than r^φ,
   and the dense model was compute-optimal under both training and deployment compute (section 3.1). Fitted per size, φ rises from below zero at 10M to
   about 0.65 at 160M, the same trend the multiplier shows. Published equal-parameter results at
   about 130M–770M correspond to a similar multiplier of 1.2–1.6, and with N counted as Schwethelm et al. count it, our
   local exponent at 40M–160M (0.32–0.50) is close to their 0.46.

5. **On text, extra loops at inference never help.** A model trained with a fixed loop count runs a fixed-length
   program whose last loop is a large "write-out" step. The whole stack trained with truncated backpropagation, the only
   truncated model probed, learns a periodic cycle instead, and a model trained with a random loop count (as in Huginn,
   with a different distribution) contracts to a fixed point. None of them gains anything from extra loops at
   inference, at any learning rate.

### On a task that needs serial steps

6. **The loop computes the chain step by step, and more loops solve longer chains.** The task is a chain of table
   lookups. Probes show the looped models advancing the chain one step per loop when trained with 8 loops. Given more
   loops at inference, they solve chains longer than any seen in training: a model trained with 8 loops goes from 0.04
   to 0.83 accuracy on 12-step chains with 16 loops (0.86 with the best larger loop count, chosen with the answers). The
   gain fades with length, by about 16 steps. Dense models fail right after the training range, even at the same
   executed depth with up to 4.4× the parameters, and so do the fixed-loop models at their training loop count. At the
   learning rate that suits a dense model, 10 of 12 looped runs did not learn the training range within the same number
   of training steps (accuracy 0.49–0.98; S2).

7. **A written chain of thought is cheaper within the training range and fails beyond it** (S5). Dense models that write
   the chain out as tokens solve the training range perfectly, 1.000 in every seed and at every learning rate tried,
   with under a third of the looped model's inference compute at its training loop count. On every longer chain they
   are near chance, with 36 layers as well. On problems with more than 8 steps, the written chain breaks within its
   first three tokens in 97–98% of problems for the 8-layer model and in 68–86% for the 36-layer model. Since longer
   chains here also mean longer prompts, this looks like a failure to generalize to longer inputs, and the comparison
   cannot tell whether a written chain would extend on inputs of fixed length. Pause tokens add nothing.

8. **Most models can decide for themselves when to stop.** The model's own confidence is a stopping rule that needs no
   answer: at a threshold of 0.99 it recovers 97–98% of the extra-loop gain for models trained with 4 or 8 loops or a
   random loop count, and with 2 loops it does better than the best fixed loop count (S8; we picked this rule after
   comparing several). The model trained with as many loops as steps extends furthest (12.8 steps), but a rule that
   waits for its state to stop changing cannot stop it, because it was never trained past the last step. Trained to keep
   looping a few loops longer, it keeps its intermediate values and reaches 0.93 accuracy on 12-step problems when it
   stops by confidence, at the cost of about one step of extrapolation (S9).

### Cost at deployment

9. **At deployment, the loop costs decoding time and memory** (S3, S6). With one KV cache per executed layer, decoding
   time and cache grow with the layers run. At equal loss (both models trained on 40 tokens per parameter), a middle
   block looped 4 times saves at most 19% of the weight memory. But it needs 2.2–2.4× the KV cache, 2.4–2.5× the
   decoding time per token and about 1.6× the peak memory of the dense model at batch 32. Sharing the KV cache across
   loops would remove most of that cost, but in a test at 20M the fixed-loop models then lose 0.09–0.17 nats, more
   than their loops had gained over the dense model: 2.2 to 7.4 times the gain. These gains come from a single seed,
   and at 20M the looped-minus-dense gap varies across seeds by 0.032–0.045 nats, so the ratios are rough.

### Interpretation

In this study, running a block again worked as test-time compute only where the task needed serial steps and the loop
was trained to iterate. On web text at these sizes, what remains is a gain in parameter efficiency that grows with
scale and data and is paid for in decoding time. We read the difference as follows: extra loops help when
the model's errors come from too few serial steps rather than from missing knowledge and the loop can run longer
without destroying its state, and using them in deployment also needs a signal for when to stop. The chain task meets
these conditions once the model is trained for them, while web text at these sizes shows no sign of errors of the
first kind. Two settings are too few to test
this reading, and section 9 lists the follow-ups that could.

---

## 1. Motivation

Looped transformers re-use a block of layers several times: Universal Transformers [1], and recently Huginn [2], Ouro
[3] and others. They promise parameter efficiency: a looped model should behave like a larger dense model, which
matters when deployment is limited by memory or bandwidth more than by arithmetic. They also promise latent reasoning:
if the loop iterates on a problem, running more loops at test time should solve harder problems without writing out a
chain of thought.

Both claims are hard to check from the literature. Studies differ in data, tokenizers, training recipes and model
families, and seeds are rarely reported. The quantity of interest, how much one more loop is worth, is usually read off
a single fit. We ran a controlled ladder of sizes with one recipe instead. We wrote the hypotheses and the analysis
down before training ([pre-registration](preregistration.md), frozen 2026-09-23), and logged every later change
([deviation log](deviation-log.md)). When the text results left the second promise untested, we built a task that needs
serial steps, froze its design and predictions, and ran it the same way.

### Related work

On how much a loop is worth, Kaplan et al. [4, App. D.2, Fig. 17] found that recurrent transformers, which re-use
parameters, "perform slightly better when comparing models with equal parameter count, but slightly worse when
accounting for reuse and comparing per FLOP". Our results agree in both directions.

The effective-parameter form we test, N_eff = N_once + r^φ·N_rec, comes from Schwethelm et al. [5], and so does the
choice of two unshared layers before and after the loop. They estimate φ = 0.46 (95% interval 0.41–0.53) on
FineWeb-Edu, with up to 8 loops, no repeated seeds reported and non-embedding parameters only. They also find that
backpropagating through only the last half of the loops lowers φ from 0.45 (their baseline refitted on the same
budgets) to 0.38. That comparison spends the saved FLOPs on more tokens, so it mixes the equal-token and equal-FLOP
accountings. Our pre-registration was built to be comparable with theirs (our N also counts the output head). Its H1
tests their finding with the two accountings separated; here it was not supported. The prelude–core–coda layout itself,
with its (2, 4, 2) split and an input injection that maps the concatenated state and prelude output, is that of Huginn
[2].

Other studies measure the value of a loop in other settings. Parcae [6] finds that looping lowers perplexity at matched
parameters and tokens (not matched compute). Lee et al. [7] find that looped models without sparse layers scale worse
than non-looped ones at equal FLOPs. Chen et al. [8] find that tying the weights of a deeper model costs a roughly
constant factor of compute in single-epoch training, while with repeated data the best loop count grows with compute.
In a controlled appendix of Ouro [3], a dense model of the same depth beats the looped one in every comparison, and in
[11] plain recursion is worse than a vanilla model in validation loss at matched compute.

For looped models without sparse layers and looped mixtures of experts alike, a bounded form extrapolates better than
r^φ to 16 loops [9]; with up to 8 loops, our data could not separate a saturating form from r^φ (H3). A 1.4B looped
model trained on 310B tokens (210B of pretraining, about 13 passes over a 16B-token corpus, then 100B of
mid-training) beats a dense model with the same parameters and data on all 12 of its benchmarks, at about 4× the
training compute per token, while a 3.9B dense model that matches its inference compute at three loops is ahead or
level on every benchmark [10]. None of these studies varies model size and tokens per parameter separately with
several seeds, which is the main thing this study adds (section 3.3).

Huginn [2], trained with a random loop count, has latent trajectories that, in qualitative
examples, settle into fixed points or orbits or drift steadily in one direction. Logan [12] reports per-token
convergence to a fixed point in a small depth-recurrent model, as we find for random loop counts (section 4). Huang et
al. [13] find that fixed-depth training with full backprop prevents sharing the KV cache across loops; in our 20M
models, sharing costs every model that gains from its loops 0.09–0.17 nats, with truncated backprop as well (S6).
Saunshi et al. [14] find that the gains of looped models concentrate on reasoning tasks, which may relate to our larger
gains on math and code.

Recurrent networks solve harder problems than they were trained on by running more iterations,
and they can "overthink" [15, 16]. Looped transformers with an input-dependent number of loops generalise to longer
inputs [17]. Kohli et al. [18] show that recurrent-depth transformers extrapolate to deeper chains of composed facts
with more loops at test time: trained on up to 12 hops, they reach 14–19. Vanilla transformers of equal parameters or
equal FLOPs do not. Zhang et al. [19] find that the positions a loop resolves per iteration scale with the training
length divided by the training loop count, within a task-dependent range, and that extra loops at test time rescue the
later positions at a fixed input length. Our synthetic results agree with both. Beyond them, we probe every
intermediate value, measure how the learning rate affects whether the task is learned within the training budget,
compare answer-free stopping rules on the same checkpoints, and compare the loop on the same problems with a chain of
thought [27] and pause tokens [28], counting the inference compute of each in layer passes.

At inference, a looped model's KV cache and decoding time grow with the number of loops unless the cache is shared. Two
loops take about twice the KV memory and 1.96× the latency in [20]. Plain looped decoding steps take 2.3×, 3.4× and
5.7× as long at 2, 3 and 5 loops in [21]. Schemes that share the cache across loops exist [3, 13, 21]: Ouro [3] shares
it during decoding only, and [21] uses a cache written by the first loop. Here they are tested only as a post-hoc change
to models trained without sharing (S6).

## 2. Setup

### Terms and labels used in this report

Every result carries one of three labels. *Pre-registered* means that the hypothesis and its decision rule were frozen
on 2026-09-23, before any training run ([pre-registration](preregistration.md)). *Pre-specified* means written in a
plan that was frozen before the runs it describes: the [synthetic design](synthetic-reasoning-design.md) or one of the
three supplementary plans, [S1–S3](supplementary-experiments.md), [S4–S9](supplementary-experiments-2.md) and
[S10](supplementary-experiments-3.md). *Exploratory* means chosen after seeing data. Every departure from a frozen
document is dated in the [deviation log](deviation-log.md). The full numbers are in [results](results.md), cited below
as "results §x".

Losses are in nats per token on held-out FineWeb-Edu, unless another validation set is named. Between three
initialization seeds, the loss of the 20M dense model varies by 0.014–0.024 nats at 10–40 tokens per parameter
(0.004–0.008 at 10M and 40M), and by 0.003–0.07 across the looped cells at 20M. Pairing runs by seed does not remove
this noise. Runs with the same seed and width start from the same weights in the layers they share and read the same
data, yet the gap between the 4-loop model and the dense model of the same seed varies across seeds by 0.005–0.020
nats at 10M, 0.032–0.045 at 20M and 0.003–0.005 at 40M (`fit/paired_noise.py`). A difference of 0.01 nats in one seed
is within this noise at 10M–20M and resolvable at 40M; the grid's results at 10M–40M are means over three seeds.
"Tokens per parameter" is training tokens divided by N, and "40N" means 40 tokens per parameter. N counts the
non-embedding parameters plus the output head.

- A *cell* is one looped configuration: which layers loop, how many times (*r*), and whether backpropagation runs
  through all loops or is truncated.
- A *rung* is one model size.
- The *middle block* loops the middle 4 of 8 layers and keeps the first two and last two unshared; the *whole stack*
  loops all 8.
- *Dense* means not looped.
- *Equal parameters* compares a looped model with the dense model of its own size trained on the same tokens.
  *Equal compute* compares it with a dense model trained with the same training FLOPs, that is, on more tokens.

### The design

| | |
|---|---|
| Model sizes | looped models at 10M, 20M, 40M and 80M (widths 320 / 448 / 640 / 896, 8 unique layers); dense reference ("ruler") also at 160M (width 1280). N = non-embedding parameters + output head of the dense model [25]: 15.4M / 26.8M / 50.1M / 92.7M / 179.6M |
| Looped cells | middle block (2 layers once, 4 layers looped, 2 layers once) or whole stack (all 8 looped); r ∈ {2, 4, 8}; full backprop, or truncated backprop through the last r/2 loops at r = 4, 8 |
| Data | FineWeb-Edu [22] sample-10BT: every run reads a prefix of one fixed-order stream, files 000–012 (10.4B tokens with our tokenizer; the longest run uses 9.9B), so no document is seen twice; 16k BPE. Validation: held-out FineWeb-Edu (main, from file 013), SlimPajama, FineMath, code |
| Training | AdamW, WSD schedule with cooldown branches [26] at 10, 20 and 40 tokens per parameter (dense: 5–80, at 160M 5–40; extended to 160 at 10M and, in S4, to 640 at 10M and 320 at 20M); learning rates from a 33-run sweep (30 dense runs, 3 looped); 3 seeds at 10M–40M, 1 at 80M, 2 at 160M |
| Measurement | N_eff = N_once + r^φ · N_rec [5], with a ruler L(N, D) = E + A·N^−α + B·D^−β fitted on the dense models [23, 24]; Huber loss in log space, 500 random starts, 1000 block-bootstrap resamples. Ruler-free: loss gaps at equal parameters and tokens, dense sizes of equal loss by interpolation, equal-compute comparisons on the raw data |
| Synthetic task | chains of 1–8 table lookups in training (32 fixed random tables on 0–96), tested up to 24 steps; dense models of 8, 20 and 36 layers and 8-layer looped models (6.5–6.6M parameters; 16.1M and 29.0M for 20 and 36 layers), 5 seeds per condition, learning rates chosen by a sweep on in-range accuracy only; [design](synthetic-reasoning-design.md) frozen 2026-10-09 |
| Compute | 4× RTX 4090; the main grid took 11 days; the extensions about 5 more; the supplementary experiments S1–S3 1.3 more, S4–S9 1.5 more and S10 0.3 more |

The pre-registration states four hypotheses:

- H1: truncated backprop lowers φ, under both iso-token and iso-FLOP accounting.
- H2: a looped middle block has a higher φ than a looped whole stack.
- H3: φ saturates in *r*.
- H4: the compute-optimal *r* is small under training compute and larger under deployment compute.

## 3. What a loop is worth on text

### 3.1 The pre-registered verdicts

| Hypothesis | Verdict |
|---|---|
| H1 truncation lowers φ | Not supported. Middle block 0.00 vs 0.13 (predicted direction, overlapping intervals); whole stack 0.17 vs 0.00 (opposite direction). Same under iso-FLOP accounting. |
| H2 middle block > whole stack | Depends on the measure. φ is mixed; raw losses at similar executed depth (36 against 32 and 20 against 16 layers) favour the middle block in 24 of 24 comparisons. |
| H3 saturation in r | Not supported. In the pre-registered extrapolation test the saturating form predicts 80M from 10M–40M better than r^φ (0.0085 vs 0.0096) but 10M from 20M–80M slightly worse (0.0203 vs 0.0200); the data do not separate the two forms. |
| H4 compute-optimal r | The dense model (r = 1) is optimal under both training and deployment compute. |

Along the way, truncated backprop turned out to help whole-stack loops with 4 loops, at every size. At 20M and 10 tokens
per parameter, a whole stack looped 4 times reached 3.646 with truncation and 3.783 with full backprop (three-seed
means); with 8 loops, truncation was slightly worse at 10M–20M. Separately, at 10M the only looped cells better than the
dense model on average over the three budgets were whole-stack loops with r = 8.

Truncation and whole-stack loops also brought loss spikes. Of the 111 runs, 30 had at least one: 25 with truncated
backprop, and the other 5 in whole stacks looped 8 times with full backprop. One run collapsed. In a follow-up, halving
the learning rate removed the spikes.

### 3.2 The exponent φ, and a multiplier read off the dense runs

φ needs a ruler: the dense-model fit that says how much loss a given number of parameters buys. Fitted on our five dense
sizes, the ruler puts its irreducible loss at the lower bound and its parameter exponent at α = 0.054, far below the
usual 0.3. Five sizes cannot pin down E, A and α separately. A joint fit over all runs gives α = 0.267 and larger φ
(0–0.41 instead of 0–0.17). The pre-registered φ values are reported as such ([results §3](results.md)), but the
conclusions below rest on quantities that need no ruler. For comparison, Schwethelm et al. [5] report φ = 0.46 with
their own ruler, and with rulers this weakly identified, absolute values of φ are hard to compare across studies.
Figure 2 shows φ fitted at each size separately, with the ruler held fixed (exploratory).

<p align="center"><img src="../code/results/figures/fig2_phi_by_rung.png" width="70%"><br><em>Figure 2. φ fitted separately at each size for the four loop types, with the pre-registered ruler held fixed and negative values allowed (exploratory); dotted lines mark the pre-registered value of each loop type. φ rises with size for three of the loop types and falls for the truncated whole stack, so a single value per loop type averages over different regimes.</em></p>

The dense runs themselves can also serve as the ruler, without fitting a law to them. For each looped model, we
interpolate between the measured dense models trained on the same tokens to find the dense size N* with the same loss
([results §5.17](results.md); exploratory). The multiplier N*/N then says what each parameter of the looped model
(output head included) is worth in dense parameters.

For the middle block looped 4 times, the multiplier rises from 0.6–1.0 at 10–20M with 10–20 tokens per parameter to
1.15–1.5 at 40M–160M. It grows by ×1.21 per doubling of size and by ×1.14 per doubling of tokens per parameter (×1.16
with 8 loops); these values use the dense models that supplementary experiment S4 trained longer, to
640 and 320 tokens per parameter at 10M and 20M (section 3.4; [results §5.18](results.md)). On a plane fitted to log M against log N and log(tokens per parameter), the multiplier reaches 1, the point where a looped model is as good as a dense model of its own size, at about N = 35M
with 20 tokens per parameter (between the 20M and 40M models). With 80 tokens per parameter it reaches 1 at about
N = 12M, which is below even the smallest model: the model called 10M has N = 15.4M. The measured points agree with the
plane at 80 tokens per parameter (1.04 for the 10M model, 1.05 on the plane), but at 20 they reach 1 earlier: the 20M
model (N = 26.8M) is already at 0.99, where the plane gives 0.93.

At 40M–160M with 20–40 tokens per parameter, the multiplier stays at 1.26–1.46. As a local exponent φ_loc, defined by N* = N_once + 4^φ_loc · N_rec, this is 0.32–0.50 when N excludes the
output head, as in Schwethelm et al. [5], close to their φ = 0.46; with the head it is
0.35–0.52. Published equal-parameter results at about 130M–770M [3, 6, 14] mostly restate to a similar 1.2–1.6, and to
up to about 2 for the small blocks of [14], which were trained on 750–1500 tokens per parameter. Ouro's 778M model, at
about 26 tokens per parameter, reads above 1.75 off a plot, while its 53M model is worth less than its parameters
despite about 377 tokens per parameter [3]. Above 770M only bounds are available: lower bounds, and 1–2.8 for the 1.4B
model of [10].

The value of a loop changes with size and data. The pre-registered single value per loop type averages two
regimes: small models trained on few tokens per parameter, where each parameter of the looped model is worth at most
about one dense parameter, and larger ones, where each is worth about 1.4. The dependence on tokens per parameter is a
loop × data effect, which the published effective-parameter laws do not have. In them the value depends on the loop
count (and, for mixtures of experts, the sparsity) but not on the training tokens [5, 9].

### 3.3 Dependence on model size and training tokens

<p align="center"><img src="../code/results/figures/fig12_scale_middle_r4.png" width="60%"><br><em>Figure 3. Looped minus dense loss for a middle block looped 4 times, against the dense model of the same size and seed, at 10, 20 and 40 tokens per parameter (seed means; bands: range over seeds; three seeds at 10M–40M, one at 80M, two at 160M). Below zero the looped model is better. At every budget the advantage grows with size.</em></p>

Figure 3 compares the same cell, a middle block looped 4 times, with the dense model of the same size and seed. At 40
tokens per parameter the loss difference is +0.018, −0.025, −0.050, −0.055 and −0.063 nats from 10M to 160M
(negative = looped better). The curve is monotone at every budget, and the two 160M seeds agree within 0.001 nats.
Expressed as φ with the ruler held fixed (exploratory), the same cell gives −0.32, 0.19, 0.39, 0.52 and 0.65. The
increments shrink in loss terms (0.043, 0.025, 0.005 and 0.008 per doubling), and since φ cannot exceed 1 the curve must
bend somewhere, though our sizes do not yet show where.

<p align="center"><img src="../code/results/figures/fig11_data_extension.png" width="90%"><br><em>Figure 4. Looped minus dense loss of the same seed against training tokens per parameter, at 10M (left) and 20M (right), for middle blocks looped 4 and 8 times and, at 10M only, two whole-stack cells (4 loops, and 8 loops with truncated backprop; seed 42). Bands: range over seeds; the labels next to the middle-block points give the number of seeds. Points at 80 and 160 tokens per parameter come from continued runs. Each doubling of the data adds 0.02–0.03 nats to the advantage of the middle-block cells.</em></p>

We also continued the finished 10M middle blocks looped 4 and 8 times to 80 and 160 tokens per parameter, and the 20M
ones to 80 (Figure 4). For the middle-block cells each doubling of the data adds 0.02–0.03 nats of advantage, with no
sign of saturation. In terms of the data a dense model needs to match the looped one, a 10M middle block looped 8 times
goes from 0.86× at 10 tokens per parameter (seed 42) to about 2.2× at 160. The two seeds give 1.95× and 2.49×, read off
the dense curves continued to 640 in S4; before S4, with the dense loss beyond 160 extrapolated, this multiplier was 1.68×. A
20M one reaches about 2.0× at 80 (1.93× and 2.07×), and whole-stack loops show the same direction.

### 3.4 Comparisons at equal training compute

A looped run costs more compute than the dense run of the same size on the same tokens. To compare the two at equal
compute, we convert each looped run's training compute into the number of tokens a dense model could have seen for it,
and read the dense loss off the measured dense curve. At the shared learning rate, the dense model is better in every
comparison that can be measured.

Before S4, 221 of the 320 comparisons could be measured, and all of them were dense wins by at least 0.016 nats; the
other 99 needed the dense loss beyond its measured curve. Two extrapolations bound the true value there. The slope of
the last measured segment flatters the dense model: in a backtest that drops the last budgets of every measured dense
curve, it predicts a dense loss 0.023–0.036 nats too low one doubling out. A power law fitted to each curve flatters
looping, predicting a dense loss 0.006–0.027 nats too high one doubling out. With the power law, looping came out ahead
in 6 of the 99, by at most 0.021 nats, all at the largest data per parameter ([results §5.17](results.md)).

S4 then continued the 10M dense models to 640 tokens per parameter and the 20M ones to 320, so that 258 of the 320
comparisons are measured. The dense model wins all of them at the shared learning rate, and the measured gaps fall about
halfway between the two extrapolations. The other 62 comparisons still need an extrapolated dense loss, 43 at 40M–160M
and 19 at 10M–20M, almost all for the third seed, whose dense curve ends at 80 tokens per parameter.

| Looped cell (size, tokens per parameter) | Training compute, relative to the dense model of the same size | Gap at equal parameters and tokens | Gap at equal compute |
|---|---|---|---|
| Middle block r = 4 (10M, 160N) | 2.2× | −0.033 | +0.016 |
| Middle block r = 8 (10M, 160N) | 3.7× | −0.051 | +0.034 |
| Middle block r = 4 (20M, 80N) | 2.2× | −0.051 | +0.022 |
| Middle block r = 8 (20M, 80N) | 3.8× | −0.064 | +0.052 |

Looped minus dense loss on the main validation set, means over seeds 42 and 43; negative = looped better. At equal
compute the dense model is trained on 2.2–3.8× the tokens ([results §5.18](results.md), S4).

The margin shrinks as the data per parameter grow. For the middle block looped 4 times at 10M it is +0.119 at 40N,
+0.062 at 80N and +0.016 at 160N, and at 20M it is +0.138, +0.075 and +0.022 from 20N to 80N.

A lower learning rate for the looped model changes one comparison. S1 trained the 10M middle r = 4 model at 0.35× the
dense rate, the lowest rate it tried, and because the loss still fell as the rate was lowered there, the best rate is
not bracketed. At 160 tokens per parameter this model is level with the dense model on average, at −0.011 and +0.005 in
two seeds, which puts the looped model ahead by 0.011 nats in one seed. It is the first equal-compute comparison in this
study that the dense model does not win. Whether it would become a looped win with still more data per parameter is not
measured. Where the loop helps at equal parameters, it reaches a given loss with fewer parameters or less data; within
the measured range it does not do so with less compute.

### 3.5 Learning rates

All looped language models used the dense models' learning rates. A check of ten looped cells at 20M and 10 tokens per
parameter, whose decision rule was written before its runs, found no better one. It tried 0.5, 0.71 and 1 times the
dense rate, and nine of the ten cells were best at 1×, the top of that range, so higher rates were not excluded.

Supplementary experiment S1 then re-tuned both the dense model and the middle block looped 4 times at 10M, 20M and 40M
and at 10–40 tokens per parameter (at 10M also 80 and 160). It tried 0.5–1.41× the dense rate, plus 0.35× where the
pre-set edge rule required it, with one seed (two at 20M), so its gaps differ slightly from the three-seed means of
section 3.3. The dense model's best rate stayed at 1× to within 0.004 nats: at 40M, 0.71× was better by 0.001 nats at 40 tokens
per parameter and by 0.004 at 80 (the dense runs reach 80).

For the looped model, the shared rate understated the advantage by at most 0.005 nats at 20M–40M, but at 10M the looped
model preferred 0.5× and below from 20 tokens per parameter on, which was worth about 0.02 nats (up to 0.024). With each side at its
best rate, the gap at 40 tokens per parameter is −0.005, −0.031 and −0.055 nats from 10M to 40M, against +0.015, −0.031
and −0.050 at the shared rate for the same seeds. The scale trend is somewhat smaller but holds by the rule fixed in
advance, and the data trend at 10M is larger, from +0.082 at 10 tokens per parameter to −0.059 at 160.

At equal compute, the dense model won 10 of the 11 comparisons of the re-tuned runs, one of them (40M at 40 tokens per
parameter) on an extrapolated dense loss. The eleventh, at 10M and 160 tokens per parameter, first used an extrapolated
dense loss. Measured in S4, it is −0.011, a looped advantage of 0.011 nats, and with a second seed (+0.005) the two are
level on average (section 3.4; [results §5.16 and §5.18](results.md)).

### 3.6 Gains per token and on other validation sets

Per token, the gain on web text is a small, broad shift: about half of the tokens improve, and the gain is not
concentrated on hard tokens beyond what two dense seeds differ by.

<p align="center"><img src="../code/results/figures/fig5_gap_by_exam.png" width="80%"><br><em>Figure 5. Looped minus dense loss on the four validation sets (FineWeb-Edu, SlimPajama, FineMath and code) at 40 tokens per parameter, for four looped cells at 20M, 40M and 80M (seed means; one seed at 80M; exploratory). The gain is largest on math and code, the sets on which the seed-to-seed noise is also largest.</em></p>

On math and code the gain is larger (Figure 5), on average about 3 and 7 times (1.5–7 and 4–11 times across cells), but
the seed-to-seed noise on those sets is 10–100 times larger too. Supplementary experiment S7 showed that the code
set is even more sensitive to the data order, which our seeds share, than to the initialization; the math set is not. In
distribution, at 20M, training the same initialization on three disjoint parts of the data stream moves the main-set
loss by 0.003–0.010 nats, 0.1–0.5× the spread between initialization seeds, so the seed intervals in this report need no
caveat. On the code set, the data order moves the loss by 0.21–0.69 nats, up to 7× the initialization spread, and the
paired looped-minus-dense gap changes sign between orders.

Because both the initialization and the documents a run happens to see move its out-of-distribution loss strongly at
this size, the out-of-distribution sets carry no fine comparison here ([results §5.18](results.md), S7). One anomaly on
these sets remains unexplained: at 160M, one looped seed is 0.2–0.6 nats worse than the other on the
out-of-distribution sets at 40 tokens per parameter, while their main-set losses are identical.

### 3.7 Depth or weight sharing (S10)

Every dense model of the ruler has eight layers, while a middle block looped 4 times executes 20, so the multiplier of
section 3.2 may measure what depth buys at a fixed size rather than what sharing buys. Deeper and thinner
models do better at a fixed size below a billion parameters [33], while at a fixed size the loss depends only weakly on
the shape over a wide range of shapes [4].

Supplementary experiment S10 ([plan](supplementary-experiments-3.md), frozen on 2026-10-27 before any of its runs)
tested this with dense models of 18 layers and about the looped model's number of parameters: 18 × 320 (N = 28.1M) at 20M and
18 × 448 (N = 51.2M) at 40M. They were trained with the grid's recipe, data, budgets and three seeds, and their
multiplier M_deep is read off the same ruler as the looped model's M_loop. The plan fixed the reading in advance:
R = log M_deep / log M_loop at 40M, averaged over the three budgets, is the share of the looped model's advantage that a
dense model of the same size and nearly the same depth recovers.

| 40M | 10N | 20N | 40N |
|---|---|---|---|
| Dense, 18 layers, loss | 3.397 | 3.232 | 3.119 |
| Looped (middle block, r = 4), loss | 3.394 | 3.217 | 3.090 |
| Looped advantage over the 18-layer model (per seed) | 0.003 (0.004, 0.004, −0.001) | 0.015 (0.017, 0.018, 0.010) | 0.029 (0.030, 0.032, 0.026) |
| M_deep | 1.10 | 1.11 | 1.11 |
| M_loop | 1.15 | 1.26 | 1.34 |
| R | 0.67 | 0.46 | 0.36 |

At 40M, R averaged over the three budgets is 0.50 (95% interval 0.41–0.62), in the band the plan reads as "partly
depth, partly sharing", as it predicted. 
What depth buys does not grow with data, while the looped model's advantage does. At 20M,
reported descriptively because M_loop is near 1 there, the 18-layer model, narrow at width 320, is worse than the
8-layer one of the same size (M_deep 0.84–0.94). There depth explains none of the looped model's advantage at 40 tokens
per parameter (M_loop 1.16).

The learning-rate check of the 18-layer models used seed 42 at 0.71 and 1.41 times the rule's rate. At 40M and 10N, 1.41
times was better by 0.014 nats, which triggered the plan's rule (it was better by 0.006 and 0.001 at 20N and 40N, and at
20M the rule's rate was best). As in S1, the readings are also reported with each model at its best tested rate. At
those rates the deep model is ahead of the looped one at 10N by 0.010 nats (M 1.37 against 1.18). At 20N and 40N, R is
0.56 and 0.34 (0.45 and 0.37 for the same seed at the rule's rate). Averaged over the three budgets, R at the best rates
is 0.94, in the band the plan reads as "most of the advantage is a depth effect". The average is carried by 10N, where
log M_loop is smallest, and the plan keeps the primary reading at the rule's rate. These readings come from a script
written after the results (`fit/supp3_best_lr.py`), because the analysis script written before them flagged the case
without computing it. The deep model's best rate was the highest tried, so its optimum is not bracketed.

The 18-layer model also needs about half the looped model's training compute per token (406M against 835M FLOPs at 40M),
and at equal compute it is ahead by 0.11–0.19 nats where its curve reaches without extrapolation. Of the plan's three
predictions, the first, R in 0.25–0.75, held. The second, that the 18-layer model would be 0.01–0.03 nats better than
the 8-layer one at 40M, held at two of three budgets (0.007, 0.016, 0.020). The third, that the rule's rate would be
within 0.01 nats of the best, held at five of six budgets ([results §5.19](results.md)).

At 40M, about half of the loop's advantage at equal parameters on text comes from depth. With each model
at its best tested learning rate, depth accounts for all of it at the shortest budget, in the one seed tried: the deep model is
0.010 nats ahead there, twice the range of this gap across the three seeds at the rule's rate. At 20M depth explains
none of it, and at neither size does depth recover the part that grows with data.

## 4. What the looped block computes on text

We probed trained models loop by loop, reading the state after each loop in three ways: with the model's own output
layers (a "logit lens"), with a trained linear map in front of them, and with a trained linear head (Figure 6).

<p align="center"><img src="../code/results/figures/fig8_loop_probes.png" width="95%"><br><em>Figure 6. Main-set loss after each of 16 loops, read three ways: by the model's own output layers (the same as running that many loops), through a trained linear map placed in front of them, and by a trained linear head. The panels show middle blocks trained with 8 loops at 20M, 40M and 80M, and the 80M whole stack trained with truncated backprop (k = 4); seed 42, trunk checkpoints at 32 tokens per parameter. Off-scale read-outs lie above the frame; Figure 7 shows the 80M models over the full range (the image title calls it
"fig 6"). The middle blocks reach their low loss only at loop 8, though through the trained map loop 4 is already close; the truncated whole stack returns to a low loss every 4 loops.</em></p>

With a fixed loop count (r = 8), the loop runs a fixed-length program. Through a trained linear map, the state after
loop 4 already reads out within 0.1–0.2 nats of the final loss, and loops 5–7 add little. The last loop is a large
non-linear move that rewrites the state into what the output layers expect: 37–47% of the tokens settle their prediction
only there. Running fewer or more loops at inference breaks the model (Figure 7). This held in all three seeds at 20M
and 40M, and at 80M (one seed).

Truncated backprop (whole stack, k = 4) leads instead to a period-4 cycle: the state returns to almost the same point
every 4 loops, so the model works at 4, 8, 12 or 16 loops and gains nothing from more.

<p align="center"><img src="../code/results/figures/fig6_depth_curve.png" width="60%"><br><em>Figure 7. Main-set loss of four 80M looped models run with 1 to 16 loops at inference; stars mark the loop count each was trained with. The models trained with full backprop reach their low loss only at their training loop count. The truncated whole stack works at every plotted multiple of its 4-loop cycle (4, 8 and 16; 12 is not plotted).</em></p>

When the loop count is drawn at random from 1–8 at every training step (Huginn also samples the loop count, from a
different distribution), the model learns a contraction to a fixed point. Its loss at 1 to 32 inference loops is flat
after about 4 loops (Figure 8). The model is robust to the loop count, but at its nominal 8 loops it is 0.05 nats worse
than fixed-r training and ties the dense model.

<p align="center"><img src="../code/results/figures/fig10_random_r_depth.png" width="55%"><br><em>Figure 8. Main-set loss of 20M middle blocks run with 1 to 32 loops at inference (trunk checkpoints at 32 tokens per parameter), trained with 8 loops or with a loop count drawn from 1–8 at every training step (full or truncated backprop); dashed: the dense model at the same tokens. Trained with a random loop count, the model is flat from about 4 loops on and no better than the dense model. Trained with 8 loops, it works only at 8.</em></p>

The learning rate changes the shape of these curves, with lower rates giving flatter curves over the inference loop
count. This showed first in the 20M models of the looped learning-rate check. S1 confirmed it at 10M, 20M and 40M
(middle block looped 4 times, checkpoints at 32 tokens per parameter): over 4–16 loops the loss varies by 0.36–0.77 nats
at 0.5× the dense rate and by 3.4–4.6 nats at 1.41×. Lower rates make the iteration closer to convergent, but on text
more loops than trained never helped, at any rate.

## 5. A task that needs serial steps

### 5.1 The task and the conditions

Predicting the next word of web text rarely needs many serial steps, so text gives the loop little reason to iterate.
We built a task that does ([design](synthetic-reasoning-design.md), frozen 2026-10-09).

A problem lists a start value and a chain of d table lookups, one per step, for example
`25 ; LUT 7 ; LUT 3 ; LUT 19 ; … =>`. Each LUT is one of 32 fixed random functions on 0–96, and the model must output
the final value in a single forward pass. Two lookups compose into one of 1024 arbitrary functions, so a model that
skips steps must memorise rapidly growing tables of compositions. The task rewards computing step by step, although, as
for any associative composition, a shallower solution exists in principle.

Models are trained on 1–8 steps and tested up to 24. Problems are drawn at random during training, so short ones repeat
(there are only 3,104 one-step problems), but chains longer than 8 steps never occur in training. The task changed twice
before the formal runs, both times before any comparison between conditions: a uniform mixture of chain lengths was not
learned, so a curriculum was added, and chains of modular arithmetic were learned too slowly, so the operations became
fixed lookup tables.

We compare dense models of 8, 20 and 36 layers with an 8-layer looped model (2 + 4×r + 2 executed layers). The looped
model is trained with r = 2, 4 or 8 loops, with a random loop count, or with as many loops as the chain has steps
(loops = steps). Every condition uses the learning rate picked by a sweep on in-distribution accuracy only, and has 5
seeds. Two further conditions were added later, in S5 (section 5.6): dense models that write the intermediate values out
as a chain of thought, and a dense model with pause tokens. S9 added one more looped regime, loops = steps plus random
extra loops (section 5.5). A planned control task, pointer chasing, was dropped: it had not been piloted, and under the
recipe calibrated on the chains none of the four conditions tried on it learned it.

### 5.2 Results at the training loop count and with more loops

| Model | Params | Largest solvable step count* | With more loops at inference | Accuracy at 12 steps: own loops / more loops** | at 16 steps** |
|---|---|---|---|---|---|
| Dense, 8 layers | 6.5M | 4.2 (4–5) | — | 0.06 | 0.05 |
| Dense, 20 layers | 16.1M | 6.4 (6–7) | — | 0.04 | 0.03 |
| Dense, 36 layers | 29.0M | 8.0 (8–8) | — | 0.02 | 0.02 |
| Looped, r = 2 | 6.6M | 6.4 (6–7) | 8.6 (6–11) | 0.05 / 0.64 | 0.04 / 0.33 |
| Looped, r = 4 | 6.6M | 8.0 (8–8) | 11.8 (10–15) | 0.03 / 0.83 | 0.03 / 0.28 |
| Looped, r = 8 | 6.6M | 9.0 (9–9) | 11.2 (11–12) | 0.04 / 0.86 | 0.03 / 0.37 |
| Looped, random loop count | 6.6M | 11.4 (10–13) | 12.2 (10–15) | 0.77 / 0.83 | 0.08 / 0.49 |
| Looped, loops = steps | 6.6M | 12.8 (11–16) | — | 0.96 | 0.37 |

\* mean (min–max) over 5 seeds of the longest chain length up to which every length is solved at ≥ 90%, at the model's
own loop count. \*\* at the model's own loop count / the best of the own and every larger inference loop count,
including d loops (for r = 2 these include 4 and 8 loops), chosen per seed with the
answers, mean over seeds. Chance is 0.01.

<p align="center"><img src="../code/results/figures/fig13_synthetic_accuracy.png" width="80%"><br><em>Figure 9. Accuracy against chain length for every condition, each run at its own loop count (mean over five seeds; bands: range; shaded: the training range of 1–8 steps). The dense and fixed-loop models solve at most one step beyond the training range; the models trained with a random loop count or with loops = steps go further.</em></p>

### 5.3 Comparison with dense models, and more loops at inference

At equal parameters, more loops solve longer chains: 4.2 → 6.4 → 8.0 → 9.0 steps. The dense models of equal executed
depth, with 2.4–4.4× the parameters, solve 1–1.6 fewer steps (the 20-layer one did not fully learn the training range).

None of the dense models gets past the training range (Figure 9): at 12 steps they score 0.02–0.06. The fixed-loop
models also stop there at their training loop count (0.03–0.05), but with more loops they reach 0.64–0.86, the
random-loop model 0.83 and the loops = steps model 0.96.

<p align="center"><img src="../code/results/figures/fig14_synthetic_inference_loops.png" width="95%"><br><em>Figure 10. Accuracy against chain length for the model trained with 8 loops (left) and the model trained with a random loop count of 1–8 (right), run with 4, 8, 16 or 32 loops at inference (mean over five seeds; bands: range; shaded: the training range). Dashed: the 36-layer dense model, as deep as 8 loops. More loops solve longer chains, but accuracy still falls with length.</em></p>

More loops at inference solve longer chains (Figure 10). A model trained with 8 loops solves 9 steps at 8 loops and
about 11 with 16 or 32, but the gain is limited: at 16 steps its accuracy is 0.37. The model trained with a random loop
count gains as well: at 16 steps its accuracy rises from 0.08 with 8 loops to 0.49 with 16. The model trained with
loops = steps, run with as many loops as the chain has steps, is 0.34–0.95 more accurate (seed means) than when run with
8 loops, at every length from 10 to 16 steps; at 9 steps, 8 loops already solve the chain and the gain is 0.03. The
pre-specified prediction P3(c) is judged by this difference (section 5.8).

### 5.4 Probes of the intermediate values

<p align="center"><img src="../code/results/figures/fig15_synthetic_probes.png" width="95%"><br><em>Figure 11. The first loop (left: looped models, mean over seeds) or layer (right: dense models, one line per seed) at which each intermediate value x_j can be decoded with probe accuracy of at least 0.5. The models trained with 8 loops or with loops = steps compute one step per loop, the other looped models pack more than one step into a loop, and the dense models compute late; a dense seed in which only x_1 and the answer are decodable appears as a single point at the last layer.</em></p>

After every loop, we train a linear probe for each intermediate value x_j (Figure 11). Each probe reads three positions
around step j in the prompt and keeps the best; reading only one position had missed values that one seed stores
elsewhere. To measure how well a model keeps its computed values as it runs more loops, we use their persistence: the
lowest probe accuracy a value reaches at the loops after it first appears, averaged over the values.

In the model trained with 8 loops, each loop computes one step: x_j appears at about loop j (x_8 at loop 8; 1.0–1.1
loops per step in all 5 seeds), and each value stays decodable afterwards (persistence 0.96). The model trained with
loops = steps also computes one step per loop and keeps its values while it runs its d loops (persistence 0.92 over
those d loops), but it loses them when run longer. It needs about d loops because it was never trained to keep looping
after the last step, while the fixed-loop model was.

Models trained with fewer loops pack more steps into each loop, about 1.6 per loop at r = 4 and 2.4 at r = 2, the
direction Zhang et al. [19] predict. The dense models compute late, from about layer 21 of 36. In 5 of the 10 seeds of
the 20- and 36-layer models, no intermediate value after the first appears at the step positions: four show only x_1
(probe accuracy 0.50–0.74) and, at the last layer, the answer, and one shows nothing.

### 5.5 Stopping rules that need no answer

Extra loops help in deployment only if their number can be set without the answer. The frozen design specified a state
rule for the loops = steps model: stop when the state at the answer position changes by less than 2% from one loop to
the next, at most 32 loops. It had been computed in every run but was not reported until 2026-10-15
([results §5.15](results.md)). For the models trained with 4 or 8 loops, it
recovers about 80% of the gain of the best fixed loop count at 12 steps and 60–70% at 16, running 11–19 loops, without
hurting the training range. The model trained with loops = steps, which extends furthest, cannot stop by it. Its state
keeps changing after d loops, and the stop lands on a wrong loop at every length (accuracy 0.02–0.08). It extends
furthest partly because the problem states the number of steps, which supplies its loop count.

S8 compared other answer-free rules on the 25 looped checkpoints of the formal runs, with the rules and thresholds
fixed in its plan; the plan was written after the state rule's results on these checkpoints were known, and frozen
before the comparison.
Rules were selected on one problem set and reported on another. One rule uses the model's own confidence: it stops when
the top probability reaches a threshold, 0.99, or 0.9 for r = 2, and is the early-stopping variant of the
maximum-confidence rule of Fan et al. [17]. Among the rules that stop, it is the best in every condition except r = 4,
where the state rule with a lower threshold (0.5%) ties it (0.816 against 0.815 at 12 steps). For the S9 model that
state rule is slightly better (0.94 against 0.93, in all five seeds), and in both cases the state rule runs far more
loops (27 against 9 and 24 against 14).

At 0.99, the confidence rule recovers 97–98% of the best fixed count's gain at 12 steps for the models trained with 4 or
8 loops or a random count. On problems within the training range it uses fewer loops than these models were trained
with. It brings the loops = steps model to 0.76, at a cost of 6–7% of the in-range problems. Stopping when the
prediction stops changing often stops on a wrong answer that no longer changes, because the fixed-loop models' read-out
turns only in the last loop. Majority votes over loop counts are not stopping rules, and they never beat the best fixed
count. They come within 0.013 of it where the plan's rule selected them (r = 4, r = 8 and the random loop count), and
fall far below it for r = 2 and loops = steps.

S9 changed the training instead, so that one model both extends far and can stop: the loops = steps model was trained
with 0–4 random extra loops after the last step. It then keeps its values when run for 16 loops (persistence
0.04 → 0.91) and extends about one step less (11.6 against 12.8 at the 90% line; 12.2 with more loops). With the
confidence rule it reaches 0.93 on 12-step problems while still solving 0.998 of the in-range problems, and on those
in-range problems it uses fewer loops than d, though at 12 steps it uses 14. By the frozen state rule, whether it stops well depends on the seed
(0.52 on average: it works in two seeds, fails in two and partly works in the fifth). None of the plan's three pre-written readings covers this outcome.
Training it costs about 1.4× the compute per training step of the loops = steps model.

| Model | Own loop count | Best fixed count (chosen with the answers) | State change < 2% | Confidence ≥ 0.99 (0.9 for r = 2); mean loops used in parentheses |
|---|---|---|---|---|
| Looped r = 2 | 0.05 | 0.62 | 0.45 | 0.70 (8.3) |
| Looped r = 4 | 0.04 | 0.83 | 0.65 | 0.81 (9.2) |
| Looped r = 8 | 0.04 | 0.86 | 0.71 | 0.84 (17) |
| Random loop count | 0.77 | 0.83 | 0.73 | 0.83 (14) |
| Loops = steps (own = d loops) | 0.96 | 0.96 | 0.03 | 0.76 (14), and 6–7% lost in range |
| Loops = steps + 0–4 extra loops (S9) | 0.90 | 0.94 | 0.52 | 0.93 (14), 0.998 kept in range |

Accuracy on 12-step chains, seed means on the S8 reporting set ([results §5.18](results.md), S8 and S9).

In the formal runs, extending far and stopping well came from different training regimes. The S9 regime gives both, at a
cost of about one step of extrapolation.

### 5.6 Comparison with a written chain of thought and pause tokens

Supplementary experiment S5 ([results §5.18](results.md)) gave the chain task to two more kinds of dense model. Some
write the intermediate values out as tokens, with 8 layers (the looped models' size) or 36 layers (their executed
depth); the other is an 8-layer model with d pause tokens [28]. Each has five seeds, with learning rates chosen as for
the other conditions. The chain-of-thought models are trained with a loss on every intermediate value and the looped and
dense models only on the answer, a difference the plan stated in advance.

| Model | Params | In range (1–8 steps) | 12 steps | 16 steps | Layer passes per 12-step problem |
|---|---|---|---|---|---|
| Dense 8 layers, chain of thought | 6.5M | 1.000 | 0.014 | 0.015 | 408 |
| Dense 36 layers, chain of thought | 29.0M | 1.000 | 0.017 | 0.013 | 1836 |
| Dense 8 layers, d pause tokens | 6.5M | 0.566 (solves 4 steps) | 0.015 | 0.012 | 416 |
| Dense 8 layers, single answer | 6.5M | 0.62 (solves 4.2 steps) | 0.064 | 0.048 | 320 |
| Looped r = 8, 8 loops / 16 loops | 6.6M | 0.998 | 0.036 / 0.827 | 0.033 / 0.285 | 1440 / 2720 |
| Looped, loops = steps, run with loops = d | 6.6M | 1.000 | 0.957 | 0.366 | 2080 |

Seed means. The inference cost counts layer passes over the tokens processed: a 12-step prompt has 40 tokens, and the
chain-of-thought models generate 12 more with a KV cache.

Within the training range the chain of thought is perfect, at 1.000 in every seed and at every rate tried, where the
single-answer 8-layer model solves 4 steps. It is also cheaper: at 12 steps it costs under a third of the looped model's
inference compute at its training loop count, 408 layer passes against 1440 (the 16 loops that solve 12-step chains cost
2720).

Beyond the range it does not extend at all. It scores 0.014 at 12 steps for 8 layers and 0.017 for 36, near the chance
level of 0.010, against 0.83–0.96 for the looped models on the same longer prompts. A diagnostic shows the chain
breaking within its first three written tokens as soon as the prompt holds more than 8 tables, in 97–98% of the problems
for 8 layers and 68–86% for 36. For 8 layers the first token is right in 81% of the 9-step and 57% of the 12-step
problems; for 36 layers it almost always is. Even when the correct earlier tokens are given, the tokens after the first
two are mostly wrong (for 8 layers, accuracy 0.04–0.10 at most positions). This is consistent with a failure to
generalize over the input length, the known weakness of transformers on inputs longer than those seen in training
[31, 32], and not with an accumulation of errors along the chain. The looped models read the same longer prompts and
still extend; why they do was not tested. The prompt of this task grows with the number of steps, however, so this
comparison cannot say whether a written chain would extend on a task whose input length does not grow. Remedies for
length generalization were not part of the plan.

Pause tokens add nothing (4 steps, as without them). On this task the better choice between loops and written tokens
depends on the length of the problem: for problems like the training ones, written tokens are cheaper and perfect, and
for longer ones the loop was the only form of extra computation that extended here.

### 5.7 Learning rates on the chain task

All looped models above were trained at 1.25e-4–2.5e-4, 4–8 times lower than the rate chosen for the 8-layer dense model
(1e-3), which has the same unique layers. (These are the rates of the embedding and output layers; the hidden layers are
scaled with the number of unique layers.)

In the learning-rate sweep, which used one seed (seed 1), the same r = 8 architecture trained at 1e-3 learned the
training range poorly, with accuracy 0.54 against 0.999 at 2.5e-4 after the same 16000 steps. It also showed no
step-by-step frontier: like the language models of section 4, it worked only at exactly 8 loops. Probes on all sweep
checkpoints show the same split. For looped r = 8, the step-by-step frontier is present at learning rates up to 5e-4
(about one step per loop at 1.25e-4 and 2.5e-4, one per 1.5 loops at 5e-4) and absent at 1e-3 and 2e-3. Looped r = 4 and
the random-loop-count model behave the same way.

Supplementary experiment S2 tested this at the full length. At 5e-4 and 1e-3, it trained the four looped conditions that
its plan had identified as extrapolating, for the full 32000 steps with three seeds. (The plan left out r = 2 on the
strength of a number corrected later; r = 2 extrapolates too, to 8.6 steps.) At 1e-3, 10 of 12 runs still did not learn
the training range (accuracy 0.49–0.98). At 5e-4, 8 of 12 did, and those mostly extended about as far as the formal
runs: 2.7 steps beyond the training range against 3.2 for looped r = 8, and 4.0 against 4.8 for loops = steps. The
exception was the random loop count, whose two learned seeds extended 2 steps against 4.2.

A high learning rate mainly keeps the looped model from learning the task in the given steps. When the model does learn
the task, it mostly learns the same extendable algorithm ([results §5.16](results.md)).

### 5.8 The pre-specified predictions

The frozen design made five predictions ([results §5.15](results.md)):

- P1 (more loops, more steps at equal parameters) holds in direction, but r = 8 did not reach 16 steps.
- P2 (iso-depth dense models at least as good) does not hold as stated, but the gaps are small: 1–1.6 steps, within P2's
  tolerance at r = 8.
- P3 (inference-time scaling) has three parts. P3(a) is half right: fewer loops than trained hurt the fixed-r
  models, as predicted, but more loops did *not* hurt the 4- and 8-loop models. P3(b) fails
  in the direction that favours looping, since the random-loop-count model, predicted not to gain, gained on long
  chains. P3(c) holds: the loops = steps model, run with d loops, beats itself run with 8 loops at every length from
  10 to 16 steps, counting from 10 steps, a choice made after the results.
- P4 (only the step-matched model's probe table is diagonal) is half right, because the fixed-r model's table is
  diagonal too, by a definition of the diagonal share fixed after the results. Read as the loop at which a value is
  decoded most accurately, the shares are 0.625 and 0.425, and P4 would hold.
- P5 (much larger gaps than on text) holds.

Of the supplementary predictions, S5 held one of three, S8 two of five and S9 three of four; the
readings are in [results §5.18](results.md).

## 6. Cost at deployment

At equal parameters, a middle block looped 2, 4 or 8 times costs 1.4–1.5×, 2.2–2.4× or 3.7–4.1× the training FLOPs per
token of the same-size dense model, and 1.5×, 2.4–2.6× or 4.5–4.6× its decoding time per token. These were measured on
one RTX 4090 in bf16, with KV-cached decoding and one cache per executed layer, so that decoding computes what the
models computed in training (supplementary experiment S3, [results §5.16](results.md)).

The fairer comparison is at equal loss, against the dense model that reaches the same loss at 40 tokens per parameter
(interpolated between the dense sizes):

| Cell | Sizes | Weight memory | KV cache per token | Decoding time, batch 1 | Peak memory, batch 1 | Peak memory, batch 32 |
|---|---|---|---|---|---|---|
| Middle block r = 2 | 20M–80M | 0.93–1.03× | 1.43–1.51× | 1.45–1.49× | 0.99–1.15× | 1.21–1.24× |
| Middle block r = 4 | 20M–160M | 0.81*–0.96× | 2.20–2.41× | 2.37–2.47× | 0.98*–1.19× | 1.57–1.58× |
| Middle block r = 8 | 20M–80M | 0.83–0.92× | 4.01–4.23× | 4.27–4.56× | 1.08–1.34× | 2.29–2.40× |
| Whole stack r = 4 | 20M–80M | 1.03–1.11× | 4.02–4.26× | 4.00–4.11× | 1.18–1.84× | 2.32–2.41× |
| Whole stack r = 8 | 20M–80M | 0.81–0.85× | 6.96–7.23× | 7.57–7.91× | 1.27–1.50× | 3.38–3.91× |

Ratios of the looped model to the dense model of equal loss; \* 160M lies beyond the largest dense model (extrapolated).

At equal loss the looped models save at most 19% of the weight memory (middle block r = 4: 4%, 14%, 17% and 19% from 20M
to 160M), and they pay roughly their extra executed layers in decoding time and KV cache. At batch 1 the larger KV cache
cancels the saving in weights: the middle block looped 4 times (r = 4) needs 1.5–2.4% less peak memory at 80M–160M and
more at 20M–40M (up to 1.2×). With more sequences, the KV cache makes looped models need more memory, 1.2–3.9× at batch
32 with a 1k context. In this range most looped models need less weight memory at equal loss (the whole stack looped
4 times is the exception, at 1.03–1.11×), none needs less decoding time, and peak memory falls only at batch 1 and only
at 80M–160M. The saving would grow where weights dominate memory, for example one sequence with a short context in a
much larger model, which we did not test.

Sharing the KV cache across loops would remove most of the KV-cache and memory cost, though not the extra decoding time.
In the scheme we tested there is one cache slot per unique layer. Each layer of the looped core keeps one cache for all
loops, in which every loop reads the earlier tokens' keys and values as written at their last loop, while the layers
outside the loop keep their own. Supplementary experiment S6 applied it to trained 20M models
([results §5.18](results.md)):

| Model (20M, seed 42, 32N trunk) | Inference loops | Cache slots, full → shared | Loss, full cache | Loss, shared cache | Δ |
|---|---|---|---|---|---|
| Middle block r = 8, full backprop | 8 | 36 → 8 | 3.443 | 3.534 | +0.091 |
| Middle block r = 8, truncated backprop | 8 | 36 → 8 | 3.451 | 3.591 | +0.140 |
| Middle block r = 8, random loop count | 8 | 36 → 8 | 3.496 | 3.496 | 0.000 |
| Middle block r = 4, full backprop | 4 | 20 → 8 | 3.442 | 3.611 | +0.170 |
| Middle block r = 4, full backprop, half the learning rate | 4 | 20 → 8 | 3.425 | 3.525 | +0.100 |

In this single-seed test, every model that gains from its loops at this size loses more than that gain when the cache
is shared, about 2.2 to 7.4 times as much: 0.09–0.17 nats, against gains of 0.019–0.046 over the dense model at the
same checkpoint and on the same tokens. The gains come from one seed, and at 20M the looped-minus-dense gap varies
across seeds by 0.032–0.045 nats (section 2), so these ratios are rough. With a shared cache, each of these models would
have a higher loss than the dense model of the same size. Truncated backprop does not make the cache shareable here,
against the finding of Huang et al. [13]. Only the models trained with a random loop count share for free; their
looped block contracts to a fixed point (section 4), and they gain nothing over the dense model.

Among the fixed-loop models, a lower learning rate loses less (0.100 against 0.170), which is consistent with training
towards convergence, or with a shared cache, as the way to get sharing, though one pair of runs cannot show it. Monea et
al. [21] pretrain looped models with a different scheme, in which only the first loop writes a cache and later loops
read it while keeping a short window of their own, and find that it costs no quality. Neither was tested here.

## 7. Practical takeaways

- Loop when the constraint is parameters rather than data, compute or latency. The advantage at equal parameters grows
  with model size and with tokens per parameter (section 3.3). At equal compute and a shared learning rate, a dense
  model is better in every measured comparison, by a margin that shrinks with data. With the lowest rate tried for the
  looped model, the two are level on average at 10M and 160 tokens per parameter (3.4); that rate was also the best
  tried, so the looped model's optimum there is not bracketed. At deployment the loop pays its executed layers in
  decoding time and KV cache (6).
- Before looping, try a deeper dense model of the same size. At 40M, 18 layers instead of 8 gave about half of the
  looped model's advantage for about half its training compute, and at 10 tokens per parameter, with its learning rate
  tuned, all of it. Only the part of the advantage that grows with data needed the loop (3.7).
- If you loop, loop a middle block, with full backprop. Middle-block loops beat whole-stack loops of similar executed
  depth in every comparison, and truncated backprop accounted for 25 of the 30 runs with loss spikes (16 of them
  whole-stack loops), though it was used in only 40 of the 100 looped runs. Halving the learning rate removed the spikes
  in a 10M follow-up.
- On text, start with r = 4 and judge at your target scale. At 40M–80M, r = 8 did about as well as r = 4, while costing
  about 1.75× the training compute and 1.8× the decoding time and KV cache. A loop is worth more at larger sizes and
  with more tokens per parameter, so ablations at small scale understate its value.
- Do not expect test-time scaling on text: fixed-loop training gives a model that only works at its training loop count,
  and random-loop training makes it robust to the loop count but not better with more loops (section 4).
- The looped model's learning rate needs its own tuning, especially at small scale and for step-by-step tasks. On text
  the dense rate was close to the looped model's best at 20M–40M, but 0.02 nats off at 10M. On the chain task, at the
  dense model's rate, most looped runs did not learn the training range in the same number of training steps (5.7).
- Sweep the loop count at inference to see whether the model can use more loops. On text none of ours could, while on
  the chain task the models whose accuracy held at more loops also solved longer chains with them. A model that works
  only at its training loop count may still compute step by step within those loops, as the loops = steps model does,
  and probes tell the two apart (5.4).
- Count the KV cache and the latency as well as the parameters. At equal loss the looped models saved at most 19% of the
  weight memory and needed 1.2–3.9× the peak memory at batch 32. Sharing the cache across loops cost the fixed-loop
  models more than their loops had gained, 2.2 to 7.4 times as much in a single seed (6).
- On a task that needs serial steps, let the lengths you need decide between loops and a chain of thought. Within the
  training range a chain of thought was cheaper and perfect, and beyond it only the loop extended (5.6). If the model
  must stop by itself, use its confidence: a 0.99 threshold recovered nearly all of the extra-loop gain (a state rule
  with a lower threshold matched it for r = 4 and was slightly better for the S9 model, but ran far more loops).
  Training with loops = steps plus a few random extra loops gave a model that both extends and stops by its confidence,
  in all five seeds (5.5).

## 8. Corrections

After the runs we checked the results and documents in several rounds, and two outside reviews (2026-10-14 and
2026-10-28) found further errors. Every
correction is dated in the [deviation log](deviation-log.md), and those named here changed a verdict, a reading or a
headline number.

On the chain task, P3(b) was first judged by the longest solvable chain; by the accuracy difference the design
specifies, the random-loop-count model gains from extra loops. P3(a) is half right and P3(c) holds, where we had judged
them not supported and partly met. S8 held two of its five predictions, where we had reported one. A stopping rule the
design specified was computed but not reported until 2026-10-15. We withdrew the claims that the loops = steps model
overwrites its values and that confidence is always the best stopping rule, and limited to deployment the claim that
extra loops need a stopping signal.

On text, the extrapolation test of H3 had kept 160M in its training set; redone as pre-registered, it no longer favours
r^φ in both directions. The joint-fit φ was corrected to 0–0.41. The equal-compute result had not said that 99 of 320
comparisons rested on an extrapolation favouring the dense model. The data-equivalent multipliers at the largest
budgets are about 2.2× and 2.0×; we had reported 1.7× and 1.65×, computed before S4 extended the dense curves. S3's
batch-1 peak memory had counted two KV caches. Sharing the cache, first compared with gains from other checkpoints,
costs 2.2 to 7.4 times the single-seed gain at the same checkpoint. Section 2 had said that a difference of 0.01 nats
between runs that share a seed is resolvable; across seeds the paired gap varies too much for that at 10M–20M, and it
is resolvable only at 40M.

## 9. Limitations

- Scale: five sizes named 10M–160M after their approximate non-embedding parameters (the model called 10M has
  N = 15.4M) and a 16k vocabulary, far below the 1B+ models where looped LMs are deployed. The scaling
  trends may bend; we have five sizes, one seed at 80M and one looped cell at 160M.
- The language models were trained on one data source (FineWeb-Edu), with no document seen twice, so multi-epoch
  regimes, where loops may matter more [8, 30], are not covered.
- Deployment cost was measured on one GPU type, for models of at most 160M parameters, with one KV cache per executed
  layer. Larger models, where weights dominate memory, were not tested, and sharing the cache across loops was measured
  only as a post-hoc change to models trained without it (S6).
- The learning rate was re-tuned for one looped cell at 10M–40M (S1), with one seed (two at 20M). The other cells, and
  80M and 160M, use the dense rates. On the chain task, the learning-rate criterion saturated (the loops = steps model
  scored 0.9999 at three rates), and ties went to the lowest rate, a rule the protocol had not specified. The 4-loop,
  random-loop and S9 models were best at the lowest rate tried, so their optimum is not bracketed, and how far a model
  extends is sensitive to the rate.
- Every dense model of the ruler has eight layers. The 18-layer dense models of S10 cover two sizes with one shape each
  (narrow at 20M); their learning rate was checked with one seed, and at 40M its best was the highest rate tried.
- The synthetic evidence comes from one task family and 6.6M-parameter models, extrapolation length varies widely
  across seeds, and the planned pointer-chasing control was dropped (section 5.1).
- The comparison with reasoning written out (S5: a chain of thought [27] and pause tokens [28]) covers one synthetic
  task, 6.5M–29M-parameter models and one training regime. In this task the prompt grows with the number of steps
  (3d + 4 tokens), so the chain-of-thought models' failure beyond the training range is a failure to generalise to
  longer inputs, which the looped models handle on the same prompts. Whether a written chain extends on a task whose
  input length does not grow with the number of steps was not tested. The chain-of-thought models were supervised on
  every intermediate value and the looped models only on the answer; the chain-of-thought models were given no help with
  length generalization, and continuous thoughts [29] were not tried.
- Seeds change only the initialisation, except in S7, which varied the data order at 20M. In distribution the data order
  matters less than the initialisation; out of distribution it matters more on the code set, and about as much or less
  on the second validation set and the math set.
- Of the 320 equal-compute comparisons, 62 rest on an extrapolated dense loss (section 3.4): 43 at 40M–160M and 19 at
  10M–20M, almost all for the third seed, whose dense curve ends at 80 tokens per parameter. The other 10M–20M dense
  curves now reach 640 and 320 tokens per parameter (S4).
- We considered four follow-ups and decided on 2026-10-17 not to run them: a second synthetic task family with a
  fixed input length, a dense control on text with the looped model's executed depth at its width (and so more
  parameters; S10's 18-layer models keep the parameter count instead), two more seeds at 80M, and dense and looped
  models trained on reasoning-heavy text. Each leaves one of the limitations above in place ([roadmap](roadmap.md),
  [paper plan](paper-plan.md)).

## 10. Reproducing

```
cd code
uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e ".[test]"
python -m pytest tests -q
```

[code/README.md](../code/README.md) lists the full sequence: data preparation, learning-rate sweep, main grid, fits,
extensions, supplementary experiments and figures. It maps every table and figure to the command that makes it.

- Every run's results are in `code/results/runs/`, and the per-run training logs in `code/results/train_logs/`.
- The fits are in `code/results/fit/`, the synthetic results in `code/results/synthetic/`, and the supplementary results
  in `code/results/supp/`, `code/results/supp2/` and `code/results/supp3/`.
- All figures are produced by `code/fit/make_figures.py`, `code/fit/make_overview.py` and `code/synthetic/analyze.py`.

## References

Venues are given only where the arXiv page states one; arXiv versions as of 2026-10-15.

[1] M. Dehghani, S. Gouws, O. Vinyals, J. Uszkoreit, Ł. Kaiser. *Universal Transformers*. ICLR 2019. [arXiv:1807.03819](https://arxiv.org/abs/1807.03819)

[2] J. Geiping, S. McLeish, N. Jain, et al. *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach*. 2025. [arXiv:2502.05171](https://arxiv.org/abs/2502.05171)

[3] R.-J. Zhu, Z. Wang, K. Hua, et al. *Scaling Latent Reasoning via Looped Language Models (Ouro)*. 2025, v6 2026. [arXiv:2510.25741](https://arxiv.org/abs/2510.25741)

[4] J. Kaplan, S. McCandlish, T. Henighan, et al. *Scaling Laws for Neural Language Models*. 2020. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)

[5] K. Schwethelm, D. Rueckert, G. Kaissis. *How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models*. 2026. [arXiv:2604.21106](https://arxiv.org/abs/2604.21106)

[6] H. Prairie, Z. Novack, T. Berg-Kirkpatrick, D. Y. Fu. *Parcae: Scaling Laws For Stable Looped Language Models*. 2026. [arXiv:2604.12946](https://arxiv.org/abs/2604.12946)

[7] R. Lee, J. Biloki, E. J. Hu, J. May. *Sparse Layers are Critical to Scaling Looped Language Models*. 2026. [arXiv:2605.09165](https://arxiv.org/abs/2605.09165)

[8] Z. Chen, A. Vegesna, S. Dahal, A. G. Wilson. *How Model Growth, Recursion, and Boundary Operators Influence Scaling Exponents*. 2026. [arXiv:2609.19107](https://arxiv.org/abs/2609.19107)

[9] Y. Chen, A. Goyal, R. Krishnamoorthi. *Scaling Laws for Looped Mixture of Experts*. 2026. [arXiv:2609.40316](https://arxiv.org/abs/2609.40316)

[10] A. Marchenko, V. Bezrukov, O. Kashurin, et al. *Closing the Loop: Practical Training Recipes for Looped Language Models*. 2026. [arXiv:2610.00673](https://arxiv.org/abs/2610.00673)

[11] S. Bae, Y. Kim, R. Bayat, et al. *Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation*. 2025. [arXiv:2507.10524](https://arxiv.org/abs/2507.10524)

[12] J. Logan. *Per-Token Fixed-Point Convergence in Depth-Recurrent Transformers*. 2026. [arXiv:2607.14427](https://arxiv.org/abs/2607.14427)

[13] B. Huang, C. Shi, J. Chen, et al. *Towards Looped Models Done Right, Part II: Rethinking at Fixed Points*. 2026. [arXiv:2610.06833](https://arxiv.org/abs/2610.06833)

[14] N. Saunshi, N. Dikkala, Z. Li, S. Kumar, S. J. Reddi. *Reasoning with Latent Thoughts: On the Power of Looped Transformers*. ICLR 2025. [arXiv:2502.17416](https://arxiv.org/abs/2502.17416)

[15] A. Schwarzschild, E. Borgnia, A. Gupta, et al. *Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks*. 2021. [arXiv:2106.04537](https://arxiv.org/abs/2106.04537)

[16] A. Bansal, A. Schwarzschild, E. Borgnia, et al. *End-to-end Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without Overthinking*. 2022. [arXiv:2202.05826](https://arxiv.org/abs/2202.05826)

[17] Y. Fan, Y. Du, K. Ramchandran, K. Lee. *Looped Transformers for Length Generalization*. ICLR 2025. [arXiv:2409.15647](https://arxiv.org/abs/2409.15647)

[18] H. Kohli, S. Parthasarathy, H. Sun, Y. Yao. *Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers*. COLM 2026. [arXiv:2604.07822](https://arxiv.org/abs/2604.07822)

[19] T. Zhang, J. Hu, Y. Peng, T. Xie. *When Does Recurrence Become an Algorithm? Convergence Selection in Weight-Tied Looped Transformers*. 2026. [arXiv:2607.20594](https://arxiv.org/abs/2607.20594)

[20] B. Wu, M. Chen, X. Luo, et al. *Parallel Loop Transformer for Efficient Test-Time Computation Scaling*. 2025. [arXiv:2510.24824](https://arxiv.org/abs/2510.24824)

[21] G. Monea, K. Ramji, Y. El-Kurdi, et al. *The Surprising Effectiveness of Shared Memory in Looped Transformers*. 2026. [arXiv:2610.02383](https://arxiv.org/abs/2610.02383)

[22] G. Penedo, H. Kydlíček, L. Ben Allal, et al. *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*. 2024. [arXiv:2406.17557](https://arxiv.org/abs/2406.17557)

[23] J. Hoffmann, S. Borgeaud, A. Mensch, et al. *Training Compute-Optimal Large Language Models*. 2022. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)

[24] T. Besiroglu, E. Erdil, M. Barnett, J. You. *Chinchilla Scaling: A replication attempt*. 2024. [arXiv:2404.10102](https://arxiv.org/abs/2404.10102)

[25] T. Porian, M. Wortsman, J. Jitsev, L. Schmidt, Y. Carmon. *Resolving Discrepancies in Compute-Optimal Scaling of Language Models*. NeurIPS 2024. [arXiv:2406.19146](https://arxiv.org/abs/2406.19146)

[26] A. Hägele, E. Bakouch, A. Kosson, L. Ben Allal, L. von Werra, M. Jaggi. *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations*. NeurIPS 2024. [arXiv:2405.18392](https://arxiv.org/abs/2405.18392)

[27] J. Wei, X. Wang, D. Schuurmans, et al. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. 2022. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)

[28] S. Goyal, Z. Ji, A. S. Rawat, A. K. Menon, S. Kumar, V. Nagarajan. *Think before you speak: Training Language Models With Pause Tokens*. ICLR 2024. [arXiv:2310.02226](https://arxiv.org/abs/2310.02226)

[29] S. Hao, S. Sukhbaatar, D. Su, et al. *Training Large Language Models to Reason in a Continuous Latent Space*. COLM 2025. [arXiv:2412.06769](https://arxiv.org/abs/2412.06769)

[30] S. Gülbahar, L. Edman, A. Fraser. *Squeezing More from Limited Data with Recursive Transformers*. 2026. [arXiv:2608.26973](https://arxiv.org/abs/2608.26973)

[31] Y. Zhou, U. Alon, X. Chen, X. Wang, R. Agarwal, D. Zhou. *Transformers Can Achieve Length Generalization But Not Robustly*. 2024. [arXiv:2402.09371](https://arxiv.org/abs/2402.09371)

[32] H. Cho, J. Cha, P. Awasthi, S. Bhojanapalli, A. Gupta, C. Yun. *Position Coupling: Improving Length Generalization of Arithmetic Transformers Using Task Structure*. NeurIPS 2024. [arXiv:2405.20671](https://arxiv.org/abs/2405.20671)

[33] Z. Liu, C. Zhao, F. Iandola, et al. *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases*. ICML 2024. [arXiv:2402.14905](https://arxiv.org/abs/2402.14905)
