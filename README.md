# What a loop buys: parameter efficiency on text, extendable depth on a serial task

[![tests](https://github.com/bethehand/looped-scaling/actions/workflows/tests.yml/badge.svg)](https://github.com/bethehand/looped-scaling/actions/workflows/tests.yml)

**A looped (recurrent-depth) transformer runs one block of layers several times. We measured what that buys, on
models of 10M to 160M parameters, with the hypotheses of the main grid frozen before its first run and every later
plan frozen before its runs. None of the four pre-registered hypotheses held as stated, and the pre-registered
effective-parameter exponent is not identified. Measured without a fitted law (exploratory), a loop on web text buys
parameter efficiency that grows with model size and training data (at 10M–20M with up to 20 training tokens per
parameter, each parameter of the looped model is worth at most about one dense parameter), not compute efficiency
anywhere we measured, and it costs decoding time. On a synthetic task that needs serial steps (chains of table
lookups, 6.6M-parameter models) it buys depth that can be extended at inference: the loop computes the chain step by
step, solves longer chains when run longer, and for most training regimes its own confidence tells it when to stop.**

**Technical report: [English](docs/report.md) · [中文](docs/report.zh.md)** · **[Paper (draft, PDF)](paper/paper.pdf)** ·
[Full results](docs/results.md) · [Pre-registration](docs/preregistration.md) · [Deviation log](docs/deviation-log.md)

> **Status (2026-10-27).** All experiments reported here are complete: 232 language models and 135 small models on the
> synthetic task, every run's results and training logs in this repository; a further supplementary experiment, S10
> ([plan](docs/supplementary-experiments-3.md)), started on 2026-10-27 (results to be added). The report covers all of
> them; the paper draft, prepared for TMLR, is in [paper/](paper/) ([PDF](paper/paper.pdf)). Results marked
> *exploratory* were not part of the pre-registration, and those marked *pre-specified* follow plans frozen before their
> runs; every correction after the two reviews (2026-10-14 and 2026-10-15) is dated in the deviation log. A third review
> (2026-10-27) corrected further errors and overstatements, also dated in the deviation log.

<p align="center">
  <img src="code/results/figures/fig0_overview.png" width="100%"><br>
  <em>A. At equal parameters and tokens the looped model is worse than the dense model at 10M, mixed at 20M, and better at every budget from 40M on. B. The advantage grows with training tokens per parameter. C. On chains of table lookups (trained on 1–8 steps), only a loop run longer at inference solves longer chains; dense models and a written chain of thought stop at the training range.</em>
</p>

## The question

Running a block *r* times costs *r* times that block's compute and adds no parameters. Two things are expected in
return: that the model behaves like a larger one (parameter efficiency), and that more loops at inference solve harder
problems without a written chain of thought (latent reasoning). The study measures both under one recipe and asks what
the loop buys, when it becomes reasoning, and what it costs.

## What we found

Losses are in nats per token on held-out web text; a difference of 0.01 nats between runs that share a seed is small
but resolvable at these sizes, one between unpaired runs is not.
φ, the dense "ruler" and the labels S1–S9 are explained in [the study](#the-study) below and in the report's
[reading guide](docs/report.md#how-to-read-this-report).

### On text, a loop buys parameter and data efficiency, not compute efficiency

1. **Looping wins at equal parameters from about 40M on, and loses at equal training compute.** At equal parameters
   and tokens, a middle block looped 4 times goes from 0.02–0.10 nats *worse* than the dense model at 10M to 0.03–0.06
   nats *better* at 160M, and at 10M–20M the advantage grows with every doubling of tokens per parameter, to 160. At
   equal training compute and the shared learning rate the dense model wins all 258 measured comparisons, up to 640
   tokens per parameter; the margin shrinks with data, and with the lowest learning rate tried for the looped model the
   two are level on average at 10M and 160 tokens per parameter (−0.011 and +0.005 in two seeds). *Exploratory; S1 and
   S4 pre-specified.*
2. **The pre-registered number φ is not identified in absolute terms** (0–0.17 or 0–0.41 depending on the fit), because
   the dense "ruler" is weakly identified. Measured without a ruler, each parameter of a model whose middle block is
   looped 4 times is worth 0.6–1.0 dense parameters at 10–20M with 10–20 tokens per parameter and 1.26–1.46 at
   40M–160M with 20–40, in line with most published equal-parameter results restated in the same unit (1.2–1.6 at
   about 130M–770M). φ is not a constant of the architecture. *Pre-registered fit; the values without a ruler are
   exploratory.*
3. **None of the four pre-registered hypotheses held as stated** (listed under [the study](#the-study)).
   *Pre-registered.*
4. **On text, the loop does not iterate towards an answer**: depending on the training regime it runs a fixed-length
   program (it works only at its training loop count), a periodic cycle, or a fixed point (robust to the loop count but
   no better with more), and never gains from extra loops at inference. *Exploratory.*

### On a task that needs serial steps, it buys depth that can be extended at inference

5. **Step by step inside the loop, longer chains with more loops, and a stop that needs no answer.** A model trained
   with 8 loops computes one chain step per loop and goes from 0.04 to 0.83 on 12-step chains when run with 16 loops
   instead of 8; dense models of the same executed depth with up to 4.4× the parameters fail right after the training
   range. Stopping when the model's own confidence reaches 0.99 recovers 97–98% of the gain of the best fixed loop count
   for the models trained with 4 or 8 loops or a random loop count (with 2 loops it does better than the best fixed
   count). The model that extends furthest, trained with as many loops as steps, cannot stop by its state; trained
   with a few random extra loops it both extends and stops by its confidence (0.93 on 12-step chains). *Pre-specified
   (synthetic design, S8, S9; the S8 plan was written after the frozen stopping rule's results on these checkpoints
   were known); the probe summaries are exploratory.*
6. **A written chain of thought did not do this here**: perfect within the training range at under a third of the compute,
   near chance level on every longer chain, because it fails to generalise to the longer prompts. Pause tokens add nothing.
   *Pre-specified (S5).*

### What it costs

7. **At deployment, loops cost latency and memory**: at equal loss, at most 19% less weight memory (4% at 20M, 14–19%
   at 40M–160M) but 2.2–2.4× the KV cache and 2.4–2.5× the decoding time per token. Sharing the cache across loops
   costs the fixed-loop models 1.8 to 3.7 times their gain. *Pre-specified (S3, S6).*

Underneath these findings is one question: when does running a block again work as test-time compute? Our reading
of the two settings, an interpretation rather than a tested law, is that three conditions separate them: headroom
(the errors come from too few serial steps, not from missing knowledge), extendable iteration (the loop can run longer
without destroying its state), and a stopping signal. The chain task has all three once the model is trained for them;
web text, at these sizes, shows no sign of headroom, and its one extendable regime, the fixed point, gains nothing
from running longer.

The [report](docs/report.md) gives the evidence for each; [results](docs/results.md) has every number.

## For practitioners

| If you want to… | What we found |
|---|---|
| **Save parameters** | Loops help, and more so at larger sizes and with more training tokens per parameter: 0.02–0.10 nats worse than the dense model at 10M, 0.03–0.06 nats better at 160M (report Figure 12), and 0.02–0.03 nats more per doubling of tokens per parameter (report Figure 11). Holds with re-tuned learning rates (S1). |
| **Save training compute** | Don't loop, within the range we measured. At equal training FLOPs and the shared learning rate the dense model won all 258 comparisons in which its loss was measured, up to 640 tokens per parameter at 10M and 320 at 20M (S4); the margin falls to 0.01–0.05 nats at the largest looped budgets, and with the lowest learning rate tried for the looped model the two are level on average at 10M and 160 tokens per parameter (−0.011 and +0.005 in two seeds). A middle block looped 2, 4, 8 times costs 1.4–1.5×, 2.2–2.4×, 3.7–4.1× the training FLOPs per token of the same-size dense model. |
| **Save memory or latency at deployment** | Not at these sizes. At equal loss, a middle block looped 4 times needs at most 19% less weight memory (4% at 20M, 14–19% at 40M–160M) but 2.2–2.4× the KV cache and 2.4–2.5× the decoding time per token, and 1.6× the peak memory at batch 32 (S3). Sharing the KV cache across loops is free only for random-loop models, whose loops gain nothing; fixed-loop models lose 0.09–0.17 nats, 1.8 to 3.7 times their gain (S6). |
| **Get better answers by running more loops at test time, on text** | Standard training does not give you this. Fixed-loop models work only at their training loop count; random-loop training (like Huginn, with a different distribution) makes them robust to the count but not better with more loops, and at 20M it erased the advantage over the dense model. |
| **Get step-by-step reasoning inside the network** | Possible, when the task needs serial steps and the model is trained for it. On chains of table lookups, looped models compute the chain step by step and solve longer chains than trained on when given more loops; dense models fail right after the training range even at the same depth with up to 4.4× the parameters (report Figures 14, 15). The learning rate decides whether this is learned within the training budget (S2). |
| **Let the model decide when to stop** | Stop when its top probability reaches 0.99: this recovers 97–98% of the extra-loop gain for models trained with 4 or 8 loops or a random loop count (with 2 loops it beats the best fixed count), using fewer loops than in training for in-range problems (S8). The model trained with loops = steps extends furthest but cannot stop by its state; trained with 0–4 random extra loops it keeps its values and stops by its confidence at 0.93 on 12-step chains, one step of extrapolation less (S9). |
| **Choose between loops and a chain of thought** | By the lengths you need. Within the training range a chain of thought was perfect at under a third of the inference compute; beyond it only the loop extended, on this task (S5). |

### A recipe

1. **Loop a middle block** and keep the first and last layers unshared (2 + 4×*r* + 2 layers in our models). At
   similar executed depth (36 against 32 and 20 against 16 layers), middle-block loops beat whole-stack loops in 24 of
   24 comparisons.
2. **Use full backpropagation through the loops.** Truncated backprop accounted for 25 of the 30 runs with loss
   spikes (16 of them whole-stack loops); halving the learning rate removed the spikes in a 10M follow-up.
3. **Start with r = 4 on text, and judge at your target scale.** At 40M–80M, r = 8 did about as well as r = 4 while
   costing about 1.75× the training compute and 1.8× the decoding time and KV cache. A loop is worth more at 160M than at 10M and more with more tokens per parameter, so a
   small-scale ablation will undersell it.
4. **Tune the learning rate of the looped model separately, especially at small scale.** On text the shared rate
   understated the looped model's advantage by at most 0.005 nats at 20M–40M, but at 10M the looped model wanted
   0.35–0.5× the dense rate on longer runs (worth about 0.02 nats). On the reasoning task the best looped rate was 4–8× lower than that of the 8-layer dense model (base rates; the hidden layers are scaled with the number of unique layers; for r = 4, the random loop count and the S9 model the lowest rate tried was best), and at
   the dense-optimal rate 10 of 12 looped runs did not learn the task in the same number of steps (S2).
5. **Check whether the model can use more loops** by sweeping the loop count at inference. On text none of ours
   could; on the chain task, models whose accuracy held at more loops also solved longer chains with them. Probes show
   whether the loop computes step by step even when the model works only at its training loop count.
6. **Count the KV cache and the latency, not just the parameters.** With one cache per executed layer, decoding time
   and KV cache grow with the executed layers (1.5×, 2.5×, 4.5× for a middle block looped 2, 4, 8 times). Loop to save
   memory only where weights dominate it, e.g. one sequence at a time with a short context in a larger model.

## The study

| | |
|---|---|
| Models | looped models at 10M, 20M, 40M and 80M (8 unique layers; a middle block or the whole stack looped r = 2, 4, 8 times; full or truncated backprop), dense models up to 160M; extensions: 160M looped, data up to 160 tokens per parameter (640 for the dense model in S4), random loop counts |
| Data | every run reads a prefix of one fixed-order stream, files 000–012 of FineWeb-Edu sample-10BT (10.4B tokens with our tokenizer; the longest run uses 9.9B), so no document is seen twice; 16k BPE; validation on held-out FineWeb-Edu (file 013), SlimPajama, FineMath and code |
| Training | AdamW, WSD schedule with cooldown branches at 10, 20 and 40 tokens per parameter; 3 seeds at 10M–40M, 1 at 80M, 2 at 160M |
| Measurement | effective size N_eff = N_once + r^φ · N_rec (the form of Schwethelm et al., 2026), read off a dense "ruler" fit; also ruler-free loss gaps, dense sizes of equal loss, and equal-compute comparisons on the raw data |
| Synthetic task | chains of 1–8 table lookups (32 fixed random tables), tested to 24 steps; dense models of 8, 20 and 36 layers and 8-layer looped models with r = 2, 4, 8, a random loop count, loops = steps, and loops = steps plus random extra loops; chain-of-thought and pause-token baselines; 5 seeds per condition ([design](docs/synthetic-reasoning-design.md), frozen 2026-10-09) |
| Pre-registration | frozen on 2026-09-23 before any training run, hash recorded in the history ([how to verify](docs/preregistration.md)); every later change is dated in the [deviation log](docs/deviation-log.md) |
| Supplementary experiments | S1 learning rates re-tuned at 10M–40M; S2 the learning rate on the synthetic task; S3 memory and decoding latency ([plan](docs/supplementary-experiments.md), frozen 2026-10-13; [results §5.16](docs/results.md)). S4 dense models to 640 tokens per parameter; S5 chain of thought and pause tokens; S6 KV-cache sharing across loops; S7 seeds that change the data order; S8 stopping rules and voting; S9 loops = steps plus random extra loops ([plan](docs/supplementary-experiments-2.md), frozen 2026-10-15; [results §5.18](docs/results.md)). All outside the pre-registration; each plan, with its readings, was frozen before its runs (pre-specified). |
| Compute | 4× RTX 4090; main grid 11 days, extensions about 5, supplementary experiments about 3 |

**Pre-registered verdicts.** None of the four hypotheses held as stated. Truncated backprop did not lower φ (it
helped whole-stack loops with 4 loops at every size; with 8 it was slightly worse at 10M–20M); a looped middle block
beat the whole stack on raw losses but not consistently on φ; a form in
which φ saturates in *r* did not extrapolate better than r^φ (the data cannot tell the two apart); and under both
training and deployment compute the dense model was optimal. The pre-registered φ itself is not identified in
absolute terms (0–0.17 in the primary fit, 0–0.41 in a joint fit) because the dense ruler is weakly identified, so the
findings above rest on quantities that need no ruler ([results §5.17](docs/results.md)).

## How this relates to other work

- The effective-parameter form N_eff = N_once + r^φ · N_rec and the choice of two unshared layers before and after the
  loop follow Schwethelm et al. ([arXiv:2604.21106](https://arxiv.org/abs/2604.21106)); the prelude–core–coda layout
  itself, with its (2, 4, 2) split and an input injection that maps the concatenated state and prelude output, is that
  of Huginn (Geiping et al., [arXiv:2502.05171](https://arxiv.org/abs/2502.05171)). This study was designed to be
  comparable with Schwethelm et al. and adds seeds, separate variation of model size and tokens per parameter,
  truncated backprop separated into equal-token and equal-FLOP accountings (they found φ lower with truncation, 0.38
  against 0.45, in a comparison that mixes the two), and whole-stack loops; here truncation did not lower φ (H1 not
  supported). Measured without a ruler, our values at 40M–160M agree with their φ = 0.46 and with most other published equal-parameter results
  restated in the same unit (1.2–1.6 dense parameters per parameter in comparable settings, up to about 2 for small
  blocks trained on far more tokens per parameter); below about 40M and 20 tokens per parameter each parameter of the
  looped model is worth at most about one dense parameter, a regime the published studies barely sample.
- That weight-shared transformers do slightly better at equal parameters and slightly worse at equal FLOPs was already
  reported by Kaplan et al. (2020, App. D.2, [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)). What this study adds
  is how the balance moves with size and data: at the most data per parameter we could afford (10M, 160 tokens per
  parameter) the dense model still wins at the shared learning rate (+0.016), and with the lowest learning rate tried
  for the looped model the two are level on average over two seeds (−0.011 and +0.005), not a looped win.
- On serial reasoning tasks, solving longer chains with more loops at test time was shown by Kohli et al.
  ([arXiv:2604.07822](https://arxiv.org/abs/2604.07822)) and Zhang et al. ([arXiv:2607.20594](https://arxiv.org/abs/2607.20594));
  our synthetic results agree with both and add probes of every intermediate value, a controlled comparison of
  answer-free stopping rules, and the comparison with a chain of thought.
- Related work in full, with references: [report §1](docs/report.md#related-work).

## Documents

| | |
|---|---|
| [docs/report.md](docs/report.md) · [中文](docs/report.zh.md) | **Technical report**: the whole study in one document, with a guide to reading it and practical takeaways |
| [docs/results.md](docs/results.md) | Full results: pre-registered fits and verdicts, robustness, every exploratory analysis and all nine supplementary experiments |
| [docs/preregistration.md](docs/preregistration.md) | Pre-registration v1.0 (English translation of the frozen original) and how to verify its hash |
| [docs/deviation-log.md](docs/deviation-log.md) | Every change to the plan, dated and append-only; frozen hyper-parameters, data and throughput |
| [docs/synthetic-reasoning-design.md](docs/synthetic-reasoning-design.md) | Design of the synthetic reasoning task and its predictions (frozen 2026-10-09) |
| [docs/supplementary-experiments.md](docs/supplementary-experiments.md), [-2.md](docs/supplementary-experiments-2.md) | The two supplementary plans, S1–S3 and S4–S9, frozen before their runs |
| [docs/roadmap.md](docs/roadmap.md) | Project phases and status |
| [docs/paper-plan.md](docs/paper-plan.md) | Paper plan: positioning, title candidates, claims ledger, outline, figures, schedule |
| [paper/](paper/) · [PDF](paper/paper.pdf) | **Paper draft** (TMLR template): LaTeX sources, figure script, bibliography |

## Repository layout

```
docs/                project documents (above)
paper/               the paper draft: LaTeX sources, figures and the script that makes them, PDF
code/
  looped/            model (prelude / looped core / coda, input injection, truncated backprop), FLOP counting,
                     data stream, WSD schedule, training loop with cooldown branches, evaluation, KV-cached inference
  scripts/           data preparation, config generation, run queue, throughput and inference benchmarks,
                     result export, loss-spike statistics, KV-sharing and chain-of-thought diagnostics
  fit/               scaling-law fits, H4 calculation, per-rung φ, ruler-free multipliers, equal-compute bounds,
                     per-token analysis, per-loop probes, supplementary analyses, figures
  synthetic/         synthetic reasoning task, trainer, multi-GPU suite runner, probes, stopping rules, analysis
  configs/           every run config and manifest (sweeps, main grid, extensions, supplementary experiments)
  results/           per-run results and training logs, collected tables, fits, synthetic and supplementary
                     outputs, figures
  tests/             unit and smoke tests (135)
```

## Reproducing

```
cd code
uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e ".[test]"
python -m pytest tests -q
```

Training ran on 4× RTX 4090 with PyTorch 2.6 (CUDA 12.4), one run per GPU; the main grid took 11 days. Fits and
figures run on a laptop CPU. [code/README.md](code/README.md) gives the full command sequence and maps every table and
figure to the command that makes it. Every run's results are in `code/results/runs/`, so the fits, tables and figures
can be rebuilt without retraining; the per-run training logs are in `code/results/train_logs/`. The analyses that read
the model checkpoints (485 GB, not included) or need a GPU (probes, per-token analyses, KV sharing, stopping rules, the
chain-of-thought diagnostic, timings) cannot be re-run from the repository; their outputs are committed, so the
numbers taken from them can be recomputed.

## Citation and license

The paper is a draft ([paper/](paper/)). Until it is published, please cite the report (also in
[CITATION.cff](CITATION.cff)):

```
Qirun Li. What a loop buys: parameter efficiency on text, extendable depth on a serial task. Technical report, 2026.
https://github.com/bethehand/looped-scaling
```

The code is released under the MIT license (see [LICENSE](LICENSE)). The code, analyses, documents and text were
drafted with an AI assistant (Claude, by Anthropic) under the author's direction; the author is responsible for every
decision and for the content. Every number can be recomputed from the results in the repository.
