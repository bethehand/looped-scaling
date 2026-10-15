# How much is a loop worth?

**A pre-registered scaling study of looped (recurrent-depth) transformers from 10M to 160M parameters**

**Technical report: [English](docs/report.md) · [中文](docs/report.zh.md)**

> **Status (2026-10-15).** All experiments are complete: 111 pre-registered training runs plus extensions and three
> supplementary experiments (frozen before they ran), about 225 language models in total and about 100 small models on
> a synthetic reasoning task. The technical report is a draft and a paper is in preparation. Results marked
> *exploratory* were not part of the pre-registration. Corrections after two reviews (2026-10-14 and 2026-10-15) are
> dated in the [deviation log](docs/deviation-log.md).

A looped transformer runs one block of layers *r* times. It pays *r* times the compute of that block and adds no
parameters. We trained models of 10M to 160M parameters under a single recipe to measure what the extra compute buys,
and when it is worth paying.

## Key findings for practitioners

| If you want to… | What we found |
|---|---|
| **Save parameters** | Loops help, and more so at larger sizes. At equal parameters and training tokens, a middle block looped 4 times goes from 0.02–0.10 nats *worse* than a dense model at 10M to 0.03–0.06 nats *better* at 160M (at 10–40 tokens per parameter), and the gain grows at every step up in size (Figure 12). This still holds at 10M–40M when both models' learning rates are re-tuned ([S1](docs/results.md)). |
| **Save memory or latency at deployment** | Not at these sizes. Every loop runs the block again and stores its keys and values again. At equal loss, a middle block looped 4 times needs 14–19% less weight memory at 40M–160M, but 2.2–2.3× the KV cache and 2.4–2.5× the decoding time per token, and its peak memory at batch 32 is about 1.6× that of the dense model ([S3](docs/results.md)). |
| **Get more out of limited data** | Loops help more the longer you train: each doubling of tokens per parameter adds 0.02–0.03 nats. At 160 tokens per parameter a 10M looped model matches a dense model trained on 1.7× as much data (Figure 11). With re-tuned learning rates the trend holds and is larger (S1). |
| **Save training compute** | Don't loop, at least within the range we measured. At equal training FLOPs the dense model won all 221 comparisons in which its loss was measured, by at least 0.016 nats. In 99 more its loss had to be extrapolated, in its favour, and at the largest budgets the margin is within that error. After re-tuning both models' learning rates it again won all 11 comparisons (S1). A middle block looped r = 2, 4, 8 times costs 1.4–1.5×, 2.2–2.4×, 3.7–4.1× the training FLOPs per token of the same-size dense model. |
| **Get better answers by running more loops at test time, on text** | Standard training does not give you this. Models trained with a fixed loop count only work at that count. Training with a random loop count (as in Huginn) makes them robust to the count, but not better with more loops, and at 20M it erased the advantage over the dense model. |
| **Get step-by-step reasoning inside the network** | Possible, when the task needs serial steps and the model is trained for it. On chains of table lookups, looped models learned to compute one step per loop and solved longer chains than they were trained on when given more loops. Dense models failed right after the training range, even at the same depth with 4.4× the parameters, and so did the looped models run at their training loop count: the gain comes from running the loop longer (Figures 14, 15). We did not compare with writing the steps out as a chain of thought. |

### A recipe

1. **Loop a middle block** and keep the first and last layers unshared (2 + 4×*r* + 2 layers in our models). At
   matched executed depth, middle-block loops beat whole-stack loops in 24 of 24 comparisons.
2. **Use full backpropagation through the loops.** Truncated backprop accounted for 25 of the 30 runs with loss
   spikes (16 of them whole-stack loops); halving the learning rate removed the spikes in a 10M follow-up.
3. **Start with r = 4 on text, and judge at your target scale.** At 40M–80M, r = 8 did about as well as r = 4 while
   costing about 1.7× more; extra loops pay off on tasks that need more serial steps. A loop is worth more at 160M than
   at 10M and more with more tokens per parameter, so a small-scale ablation will undersell it.
4. **Tune the learning rate of the looped model separately, especially at small scale.** On text the dense rate cost
   the looped model at most 0.006 nats at 20M–40M, but at 10M the looped model wanted 0.35–0.5× the dense rate on
   longer runs (worth 0.02 nats), while the dense model's best stayed at 1×. On the reasoning task the best looped
   rate was 2–8× lower than the dense one. At the dense-optimal rate, 10 of 12 looped runs did not learn the task in
   the same number of steps; at half that rate 8 of 12 did, and those mostly extended about as far as the models
   trained at the low rate ([S2](docs/results.md)).
5. **Check whether the model can use more loops** by sweeping the loop count at inference. On text none of ours
   could; on the chain task, models whose accuracy held at more loops also solved longer chains with them. Probes show
   whether the loop computes step by step even when the model works only at its training loop count.
6. **Count the KV cache and the latency, not just the parameters.** With one cache per executed layer, decoding time
   and KV cache grow with the executed layers (1.5×, 2.5×, 4.5× for a middle block looped 2, 4, 8 times). So at equal
   loss a looped model decodes more slowly. Its peak memory at batch 1 was about the same as the dense model's only for
   a middle block looped 2 or 4 times at 80M–160M (1–2% lower) and higher in the other cells, and it was higher with
   more sequences (1.6× at batch 32) ([S3](docs/results.md)). Loop to save
   memory only where weights dominate it, e.g. one sequence at a time with a short context in a larger model. We did
   not test schemes that share the cache across loops.

## Results at a glance

<p align="center">
  <img src="code/results/figures/fig12_scale_middle_r4.png" width="55%"><br>
  <em>The same looped cell against the dense model of equal size: worse at 10M, about even at 20M, better from 40M on, and the gap keeps growing to 160M.</em>
</p>

<p align="center">
  <img src="code/results/figures/fig14_synthetic_inference_loops.png" width="95%"><br>
  <em>On a step-by-step task, more loops at inference solve longer chains than seen in training (trained on 1–8 steps); a dense model of the same depth does not.</em>
</p>

**On text, the loop does not reason.** Probing trained language models loop by loop, a fixed-loop model runs a
fixed-length program whose last loop rewrites the state for the output layers; truncated backprop learns a periodic
cycle; random-loop training converges to a fixed point. None gains from extra loops at inference, because predicting
web text rarely needs many serial steps.

**On a task that needs serial steps, it does.** Mean over 5 seeds (min–max):

| Model | Params | Longest chain solved* | With more loops at inference | Accuracy at 12 steps: own loops / more loops** |
|---|---|---|---|---|
| Dense, 8 layers | 6.5M | 4.2 (4–5) | — | 0.06 |
| Dense, 36 layers | 29.0M | 8.0 (8–8) | — | 0.02 |
| Looped, middle block × 8 loops (36 layers executed) | 6.6M | 9.0 (9–9) | 11.2 (11–12) | 0.04 / 0.86 |
| Looped, random loop count in training | 6.6M | 11.4 (10–13) | 12.2 (10–15) | 0.77 / 0.83 |
| Looped, as many loops as steps in training | 6.6M | 12.8 (11–16) | — | 0.96 |

\* every chain length up to this one solved at ≥ 90% accuracy, at the model's own loop count; trained on 1–8 steps.
\*\* at the model's own loop count / the best of the larger inference loop counts.

## The study

| | |
|---|---|
| Models | looped models at 10M, 20M, 40M and 80M (8 unique layers; a middle block or the whole stack looped r = 2, 4, 8 times; full or truncated backprop), dense models up to 160M; extensions: 160M looped, data up to 160 tokens per parameter, random loop counts, synthetic tasks |
| Data | FineWeb-Edu sample-10BT (10.4B tokens, one epoch), 16k BPE; validation on held-out FineWeb-Edu, SlimPajama, FineMath and code |
| Training | AdamW, WSD schedule with cooldown branches at 10, 20 and 40 tokens per parameter; 3 seeds at 10M–40M |
| Measurement | effective size N_eff = N_once + r^φ · N_rec (the form of Schwethelm et al., 2026), read off a dense "ruler" fit; also ruler-free loss gaps and equal-compute comparisons |
| Pre-registration | frozen on 2026-09-23 before any training run, hash recorded in the history ([how to verify](docs/preregistration.md)); every later change is dated in the [deviation log](docs/deviation-log.md) |
| Supplementary experiments | S1: learning rates of the dense and the looped model re-tuned at 10M–40M; S2: the learning rate on the synthetic task, 3 seeds; S3: memory and decoding latency. Exploratory; [plan](docs/supplementary-experiments.md) frozen on 2026-10-13 before any of their runs; results in [results §5.16](docs/results.md) |

**Pre-registered verdicts.** None of the four hypotheses held as stated. Truncated backprop did not lower φ (it
helped whole-stack loops); a looped middle block beat the whole stack on raw losses but not consistently on φ; a form in
which φ saturates in *r* did not extrapolate better than r^φ (the data cannot tell the two apart); and under both training and deployment compute the dense model was optimal. The pre-registered
φ itself is not identified in absolute terms (0–0.17 in the primary fit, 0–0.41 in a joint fit) because the dense
ruler is weakly identified, so the findings above rest on quantities that need no ruler. Details: [report](docs/report.md),
[full results](docs/results.md).

## How this relates to other work

- The effective-parameter form N_eff = N_once + r^φ · N_rec and the layout with unshared layers before and after the
  loop follow Schwethelm et al. ([arXiv:2604.21106](https://arxiv.org/abs/2604.21106)). This study was designed to be
  comparable with theirs and adds seeds, separate variation of model size and tokens per parameter, truncated backprop
  under two accountings, and whole-stack loops.
- That weight-shared transformers do slightly better at equal parameters and slightly worse at equal FLOPs was already
  reported by Kaplan et al. (2020, App. D.2, [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)). What this study adds
  is how the balance moves with size and data.
- On serial reasoning tasks, solving longer chains with more loops at test time was shown by Kohli et al.
  ([arXiv:2604.07822](https://arxiv.org/abs/2604.07822)) and Zhang et al. ([arXiv:2607.20594](https://arxiv.org/abs/2607.20594));
  our synthetic results agree with both.
- Related work in full, with references: [report §1](docs/report.md#related-work).

## Documents

| | |
|---|---|
| [docs/report.md](docs/report.md) · [中文](docs/report.zh.md) | **Technical report**: the whole study in one readable document, with practical takeaways |
| [docs/results.md](docs/results.md) | Full results: pre-registered fits and verdicts, robustness, and every exploratory analysis |
| [docs/preregistration.md](docs/preregistration.md) | Pre-registration v1.0 (English translation of the frozen original) and how to verify its hash |
| [docs/deviation-log.md](docs/deviation-log.md) | Every change to the plan, dated and append-only; frozen hyper-parameters, data and throughput |
| [docs/synthetic-reasoning-design.md](docs/synthetic-reasoning-design.md) | Design of the synthetic reasoning tasks (frozen 2026-10-09) |
| [docs/roadmap.md](docs/roadmap.md) | Project phases and status |

## Repository layout

```
docs/                project documents (above)
code/
  looped/            model (prelude / looped core / coda, input injection, truncated backprop), FLOP counting,
                     data stream, WSD schedule, training loop with cooldown branches, evaluation
  scripts/           data preparation, config generation, run queue, throughput measurement, result export,
                     loss-spike statistics
  fit/               scaling-law fits, H4 calculation, per-rung φ, per-token analysis, per-loop probes, figures
  synthetic/         synthetic reasoning tasks, trainer, multi-GPU suite runner, intermediate-value probes, analysis
  configs/           every run config and manifest (sweeps, main grid, extensions)
  results/           per-run results, collected tables, fits, probe outputs, figures
  tests/             unit and smoke tests
```

## Reproducing

```
cd code
uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e .
python -m pytest tests -q
```

Training ran on 4× RTX 4090 with PyTorch 2.6 (CUDA 12.4), one run per GPU; the main grid took 11 days. Fits and
figures run on a laptop CPU. [code/README.md](code/README.md) gives the full command sequence and maps every table and
figure to the command that makes it. Every run's results are in `code/results/runs/`, so every number can be
recomputed without retraining; the per-run training logs will be added in `code/results/train_logs/`.

## Citation and license

A paper is in preparation; citation details will be added here. The code is released under the MIT license
(see [LICENSE](LICENSE)).
