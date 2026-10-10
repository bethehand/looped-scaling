# Pre-registration v1.0 (English translation)

**Pre-registration document: the loop-equivalence exponent φ of looped Transformers and its design factors**

> **About this document.** This is an English translation, made on 2026-10-11, of the pre-registration
> `01_预注册文档.md`, which is written in Chinese and is the binding document; if the two differ, the original
> prevails. The original was frozen as version 1.0 on 2026-09-23, before any training run. Its SHA-256,
> `1b01a66f19cd188b8fadf0cb517b91ae89c9fed1eb0f02f0a369f6d5c10a2430` (computed 2026-09-23 16:40 CST), was first
> committed in `d2c6023` (2026-09-23 23:01 +0800). To verify:
>
> ```
> git show d2c6023:01_预注册文档.v1.0.sha256               # the recorded hash
> git show 1564898:01_预注册文档.md | shasum -a 256         # the frozen text: 1b01a66f...
> git log --oneline -- 01_预注册文档.md                      # every commit that touched the file
> ```
>
> Before the deviation log became a separate file, entries were appended to section 6 of this document, as that
> section allows: commit `d2c6023` carries two entries dated 2026-09-23 and `d14da22` adds a third dated
> 2026-09-24. Commit `1564898` (2026-09-24) moved all three to the [deviation log](deviation-log.md), restoring the
> frozen bytes, and the text has not changed since (`git diff d2c6023 1564898 -- 01_预注册文档.md` shows only the
> removed entries). The Chinese
> original left the working tree when the repository was converted to English on 2026-10-11 and remains in the
> history. Deviations from this plan are listed in the [deviation log](deviation-log.md); results are in
> [results.md](results.md).

Version: **v1.0, frozen**, 2026-09-23. Decisions confirmed by the author: 160M looped cells postponed; one 160M dense-ruler point kept; 10M rung added; truncation only at r=4/8; no multi-epoch branches; budget about 47 GPU-days. From this point on the main text is no longer modified; any change may only be appended to Section 6, "Deviation log".
Basis: our internal notes on 9 papers; each decision is marked with its source numbers [01]–[09].

---

## 1. Research questions

**Core quantity**: the loop-equivalence exponent φ, defined as the φ in L = E + A·(N_once + r^φ·N_rec)^(−α) + B·D^(−β), i.e. "how many added independent parameters one more loop of the looped core is equivalent to". With fixed r, fully unrolled backprop and r∈{1,2,4,8}, iso-depth measured φ = 0.459, 95% interval [0.41, 0.53] [01].

- RQ1 Functional form: which is more accurate in leave-one-rung-out extrapolation, the r^φ form or the geometric-saturation form N_eff = N_once + N_rec·(1−ρ^r)/(1−ρ)?
- RQ2 Backprop truncation: with backprop only through the last k=⌈r/2⌉ loops, how much does φ decrease? How much under "iso-token" accounting and under "iso-training-FLOP" accounting, respectively? (iso-depth's 0.38 mixes the two together [01])
- RQ3 Loop placement: does φ differ between looping the middle block (2+4+2) and looping the whole stack (all 8 layers looped)?
- RQ4 Interaction: does the effect of truncation depend on loop placement?
- RQ5 Optimal loop count: what is the fitted optimal r under training-FLOP-matched accounting and under deployment-FLOP-matched accounting, respectively?
- RQ6 Scale trend: do the factor effects grow or shrink within 10M→80M?

**Explicitly out of scope**: MoE; sampling r during training; random initial state; the multi-epoch / data-constrained regime (φ depends on the training regime [05]; this study draws conclusions only for a single epoch); looped cells at 160M and above (postponed because of compute limits; the dense ruler keeps one 160M point; if they are added later, they are run under the same protocol as this document and recorded in the deviation log); hyper-connections; conclusive statements about downstream benchmarks.

## 2. Pre-registered hypotheses and decision rules

| Hypothesis | Content | Decision rule for support |
|---|---|---|
| H1 | Truncated backprop lowers φ | φ_trunc < φ_full, and the 95% bootstrap intervals of the two do not overlap; it counts as supported only if this **holds under both iso-token and iso-FLOP accounting**. If it holds only under iso-token accounting, the conclusion is "the effect comes from FLOP reallocation". |
| H2 | Looping the middle block gives a higher φ than looping the whole stack | φ_mid > φ_whole, with non-overlapping 95% intervals; also report the raw loss pairs with similar executed depth (middle block r=8 vs whole stack r=4; middle block r=4 vs whole stack r=2); if the two views disagree, report this truthfully as "depends on the view". |
| H3 | φ saturates with r; the geometric-saturation form is better than r^φ | The geometric form has lower error in leave-one-rung-out extrapolation in both directions (10–40M predicting 80M; 20–80M predicting 10M), and the AIC difference is > 10. |
| H4 | With training FLOP matched, the optimal r ≤ 2; with deployment FLOP matched, the optimal r is larger | Computed from the fitted formula and reported directly; no hypothesis test. |

**Minimum threshold for a significant effect**: the effect of a factor on φ must be larger than both the bootstrap interval half-width and the spread of φ measured with three seeds at 20M/40M; otherwise it is reported as "indistinguishable". SMELT's placement effect is only 0.006–0.008 nats and single-seed [03]; this study does not accept single-seed conclusions about placement.

## 3. Models and grid

**Common settings**: 8 unique layers; width 320 / 448 / 640 / 896 (about 10M / 19M / 39M / 77M non-embedding parameters; exact values to be filled in after implementation), plus one extra point at width 1280 (about 157M) for the dense ruler; RMSNorm pre-norm; SwiGLU; RoPE; no biases; input and output embeddings untied; vocabulary 16k (self-trained SentencePiece BPE); sequence length 1024; bf16.

**Parameter counting** [08][09]: N = non-embedding parameters + output head; the input embedding is recorded separately and not counted. N_once = prelude + coda + output head; N_rec = looped core + injection adapter (a 2d² linear layer, following iso-depth [01]). In the whole-stack scheme, N_once = output head.

**Loop mechanism**: in each loop, [s, e] is concatenated and projected back to d dimensions by a linear layer (e is the prelude output; in the whole-stack scheme, the embedding output); initial state s₀ = 0; fixed r is used in both training and evaluation.

**Dense ruler**: a dense 8-layer model with r = 1, without the injection adapter (it differs from the looped model by 2d², < 2% of N; this is recorded).

**Residual scaling and learning rate** [06]: the output of each attention/MLP branch is multiplied by ε; for looped-core branches ε = (1/r)·(L/12)^(−1/2), for non-looped branches ε = (L/12)^(−1/2), L = 8. The hidden-layer learning rate is η = η₀(d)·(L/12)^(−1/2) and **does not change with r**; the embedding and output-head learning rate = η₀(d). η₀(d) is swept separately on the dense ruler at each width (5 points, 2× spacing, 10N token budget), because [06] only verified transfer across r and across the number of layers, not across width [06][09]; a ±2× cross-check is done once at 80M, r=4.

**Optimizer and schedule** [06][07][09]: AdamW(0.9, 0.95), weight decay 0.1, gradient clipping 1.0; warmup = 2N tokens; WSD: constant learning rate on the trunk, cooldown branches split off at specified token counts, linear decay to 0, **cooldown length = 20% of the branch point**; tokens per step ∝ N^(2/3): 0.125M / 0.2M / 0.32M / 0.5M. For β₂, 0.99 is also tested during the 20M sweep; if it is better, 20M/40M use 0.99.

**Data**: training uses FineWeb-Edu sample-100BT (the same source as the iso-depth, Parcae, Sparse Layers and residual-scaling papers [01][02][04][06]), with a fixed shard order, the same order for all jobs, single epoch; the main validation set is 50M held-out FineWeb-Edu tokens, fixed and unchanged; the second validation set is 20M held-out tokens from another public web corpus (SlimPajama or DCLM-baseline), used to test whether φ depends on the training distribution [05][07]. **Exploratory validation sets** (not used in the φ fits, not entered into the hypothesis tests, results listed separately as exploratory): 20M held-out tokens of math text (FineMath or OpenWebMath) and 20M held-out tokens of code (the-stack), used only to report "whether the loss reduction brought by looping differs by domain", corresponding to the observations that SMELT's gain is largest on code [03] and that Huginn benefits more on math tasks [01 related work]. Vocabulary 16k self-trained BPE (smaller than the 32k–128k commonly used in the literature, so that the output head does not dominate the parameter count at the 10M/20M rungs; absolute loss values are therefore not directly comparable with other papers, while φ, a relative quantity within the study, is unaffected).

**Main grid**

| Dimension | Values |
|---|---|
| Rungs | Looped cells: 10M, 20M, 40M, 80M; dense ruler: 10M, 20M, 40M, 80M, 160M |
| r | 1 (ruler), 2, 4, 8 |
| Loop placement | Middle block 2+4+2; whole stack 8 |
| Backprop | Fully unrolled; truncated k = ⌈r/2⌉ (set only at r=4, 8; at r=2, k=1 has already been reported by [01] as a systematic mismatch, so it is not run) |
| Data-amount branches (dense ruler) | 5 / 10 / 20 / 40 / 80 tokens per unit of N; 5 points spanning 16× to stabilize β [07][08][09] |
| Data-amount branches (looped models) | 10 / 20 / 40 tokens per unit of N; truncated jobs get additional branches at the token counts corresponding to iso-FLOP (trunk extended to about 1.3×) |
| Seeds | 3 each for all cells at 10M, 20M and 40M (42/43/44); 1 for the bulk of 80M, increased to 3, if time allows, for the 4 cells that decide H1/H2 (middle block and whole stack, r=4, fully unrolled/truncated); 1 for the 160M ruler |
| Handling rule for the 10M rung | If the 10M rung's residual in the dense-ruler fit exceeds 2× that of the other rungs, report fits both with and without 10M, with the fit without it as primary |

**Number of jobs**: per rung, ruler 1 + middle block 5 (r2, r4 full/truncated, r8 full/truncated) + whole stack 5 = 11, 44 over the four rungs; 160M ruler 1; extra seeds 66 (22 each at 10M/20M/40M); learning-rate sweep about 32 → about 143 jobs in total, of which 111 produce fitting data.

**Compute budget** (computed at 5×10¹³ FLOP/s per GPU, adjusted upward for small widths because of lower utilization): 80M about 18.5 GPU-days, 40M about 5, 20M about 1.5, 10M about 0.6, 160M ruler about 4, seeds for the three rungs about 14, learning-rate sweep about 3; total about 47 GPU-days, about 12 days of pure running time on 4 GPUs, about 2.5 weeks with 1.4× overhead. The optional extra seeds for the four 80M cells need about 11 more GPU-days, about 3.5 weeks in total.

**If time is insufficient, cut in this order**: ① drop the optional extra seeds at 80M; ② remove 80M whole-stack r=8 truncated; ③ reduce 40M seeds to 2. **If 160M looped cells are added later**, the priority is middle block r=2/4/8 fully unrolled → middle block truncated → whole stack r=2/4 → whole stack r=8 (about 15 GPU-days each). Any change is written to the deviation log.

## 4. Evaluation and fitting protocol

**Data points**: the validation loss at the end of each cooldown branch (mean of the last 3 checkpoints, nats/token). Each row records: rung, N_once, N_rec, r, placement, backprop, D, seed, loss, training FLOP (counted over executed layers, including the output head and the attention terms [08][09]), deployment FLOP/token, stored parameters.

**Fitting** [01][03][08]: Huber loss in log space, δ = 10⁻³, LSE parameterization; L-BFGS-B, 500 starting points, with bounds; first fit E, A, α, B, β on the dense ruler, then fix them and fit φ cell by cell (primary analysis); joint fit as secondary analysis [05]. Confidence intervals: block bootstrap by (rung, r, budget), 1000 resamples. Also report restricted fits with φ fixed at 0 and at 1, a refit without r=8, and a half-budget refit. **R² is not used as the basis for form selection** (loss differences across architectures are only about 0.1 nats [01]).

**Form selection**: F1 (r^φ), F2 (geometric saturation ρ), F3 (cell-specific φ or ρ, other parameters shared). The selection is based on leave-one-rung-out extrapolation error and AIC; see H3.

**Robustness**: seed spread; refit on the second validation set; dropping r=8; the two truncation accountings, iso-token and iso-FLOP.

## 5. Pre-run checks that must be passed

1. Truncated-backprop gradients agree with a manual derivation; the r=1 looped model is bitwise identical to the dense model (with the adapter switched off).
2. At 40M with matched parameters, r=4 is better than r=1; the dense-ruler loss decreases monotonically with width.
3. The FLOP counter agrees with the FLOPs measured in PyTorch (error < 5%).
4. Measure throughput at the five widths and rearrange the schedule; if the measured value at 80M is more than 30% below the estimate, trigger step ① of the cutting order.
5. The learning-rate sweep is completed and frozen, and written into Appendix C of this document.

## 6. Deviation log

(Appended after the runs start; format: date / deviation / reason / impact on the conclusions)

---

## Appendix A: Basis for each decision

| Decision | Basis |
|---|---|
| Fixed r, r∈{1,2,4,8}, prelude/coda = 2/2 | Aligned with the iso-depth main experiment so that φ can be compared [01] |
| Truncation k=⌈r/2⌉ | iso-depth's truncation setting, from which its 0.38 was obtained [01] |
| Truncation split into iso-token / iso-FLOP | iso-depth converted the saved FLOPs into 1.3× tokens, confounding the two variables [01] |
| Cooldown 20%, linear to 0 | Hägele Fig. 5: gains saturate at 20%; 5%–15% is insufficient [07] |
| 5 budget points spanning 16× | 3 points spanning 4× cannot fit β [07][08][09] |
| N includes the output head and excludes the input embedding; FLOP includes the output head | Porian points out that not counting output-head FLOPs shifts the exponent by +0.13 [09]; Chinchilla Appendix F [08] |
| warmup = 2N tokens | Porian: N–4N is appropriate; a fixed-length warmup shifts the exponent by +0.10 [09] |
| batch ∝ N^(2/3), learning rate re-swept per width | Porian: not retuning shifts the exponent by +0.07–0.10 [09]; [06] did not verify transfer across width |
| ε = 1/r, learning rate does not change with r | [06] Tables 1/3/4 and Fig. 5: under 1/r the optimal LR is constant across r∈{1,2,4,8}; under 1/√r it drifts |
| E, α, β fitted on the ruler first, then shared | Chen–Wilson's approach; prevents E from being pulled by the looped cells [05] |
| Huber δ=10⁻³, 500 starting points, block bootstrap | iso-depth and Chinchilla fitting protocols [01][08]; the Besiroglu replication points out that the Huber loss is summed, not averaged [08] |
| Placement conclusions require multiple seeds plus execution-depth-matched pairs | SMELT placement effect 0.006–0.008 nats, single seed, confounded with width [03] |
| No 4-point power law without E; embeddings not counted in N | The two methodological problems of Sparse Layers [04] |
| Conclusions drawn only for a single epoch | With multiple epochs, the optimal loop count grows with compute, and φ differs [05] |

## Appendix B: Gaps relative to the literature

Neither iso-depth nor Parcae looped the whole stack, swept truncation, reported seeds or released code [01][02]; SMELT and Sparse Layers have no φ for dense looping, and their placement evidence is single-seed [03][04]; Chen–Wilson and the residual-scaling paper did not test truncation or placement [05][06]. The four additions of this study: the whole-stack comparison, separating truncation under the two accountings, multiple seeds, and all logs made public.

## Appendix C: Frozen hyperparameters (to be filled in during week 3)

(η₀ for each of the five widths, β₂, measured throughput per rung, exact parameter counts)
