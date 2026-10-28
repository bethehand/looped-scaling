# Paper plan (step 5)

> **Status 2026-10-17.** The first draft follows this plan with the revised positioning of section 2 and title candidate
> 2 in its singular form; it is in [paper/](../paper/) ([PDF](../paper/paper.pdf)).
>
> **Status 2026-10-28.** The paper was reorganised along an argument of 13 steps decided by the author (two claims, each
> tested on text and on the chain task; subsection headings state the claims; an introduction table puts the two claims
> and the two tasks side by side), and S10 (results §5.19) was added as Section 4.3. The outline in section 5 below is
> the plan of 2026-10-17; the paper itself is the current outline.

Drafted 2026-10-17 for the author's approval, after the author decided not to run any of the optional step-4
experiments ([roadmap](roadmap.md)). The paper is written from the results as they stand: [results](results.md),
[report](report.md), [deviation log](deviation-log.md). The report is already a prose draft of most sections; the paper
re-orders it around one question and adds the figures listed in section 6.

## 1. Decisions needed from the author

1. **Positioning**: A or B below (recommendation, revised 2026-10-17 evening: the measurement as the spine, B's question as
   the frame; end of section 2).
2. **Title**: one of the three candidates in section 3 (recommendation, revised 2026-10-17 evening: the second).
3. **Authors**: names, affiliations, contact e-mail; whether to post on arXiv at submission or after the decision;
   the wording of the AI-assistance statement (section 9).

Everything else in this plan is a proposal that the author can change by commenting on it.

## 2. Positioning

**A. A pre-registered, multi-seed replication and extension of the recurrence-scaling study.** The text results are
the paper; the synthetic task is an extension. Safe and easy to review, but the headline would be the number φ, which
the data did not identify in absolute terms.

**B. When does recurrent depth work as test-time compute?** One question, two settings. On text, where the task rarely
needs serial steps, the loop buys parameter and data efficiency that grows with scale and tokens per parameter, never
compute efficiency, and costs decoding time; the block runs a fixed-length program and gains nothing from extra loops.
On a task that needs serial steps, the same architecture learns one loop per step, extends at inference, stops by its
own confidence, and does what a written chain of thought did not do here. Three conditions make the difference:
headroom (errors come from too few steps, not from missing knowledge), extendable iteration (the loop can be run longer
without destroying its state), and a stopping signal. The pre-registration is the method, not the headline.

**Recommendation (2026-10-17 morning): B.** After S5, S8 and S9 the paper has a complete answer to this question; under A
the same material appears with the synthetic half demoted to an extension. The claims ledger (section 4) is the same
under both; only the order of sections 4–6 of the outline changes.

**Revised recommendation (2026-10-17 evening, after a full re-read of the results): the measurement as the spine, B's
question as the frame, title candidate 2.** Three reasons. (i) The evidence is uneven. The text results rest on 111
pre-registered runs with three seeds, model size and tokens per parameter varied separately, a 160M extension, re-tuned
learning rates (S1) and dense curves to 640 tokens per parameter (S4). The synthetic results rest on one task family at
6.6M parameters; the basic finding, that more loops at inference solve longer chains, is already in Kohli et al. and
Zhang et al.; and the chain-of-thought comparison is confounded by input length. (ii) Title 1 promises general
conditions. With the step-4 experiments not run (no second task family, no reasoning-heavy text), the three conditions
rest on two case studies and should be presented as the reading the data suggest, not as a finding; a question title
invites reviewers to test it. (iii) Title 2 states exactly the two findings that are fully supported, and it continues
the repository's title, so paper and repository carry one name. Under this choice the abstract still opens with B's
question, sections 4–6 keep the outline's order (text, mechanism, chain task), and the three conditions close the
discussion, labelled as an interpretation. Venue unchanged: TMLR, with the arXiv preprint posted at submission.

## 3. Title candidates

The working title *How much is a loop worth?* collides with Schwethelm et al. (2026), *How Much Is One Recurrence
Worth?*, whose effective-parameter form we test; it has to change.

1. **When Does Recurrent Depth Work as Test-Time Compute? A Pre-Registered Study of Looped Transformers** (fits B;
   recommended on 2026-10-17 morning).
2. **What a Loop Buys: Parameter Efficiency on Text, Extendable Depth on a Serial Task** (states the two findings; fits
   A or B; recommended since 2026-10-17 evening, section 2; singular since 2026-10-17 night, the author's decision, because
   the evidence is one task family).
3. **Looped Transformers from 10M to 160M: A Pre-Registered Replication and Extension of Recurrence Scaling** (fits A).

## 4. Claims ledger

Every claim the paper will make, with where its evidence is and what caveat accompanies it. "Pre-registered" means
the hypothesis and decision rule were frozen before training (2026-09-23); "pre-specified" means written in a frozen
plan before the runs (synthetic design v2, supplementary plans 1 and 2); "exploratory" means chosen after seeing data.
Section numbers in the evidence column refer to [results](results.md).

| # | Claim | Evidence | Status | Caveat to state | Paper § |
|---|---|---|---|---|---|
| C1 | φ is not identified in absolute terms: 0–0.17 per cell (primary), 0–0.41 (joint), because the dense ruler is weakly identified | §3; report §3.2 | pre-registered analysis | the trend across sizes is robust (below zero at 10M to about 0.65 at 160M) | 4.1–4.2 |
| C2 | H1 not supported: truncated backprop did not lower φ; it helped whole-stack loops with 4 loops at every size (with 8, slightly worse at 10M–20M) | §4, 5.3 | pre-registered | both accountings agree | 4.1 |
| C3 | H2: a looped middle block beats a looped whole stack in raw loss at similar executed depth (36 against 32 and 20 against 16 layers) in every comparison; the verdict depends on the measure (φ is mixed) | §4 | pre-registered | | 4.1 |
| C4 | H3 not supported: the data cannot separate a saturating φ(r) from r^φ; neither extrapolates better in both directions | §4; deviation log 2026-10-14/15 | pre-registered | the AIC proxy is no longer cited | 4.1 |
| C5 | H4: the dense model is compute-optimal under training and under deployment accounting | §4, 5.11 | pre-registered | | 4.1, 4.3 |
| C6 | At equal parameters and tokens, a middle block looped 4 times goes from 0.02–0.10 nats worse at 10M to 0.03–0.06 nats better at 160M; monotone across five sizes | 5.1, 5.12, 5.16 (S1 Q1) | exploratory; S1 pre-specified | 3 seeds at 10–40M, 1 at 80M, 2 at 160M; one 160M seed is anomalous out of distribution | 4.3 |
| C7 | The advantage grows with tokens per parameter, 0.02–0.03 nats per doubling; at 160 tokens per parameter a 10M model whose middle block is looped 8 times matches a dense model trained on about 2.2× the tokens (two seeds, on the dense curves extended in S4; 4 loops: 1.56–1.82×) | 5.10, 5.16 (S1 Q2), 5.18 (S4) | exploratory; S1 pre-specified | 10M and 20M only; two seeds at 80N/160N | 4.3 |
| C8 | Measured without a ruler, each parameter of a model whose middle block is looped 4 times (output head included) is worth 0.6–1.0 dense parameters at 10–20M with 10–20 tokens per parameter and 1.26–1.46 at 40M–160M with 20–40 (local φ 0.35–0.52 with the output head, 0.32–0.50 without, as counted by Schwethelm et al.); ×1.21 per doubling of size, ×1.14–1.16 per doubling of tokens per parameter (with the S4 points); break-even on the fitted plane near N = 35M at 20 tokens per parameter and N = 12M at 80 (below the smallest model, N = 15.4M); the measured points agree at 80 (1.04 against 1.05 on the plane) and reach M = 1 earlier at 20 (0.99 for the 20M model, N = 26.8M, against 0.93) | 5.17, 5.18 (S4) | exploratory (step 2) | published equal-parameter results restate mostly to 1.2–1.6 at about 130M–770M in comparable settings (Ouro's 778M model above 1.75 and its 53M model below 1 despite about 377 tokens per parameter; above 770M only bounds: lower bounds, and 1–2.8 for Marchenko et al.'s 1.4B model), up to about 2 for small blocks at 750–1500 tokens per parameter (Saunshi et al.) | 4.3 |
| C9 | At equal training compute the dense model wins all 258 measured comparisons, up to 640 tokens per parameter at 10M and 320 at 20M; the margin shrinks with data (0.01–0.05 nats at the largest looped budgets); with the lowest learning rate tried for the looped model the two are level on average at 10M/160N (−0.011, +0.005 in two seeds) | 5.11, 5.17(b), 5.18 (S4) | exploratory; S4 pre-specified | 62 comparisons still rest on extrapolation (43 at 40M–160M, 19 at 10M–20M, mostly the third seed), bounded by two methods; the measured comparisons are not independent; the looped rate at 10M (0.35×, the lowest tried) is not bracketed | 4.3 |
| C10 | Re-tuning the learning rate for both models at 10M–40M leaves C6, C7 and C9 intact; the shared rate understated the looped model by up to 0.024 nats at 10M | 5.16 (S1) | S1 pre-specified | one seed (two at 20M); one looped cell | 4.3, 9 |
| C11 | Gains are larger on math and code text, but out of distribution the seed noise is large, and on the code set the data order, which the seeds share, matters even more (on the math set it does not): it matters 0.1–0.5× as much as the initialisation in distribution and up to 7× on the code set | 5.6, 5.18 (S7) | exploratory; S7 pre-specified | out-of-distribution differences are not reported as results | 4.4 |
| C12 | The per-token gain is a diffuse downward shift, not concentrated on hard tokens | 5.7 | exploratory | | 4.4 |
| C13 | On text the looped block does not iterate towards an answer: fixed loop count = fixed-length program with a large write-out last loop; truncated backprop = periodic cycle; random loop count = contraction to a fixed point; the information a probe can read saturates by about 4 loops; none gains from extra loops at inference | 5.8, 5.9, 5.14 | exploratory | 20M–80M, one to three checkpoints per cell; the random-loop range (1–8) differs from Huginn's | 5 |
| C14 | Truncated backprop accounts for 25 of the 30 runs with loss spikes; halving the learning rate removed them in a 10M follow-up | 5.4 | exploratory | | 4.1, App. B |
| C15 | On the chain task, looped models learn to advance one step per loop (probes) and solve longer chains with more loops at inference: r = 8 at 12 steps 0.04 → 0.83 with 16 loops (0.86 with the best larger loop count); dense models of equal depth with 2.4–4.4× the parameters fail right after the training range, as do fixed-loop models at their training loop count | 5.15 | pre-specified predictions P1–P5 (verdicts per 5.15) | 6.6M-parameter models, one task family; accuracy falls with length (16 steps 0.37) and varies widely across seeds | 6.2–6.4 |
| C16 | The learning rate decides whether the task is learned within the training budget: at the dense-optimal rate 10 of 12 looped runs did not learn it in the same number of steps; runs that learned it at a higher rate mostly learned the same extendable algorithm | 5.15, 5.16 (S2) | exploratory; S2 pre-specified | | 6.5 |
| C17 | The model trained with loops = steps extends furthest (12.8 steps) but cannot stop by its state (persistence 0.04) | 5.15, 5.18 (S8) | pre-specified (adaptive stop) | | 6.6 |
| C18 | The model's own confidence (max probability ≥ 0.99; ≥ 0.9 for r = 2) is the best answer-free rule that stops in every S8 condition except r = 4, where a state rule with a 0.005 threshold ties it with far more loops (for the S9 model that state rule is slightly better): it recovers 97–98% of the extra-loop gain for models trained with 4 or 8 loops or a random loop count (with 2 loops it beats the best fixed count), using fewer loops than training; voting over loop counts is no better than the best fixed count | 5.18 (S8) | S8 pre-specified (plan written after the state rule's results on these checkpoints were known); rules chosen on a selection set, reported on a separate set | | 6.6 |
| C19 | Trained with loops = steps plus 0–4 random extra loops, the model keeps its values (persistence 0.91), stops by confidence at 0.93 on 12-step problems with 0.998 in range, and loses about one step of extrapolation (11.6 vs 12.8) at about 1.4× the training compute | 5.18 (S9) | S9 pre-specified; the confidence reading is outside the plan's three pre-written readings (deviation log 2026-10-17) | the plan's state rule gives 0.52, seed-dependent | 6.6 |
| C20 | Against reasoning written out: dense models with a chain of thought solve the training range perfectly (1.000 in every seed and learning rate) at under a third of the looped model's inference compute, and are near chance beyond it (12 steps 0.014 and 0.017; chance 0.010), 36 layers included; the chain breaks within its first three written tokens (97–98% of the problems for 8 layers, whose first token is right in 81% of the 9-step and 57% of the 12-step problems), and with a correct prefix the later tokens are still mostly wrong; pause tokens add nothing | 5.18 (S5) | S5 pre-specified; the breaking-point diagnostic is exploratory | the prompt grows with the number of steps, so the failure is one of input-length generalisation; a fixed-input-length task was not run; the chain of thought is supervised on every intermediate value, the looped models only on the answer (stated in the S5 plan) | 6.7 |
| C21 | At equal loss the looped middle block needs at most 19% less weight memory (4% at 20M, 14–19% at 40M–160M) but 2.2–2.4× the KV cache and 2.4–2.5× the decoding time per token; peak memory at batch 32 about 1.6× | 5.16 (S3) | S3 pre-specified | one GPU type, ≤160M | 7 |
| C22 | Sharing the KV cache across loops costs the fixed-loop models 0.09–0.17 nats, 1.8 to 3.7 times their gain, and is free only for the random-loop models, whose loops gain nothing | 5.18 (S6) | S6 pre-specified | post-hoc change to models trained without sharing; 20M only | 7 |
| C23 | Method: four pre-registered hypotheses, none held as stated; every departure dated in the deviation log; an external review (2026-10-14) found five errors and a second review (2026-10-15) about 35 errors and overstatements, and a third review (2026-10-27) corrected further ones; all numbers re-computable from the results in the repository | deviation log; report §8 | | | 1, 10, App. B |

Claims the paper will **not** make: that the trends continue beyond 160M or cross at equal compute; that looping
helps on text at inference; that the chain-of-thought result generalises beyond this task; anything about
multi-epoch training.

## 5. Outline (positioning B; under A, sections 4–6 are re-ordered 4, 6, 5 and section 6 is titled "Extension")

**Abstract.** Question; design (111-run pre-registered grid, extensions, nine supplementary experiments with frozen
plans); the two findings (text: parameter efficiency that grows with scale and data, not compute efficiency, at a
cost in decoding time; serial task: one loop per step, extension at inference, stopping by confidence, what a chain of
thought did not do); the three conditions; everything reproducible.

**1 Introduction.** The two promises of recurrent depth (parameter efficiency, latent reasoning); why the literature
cannot settle them (recipes differ, seeds rare, one fit); what we did; the three conditions as the organising
idea; contributions (five bullets: C1–C5 verdicts; C6–C9 trends; C13 mechanism; C15–C20 serial task; C21–C22 cost);
Figure 1.

**2 Related work.** From report §1 (how much a loop is worth; what the loop computes; serial reasoning; cost at
inference), with the restated multipliers of C8 as the bridge to published results.

**3 Setup.** Report §2: sizes, cells, data, training, measurement, compute; the four hypotheses and the decision
rules; what "pre-registered", "pre-specified" and "exploratory" mean in this paper; Table 1.

**4 What a loop is worth on text.** 4.1 the pre-registered verdicts (C2–C5, C14; Table 2: φ per cell with intervals,
H1–H4). 4.2 why φ is not identified (C1; the ruler). 4.3 the value grows with scale and with data (C6–C10; Figure 2
scale and data trends; Figure 3 ruler-free multiplier; Figure 4 equal-compute gap with the S4 points and the two
extrapolation bounds). 4.4 where the gain lands (C11, C12).

**5 What the looped block computes on text.** C13: per-loop probes for the three training regimes (Figure 5); the
depth curve at inference; why extra loops do not help on text.

**6 Loops as extendable depth on a serial task.** 6.1 the chain task, conditions, controls of equal parameters and
equal depth, the pre-specified predictions. 6.2 verdicts on P1–P5 (Table 3). 6.3 extension at inference (C15;
Figure 6 accuracy vs steps at the training loop count and with more loops, dense controls). 6.4 probes: one loop per
step (Figure 7). 6.5 the learning rate decides whether the task is learned within the training budget (C16). 6.6 stopping: rules, S8 verdict,
loops = steps vs loops = steps + extra (C17–C19; Figure 8; Table 4). 6.7 against a chain of thought and pause tokens
(C20; Figure 9 accuracy vs steps with inference compute; the breaking-point diagnostic).

**7 Cost at deployment.** C21, C22 (Table 5).

**8 Discussion: when recurrent depth is test-time compute.** The three conditions and which experiments speak to
each; what text lacks (headroom) and what the chain task has; training decides the regime (fixed program, fixed
point, extendable algorithm) more than the architecture does; φ is not a constant of the architecture; practical
takeaways (report §7, condensed).

**9 Limitations.** Report §9, plus the step-4 items not run (section 8 of this plan).

**10 Pre-registration, deviations and reproducibility.** What was frozen and when (SHA-256 in the repository), how
departures were logged, what the two reviews changed (report §8 condensed), the reproduction map, the AI-assistance
statement.

**Appendices.** A the pre-registration text and hypotheses verbatim; B what went wrong and what we changed (report §8
in full); C fits and the robustness variants (results §3, 5.17); D out-of-distribution and per-token analyses (5.5–5.7,
S7); E the synthetic task (design, learning-rate sweeps, all conditions and seeds, halting-rule tables, the
chain-of-thought diagnostic); F deployment measurements (S3, S6 details); G training logs, spikes and the
learning-rate re-tuning (S1).

## 6. Figures and tables

Existing figures are in `code/results/figures/` and are made by `code/fit/make_figures.py` and
`code/synthetic/analyze.py`; new ones need scripts in the same places so that the reproduction map stays complete.

| Paper figure | Content | Source | Status |
|---|---|---|---|
| Fig. 1 | two panels: equal-parameter advantage vs size and vs tokens per parameter; chain-task accuracy vs steps with more loops | fig12, fig11, fig14 | compose from existing |
| Fig. 2 | scale trend and data trend with seed ranges | fig12, fig11 | existing |
| Fig. 3 | ruler-free multiplier M(N, tokens per parameter) with bootstrap intervals | `results/fit/local_phi*.csv` (5.17) | **new** |
| Fig. 4 | equal-compute gap vs budget, measured points (S4 included) and the two extrapolation bounds | `results/supp2/iso_flop_*` (5.11, 5.17b, S4) | **new** |
| Fig. 5 | per-loop probes for the three regimes | fig8, fig8b, fig9, fig10 | existing, re-composed |
| Fig. 6 | chain task: accuracy vs steps, training loop count and best loop count, dense controls | fig13, fig14 | existing |
| Fig. 7 | chain task probes (one loop per step) | fig15 | existing |
| Fig. 8 | stopping: accuracy and loops used under each rule, S8 and S9 models | `results/synthetic/halting*/` | **new** |
| Fig. 9 | chain of thought and pause tokens vs loops: accuracy vs steps, inference compute | `results/synthetic/supp2/` (S5) | **new** |
| Fig. 10 | deployment cost at equal loss (S3) and the KV-sharing penalty (S6) | `results/supp*/` | **new** |
| App. | loss vs tokens (fig3), spikes (fig4), gap by exam (fig5), depth curve (fig6), gain by difficulty (fig7), φ by rung (fig2) | existing | existing |

Tables: 1 setup; 2 φ and H1–H4; 3 synthetic predictions P1–P5; 4 stopping rules (S8/S9); 5 deployment (S3/S6);
appendix tables for S1, S4, S7 and the learning-rate choices.

## 7. Writing plan and schedule

| Step | Content | Time |
|---|---|---|
| 1 | Author approves positioning, title, this ledger and outline | — |
| 2 | LaTeX skeleton (TMLR template) in `paper/`; sections 3, 4, 5, 7 from the report; Tables 1, 2, 5 | 3 days |
| 3 | Section 6 and its tables from results 5.15–5.18; appendices E, F | 3 days |
| 4 | The five new figures (scripts committed), re-composed Figures 1 and 5 | 2 days |
| 5 | Sections 1, 2, 8, 9, 10; abstract | 2 days |
| 6 | Number audit: every number in the paper traced to a results file by script; referee pass; limitations check | 2 days |
| 7 | Anonymised repository snapshot, AI-assistance statement, submission to TMLR; arXiv at the author's chosen time | 2 days |

About two weeks to a submitted draft, writing in English; the Chinese working documents stay in `zh/`.

## 8. What the skipped step 4 means for the paper

| Item not run | Limitation the paper states |
|---|---|
| Second synthetic task family with fixed input length | the chain-of-thought comparison (C20) is confounded with input-length generalisation; the paper says so and does not claim that a chain of thought cannot extend in general |
| Dense control of equal executed depth on text | the paper's text comparisons are equal-parameter and equal-compute only; the equal-depth control exists only on the synthetic task |
| Two more seeds at 80M | the scale trend has one seed at 80M and two at 160M; stated in §9 |
| Reasoning-heavy text at 20M–40M | the larger out-of-distribution gains on math and code (C11) stay an observation with a data-order caveat (S7); no claim about equal-compute parity on such text |

## 9. Venue, format and disclosure

TMLR (double-blind review: the repository is linked through an anonymised snapshot; the public repository and
arXiv version carry the author information). The paper states that the code, analyses, documents and text were
drafted with an AI assistant (Claude, by Anthropic) under the author's direction, that the author is responsible for
every decision and for the content, that every number can be recomputed from the results in the repository, and how
the pre-registration and deviation log were kept; the author chooses the wording.
