# Synthetic depth-controlled reasoning tasks: design v2 (English translation)

**Synthetic "controllable step count" reasoning tasks: design draft v2 (extension experiment C)**

> **About this document.** English translation, made on 2026-10-11, of `06_合成推理任务设计草案.md` (Chinese, the
> binding version; if the two differ, the original prevails). Design v2 was frozen on 2026-10-09 before the pilot,
> with SHA-256 `12f1240ad7c56bd8cb20cd5fadc6079da5c7b74567bd7c56eab841ce2f188243`, committed in `4f3db9f`
> (2026-10-09 12:06 +0800); verify with `git show 4f3db9f:06_合成推理任务设计草案.md | shasum -a 256`. Changes made
> after the freeze (curriculum, lookup-table operations, the formal-suite protocol) are dated in the
> [deviation log](deviation-log.md) (entries of 2026-10-10 and 2026-10-11); results are in [results.md](results.md)
> (sections 5.13 and later).

Status: **v2 frozen (2026-10-09, confirmed by the author: "implement it the way you said most recently"; SHA-256: see `06_合成推理任务设计草案.v2.sha256`)**; v1 was written on 10-08, and this version was rewritten following "think it over more carefully". Changes after freezing may only be recorded in docs/deviation-log.md. The goal is unchanged: to turn the author's core question, "is putting reasoning into the architecture really useful", into an experiment that can be tested directly. The formal suite is run only after the predictions have been fixed in writing, the author has confirmed, and the SHA-256 has been recorded; the pilot is used only to calibrate parameters and is explicitly labeled as an extension outside the pre-registration.

## 0. What changed from v1 to v2, and why

1. **Add an "iso-depth dense model" control.** v1 only compared "iso-parameter" (8-layer dense vs 8-layer looped r times). But the executed depth of the looped model is 2+4r+2 layers; to tell whether the gain really comes from "depth" or from something else, a dense model with 2+4r+2 layers and no weight sharing (2.5 to 4.5 times the parameters of the looped model) is needed as an upper bound. If looped ≈ iso-depth dense ≫ iso-parameter dense, the conclusion is "what looping buys is depth itself; parameters do not matter", which is the cleanest one-sentence statement.
2. **Remove the S5 permutation-group word problem.** Composition of permutations is associative, so a transformer can complete the computation in log depth with "prefix doubling"; Liu et al. 2023 have proved, and observed, that models learn this shortcut, so it is not a "number of steps equals depth" task. It is replaced as the main task by a task whose composition is incompressible (see T1); pointer chasing is kept as a control for "is looping still useful when a shortcut exists" (T2). In theory **any** task of this kind has a log-depth parallel solution (they are all in NC), so this experiment measures "what small models actually learn with limited data", and the predictions must be written honestly as empirical ones.
3. **Add a "step-matched loop count" training mode.** v1 had only fixed loop count and random loop count. Section 5.9 has already shown that random loop-count training converges to a fixed point and that more loops do not help; this is natural, because during training the loop count has nothing to do with problem difficulty. The key approach of Fan et al. 2024 (Looped Transformers for Length Generalization) is to make the loop count equal to the number of steps of the problem during training, and then to loop more on longer problems at test time. This is the most direct and most favorable condition for "thinking a few more loops to solve deeper problems"; without testing it, the author's question has not been answered.
4. **Compute the loss only on answer tokens, with a separate small training script.** The problem-statement tokens are random and should not enter the loss; the main training code is currently running 160M and is not changed. Training on the synthetic tasks is short (a few minutes to an hour) and uses a separate script under `code/synthetic/`: it reuses `LoopedLM`, computes the loss at the answer position, has built-in accuracy evaluation by step count and by inference loop count, and generates data online (unlimited data, no memorization of problems).
5. **Add "intermediate-value probes".** The synthetic tasks have true intermediate results x₁…x_d. Training linear probes after each loop to decode x_j shows directly "which step the model has computed up to after loop i". This is direct evidence (or counter-evidence) for "reasoning has been put into the architecture", more convincing than accuracy curves, and fully consistent with the per-loop probe method of 5.8.
6. **Narrow the scope.** Two tasks, one model size, two seeds; half a day for the pilot, within one day for the formal suite.

## 1. Tasks

Two tasks, each with an adjustable step-count parameter d, as token sequences over a small vocabulary, generated randomly by a program; every problem is independently random (no repeats between training and testing).

**T1 Modular operation chain (main task, incompressible composition)**: `x0=a ; op1 ; op2 ; … ; opd ; => ?`, modulo p = 97. Operations are drawn from {add c, multiply by c, square then add c} (c random); each step depends on the result of the previous step, and the answer is the final value (one token). The operations are globally fixed (no table is given in the context); the model must memorize the operation table in its weights; the composition of two steps is an arbitrary function Z₉₇→Z₉₇ with no compact representation, so the shortcut of "computing two steps in one layer" is extremely costly; this is the reason it was chosen as the main task. d is 1 to 8 in training and 1 to 24 in testing. The intermediate results x₁…x_d are saved for the probes.

**T2 Pointer chasing (control, with a known log-depth shortcut)**: a random function f on 32 nodes is given as `a>b` pairs in shuffled order; the question is "start from x and take k steps", and the answer is f^k(x). k is 1 to 8 in training and 1 to 16 in testing. The model can learn "prefix doubling" (layer 1 computes f, layer 2 computes f², ...), so a dense 8-layer model may also be able to solve k=16; it answers "when the task has a parallel shortcut, does the extra depth of looping still have value".

## 2. Model and training modes

- Same architecture as the main grid: 8 unique layers, middle block 2 + 4×r + 2, width 256 (about 5M parameters, vocabulary ~110).
- Conditions (all run for each task):
  - A. Dense, 8 layers (iso-parameter baseline)
  - B. Iso-depth dense: 20 layers (corresponding to r=4) and 36 layers (corresponding to r=8), no weight sharing (parameter upper bound)
  - C. Looped, fixed loop count: r = 2, 4, 8
  - D. Looped, random loop count: nominal r=8, drawn uniformly from 1 to 8 at each step (same as 5.9)
  - E. Looped, step-matched: during training, loop count = the problem's step count d (d between 1 and 8); this is the approach of Fan et al.
- Training: AdamW, learning rate of the same order of magnitude as the frozen value of the 10M rung (the pilot sweeps 3 values), linear warmup + cosine decay, 8000 steps with 512 problems per step (about 4 million problems, one sequence per problem), loss only at the answer position. About 5 to 20 minutes per run on a single GPU.
- Evaluation: 2000 new problems for each d, reporting answer accuracy; looped models are additionally evaluated at inference loop counts r_eval ∈ {1,2,4,8,12,16,24,32}; for E, two more settings are added: "loop count = d (oracle)" and "loop count = adaptive stopping (state change between consecutive loops < threshold)".
- Seeds: 1 for the pilot, 2 for the formal suite.
- Number of runs: 2 tasks × 8 conditions × 2 seeds = 32 training runs, about 10 GPU-hours; pilot: T1 × conditions A, B36, C8, D, E × 1 seed, about 2 hours.

## 3. Intermediate-value probes (analysis)

On the checkpoints of the looped models for T1, train linear probes on the state after each loop to decode x₁…x_d separately (on problems with d fixed at 8), which gives a "loop count × step count" table of decoding accuracy:
- if the table for condition E is close to diagonal (x_i is decoded most accurately at loop i), this indicates that the model has learned "one step per loop", which is the strongest evidence that "there is reasoning in the architecture";
- if the table for condition C (fixed 8 loops) shows "nothing can be decoded in the first few loops, everything appears in the last loop", this corresponds to the final-loop write-out ("turn") seen in 5.8;
- if under all conditions only the final answer can be decoded and the intermediate values cannot, this indicates that the model is not using a step-by-step algorithm.

## 4. Predictions fixed in writing (before the runs start; this section is authoritative for the formal suite)

- **P1 (steps for capability, iso-parameter)**: on T1, the "largest solvable step count" (the largest d with accuracy ≥ 90%) increases in the order dense 8 layers < r=2 < r=4 < r=8; the upper limit of the dense 8-layer model does not exceed 8 (partial shortcuts allowed), and r=8 is not lower than 16.
- **P2 (depth, not parameters)**: the iso-depth dense model is not lower than the corresponding looped model, and the gap between the two is ≤ one quarter of the gap between the dense 8-layer model and the looped model; i.e. the capability comes from executed depth, and the cost of weight sharing is small.
- **P3 (inference-time scaling)**:
  - the fixed-loop-count model's accuracy drops sharply when r_eval ≠ r_train (consistent with 5.7 and 5.8);
  - the random-loop-count model is robust to r_eval, but on problems with d > 8 it does **not** improve as r_eval increases (consistent with 5.9: it converges to a fixed point);
  - on problems with d > 8, the step-matched model's accuracy with loop count = d is significantly higher than with loop count = 8, and this extends with d to at least 16; this is the condition under which "thinking a few more loops to solve deeper problems" holds. If it does not hold, the conclusion is "even when trained specifically this way, there is no extrapolation at this scale".
- **P4 (intermediate values)**: the probe table of the step-matched model is close to diagonal; that of the fixed-loop-count model is not.
- **P5 (comparison with text)**: on T1, the gap between the iso-parameter looped and dense models is tens of percentage points of accuracy, which in nats is far larger than the 0.01–0.07 on text; the reading, combined with 5.7–5.10, is: looping in the architecture is "depth used on demand", which web text rarely needs, so in the language-model loss it shows up only as a parameter-efficiency gain that grows with the amount of data.
- **No directional prediction for T2**; only report: if the dense 8-layer model can already solve k=16, this shows that the depth advantage of looping disappears when a shortcut exists, which is in itself an answer to "when is looping useful".
- Decision quantities: for each condition, the "largest solvable step count", the "accuracy difference between loop count = d and loop count = 8 for d > 8", and the diagonal share of the probe table. The number of problems is sufficient, so no significance tests are done (the standard error of accuracy on 2000 problems is ≤ 1.1 percentage points).

## 5. Merging with the main paper (conclusion unchanged, additional reasons)

The four issues of v1 Section 5 (two stories, length, novelty, risk) still hold; the recommendation is still one paper, submitted to TMLR. v2 makes the synthetic part serve the main question more tightly: 5.8 says the fixed-loop-count model is a "fixed-length program", 5.9 says the random-loop-count model is a "fixed point", 5.10 says the value on text is a parameter efficiency that grows with data; C supplies the last piece: **on tasks that really require serial steps, can looping become "depth on demand", and how must it be trained to become that**. On novelty: the iso-depth control, step-matched training and intermediate-value probes, three things done together on the same set of models, have no ready-made counterpart in the literature.

The reasons for splitting into two papers have not changed either: the first paper already has enough material. But the pilot of C takes only half a day; the recommendation is to run the pilot first and then decide whether to merge.

## 6. Risks and responses

- The dense 8-layer model can also solve very deep problems on T1 (it has learned the composition table): then P1 does not hold; report it, and retry once with p raised to 997 (larger composition table).
- The step-matched model does not extrapolate: the third item of P3 does not hold; this is negative evidence against Huginn-style claims; write it up as it is.
- The intermediate-value probes cannot decode any intermediate value: this indicates that the model is not using a step-by-step algorithm, and P4 does not hold; also write this up as it is; in this case P1 may still hold (depth is useful, but not step by step).
- Learning-rate sensitivity: the pilot sweeps 3 values; in the formal suite each condition uses its own optimal value, which is reported.

## 7. Next steps

1. The author confirms the tasks, conditions and predictions → freeze this file (record the SHA-256).
2. Write `code/synthetic/`: generator (including intermediate values), separate training script (answer-position loss, three loop-count modes, built-in evaluation), probe script; CPU tests cover generation correctness and a training smoke test.
3. Pilot on GPU 3 (after the data-extension seed 43 finishes, around tonight); the formal suite within one day; results are written to docs/results.md, Section 5 "Extensions".
