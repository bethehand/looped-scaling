# looped-scaling: code

A controlled small-scale scaling study of looped / recurrent-depth Transformers (the loop-equivalence exponent φ and
ruler-free measurements on text) and a synthetic test of step-by-step reasoning inside the loop.
The experimental settings follow `../docs/preregistration.md` (v1.0, frozen; English translation, the frozen original is Chinese).
The findings are in `../docs/report.md` and `../docs/results.md`.

## Layout
- `looped/`    model (prelude / looped core / coda), FLOP counting, data stream, learning-rate schedule, training loop,
               evaluation, KV-cached inference
- `scripts/`   data preparation, config generation, queue runner, throughput and inference benchmarks, result export,
               loss-spike statistics
- `fit/`       result collection, scaling-law fitting (F1 / F2 / F3, Huber + L-BFGS-B, bootstrap, extrapolation tests),
               H4, per-rung φ, ruler-free multipliers, equal-compute check and its bounds, published results restated,
               per-token analysis, per-loop probes, supplementary analysis, figures
- `synthetic/` synthetic reasoning tasks, trainer, multi-GPU suite runner, intermediate-value probes, analysis
- `configs/`   every run config and manifest (sweeps, main grid, extensions, supplementary experiments)
- `results/`   everything the documents cite: `runs/` (per-run results), `train_logs/` (per-run training logs), the
               collected tables (`all_results.csv`, `ext_*.csv`, ...), `fit/`, `synthetic/`, `supp/`, `bench/`, the probe
               and per-token outputs, and `figures/`
- `tests/`     pre-run checks (truncated-backprop gradients, r=1 equivalence, FLOP and parameter counting) and tests of
               training, inference and every analysis

## Environment
    uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e ".[test]"
    python -m pytest tests -q

## Where every number comes from

Run from `code/`. Each command below rebuilds a committed file from other committed files, on a laptop CPU. Those
marked ✓ were re-run on 2026-10-15 and reproduced the committed file exactly (figures are rebuilt with the same content
but are not byte-identical, because the files carry a timestamp).

| Output | Used in | Command |
|---|---|---|
| `results/all_results.csv` (main grid) ✓ | everything | `python fit/collect_results.py --runs results/runs --manifest configs/manifest.csv --out results/all_results.csv` |
| `results/ext_data.csv`, `ext_160m.csv`, `ext_160m_dense.csv`, `ext_randr.csv` ✓ | results 5.9–5.12 | the same with `--manifest configs/manifest_ext_<name>.csv --out results/ext_<name>.csv` |
| `results/supp_lr.csv` ✓ | S1 | the same with `--manifest configs/manifest_supp_lr.csv --out results/supp_lr.csv` |
| `results/sweep.csv`, `results/looplr.csv` | learning-rate sweep, loop learning-rate check | the same with `manifest_sweep.csv` / `manifest_looplr.csv` (the committed files predate two added columns; the values are identical) |
| `results/spikes.csv` ✓ | results §1, 5.4 | `python scripts/spike_stats.py --runs results/train_logs` (on the training machine `--runs runs`) |
| `results/fit/primary/` ✓¹ | results §3–4 (H1–H3) | `python fit/fit_laws.py --out results/fit/primary` (20–25 minutes); the same with `--extrapolation-only` writes `extrapolation_preregistered_*.json` |
| `results/fit/<variant>/` ✓¹ | results §3, robustness | `python fit/fit_laws.py --out results/fit/<variant>` plus: no_spikes `--exclude-spiking`; valavg `--loss-col valavg_fwe`; second `--loss-col val_second`; no_r8 `--exclude-r 8`; half_budget `--half-budget`; no_head `--n-def no_head`; iso_flop `--accounting iso_flop`; finemath `--loss-col val_finemath`; code `--loss-col val_code` |
| `results/fit/h4_optimal_r_primary.json`, `h4_optimal_r_joint.json` ✓ | H4 | `python fit/h4_optimal_r.py --out results/fit/h4_optimal_r_primary.json`; `python fit/h4_optimal_r.py --stage joint --out results/fit/h4_optimal_r_joint.json` |
| `results/fit/exploratory_phi_by_rung.json` ✓ | results 5.1–5.2 | `python fit/phi_by_rung.py` |
| `results/fit/exploratory_phi_by_rung_with_160m.json` ✓ | results 5.12 | `python fit/phi_by_rung.py --results results/all_results.csv results/ext_160m.csv results/ext_160m_dense.csv --out results/fit/exploratory_phi_by_rung_with_160m.json` |
| `results/fit/iso_flop_raw.csv` ✓ | results 5.11, report 3.3 | `python fit/iso_flop_raw.py` |
| `results/supp/` (S1–S3 tables and readings) ✓ | results 5.16 | `python fit/supp_analysis.py` |
| `results/synthetic/formal_summary.csv`, `formal_probe_frontier.csv`, `formal_verdicts.json`, `sweep_lr_table.csv`, figures 13–15 ✓ | results 5.15, report 5 | `python synthetic/analyze.py` |
| `results/fit/local_phi.csv`, `local_phi_trends.json` ✓ | results 5.17(a), report 3.2 | `python fit/local_phi.py` (robustness: `--n-def no_head`, `--exclude-spiking`, with `--out`/`--trends-out`) |
| `results/fit/iso_flop_bounds.csv`, `iso_flop_bounds.json` ✓ | results 5.17(b), report 3.3 | `python fit/iso_flop_bounds.py` |
| `results/fit/external_multipliers.csv` ✓ | results 5.17(c) | `python fit/external_multipliers.py` (published numbers, written in the script with their sources) |
| `results/figures/fig1`–`fig12` | report, results | `python fit/make_figures.py` |
| `results/figures/fig0_overview.png` | README, report | `python fit/make_overview.py` |
| `results/supp2_s4.csv`, `results/supp2_s7.csv` | S4, S7 | `python fit/collect_results.py --runs results/runs --manifest configs/manifest_supp2_s4.csv --out results/supp2_s4.csv` (and `manifest_supp2_s7.csv` → `supp2_s7.csv`) |
| `results/supp2/` (S4–S9 tables and readings, incl. `iso_flop_*_s4`, `local_phi_s4*`) | results 5.18 | `python fit/supp2_analysis.py` (one part: `--part s4` … `--part s9`) |

¹ All ten fits were re-run on 2026-10-15. Since 2026-10-14 `fit_laws.py` also writes the pre-registered extrapolation
test of H3 into the fit file, so a re-run fit file has one more entry per form (`extrapolation_preregistered`) than the
committed one. That entry equals the committed `extrapolation_preregistered_*.json` where there is one (the primary fit
and seven variants); everything else is identical.

These steps need the model checkpoints (485 GB, not in the repository) or a GPU; they ran on the training machine and
their outputs are committed:

| Output | Used in | Command |
|---|---|---|
| `results/token_analysis/` (per-token tables, inference loop-count curves) | results 5.6–5.7, 5.9 | `python fit/token_analysis.py --runs <run names> --stage all`, then `python fit/raw_means.py --pattern <run name part> --out <csv>` |
| `results/token_analysis_looplr/` | results 5.14 | the same on the runs of the loop learning-rate check |
| `results/loop_probe/` | results 5.8, figures 8–9 | `python fit/loop_probe.py --runs <run names> --r-max 16 --device cuda` |
| `results/synthetic/chain/`, `formal/` (training runs and probes) | results 5.13–5.15 | `python -u synthetic/suite.py --stage all --gpus 0,1,2,3` (sweep, choice of learning rate, formal runs, probes) |
| `results/synthetic/lr_seeds/`, `results/supp/depth_curve.csv` | results 5.16 (S1 Q5, S2) | the supplementary queue below |
| `results/bench/inference.json` | results 5.16 (S3) | `python scripts/bench_inference.py --out results/bench/inference.json` (random weights, one GPU) |
| `results/runs/10M_dense_r1_d640_*`, `20M_dense_r1_d320_*`, `10M_middle_r4_full_lrs0.35_*` (S4); `results/runs/20M_*_do3b_s42`, `*_do6b_s42` (S7) | results 5.18 (S4, S7) | `python scripts/make_supp2.py --part s4` then `python -u scripts/run_queue.py --manifest configs/manifest_supp2_s4.csv --gpus 0,1`; the same with `--part s7` and `manifest_supp2_s7.csv` |
| `results/synthetic/supp2/` (S5 and S9 training runs, sweeps and probes; `lr_choice_supp2.json`) | results 5.18 (S5, S9) | `python -u synthetic/suite.py --stage supp2 --gpus 0,1,2,3` |
| `results/synthetic/halting/`, `results/synthetic/halting_supp2/` (S8, S9 stopping rules) | results 5.18 (S8, S9) | `python synthetic/halting.py --ckpt-dir runs/synthetic/formal/chain --out results/synthetic/halting --device cuda`; the same with `runs/synthetic/supp2/formal/chain` → `results/synthetic/halting_supp2` |
| `results/supp2/kv_share.json` (S6) | results 5.18 (S6) | `python scripts/kv_share_eval.py --runs <20M run name>[@<inference loops>] … --out results/supp2/kv_share.json` |
| `results/synthetic/supp2/cot_length_probe.json` (S5 diagnostic) | results 5.18 (S5) | `python scripts/cot_length_probe.py --ckpt runs/synthetic/supp2/formal/chain/dense*_cot_*.pt --out results/synthetic/supp2/cot_length_probe.json --device cuda` |

## Training from scratch (weeks 1 to 3, then the extensions)
    python -m pytest tests -q                                   # checks 1 and 3 of the five, plus a training smoke test
    SRC=data/raw/fineweb-edu/sample/10BT                        # training data: FineWeb-Edu sample-10BT, see ../docs/deviation-log.md
    python scripts/prepare_data.py download --skip-fwe          # download only the three small validation sets
    python scripts/prepare_data.py --src $SRC tokenizer --chars 2e9
    python scripts/prepare_data.py --src $SRC tokenize --workers 32
    python scripts/prepare_data.py --src $SRC valsets --workers 16
    python scripts/measure_throughput.py --device cuda          # check 4: measured throughput -> configs/throughput.json
    python scripts/make_configs.py --sweep                      # week 3: learning-rate sweep configs
    python scripts/run_queue.py --manifest configs/manifest_sweep.csv --gpus 0,1,2,3
    python fit/collect_results.py --manifest configs/manifest_sweep.csv --out results/sweep.csv
    python scripts/pick_lr.py --results results/sweep.csv     # -> configs/lr_table.json
    python scripts/make_configs.py --lr-table configs/lr_table.json   # main grid: 111 configs + manifest.csv
    python scripts/run_queue.py --manifest configs/manifest.csv --gpus 0,1,2,3
    python fit/collect_results.py && python fit/fit_laws.py     # weeks 8 to 9
    python scripts/make_configs.py --ext --lr-table configs/lr_table_frozen.json --compile   # extension experiments (2026-10-08): configs/ext + manifest_ext_*.csv
    python scripts/run_queue.py --manifest configs/manifest_ext_160m.csv --gpus 0,1       # 160M middle-block r=4, two seeds
    python scripts/run_queue.py --manifest configs/manifest_ext_data.csv --gpus 2         # continue trunks from earlier checkpoints to 80N/160N (init_from in the config)
    python scripts/run_queue.py --manifest configs/manifest_ext_randr.csv --gpus 3        # random loop count at every step (r_sample in the config)
    python -u synthetic/suite.py --stage all --gpus 0,1,2,3                              # synthetic task (extension C)
    python scripts/make_supp.py                                                          # supplementary S1-S3: configs and one queue
    python -u scripts/run_queue.py --manifest configs/queue_supp.csv --follow --gpus 0,1,2,3
    python scripts/make_supp2.py --part s4 && python -u scripts/run_queue.py --manifest configs/manifest_supp2_s4.csv --gpus 0,1   # S4 (2026-10-15)
    python scripts/make_supp2.py --part s7 && python -u scripts/run_queue.py --manifest configs/manifest_supp2_s7.csv --gpus 2,3   # S7
    python -u synthetic/suite.py --stage supp2 --gpus 0,1,2,3                            # S5, S9 (sweep, choice of learning rate, formal runs, probes)
    python synthetic/halting.py --device cuda                                           # S8 on the formal checkpoints; S9: --ckpt-dir runs/synthetic/supp2/formal/chain --out results/synthetic/halting_supp2
    python scripts/kv_share_eval.py --runs <20M runs> --out results/supp2/kv_share.json  # S6
    python fit/supp2_analysis.py                                                        # readings for S4-S9 -> results/supp2/

The loop learning-rate check of 2026-09-25 is in `configs/manifest_looplr.csv` and `scripts/pick_loop_lr.py`.

## Running in tmux: live output with a saved log
    mkdir -p logs
    python -u scripts/xxx.py ... 2>&1 | tee -a logs/xxx.log
- `-u` makes Python print each line immediately; `2>&1` includes error output as well; `tee -a` shows the output on screen and appends it to the log at the same time.
- The queue prints the output of each training job to the screen in real time with a `[gpuN]` prefix, and also writes it to `runs/<job name>/stdout.log`.
- To leave tmux and keep it running: press Ctrl-b, then d; to come back: `tmux attach`.

## Queue and progress
- Queue progress is written to `runs/queue_<manifest file name>`, e.g. `runs/queue_manifest_sweep.csv`; the manifest files tracked by Git are not modified.
- Number of finished jobs: `grep -c ,done, runs/queue_manifest_sweep.csv`; failed jobs: `grep ,failed, runs/queue_manifest_sweep.csv`.
- A job that still fails after the maximum number of retries is marked failed. After investigating, change failed to pending on that job's line in the state file and start the queue once more; finished jobs are not rerun.
- Result metrics: `val_*` is the end-of-cooldown loss (primary metric), `valavg_*` is the mean of the last three evaluation points (robustness metric).

## Pushing results back to GitHub (on the GPU machine)
    python scripts/export_results.py          # copies runs/*/results.jsonl and run_info.json to results/runs/
    git add results && git commit -m "results: ..."
    git pull --rebase --autostash && git push # first pull in the new commits already on GitHub, then push
To read them on the Mac: after `git pull`, run `python fit/collect_results.py --runs results/runs --manifest configs/manifest_sweep.csv --out results/sweep.csv`.
The training logs (`train_log.csv` and `stdout.log` of all 232 runs in `results/runs/`, 47 MB) were copied to `results/train_logs/`
on 2026-10-15 (the nine S4 and S7 runs on 2026-10-17).
