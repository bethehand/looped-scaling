# looped-scaling: code

A controlled small-scale scaling study: the loop-equivalence exponent φ of looped / recurrent-depth Transformers.
The experimental settings follow `../docs/preregistration.md` (v1.0, frozen; English translation, the frozen original is Chinese).

## Layout
- `looped/`  model, FLOP counting, data stream, learning-rate schedule, training loop, evaluation
- `scripts/` data preparation, config generation, queue runner, throughput measurement
- `fit/`     scaling-law fitting (F1 / F2 / F3, Huber + L-BFGS-B, bootstrap, leave-one-rung-out extrapolation)
- `tests/`   pre-run checks (truncated-backprop gradients, r=1 equivalence, FLOP counting, parameter counting)
- `configs/` auto-generated run configs and manifests

## Environment
    uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e .
    python -m pytest tests -q

## Order of use (weeks 1 to 3)
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

