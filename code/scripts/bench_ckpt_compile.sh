#!/usr/bin/env bash
# Re-measure the six checkpointed loop cells on a healthy machine: eager, then per-block compile, then one
# compiled loop step ("region"), one loop width per GPU, all four widths in parallel (~20-40 min).
# The earlier decision to run these cells eager was made while GS01's CPUs were power-capped (03_偏离记录.md).
#
#   cd ~/looped-scaling/code && bash scripts/bench_ckpt_compile.sh 2>&1 | tee -a logs/bench_ckpt.log
#
# Results: results/throughput_ckpt/w<width>.json (eager baseline measured in the same session).
set -u
CELLS=middle_r8_k0,middle_r8_k4,whole_r4_k0,whole_r4_k2,whole_r8_k0,whole_r8_k4
WIDTHS=(320 448 640 896)
mkdir -p logs results/throughput_ckpt
echo "[bench] gradients of compiled (blocks, region) vs eager on this GPU"
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_compile.py -q 2>&1 | tail -3
for i in 0 1 2 3; do
  w=${WIDTHS[$i]}
  out=results/throughput_ckpt/w$w.json
  rm -f "$out"
  (
    CUDA_VISIBLE_DEVICES=$i python -u scripts/measure_throughput.py --widths "$w" --cells "$CELLS" --out "$out"
    for mode in blocks region; do
      CUDA_VISIBLE_DEVICES=$i python -u scripts/measure_throughput.py --widths "$w" --cells "$CELLS" --out "$out" \
        --compile --compile-mode "$mode"
    done
  ) 2>&1 | sed -u "s/^/[gpu$i w$w] /" &
done
wait
echo "[bench] all widths done: results/throughput_ckpt/"
