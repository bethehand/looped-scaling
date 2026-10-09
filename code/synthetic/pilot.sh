#!/usr/bin/env bash
# Pilot of extension C (06_合成推理任务设计草案.md v2, section 2): task T1 "chain", seed 1,
# the five conditions at lr 1e-3, then a three-point learning-rate check on dense8 and loop_r8.
# Calibration only (learning rate, steps, p); the pre-registered suite runs afterwards with the chosen values.
#
#   PY=/mnt/nvme0/looped/venv/bin/python CUDA_VISIBLE_DEVICES=3 bash synthetic/pilot.sh --device cuda
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
for c in dense8 dense36 loop_r8 loop_rand loop_step; do
  $PY -u synthetic/train_synth.py --task chain --cond "$c" --seed 1 --lr 1e-3 "$@"
done
for lr in 5e-4 2e-3; do
  for c in dense8 loop_r8; do
    $PY -u synthetic/train_synth.py --task chain --cond "$c" --seed 1 --lr "$lr" "$@"
  done
done
echo "pilot done"
