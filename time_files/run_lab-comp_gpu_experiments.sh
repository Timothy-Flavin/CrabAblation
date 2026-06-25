#!/usr/bin/env bash
# Auto-generated multi-process schedule for lab-comp_gpu
# Capacity: 4
set -euo pipefail

source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS || true
nvidia-cuda-mps-control -d || true
sleep 2

# --- Concurrency Queue 0 ---
(
  echo "[lab-comp_gpu - Q0] Running sac on nchain | Abl 0 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo sac --env_name nchain --ablation 0 --run 7 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running sac on nchain | Abl 0 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo sac --env_name nchain --ablation 0 --run 12 --device_name lab-comp_gpu
) &

# --- Concurrency Queue 1 ---
(
  echo "[lab-comp_gpu - Q1] Running sac on nchain | Abl 0 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo sac --env_name nchain --ablation 0 --run 9 --device_name lab-comp_gpu
) &

# --- Concurrency Queue 2 ---
(
  echo "[lab-comp_gpu - Q2] Running sac on nchain | Abl 0 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo sac --env_name nchain --ablation 0 --run 10 --device_name lab-comp_gpu
) &

# --- Concurrency Queue 3 ---
(
  echo "[lab-comp_gpu - Q3] Running sac on nchain | Abl 0 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo sac --env_name nchain --ablation 0 --run 11 --device_name lab-comp_gpu
) &

echo "All queues launched. Waiting for completion..."
wait

echo quit | nvidia-cuda-mps-control || true
sudo nvidia-smi -i 0 -c DEFAULT || true
echo "All tasks completed on this machine."
