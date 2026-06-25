#!/usr/bin/env bash
# Auto-generated multi-process schedule for timpc_gpu
# Capacity: 2
set -euo pipefail

source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS || true
nvidia-cuda-mps-control -d || true
sleep 2

# --- Concurrency Queue 0 ---
(
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 0 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 0 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 0 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 0 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 0 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 0 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 0 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 0 --run 12 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 0 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 0 --run 14 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 1 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 1 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 1 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 1 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 1 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 1 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 1 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 1 --run 12 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 1 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 1 --run 14 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 2 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 2 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 2 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 2 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 2 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 2 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 2 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 2 --run 12 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 2 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 2 --run 14 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 3 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 3 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 3 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 3 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 3 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 3 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 3 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 3 --run 12 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 3 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 3 --run 14 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 4 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 4 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 4 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 4 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 4 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 4 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 4 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 4 --run 12 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 4 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 4 --run 14 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 5 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 5 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 5 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 5 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 5 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 5 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 5 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 5 --run 12 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 5 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 5 --run 14 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 6 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 6 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 6 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 6 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 6 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 6 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 6 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 6 --run 12 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on nchain | Abl 6 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name nchain --ablation 6 --run 14 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 0 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 0 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 0 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 0 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 0 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 0 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 0 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 0 --run 12 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 0 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 0 --run 14 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 1 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 1 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 1 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 1 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 1 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 1 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 1 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 1 --run 12 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 1 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 1 --run 14 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 2 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 2 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 2 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 2 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 2 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 2 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 2 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 2 --run 12 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 2 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 2 --run 14 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 3 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 3 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 3 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 3 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 3 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 3 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 3 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 3 --run 12 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 3 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 3 --run 14 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 4 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 4 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 4 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 4 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 4 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 4 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 4 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 4 --run 12 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 4 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 4 --run 14 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 5 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 5 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 5 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 5 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 5 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 5 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 5 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 5 --run 12 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 5 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 5 --run 14 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 6 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 6 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 6 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 6 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 6 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 6 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 6 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 6 --run 12 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on nchain | Abl 6 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name nchain --ablation 6 --run 14 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 0 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 0 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 0 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 0 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 0 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 0 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 0 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 0 --run 12 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 0 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 0 --run 14 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 1 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 1 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 1 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 1 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 1 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 1 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 1 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 1 --run 12 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 1 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 1 --run 14 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 2 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 2 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 2 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 2 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 2 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 2 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 2 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 2 --run 12 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 2 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 2 --run 14 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 3 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 3 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 3 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 3 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 3 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 3 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 3 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 3 --run 12 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 3 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 3 --run 14 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 4 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 4 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 4 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 4 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 4 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 4 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 4 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 4 --run 12 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 4 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 4 --run 14 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 5 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 5 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 5 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 5 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 5 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 5 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 5 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 5 --run 12 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 5 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 5 --run 14 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 6 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 6 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 6 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 6 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 6 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 6 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 6 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 6 --run 12 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on nchain | Abl 6 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name nchain --ablation 6 --run 14 --device_name timpc_gpu
) &

# --- Concurrency Queue 1 ---
(
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 0 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 0 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 0 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 0 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 0 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 0 --run 11 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 0 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 0 --run 13 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 0 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 0 --run 15 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 1 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 1 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 1 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 1 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 1 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 1 --run 11 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 1 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 1 --run 13 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 1 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 1 --run 15 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 2 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 2 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 2 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 2 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 2 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 2 --run 11 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 2 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 2 --run 13 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 2 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 2 --run 15 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 3 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 3 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 3 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 3 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 3 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 3 --run 11 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 3 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 3 --run 13 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 3 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 3 --run 15 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 4 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 4 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 4 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 4 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 4 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 4 --run 11 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 4 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 4 --run 13 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 4 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 4 --run 15 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 5 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 5 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 5 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 5 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 5 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 5 --run 11 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 5 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 5 --run 13 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 5 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 5 --run 15 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 6 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 6 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 6 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 6 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 6 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 6 --run 11 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 6 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 6 --run 13 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on nchain | Abl 6 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name nchain --ablation 6 --run 15 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 0 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 0 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 0 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 0 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 0 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 0 --run 11 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 0 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 0 --run 13 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 0 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 0 --run 15 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 1 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 1 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 1 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 1 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 1 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 1 --run 11 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 1 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 1 --run 13 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 1 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 1 --run 15 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 2 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 2 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 2 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 2 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 2 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 2 --run 11 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 2 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 2 --run 13 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 2 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 2 --run 15 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 3 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 3 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 3 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 3 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 3 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 3 --run 11 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 3 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 3 --run 13 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 3 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 3 --run 15 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 4 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 4 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 4 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 4 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 4 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 4 --run 11 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 4 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 4 --run 13 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 4 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 4 --run 15 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 5 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 5 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 5 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 5 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 5 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 5 --run 11 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 5 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 5 --run 13 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 5 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 5 --run 15 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 6 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 6 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 6 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 6 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 6 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 6 --run 11 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 6 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 6 --run 13 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on nchain | Abl 6 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name nchain --ablation 6 --run 15 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 0 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 0 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 0 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 0 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 0 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 0 --run 11 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 0 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 0 --run 13 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 0 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 0 --run 15 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 1 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 1 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 1 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 1 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 1 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 1 --run 11 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 1 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 1 --run 13 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 1 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 1 --run 15 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 2 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 2 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 2 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 2 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 2 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 2 --run 11 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 2 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 2 --run 13 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 2 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 2 --run 15 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 3 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 3 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 3 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 3 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 3 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 3 --run 11 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 3 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 3 --run 13 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 3 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 3 --run 15 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 4 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 4 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 4 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 4 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 4 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 4 --run 11 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 4 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 4 --run 13 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 4 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 4 --run 15 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 5 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 5 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 5 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 5 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 5 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 5 --run 11 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 5 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 5 --run 13 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 5 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 5 --run 15 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 6 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 6 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 6 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 6 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 6 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 6 --run 11 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 6 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 6 --run 13 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on nchain | Abl 6 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name nchain --ablation 6 --run 15 --device_name timpc_gpu
) &

echo "All queues launched. Waiting for completion..."
wait

echo quit | nvidia-cuda-mps-control || true
sudo nvidia-smi -i 0 -c DEFAULT || true
echo "All tasks completed on this machine."
