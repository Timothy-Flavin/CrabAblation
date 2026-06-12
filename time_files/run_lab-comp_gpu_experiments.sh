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
  echo "[lab-comp_gpu - Q0] Running dqn on minigrid | Abl 0 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo dqn --env_name minigrid --ablation 0 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running dqn on minigrid | Abl 0 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo dqn --env_name minigrid --ablation 0 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running dqn on minigrid | Abl 1 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo dqn --env_name minigrid --ablation 1 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running dqn on minigrid | Abl 2 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo dqn --env_name minigrid --ablation 2 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running dqn on minigrid | Abl 3 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo dqn --env_name minigrid --ablation 3 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running dqn on minigrid | Abl 5 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo dqn --env_name minigrid --ablation 5 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running dqn on minigrid | Abl 5 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo dqn --env_name minigrid --ablation 5 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running sac on minigrid | Abl 0 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo sac --env_name minigrid --ablation 0 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running sac on minigrid | Abl 1 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo sac --env_name minigrid --ablation 1 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running sac on minigrid | Abl 2 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo sac --env_name minigrid --ablation 2 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running sac on minigrid | Abl 3 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo sac --env_name minigrid --ablation 3 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running sac on minigrid | Abl 3 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo sac --env_name minigrid --ablation 3 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running sac on minigrid | Abl 5 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo sac --env_name minigrid --ablation 5 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running ppo on minigrid | Abl 0 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo ppo --env_name minigrid --ablation 0 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running ppo on minigrid | Abl 1 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo ppo --env_name minigrid --ablation 1 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running ppo on minigrid | Abl 2 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo ppo --env_name minigrid --ablation 2 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running ppo on minigrid | Abl 2 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo ppo --env_name minigrid --ablation 2 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running ppo on minigrid | Abl 3 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo ppo --env_name minigrid --ablation 3 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running ppo on minigrid | Abl 5 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo ppo --env_name minigrid --ablation 5 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running dqn on cartpole | Abl 1 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo dqn --env_name cartpole --ablation 1 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running sac on cartpole | Abl 0 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo sac --env_name cartpole --ablation 0 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running sac on cartpole | Abl 1 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo sac --env_name cartpole --ablation 1 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running sac on cartpole | Abl 1 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo sac --env_name cartpole --ablation 1 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running sac on cartpole | Abl 2 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo sac --env_name cartpole --ablation 2 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running sac on cartpole | Abl 3 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo sac --env_name cartpole --ablation 3 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running sac on cartpole | Abl 5 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo sac --env_name cartpole --ablation 5 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running ppo on cartpole | Abl 0 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo ppo --env_name cartpole --ablation 0 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running ppo on cartpole | Abl 0 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo ppo --env_name cartpole --ablation 0 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running ppo on cartpole | Abl 1 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo ppo --env_name cartpole --ablation 1 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running ppo on cartpole | Abl 2 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo ppo --env_name cartpole --ablation 2 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running ppo on cartpole | Abl 3 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo ppo --env_name cartpole --ablation 3 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running ppo on cartpole | Abl 5 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo ppo --env_name cartpole --ablation 5 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running ppo on cartpole | Abl 5 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo ppo --env_name cartpole --ablation 5 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running dqn on mujoco | Abl 0 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo dqn --env_name mujoco --ablation 0 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running dqn on mujoco | Abl 1 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo dqn --env_name mujoco --ablation 1 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running dqn on mujoco | Abl 2 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo dqn --env_name mujoco --ablation 2 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running dqn on mujoco | Abl 3 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo dqn --env_name mujoco --ablation 3 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running dqn on mujoco | Abl 3 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo dqn --env_name mujoco --ablation 3 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running dqn on mujoco | Abl 5 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo dqn --env_name mujoco --ablation 5 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running sac on mujoco | Abl 0 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo sac --env_name mujoco --ablation 0 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running sac on mujoco | Abl 1 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo sac --env_name mujoco --ablation 1 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running sac on mujoco | Abl 2 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo sac --env_name mujoco --ablation 2 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running sac on mujoco | Abl 2 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo sac --env_name mujoco --ablation 2 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running sac on mujoco | Abl 3 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo sac --env_name mujoco --ablation 3 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running sac on mujoco | Abl 4 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo sac --env_name mujoco --ablation 4 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running sac on mujoco | Abl 5 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo sac --env_name mujoco --ablation 5 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running ppo on mujoco | Abl 0 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo ppo --env_name mujoco --ablation 0 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running ppo on mujoco | Abl 0 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo ppo --env_name mujoco --ablation 0 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running ppo on mujoco | Abl 1 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo ppo --env_name mujoco --ablation 1 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running ppo on mujoco | Abl 2 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo ppo --env_name mujoco --ablation 2 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running ppo on mujoco | Abl 3 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo ppo --env_name mujoco --ablation 3 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running ppo on mujoco | Abl 5 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo ppo --env_name mujoco --ablation 5 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q0] Running ppo on mujoco | Abl 5 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 16-19,48-51 python runner.py --algo ppo --env_name mujoco --ablation 5 --run 15 --device_name lab-comp_gpu
) &

# --- Concurrency Queue 1 ---
(
  echo "[lab-comp_gpu - Q1] Running dqn on minigrid | Abl 0 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo dqn --env_name minigrid --ablation 0 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running dqn on minigrid | Abl 1 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo dqn --env_name minigrid --ablation 1 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running dqn on minigrid | Abl 1 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo dqn --env_name minigrid --ablation 1 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running dqn on minigrid | Abl 2 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo dqn --env_name minigrid --ablation 2 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running dqn on minigrid | Abl 3 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo dqn --env_name minigrid --ablation 3 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running dqn on minigrid | Abl 5 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo dqn --env_name minigrid --ablation 5 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running sac on minigrid | Abl 0 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo sac --env_name minigrid --ablation 0 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running sac on minigrid | Abl 0 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo sac --env_name minigrid --ablation 0 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running sac on minigrid | Abl 1 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo sac --env_name minigrid --ablation 1 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running sac on minigrid | Abl 2 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo sac --env_name minigrid --ablation 2 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running sac on minigrid | Abl 3 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo sac --env_name minigrid --ablation 3 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running sac on minigrid | Abl 5 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo sac --env_name minigrid --ablation 5 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running sac on minigrid | Abl 5 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo sac --env_name minigrid --ablation 5 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running ppo on minigrid | Abl 0 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo ppo --env_name minigrid --ablation 0 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running ppo on minigrid | Abl 1 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo ppo --env_name minigrid --ablation 1 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running ppo on minigrid | Abl 2 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo ppo --env_name minigrid --ablation 2 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running ppo on minigrid | Abl 3 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo ppo --env_name minigrid --ablation 3 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running ppo on minigrid | Abl 3 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo ppo --env_name minigrid --ablation 3 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running ppo on minigrid | Abl 5 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo ppo --env_name minigrid --ablation 5 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running dqn on cartpole | Abl 1 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo dqn --env_name cartpole --ablation 1 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running sac on cartpole | Abl 0 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo sac --env_name cartpole --ablation 0 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running sac on cartpole | Abl 1 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo sac --env_name cartpole --ablation 1 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running sac on cartpole | Abl 2 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo sac --env_name cartpole --ablation 2 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running sac on cartpole | Abl 2 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo sac --env_name cartpole --ablation 2 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running sac on cartpole | Abl 3 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo sac --env_name cartpole --ablation 3 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running sac on cartpole | Abl 5 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo sac --env_name cartpole --ablation 5 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running ppo on cartpole | Abl 0 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo ppo --env_name cartpole --ablation 0 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running ppo on cartpole | Abl 1 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo ppo --env_name cartpole --ablation 1 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running ppo on cartpole | Abl 1 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo ppo --env_name cartpole --ablation 1 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running ppo on cartpole | Abl 2 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo ppo --env_name cartpole --ablation 2 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running ppo on cartpole | Abl 3 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo ppo --env_name cartpole --ablation 3 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running ppo on cartpole | Abl 5 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo ppo --env_name cartpole --ablation 5 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running dqn on mujoco | Abl 0 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo dqn --env_name mujoco --ablation 0 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running dqn on mujoco | Abl 0 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo dqn --env_name mujoco --ablation 0 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running dqn on mujoco | Abl 1 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo dqn --env_name mujoco --ablation 1 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running dqn on mujoco | Abl 2 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo dqn --env_name mujoco --ablation 2 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running dqn on mujoco | Abl 3 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo dqn --env_name mujoco --ablation 3 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running dqn on mujoco | Abl 5 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo dqn --env_name mujoco --ablation 5 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running dqn on mujoco | Abl 5 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo dqn --env_name mujoco --ablation 5 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running sac on mujoco | Abl 0 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo sac --env_name mujoco --ablation 0 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running sac on mujoco | Abl 1 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo sac --env_name mujoco --ablation 1 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running sac on mujoco | Abl 2 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo sac --env_name mujoco --ablation 2 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running sac on mujoco | Abl 3 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo sac --env_name mujoco --ablation 3 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running sac on mujoco | Abl 3 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo sac --env_name mujoco --ablation 3 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running sac on mujoco | Abl 4 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo sac --env_name mujoco --ablation 4 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running sac on mujoco | Abl 5 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo sac --env_name mujoco --ablation 5 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running ppo on mujoco | Abl 0 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo ppo --env_name mujoco --ablation 0 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running ppo on mujoco | Abl 1 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo ppo --env_name mujoco --ablation 1 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running ppo on mujoco | Abl 1 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo ppo --env_name mujoco --ablation 1 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running ppo on mujoco | Abl 2 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo ppo --env_name mujoco --ablation 2 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running ppo on mujoco | Abl 3 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo ppo --env_name mujoco --ablation 3 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q1] Running ppo on mujoco | Abl 5 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 20-23,52-55 python runner.py --algo ppo --env_name mujoco --ablation 5 --run 12 --device_name lab-comp_gpu
) &

# --- Concurrency Queue 2 ---
(
  echo "[lab-comp_gpu - Q2] Running dqn on minigrid | Abl 0 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo dqn --env_name minigrid --ablation 0 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running dqn on minigrid | Abl 1 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo dqn --env_name minigrid --ablation 1 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running dqn on minigrid | Abl 2 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo dqn --env_name minigrid --ablation 2 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running dqn on minigrid | Abl 2 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo dqn --env_name minigrid --ablation 2 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running dqn on minigrid | Abl 3 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo dqn --env_name minigrid --ablation 3 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running dqn on minigrid | Abl 5 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo dqn --env_name minigrid --ablation 5 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running sac on minigrid | Abl 0 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo sac --env_name minigrid --ablation 0 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running sac on minigrid | Abl 1 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo sac --env_name minigrid --ablation 1 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running sac on minigrid | Abl 1 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo sac --env_name minigrid --ablation 1 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running sac on minigrid | Abl 2 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo sac --env_name minigrid --ablation 2 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running sac on minigrid | Abl 3 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo sac --env_name minigrid --ablation 3 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running sac on minigrid | Abl 5 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo sac --env_name minigrid --ablation 5 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running ppo on minigrid | Abl 0 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo ppo --env_name minigrid --ablation 0 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running ppo on minigrid | Abl 0 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo ppo --env_name minigrid --ablation 0 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running ppo on minigrid | Abl 1 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo ppo --env_name minigrid --ablation 1 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running ppo on minigrid | Abl 2 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo ppo --env_name minigrid --ablation 2 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running ppo on minigrid | Abl 3 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo ppo --env_name minigrid --ablation 3 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running ppo on minigrid | Abl 5 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo ppo --env_name minigrid --ablation 5 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running ppo on minigrid | Abl 5 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo ppo --env_name minigrid --ablation 5 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running dqn on cartpole | Abl 2 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo dqn --env_name cartpole --ablation 2 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running sac on cartpole | Abl 0 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo sac --env_name cartpole --ablation 0 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running sac on cartpole | Abl 1 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo sac --env_name cartpole --ablation 1 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running sac on cartpole | Abl 2 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo sac --env_name cartpole --ablation 2 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running sac on cartpole | Abl 3 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo sac --env_name cartpole --ablation 3 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running sac on cartpole | Abl 3 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo sac --env_name cartpole --ablation 3 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running sac on cartpole | Abl 5 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo sac --env_name cartpole --ablation 5 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running ppo on cartpole | Abl 0 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo ppo --env_name cartpole --ablation 0 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running ppo on cartpole | Abl 1 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo ppo --env_name cartpole --ablation 1 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running ppo on cartpole | Abl 2 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo ppo --env_name cartpole --ablation 2 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running ppo on cartpole | Abl 2 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo ppo --env_name cartpole --ablation 2 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running ppo on cartpole | Abl 3 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo ppo --env_name cartpole --ablation 3 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running ppo on cartpole | Abl 5 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo ppo --env_name cartpole --ablation 5 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running dqn on mujoco | Abl 0 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo dqn --env_name mujoco --ablation 0 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running dqn on mujoco | Abl 1 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo dqn --env_name mujoco --ablation 1 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running dqn on mujoco | Abl 1 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo dqn --env_name mujoco --ablation 1 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running dqn on mujoco | Abl 2 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo dqn --env_name mujoco --ablation 2 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running dqn on mujoco | Abl 3 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo dqn --env_name mujoco --ablation 3 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running dqn on mujoco | Abl 5 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo dqn --env_name mujoco --ablation 5 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running sac on mujoco | Abl 0 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo sac --env_name mujoco --ablation 0 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running sac on mujoco | Abl 0 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo sac --env_name mujoco --ablation 0 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running sac on mujoco | Abl 1 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo sac --env_name mujoco --ablation 1 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running sac on mujoco | Abl 2 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo sac --env_name mujoco --ablation 2 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running sac on mujoco | Abl 3 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo sac --env_name mujoco --ablation 3 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running sac on mujoco | Abl 4 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo sac --env_name mujoco --ablation 4 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running sac on mujoco | Abl 5 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo sac --env_name mujoco --ablation 5 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running sac on mujoco | Abl 5 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo sac --env_name mujoco --ablation 5 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running ppo on mujoco | Abl 0 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo ppo --env_name mujoco --ablation 0 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running ppo on mujoco | Abl 1 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo ppo --env_name mujoco --ablation 1 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running ppo on mujoco | Abl 2 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo ppo --env_name mujoco --ablation 2 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running ppo on mujoco | Abl 2 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo ppo --env_name mujoco --ablation 2 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running ppo on mujoco | Abl 3 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo ppo --env_name mujoco --ablation 3 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q2] Running ppo on mujoco | Abl 5 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 24-27,56-59 python runner.py --algo ppo --env_name mujoco --ablation 5 --run 13 --device_name lab-comp_gpu
) &

# --- Concurrency Queue 3 ---
(
  echo "[lab-comp_gpu - Q3] Running dqn on minigrid | Abl 0 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo dqn --env_name minigrid --ablation 0 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running dqn on minigrid | Abl 1 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo dqn --env_name minigrid --ablation 1 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running dqn on minigrid | Abl 2 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo dqn --env_name minigrid --ablation 2 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running dqn on minigrid | Abl 3 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo dqn --env_name minigrid --ablation 3 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running dqn on minigrid | Abl 3 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo dqn --env_name minigrid --ablation 3 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running dqn on minigrid | Abl 5 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo dqn --env_name minigrid --ablation 5 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running sac on minigrid | Abl 0 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo sac --env_name minigrid --ablation 0 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running sac on minigrid | Abl 1 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo sac --env_name minigrid --ablation 1 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running sac on minigrid | Abl 2 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo sac --env_name minigrid --ablation 2 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running sac on minigrid | Abl 2 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo sac --env_name minigrid --ablation 2 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running sac on minigrid | Abl 3 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo sac --env_name minigrid --ablation 3 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running sac on minigrid | Abl 5 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo sac --env_name minigrid --ablation 5 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running ppo on minigrid | Abl 0 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo ppo --env_name minigrid --ablation 0 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running ppo on minigrid | Abl 1 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo ppo --env_name minigrid --ablation 1 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running ppo on minigrid | Abl 1 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo ppo --env_name minigrid --ablation 1 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running ppo on minigrid | Abl 2 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo ppo --env_name minigrid --ablation 2 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running ppo on minigrid | Abl 3 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo ppo --env_name minigrid --ablation 3 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running ppo on minigrid | Abl 5 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo ppo --env_name minigrid --ablation 5 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running dqn on cartpole | Abl 1 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo dqn --env_name cartpole --ablation 1 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running sac on cartpole | Abl 0 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo sac --env_name cartpole --ablation 0 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running sac on cartpole | Abl 0 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo sac --env_name cartpole --ablation 0 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running sac on cartpole | Abl 1 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo sac --env_name cartpole --ablation 1 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running sac on cartpole | Abl 2 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo sac --env_name cartpole --ablation 2 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running sac on cartpole | Abl 3 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo sac --env_name cartpole --ablation 3 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running sac on cartpole | Abl 5 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo sac --env_name cartpole --ablation 5 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running sac on cartpole | Abl 5 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo sac --env_name cartpole --ablation 5 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running ppo on cartpole | Abl 0 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo ppo --env_name cartpole --ablation 0 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running ppo on cartpole | Abl 1 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo ppo --env_name cartpole --ablation 1 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running ppo on cartpole | Abl 2 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo ppo --env_name cartpole --ablation 2 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running ppo on cartpole | Abl 3 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo ppo --env_name cartpole --ablation 3 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running ppo on cartpole | Abl 3 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo ppo --env_name cartpole --ablation 3 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running ppo on cartpole | Abl 5 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo ppo --env_name cartpole --ablation 5 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running dqn on mujoco | Abl 0 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo dqn --env_name mujoco --ablation 0 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running dqn on mujoco | Abl 1 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo dqn --env_name mujoco --ablation 1 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running dqn on mujoco | Abl 2 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo dqn --env_name mujoco --ablation 2 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running dqn on mujoco | Abl 2 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo dqn --env_name mujoco --ablation 2 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running dqn on mujoco | Abl 3 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo dqn --env_name mujoco --ablation 3 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running dqn on mujoco | Abl 5 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo dqn --env_name mujoco --ablation 5 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running sac on mujoco | Abl 0 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo sac --env_name mujoco --ablation 0 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running sac on mujoco | Abl 1 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo sac --env_name mujoco --ablation 1 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running sac on mujoco | Abl 1 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo sac --env_name mujoco --ablation 1 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running sac on mujoco | Abl 2 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo sac --env_name mujoco --ablation 2 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running sac on mujoco | Abl 3 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo sac --env_name mujoco --ablation 3 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running sac on mujoco | Abl 4 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo sac --env_name mujoco --ablation 4 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running sac on mujoco | Abl 5 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo sac --env_name mujoco --ablation 5 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running sac on mujoco | Abl 6 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo sac --env_name mujoco --ablation 6 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running ppo on mujoco | Abl 0 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo ppo --env_name mujoco --ablation 0 --run 14 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running ppo on mujoco | Abl 1 | Run 13"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo ppo --env_name mujoco --ablation 1 --run 13 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running ppo on mujoco | Abl 2 | Run 12"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo ppo --env_name mujoco --ablation 2 --run 12 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running ppo on mujoco | Abl 3 | Run 11"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo ppo --env_name mujoco --ablation 3 --run 11 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running ppo on mujoco | Abl 3 | Run 15"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo ppo --env_name mujoco --ablation 3 --run 15 --device_name lab-comp_gpu
  echo "[lab-comp_gpu - Q3] Running ppo on mujoco | Abl 5 | Run 14"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 numactl --preferred=1 taskset -c 28-31,60-63 python runner.py --algo ppo --env_name mujoco --ablation 5 --run 14 --device_name lab-comp_gpu
) &

echo "All queues launched. Waiting for completion..."
wait

echo quit | nvidia-cuda-mps-control || true
sudo nvidia-smi -i 0 -c DEFAULT || true
echo "All tasks completed on this machine."
