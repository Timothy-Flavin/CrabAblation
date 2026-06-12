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
  echo "[timpc_gpu - Q0] Running ppo on minigrid | Abl 0 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name minigrid --ablation 0 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on minigrid | Abl 0 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name minigrid --ablation 0 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on minigrid | Abl 0 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name minigrid --ablation 0 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on minigrid | Abl 1 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name minigrid --ablation 1 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on minigrid | Abl 1 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name minigrid --ablation 1 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on minigrid | Abl 2 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name minigrid --ablation 2 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on minigrid | Abl 3 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name minigrid --ablation 3 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on minigrid | Abl 3 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name minigrid --ablation 3 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on minigrid | Abl 3 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name minigrid --ablation 3 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on minigrid | Abl 5 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name minigrid --ablation 5 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on minigrid | Abl 5 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name minigrid --ablation 5 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on cartpole | Abl 1 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name cartpole --ablation 1 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on cartpole | Abl 1 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name cartpole --ablation 1 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on cartpole | Abl 0 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name cartpole --ablation 0 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on cartpole | Abl 0 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name cartpole --ablation 0 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on cartpole | Abl 0 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name cartpole --ablation 0 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on cartpole | Abl 1 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name cartpole --ablation 1 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on cartpole | Abl 1 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name cartpole --ablation 1 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on cartpole | Abl 2 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name cartpole --ablation 2 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on cartpole | Abl 2 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name cartpole --ablation 2 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on cartpole | Abl 2 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name cartpole --ablation 2 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on cartpole | Abl 3 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name cartpole --ablation 3 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on cartpole | Abl 3 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name cartpole --ablation 3 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on cartpole | Abl 4 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name cartpole --ablation 4 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on cartpole | Abl 5 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name cartpole --ablation 5 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on cartpole | Abl 5 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name cartpole --ablation 5 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on mujoco | Abl 1 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name mujoco --ablation 1 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on mujoco | Abl 1 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name mujoco --ablation 1 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on mujoco | Abl 2 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name mujoco --ablation 2 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on mujoco | Abl 2 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name mujoco --ablation 2 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on mujoco | Abl 2 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name mujoco --ablation 2 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on mujoco | Abl 3 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name mujoco --ablation 3 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on mujoco | Abl 3 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name mujoco --ablation 3 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on mujoco | Abl 5 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name mujoco --ablation 5 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on mujoco | Abl 5 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name mujoco --ablation 5 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running dqn on mujoco | Abl 5 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo dqn --env_name mujoco --ablation 5 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on mujoco | Abl 0 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name mujoco --ablation 0 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on mujoco | Abl 0 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name mujoco --ablation 0 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on mujoco | Abl 1 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name mujoco --ablation 1 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on mujoco | Abl 1 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name mujoco --ablation 1 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on mujoco | Abl 1 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name mujoco --ablation 1 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on mujoco | Abl 2 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name mujoco --ablation 2 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on mujoco | Abl 2 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name mujoco --ablation 2 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on mujoco | Abl 3 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name mujoco --ablation 3 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on mujoco | Abl 3 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name mujoco --ablation 3 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on mujoco | Abl 3 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name mujoco --ablation 3 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running sac on mujoco | Abl 5 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo sac --env_name mujoco --ablation 5 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on mujoco | Abl 0 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name mujoco --ablation 0 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on mujoco | Abl 0 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name mujoco --ablation 0 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on mujoco | Abl 0 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name mujoco --ablation 0 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on mujoco | Abl 1 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name mujoco --ablation 1 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on mujoco | Abl 1 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name mujoco --ablation 1 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on mujoco | Abl 2 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name mujoco --ablation 2 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on mujoco | Abl 2 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name mujoco --ablation 2 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on mujoco | Abl 2 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name mujoco --ablation 2 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on mujoco | Abl 3 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name mujoco --ablation 3 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on mujoco | Abl 3 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name mujoco --ablation 3 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on mujoco | Abl 4 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name mujoco --ablation 4 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on mujoco | Abl 4 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name mujoco --ablation 4 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on mujoco | Abl 4 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name mujoco --ablation 4 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on mujoco | Abl 5 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name mujoco --ablation 5 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q0] Running ppo on mujoco | Abl 5 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 0-7,16-23 python runner.py --algo ppo --env_name mujoco --ablation 5 --run 9 --device_name timpc_gpu
) &

# --- Concurrency Queue 1 ---
(
  echo "[timpc_gpu - Q1] Running ppo on minigrid | Abl 0 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name minigrid --ablation 0 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on minigrid | Abl 0 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name minigrid --ablation 0 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on minigrid | Abl 1 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name minigrid --ablation 1 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on minigrid | Abl 1 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name minigrid --ablation 1 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on minigrid | Abl 1 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name minigrid --ablation 1 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on minigrid | Abl 2 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name minigrid --ablation 2 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on minigrid | Abl 3 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name minigrid --ablation 3 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on minigrid | Abl 3 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name minigrid --ablation 3 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on minigrid | Abl 5 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name minigrid --ablation 5 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on minigrid | Abl 5 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name minigrid --ablation 5 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on minigrid | Abl 5 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name minigrid --ablation 5 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on cartpole | Abl 1 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name cartpole --ablation 1 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on cartpole | Abl 1 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name cartpole --ablation 1 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on cartpole | Abl 0 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name cartpole --ablation 0 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on cartpole | Abl 0 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name cartpole --ablation 0 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on cartpole | Abl 1 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name cartpole --ablation 1 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on cartpole | Abl 1 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name cartpole --ablation 1 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on cartpole | Abl 1 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name cartpole --ablation 1 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on cartpole | Abl 2 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name cartpole --ablation 2 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on cartpole | Abl 2 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name cartpole --ablation 2 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on cartpole | Abl 3 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name cartpole --ablation 3 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on cartpole | Abl 3 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name cartpole --ablation 3 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on cartpole | Abl 3 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name cartpole --ablation 3 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on cartpole | Abl 5 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name cartpole --ablation 5 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on cartpole | Abl 5 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name cartpole --ablation 5 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on cartpole | Abl 5 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name cartpole --ablation 5 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on mujoco | Abl 1 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name mujoco --ablation 1 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on mujoco | Abl 1 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name mujoco --ablation 1 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on mujoco | Abl 2 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name mujoco --ablation 2 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on mujoco | Abl 2 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name mujoco --ablation 2 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on mujoco | Abl 3 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name mujoco --ablation 3 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on mujoco | Abl 3 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name mujoco --ablation 3 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on mujoco | Abl 3 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name mujoco --ablation 3 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on mujoco | Abl 5 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name mujoco --ablation 5 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running dqn on mujoco | Abl 5 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo dqn --env_name mujoco --ablation 5 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on mujoco | Abl 0 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name mujoco --ablation 0 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on mujoco | Abl 0 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name mujoco --ablation 0 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on mujoco | Abl 0 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name mujoco --ablation 0 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on mujoco | Abl 1 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name mujoco --ablation 1 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on mujoco | Abl 1 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name mujoco --ablation 1 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on mujoco | Abl 2 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name mujoco --ablation 2 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on mujoco | Abl 2 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name mujoco --ablation 2 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on mujoco | Abl 2 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name mujoco --ablation 2 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on mujoco | Abl 3 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name mujoco --ablation 3 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on mujoco | Abl 3 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name mujoco --ablation 3 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on mujoco | Abl 5 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name mujoco --ablation 5 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running sac on mujoco | Abl 5 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo sac --env_name mujoco --ablation 5 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on mujoco | Abl 0 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name mujoco --ablation 0 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on mujoco | Abl 0 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name mujoco --ablation 0 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on mujoco | Abl 1 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name mujoco --ablation 1 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on mujoco | Abl 1 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name mujoco --ablation 1 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on mujoco | Abl 1 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name mujoco --ablation 1 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on mujoco | Abl 2 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name mujoco --ablation 2 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on mujoco | Abl 2 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name mujoco --ablation 2 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on mujoco | Abl 3 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name mujoco --ablation 3 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on mujoco | Abl 3 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name mujoco --ablation 3 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on mujoco | Abl 3 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name mujoco --ablation 3 --run 10 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on mujoco | Abl 4 | Run 7"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name mujoco --ablation 4 --run 7 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on mujoco | Abl 4 | Run 9"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name mujoco --ablation 4 --run 9 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on mujoco | Abl 5 | Run 6"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name mujoco --ablation 5 --run 6 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on mujoco | Abl 5 | Run 8"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name mujoco --ablation 5 --run 8 --device_name timpc_gpu
  echo "[timpc_gpu - Q1] Running ppo on mujoco | Abl 5 | Run 10"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 taskset -c 8-15,24-31 python runner.py --algo ppo --env_name mujoco --ablation 5 --run 10 --device_name timpc_gpu
) &

echo "All queues launched. Waiting for completion..."
wait

echo quit | nvidia-cuda-mps-control || true
sudo nvidia-smi -i 0 -c DEFAULT || true
echo "All tasks completed on this machine."
