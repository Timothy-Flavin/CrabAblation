#!/usr/bin/env bash
# Auto-generated multi-process schedule for white-machine_gpu
# Capacity: 2
set -euo pipefail

source .venv/bin/activate

# --- Concurrency Queue 0 ---
(
  echo "[white-machine_gpu - Q0] Running dqn on minigrid | Abl 1 | Run 6"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 1 --run 6 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q0] Running dqn on minigrid | Abl 2 | Run 7"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 2 --run 7 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q0] Running dqn on minigrid | Abl 2 | Run 9"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 2 --run 9 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q0] Running dqn on minigrid | Abl 6 | Run 9"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 6 --run 9 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q0] Running ppo on minigrid | Abl 6 | Run 8"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo ppo --env_name minigrid --ablation 6 --run 8 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q0] Running dqn on cartpole | Abl 6 | Run 6"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name cartpole --ablation 6 --run 6 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q0] Running dqn on cartpole | Abl 6 | Run 8"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name cartpole --ablation 6 --run 8 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q0] Running dqn on cartpole | Abl 6 | Run 10"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name cartpole --ablation 6 --run 10 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q0] Running sac on cartpole | Abl 6 | Run 8"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo sac --env_name cartpole --ablation 6 --run 8 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q0] Running sac on cartpole | Abl 6 | Run 10"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo sac --env_name cartpole --ablation 6 --run 10 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q0] Running sac on mujoco | Abl 4 | Run 7"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo sac --env_name mujoco --ablation 4 --run 7 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q0] Running sac on mujoco | Abl 4 | Run 9"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo sac --env_name mujoco --ablation 4 --run 9 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q0] Running sac on mujoco | Abl 5 | Run 8"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo sac --env_name mujoco --ablation 5 --run 8 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q0] Running ppo on mujoco | Abl 6 | Run 6"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo ppo --env_name mujoco --ablation 6 --run 6 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q0] Running ppo on mujoco | Abl 6 | Run 8"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo ppo --env_name mujoco --ablation 6 --run 8 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q0] Running ppo on mujoco | Abl 6 | Run 10"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo ppo --env_name mujoco --ablation 6 --run 10 --device_name white-machine_gpu
) &

# --- Concurrency Queue 1 ---
(
  echo "[white-machine_gpu - Q1] Running dqn on minigrid | Abl 2 | Run 6"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 2 --run 6 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q1] Running dqn on minigrid | Abl 2 | Run 8"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 2 --run 8 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q1] Running dqn on minigrid | Abl 6 | Run 8"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 6 --run 8 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q1] Running ppo on minigrid | Abl 6 | Run 7"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo ppo --env_name minigrid --ablation 6 --run 7 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q1] Running ppo on minigrid | Abl 6 | Run 10"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo ppo --env_name minigrid --ablation 6 --run 10 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q1] Running dqn on cartpole | Abl 6 | Run 7"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name cartpole --ablation 6 --run 7 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q1] Running dqn on cartpole | Abl 6 | Run 9"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name cartpole --ablation 6 --run 9 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q1] Running sac on cartpole | Abl 6 | Run 7"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo sac --env_name cartpole --ablation 6 --run 7 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q1] Running sac on cartpole | Abl 6 | Run 9"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo sac --env_name cartpole --ablation 6 --run 9 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q1] Running sac on mujoco | Abl 4 | Run 6"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo sac --env_name mujoco --ablation 4 --run 6 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q1] Running sac on mujoco | Abl 4 | Run 8"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo sac --env_name mujoco --ablation 4 --run 8 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q1] Running sac on mujoco | Abl 4 | Run 10"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo sac --env_name mujoco --ablation 4 --run 10 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q1] Running sac on mujoco | Abl 5 | Run 10"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo sac --env_name mujoco --ablation 5 --run 10 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q1] Running ppo on mujoco | Abl 6 | Run 7"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo ppo --env_name mujoco --ablation 6 --run 7 --device_name white-machine_gpu
  echo "[white-machine_gpu - Q1] Running ppo on mujoco | Abl 6 | Run 9"
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo ppo --env_name mujoco --ablation 6 --run 9 --device_name white-machine_gpu
) &

echo "All queues launched. Waiting for completion..."
wait

echo "All tasks completed on this machine."
