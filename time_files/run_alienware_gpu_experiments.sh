#!/usr/bin/env bash
# Auto-generated multi-process schedule for alienware_gpu
# Capacity: 2
set -euo pipefail

source .venv/bin/activate

# --- Concurrency Queue 0 ---
(
  echo "[alienware_gpu - Q0] Running ppo on minigrid | Abl 6 | Run 6"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo ppo --env_name minigrid --ablation 6 --run 6 --device_name alienware_gpu
  echo "[alienware_gpu - Q0] Running sac on cartpole | Abl 0 | Run 6"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name cartpole --ablation 0 --run 6 --device_name alienware_gpu
  echo "[alienware_gpu - Q0] Running sac on cartpole | Abl 0 | Run 8"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name cartpole --ablation 0 --run 8 --device_name alienware_gpu
  echo "[alienware_gpu - Q0] Running sac on cartpole | Abl 0 | Run 10"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name cartpole --ablation 0 --run 10 --device_name alienware_gpu
  echo "[alienware_gpu - Q0] Running sac on cartpole | Abl 2 | Run 6"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name cartpole --ablation 2 --run 6 --device_name alienware_gpu
  echo "[alienware_gpu - Q0] Running sac on cartpole | Abl 2 | Run 8"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name cartpole --ablation 2 --run 8 --device_name alienware_gpu
  echo "[alienware_gpu - Q0] Running sac on cartpole | Abl 2 | Run 10"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name cartpole --ablation 2 --run 10 --device_name alienware_gpu
  echo "[alienware_gpu - Q0] Running sac on cartpole | Abl 3 | Run 7"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name cartpole --ablation 3 --run 7 --device_name alienware_gpu
  echo "[alienware_gpu - Q0] Running sac on cartpole | Abl 3 | Run 9"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name cartpole --ablation 3 --run 9 --device_name alienware_gpu
  echo "[alienware_gpu - Q0] Running sac on cartpole | Abl 5 | Run 6"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name cartpole --ablation 5 --run 6 --device_name alienware_gpu
  echo "[alienware_gpu - Q0] Running sac on cartpole | Abl 5 | Run 8"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name cartpole --ablation 5 --run 8 --device_name alienware_gpu
  echo "[alienware_gpu - Q0] Running sac on cartpole | Abl 5 | Run 10"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name cartpole --ablation 5 --run 10 --device_name alienware_gpu
  echo "[alienware_gpu - Q0] Running dqn on mujoco | Abl 0 | Run 7"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo dqn --env_name mujoco --ablation 0 --run 7 --device_name alienware_gpu
  echo "[alienware_gpu - Q0] Running dqn on mujoco | Abl 0 | Run 9"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo dqn --env_name mujoco --ablation 0 --run 9 --device_name alienware_gpu
  echo "[alienware_gpu - Q0] Running sac on mujoco | Abl 6 | Run 6"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name mujoco --ablation 6 --run 6 --device_name alienware_gpu
  echo "[alienware_gpu - Q0] Running sac on mujoco | Abl 6 | Run 8"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name mujoco --ablation 6 --run 8 --device_name alienware_gpu
  echo "[alienware_gpu - Q0] Running sac on mujoco | Abl 6 | Run 10"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name mujoco --ablation 6 --run 10 --device_name alienware_gpu
) &

# --- Concurrency Queue 1 ---
(
  echo "[alienware_gpu - Q1] Running ppo on minigrid | Abl 6 | Run 9"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo ppo --env_name minigrid --ablation 6 --run 9 --device_name alienware_gpu
  echo "[alienware_gpu - Q1] Running sac on cartpole | Abl 0 | Run 7"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 0 --run 7 --device_name alienware_gpu
  echo "[alienware_gpu - Q1] Running sac on cartpole | Abl 0 | Run 9"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 0 --run 9 --device_name alienware_gpu
  echo "[alienware_gpu - Q1] Running sac on cartpole | Abl 1 | Run 10"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 1 --run 10 --device_name alienware_gpu
  echo "[alienware_gpu - Q1] Running sac on cartpole | Abl 2 | Run 7"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 2 --run 7 --device_name alienware_gpu
  echo "[alienware_gpu - Q1] Running sac on cartpole | Abl 2 | Run 9"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 2 --run 9 --device_name alienware_gpu
  echo "[alienware_gpu - Q1] Running sac on cartpole | Abl 3 | Run 6"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 3 --run 6 --device_name alienware_gpu
  echo "[alienware_gpu - Q1] Running sac on cartpole | Abl 3 | Run 8"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 3 --run 8 --device_name alienware_gpu
  echo "[alienware_gpu - Q1] Running sac on cartpole | Abl 3 | Run 10"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 3 --run 10 --device_name alienware_gpu
  echo "[alienware_gpu - Q1] Running sac on cartpole | Abl 5 | Run 7"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 5 --run 7 --device_name alienware_gpu
  echo "[alienware_gpu - Q1] Running sac on cartpole | Abl 5 | Run 9"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 5 --run 9 --device_name alienware_gpu
  echo "[alienware_gpu - Q1] Running dqn on mujoco | Abl 0 | Run 6"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo dqn --env_name mujoco --ablation 0 --run 6 --device_name alienware_gpu
  echo "[alienware_gpu - Q1] Running dqn on mujoco | Abl 0 | Run 8"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo dqn --env_name mujoco --ablation 0 --run 8 --device_name alienware_gpu
  echo "[alienware_gpu - Q1] Running dqn on mujoco | Abl 0 | Run 10"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo dqn --env_name mujoco --ablation 0 --run 10 --device_name alienware_gpu
  echo "[alienware_gpu - Q1] Running sac on mujoco | Abl 6 | Run 7"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name mujoco --ablation 6 --run 7 --device_name alienware_gpu
  echo "[alienware_gpu - Q1] Running sac on mujoco | Abl 6 | Run 9"
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name mujoco --ablation 6 --run 9 --device_name alienware_gpu
) &

echo "All queues launched. Waiting for completion..."
wait

echo "All tasks completed on this machine."
