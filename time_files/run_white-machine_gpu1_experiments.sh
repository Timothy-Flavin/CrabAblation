#!/usr/bin/env bash
# Auto-generated schedule for white-machine_gpu1
set -euo pipefail

source .venv/bin/activate

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 0 | Run 2"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 0 --run 2 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 4 | Run 3"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 4 --run 3 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 6 | Run 1"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 6 --run 1 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 6 | Run 3"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 6 --run 3 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on mujoco | Ablation 6 | Run 4"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name mujoco --ablation 6 --run 4 --device_name white-machine_gpu1

