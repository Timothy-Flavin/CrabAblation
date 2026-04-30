#!/usr/bin/env bash
# Auto-generated schedule for white-machine_gpu1
set -euo pipefail

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 0 | Run 1"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 0 --run 1 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 0 | Run 2"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 0 --run 2 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 0 | Run 3"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 0 --run 3 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 0 | Run 4"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 0 --run 4 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 0 | Run 5"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 0 --run 5 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 1 | Run 1"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 1 --run 1 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 1 | Run 2"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 1 --run 2 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 1 | Run 3"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 1 --run 3 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 1 | Run 4"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 1 --run 4 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 1 | Run 5"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 1 --run 5 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 2 | Run 1"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 2 --run 1 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 2 | Run 2"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 2 --run 2 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 2 | Run 3"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 2 --run 3 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 2 | Run 4"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 2 --run 4 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 2 | Run 5"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 2 --run 5 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 5 | Run 3"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 5 --run 3 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 5 | Run 5"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 5 --run 5 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on mujoco | Ablation 0 | Run 4"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name mujoco --ablation 0 --run 4 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on mujoco | Ablation 1 | Run 2"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name mujoco --ablation 1 --run 2 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on mujoco | Ablation 1 | Run 3"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name mujoco --ablation 1 --run 3 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on mujoco | Ablation 1 | Run 4"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name mujoco --ablation 1 --run 4 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on mujoco | Ablation 1 | Run 5"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name mujoco --ablation 1 --run 5 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running sac on mujoco | Ablation 0 | Run 1"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo sac --env_name mujoco --ablation 0 --run 1 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running sac on mujoco | Ablation 0 | Run 2"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo sac --env_name mujoco --ablation 0 --run 2 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running sac on mujoco | Ablation 0 | Run 3"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo sac --env_name mujoco --ablation 0 --run 3 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running sac on mujoco | Ablation 0 | Run 4"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo sac --env_name mujoco --ablation 0 --run 4 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running sac on mujoco | Ablation 0 | Run 5"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo sac --env_name mujoco --ablation 0 --run 5 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running sac on mujoco | Ablation 2 | Run 5"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo sac --env_name mujoco --ablation 2 --run 5 --device_name white-machine_gpu1

