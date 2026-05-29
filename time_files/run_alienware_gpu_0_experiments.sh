#!/usr/bin/env bash
# Auto-generated schedule for alienware_gpu_0
set -euo pipefail

source .venv/bin/activate

echo "[alienware_gpu_0] Running dqn on cartpole | Ablation 3 | Run 1"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo dqn --env_name cartpole --ablation 3 --run 1 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running dqn on cartpole | Ablation 3 | Run 2"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo dqn --env_name cartpole --ablation 3 --run 2 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running dqn on mujoco | Ablation 0 | Run 1"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo dqn --env_name mujoco --ablation 0 --run 1 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running dqn on mujoco | Ablation 0 | Run 2"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo dqn --env_name mujoco --ablation 0 --run 2 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running dqn on mujoco | Ablation 0 | Run 3"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo dqn --env_name mujoco --ablation 0 --run 3 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running dqn on mujoco | Ablation 0 | Run 4"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo dqn --env_name mujoco --ablation 0 --run 4 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running dqn on mujoco | Ablation 0 | Run 5"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo dqn --env_name mujoco --ablation 0 --run 5 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running dqn on mujoco | Ablation 2 | Run 1"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo dqn --env_name mujoco --ablation 2 --run 1 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running dqn on mujoco | Ablation 2 | Run 4"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo dqn --env_name mujoco --ablation 2 --run 4 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running dqn on mujoco | Ablation 5 | Run 3"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo dqn --env_name mujoco --ablation 5 --run 3 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running dqn on mujoco | Ablation 5 | Run 5"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo dqn --env_name mujoco --ablation 5 --run 5 --device_name alienware_gpu_0

