#!/usr/bin/env bash
# Auto-generated schedule for alienware_gpu_1
set -euo pipefail

echo "[alienware_gpu_1] Running dqn on cartpole | Ablation 1 | Run 4"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo dqn --env_name cartpole --ablation 1 --run 4 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running dqn on cartpole | Ablation 2 | Run 1"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo dqn --env_name cartpole --ablation 2 --run 1 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running dqn on cartpole | Ablation 2 | Run 2"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo dqn --env_name cartpole --ablation 2 --run 2 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running dqn on cartpole | Ablation 2 | Run 3"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo dqn --env_name cartpole --ablation 2 --run 3 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running dqn on cartpole | Ablation 2 | Run 4"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo dqn --env_name cartpole --ablation 2 --run 4 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running dqn on cartpole | Ablation 4 | Run 1"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo dqn --env_name cartpole --ablation 4 --run 1 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running dqn on cartpole | Ablation 4 | Run 2"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo dqn --env_name cartpole --ablation 4 --run 2 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running dqn on cartpole | Ablation 4 | Run 3"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo dqn --env_name cartpole --ablation 4 --run 3 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running dqn on cartpole | Ablation 4 | Run 4"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo dqn --env_name cartpole --ablation 4 --run 4 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running dqn on cartpole | Ablation 4 | Run 5"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo dqn --env_name cartpole --ablation 4 --run 5 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running dqn on cartpole | Ablation 5 | Run 1"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo dqn --env_name cartpole --ablation 5 --run 1 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running dqn on cartpole | Ablation 5 | Run 2"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo dqn --env_name cartpole --ablation 5 --run 2 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running dqn on cartpole | Ablation 5 | Run 3"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo dqn --env_name cartpole --ablation 5 --run 3 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running dqn on cartpole | Ablation 5 | Run 4"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo dqn --env_name cartpole --ablation 5 --run 4 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running dqn on cartpole | Ablation 5 | Run 5"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo dqn --env_name cartpole --ablation 5 --run 5 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running dqn on mujoco | Ablation 2 | Run 1"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo dqn --env_name mujoco --ablation 2 --run 1 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running dqn on mujoco | Ablation 3 | Run 4"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo dqn --env_name mujoco --ablation 3 --run 4 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running dqn on mujoco | Ablation 5 | Run 2"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo dqn --env_name mujoco --ablation 5 --run 2 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running ppo on mujoco | Ablation 4 | Run 1"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo ppo --env_name mujoco --ablation 4 --run 1 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running ppo on mujoco | Ablation 4 | Run 2"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo ppo --env_name mujoco --ablation 4 --run 2 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running ppo on mujoco | Ablation 4 | Run 3"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo ppo --env_name mujoco --ablation 4 --run 3 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running ppo on mujoco | Ablation 4 | Run 4"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo ppo --env_name mujoco --ablation 4 --run 4 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running ppo on mujoco | Ablation 4 | Run 5"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo ppo --env_name mujoco --ablation 4 --run 5 --device_name alienware_gpu_1

