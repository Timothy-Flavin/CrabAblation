#!/usr/bin/env bash
# Auto-generated schedule for alienware_gpu_0
set -euo pipefail

source .venv/bin/activate

echo "[alienware_gpu_0] Running sac on cartpole | Ablation 0 | Run 7"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name cartpole --ablation 0 --run 7 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running sac on cartpole | Ablation 0 | Run 8"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name cartpole --ablation 0 --run 8 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running sac on cartpole | Ablation 0 | Run 9"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name cartpole --ablation 0 --run 9 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running sac on cartpole | Ablation 0 | Run 10"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name cartpole --ablation 0 --run 10 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running sac on cartpole | Ablation 2 | Run 6"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name cartpole --ablation 2 --run 6 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running sac on cartpole | Ablation 2 | Run 7"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name cartpole --ablation 2 --run 7 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running sac on cartpole | Ablation 2 | Run 8"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name cartpole --ablation 2 --run 8 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running sac on cartpole | Ablation 2 | Run 9"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name cartpole --ablation 2 --run 9 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running sac on cartpole | Ablation 2 | Run 10"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name cartpole --ablation 2 --run 10 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running sac on cartpole | Ablation 5 | Run 7"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name cartpole --ablation 5 --run 7 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running dqn on mujoco | Ablation 0 | Run 6"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo dqn --env_name mujoco --ablation 0 --run 6 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running dqn on mujoco | Ablation 0 | Run 7"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo dqn --env_name mujoco --ablation 0 --run 7 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running dqn on mujoco | Ablation 0 | Run 9"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo dqn --env_name mujoco --ablation 0 --run 9 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running dqn on mujoco | Ablation 0 | Run 10"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo dqn --env_name mujoco --ablation 0 --run 10 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running dqn on mujoco | Ablation 1 | Run 8"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo dqn --env_name mujoco --ablation 1 --run 8 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running dqn on mujoco | Ablation 1 | Run 10"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo dqn --env_name mujoco --ablation 1 --run 10 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running sac on mujoco | Ablation 6 | Run 6"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name mujoco --ablation 6 --run 6 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running sac on mujoco | Ablation 6 | Run 7"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name mujoco --ablation 6 --run 7 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running sac on mujoco | Ablation 6 | Run 8"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name mujoco --ablation 6 --run 8 --device_name alienware_gpu_0

echo "[alienware_gpu_0] Running sac on mujoco | Ablation 6 | Run 9"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,4,5 python runner.py --algo sac --env_name mujoco --ablation 6 --run 9 --device_name alienware_gpu_0

