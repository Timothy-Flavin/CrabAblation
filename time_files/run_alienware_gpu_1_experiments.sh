#!/usr/bin/env bash
# Auto-generated schedule for alienware_gpu_1
set -euo pipefail

source .venv/bin/activate

echo "[alienware_gpu_1] Running sac on cartpole | Ablation 1 | Run 6"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 1 --run 6 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on cartpole | Ablation 1 | Run 7"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 1 --run 7 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on cartpole | Ablation 1 | Run 8"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 1 --run 8 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on cartpole | Ablation 1 | Run 9"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 1 --run 9 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on cartpole | Ablation 1 | Run 10"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 1 --run 10 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on cartpole | Ablation 5 | Run 6"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 5 --run 6 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on cartpole | Ablation 5 | Run 8"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 5 --run 8 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on cartpole | Ablation 5 | Run 9"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 5 --run 9 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on cartpole | Ablation 5 | Run 10"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 5 --run 10 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on mujoco | Ablation 1 | Run 7"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name mujoco --ablation 1 --run 7 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on mujoco | Ablation 1 | Run 9"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name mujoco --ablation 1 --run 9 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on mujoco | Ablation 1 | Run 10"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name mujoco --ablation 1 --run 10 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on mujoco | Ablation 3 | Run 6"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name mujoco --ablation 3 --run 6 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on mujoco | Ablation 3 | Run 7"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name mujoco --ablation 3 --run 7 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on mujoco | Ablation 3 | Run 8"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name mujoco --ablation 3 --run 8 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on mujoco | Ablation 3 | Run 9"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name mujoco --ablation 3 --run 9 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on mujoco | Ablation 3 | Run 10"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name mujoco --ablation 3 --run 10 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on mujoco | Ablation 6 | Run 10"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name mujoco --ablation 6 --run 10 --device_name alienware_gpu_1

