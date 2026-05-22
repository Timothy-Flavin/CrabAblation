#!/usr/bin/env bash
# Auto-generated schedule for alienware_gpu_1
set -euo pipefail

source .venv/bin/activate

echo "[alienware_gpu_1] Running sac on minigrid | Ablation 0 | Run 4"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name minigrid --ablation 0 --run 4 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on minigrid | Ablation 1 | Run 5"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name minigrid --ablation 1 --run 5 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on minigrid | Ablation 3 | Run 1"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name minigrid --ablation 3 --run 1 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on minigrid | Ablation 3 | Run 2"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name minigrid --ablation 3 --run 2 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on minigrid | Ablation 3 | Run 3"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name minigrid --ablation 3 --run 3 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on minigrid | Ablation 3 | Run 5"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name minigrid --ablation 3 --run 5 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on minigrid | Ablation 4 | Run 5"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name minigrid --ablation 4 --run 5 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on cartpole | Ablation 1 | Run 1"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 1 --run 1 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on cartpole | Ablation 1 | Run 2"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 1 --run 2 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on cartpole | Ablation 1 | Run 3"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 1 --run 3 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on cartpole | Ablation 1 | Run 4"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 1 --run 4 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on cartpole | Ablation 1 | Run 5"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 1 --run 5 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on cartpole | Ablation 2 | Run 1"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 2 --run 1 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on cartpole | Ablation 2 | Run 2"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 2 --run 2 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on cartpole | Ablation 2 | Run 3"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 2 --run 3 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on cartpole | Ablation 2 | Run 4"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 2 --run 4 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on cartpole | Ablation 2 | Run 5"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 2 --run 5 --device_name alienware_gpu_1

echo "[alienware_gpu_1] Running sac on cartpole | Ablation 3 | Run 4"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 taskset -c 2,3,6,7 python runner.py --algo sac --env_name cartpole --ablation 3 --run 4 --device_name alienware_gpu_1

