#!/usr/bin/env bash
# Auto-generated schedule for white-machine_gpu0
set -euo pipefail

source .venv/bin/activate

echo "[white-machine_gpu0] Running dqn on minigrid | Ablation 0 | Run 1"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 0 --run 1 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on minigrid | Ablation 0 | Run 2"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 0 --run 2 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on minigrid | Ablation 0 | Run 5"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 0 --run 5 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on minigrid | Ablation 1 | Run 1"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 1 --run 1 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on minigrid | Ablation 1 | Run 2"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 1 --run 2 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on minigrid | Ablation 1 | Run 4"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 1 --run 4 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on minigrid | Ablation 1 | Run 5"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 1 --run 5 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on minigrid | Ablation 2 | Run 1"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 2 --run 1 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on minigrid | Ablation 2 | Run 2"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 2 --run 2 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on minigrid | Ablation 2 | Run 3"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 2 --run 3 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on minigrid | Ablation 2 | Run 4"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 2 --run 4 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on minigrid | Ablation 2 | Run 5"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 2 --run 5 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on cartpole | Ablation 3 | Run 3"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name cartpole --ablation 3 --run 3 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on cartpole | Ablation 3 | Run 4"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name cartpole --ablation 3 --run 4 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on cartpole | Ablation 6 | Run 3"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name cartpole --ablation 6 --run 3 --device_name white-machine_gpu0

