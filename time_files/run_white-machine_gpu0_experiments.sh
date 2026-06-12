#!/usr/bin/env bash
# Auto-generated schedule for white-machine_gpu0
set -euo pipefail

source .venv/bin/activate

echo "[white-machine_gpu0] Running dqn on minigrid | Ablation 0 | Run 6"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 0 --run 6 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on minigrid | Ablation 0 | Run 7"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 0 --run 7 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on minigrid | Ablation 0 | Run 8"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 0 --run 8 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on minigrid | Ablation 0 | Run 9"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 0 --run 9 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on minigrid | Ablation 1 | Run 6"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 1 --run 6 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on minigrid | Ablation 1 | Run 7"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 1 --run 7 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on minigrid | Ablation 1 | Run 8"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 1 --run 8 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on minigrid | Ablation 1 | Run 9"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 1 --run 9 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on minigrid | Ablation 1 | Run 10"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 1 --run 10 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on minigrid | Ablation 2 | Run 6"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 2 --run 6 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on minigrid | Ablation 2 | Run 7"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 2 --run 7 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on minigrid | Ablation 2 | Run 8"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 2 --run 8 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on minigrid | Ablation 2 | Run 9"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 2 --run 9 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running dqn on minigrid | Ablation 2 | Run 10"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name minigrid --ablation 2 --run 10 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running sac on mujoco | Ablation 5 | Run 8"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo sac --env_name mujoco --ablation 5 --run 8 --device_name white-machine_gpu0

