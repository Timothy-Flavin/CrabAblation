#!/usr/bin/env bash
# Auto-generated schedule for white-machine_gpu0
set -euo pipefail

source .venv/bin/activate

echo "[white-machine_gpu0] Running sac on minigrid | Ablation 2 | Run 2"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo sac --env_name minigrid --ablation 2 --run 2 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running sac on minigrid | Ablation 2 | Run 4"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo sac --env_name minigrid --ablation 2 --run 4 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running sac on minigrid | Ablation 4 | Run 2"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo sac --env_name minigrid --ablation 4 --run 2 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running sac on minigrid | Ablation 4 | Run 3"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo sac --env_name minigrid --ablation 4 --run 3 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running sac on minigrid | Ablation 4 | Run 4"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo sac --env_name minigrid --ablation 4 --run 4 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running sac on minigrid | Ablation 5 | Run 1"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo sac --env_name minigrid --ablation 5 --run 1 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running sac on minigrid | Ablation 5 | Run 2"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo sac --env_name minigrid --ablation 5 --run 2 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running sac on minigrid | Ablation 5 | Run 3"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo sac --env_name minigrid --ablation 5 --run 3 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running sac on minigrid | Ablation 5 | Run 4"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo sac --env_name minigrid --ablation 5 --run 4 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running sac on minigrid | Ablation 5 | Run 5"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo sac --env_name minigrid --ablation 5 --run 5 --device_name white-machine_gpu0

echo "[white-machine_gpu0] Running sac on cartpole | Ablation 4 | Run 1"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo sac --env_name cartpole --ablation 4 --run 1 --device_name white-machine_gpu0

