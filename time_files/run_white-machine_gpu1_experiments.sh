#!/usr/bin/env bash
# Auto-generated schedule for white-machine_gpu1
set -euo pipefail

source .venv/bin/activate

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 0 | Run 10"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 0 --run 10 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 5 | Run 6"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 5 --run 6 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 5 | Run 7"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 5 --run 7 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 5 | Run 8"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 5 --run 8 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 5 | Run 9"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 5 --run 9 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running dqn on minigrid | Ablation 5 | Run 10"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name minigrid --ablation 5 --run 10 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running sac on minigrid | Ablation 2 | Run 6"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo sac --env_name minigrid --ablation 2 --run 6 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running sac on minigrid | Ablation 2 | Run 7"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo sac --env_name minigrid --ablation 2 --run 7 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running sac on minigrid | Ablation 2 | Run 8"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo sac --env_name minigrid --ablation 2 --run 8 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running sac on minigrid | Ablation 2 | Run 9"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo sac --env_name minigrid --ablation 2 --run 9 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running sac on minigrid | Ablation 2 | Run 10"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo sac --env_name minigrid --ablation 2 --run 10 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running sac on minigrid | Ablation 3 | Run 9"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo sac --env_name minigrid --ablation 3 --run 9 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running sac on cartpole | Ablation 0 | Run 6"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo sac --env_name cartpole --ablation 0 --run 6 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running sac on cartpole | Ablation 3 | Run 6"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo sac --env_name cartpole --ablation 3 --run 6 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running sac on cartpole | Ablation 3 | Run 7"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo sac --env_name cartpole --ablation 3 --run 7 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running sac on cartpole | Ablation 3 | Run 8"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo sac --env_name cartpole --ablation 3 --run 8 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running sac on cartpole | Ablation 3 | Run 9"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo sac --env_name cartpole --ablation 3 --run 9 --device_name white-machine_gpu1

echo "[white-machine_gpu1] Running sac on cartpole | Ablation 3 | Run 10"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo sac --env_name cartpole --ablation 3 --run 10 --device_name white-machine_gpu1

