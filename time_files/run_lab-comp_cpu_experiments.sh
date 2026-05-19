#!/usr/bin/env bash
# Auto-generated schedule for lab-comp_cpu
set -euo pipefail

source .venv/bin/activate

echo "[lab-comp_cpu] Running dqn on minigrid | Ablation 0 | Run 1"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES="" numactl --cpunodebind=0 --membind=0 python runner.py --algo dqn --env_name minigrid --ablation 0 --run 1 --device_name lab-comp_cpu

echo "[lab-comp_cpu] Running dqn on minigrid | Ablation 0 | Run 2"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES="" numactl --cpunodebind=0 --membind=0 python runner.py --algo dqn --env_name minigrid --ablation 0 --run 2 --device_name lab-comp_cpu

echo "[lab-comp_cpu] Running dqn on minigrid | Ablation 0 | Run 3"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES="" numactl --cpunodebind=0 --membind=0 python runner.py --algo dqn --env_name minigrid --ablation 0 --run 3 --device_name lab-comp_cpu

echo "[lab-comp_cpu] Running dqn on minigrid | Ablation 0 | Run 4"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES="" numactl --cpunodebind=0 --membind=0 python runner.py --algo dqn --env_name minigrid --ablation 0 --run 4 --device_name lab-comp_cpu

echo "[lab-comp_cpu] Running dqn on minigrid | Ablation 0 | Run 5"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES="" numactl --cpunodebind=0 --membind=0 python runner.py --algo dqn --env_name minigrid --ablation 0 --run 5 --device_name lab-comp_cpu

echo "[lab-comp_cpu] Running dqn on minigrid | Ablation 1 | Run 2"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES="" numactl --cpunodebind=0 --membind=0 python runner.py --algo dqn --env_name minigrid --ablation 1 --run 2 --device_name lab-comp_cpu

echo "[lab-comp_cpu] Running dqn on minigrid | Ablation 1 | Run 5"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES="" numactl --cpunodebind=0 --membind=0 python runner.py --algo dqn --env_name minigrid --ablation 1 --run 5 --device_name lab-comp_cpu

echo "[lab-comp_cpu] Running dqn on minigrid | Ablation 4 | Run 1"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES="" numactl --cpunodebind=0 --membind=0 python runner.py --algo dqn --env_name minigrid --ablation 4 --run 1 --device_name lab-comp_cpu

echo "[lab-comp_cpu] Running dqn on minigrid | Ablation 4 | Run 2"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES="" numactl --cpunodebind=0 --membind=0 python runner.py --algo dqn --env_name minigrid --ablation 4 --run 2 --device_name lab-comp_cpu

echo "[lab-comp_cpu] Running dqn on minigrid | Ablation 4 | Run 3"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES="" numactl --cpunodebind=0 --membind=0 python runner.py --algo dqn --env_name minigrid --ablation 4 --run 3 --device_name lab-comp_cpu

echo "[lab-comp_cpu] Running dqn on minigrid | Ablation 4 | Run 5"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES="" numactl --cpunodebind=0 --membind=0 python runner.py --algo dqn --env_name minigrid --ablation 4 --run 5 --device_name lab-comp_cpu

echo "[lab-comp_cpu] Running ppo on minigrid | Ablation 4 | Run 2"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES="" numactl --cpunodebind=0 --membind=0 python runner.py --algo ppo --env_name minigrid --ablation 4 --run 2 --device_name lab-comp_cpu

echo "[lab-comp_cpu] Running ppo on minigrid | Ablation 4 | Run 3"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES="" numactl --cpunodebind=0 --membind=0 python runner.py --algo ppo --env_name minigrid --ablation 4 --run 3 --device_name lab-comp_cpu

echo "[lab-comp_cpu] Running ppo on minigrid | Ablation 4 | Run 5"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES="" numactl --cpunodebind=0 --membind=0 python runner.py --algo ppo --env_name minigrid --ablation 4 --run 5 --device_name lab-comp_cpu

echo "[lab-comp_cpu] Running ppo on minigrid | Ablation 5 | Run 1"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES="" numactl --cpunodebind=0 --membind=0 python runner.py --algo ppo --env_name minigrid --ablation 5 --run 1 --device_name lab-comp_cpu

echo "[lab-comp_cpu] Running ppo on minigrid | Ablation 5 | Run 4"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES="" numactl --cpunodebind=0 --membind=0 python runner.py --algo ppo --env_name minigrid --ablation 5 --run 4 --device_name lab-comp_cpu

