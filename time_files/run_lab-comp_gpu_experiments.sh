#!/usr/bin/env bash
# Auto-generated schedule for lab-comp_gpu
set -euo pipefail

echo "[lab-comp_gpu] Running dqn on minigrid | Ablation 3 | Run 1"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name minigrid --ablation 3 --run 1 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on minigrid | Ablation 3 | Run 2"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name minigrid --ablation 3 --run 2 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on minigrid | Ablation 3 | Run 3"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name minigrid --ablation 3 --run 3 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on minigrid | Ablation 3 | Run 4"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name minigrid --ablation 3 --run 4 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on minigrid | Ablation 3 | Run 5"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name minigrid --ablation 3 --run 5 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 0 | Run 1"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 0 --run 1 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 0 | Run 2"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 0 --run 2 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 0 | Run 3"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 0 --run 3 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 0 | Run 4"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 0 --run 4 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 0 | Run 5"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 0 --run 5 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 1 | Run 1"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 1 --run 1 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 1 | Run 2"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 1 --run 2 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 1 | Run 3"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 1 --run 3 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 1 | Run 4"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 1 --run 4 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 1 | Run 5"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 1 --run 5 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on cartpole | Ablation 1 | Run 1"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name cartpole --ablation 1 --run 1 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 1 | Run 1"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 1 --run 1 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 3 | Run 1"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 3 --run 1 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 3 | Run 2"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 3 --run 2 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 3 | Run 3"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 3 --run 3 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 3 | Run 4"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 3 --run 4 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 3 | Run 5"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 3 --run 5 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 0 | Run 1"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 0 --run 1 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 0 | Run 2"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 0 --run 2 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 0 | Run 3"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 0 --run 3 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 0 | Run 4"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 0 --run 4 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 2 | Run 1"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 2 --run 1 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 2 | Run 2"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 2 --run 2 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 2 | Run 3"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 2 --run 3 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 2 | Run 4"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 2 --run 4 --device_name lab-comp_gpu

