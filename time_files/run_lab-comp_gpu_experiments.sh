#!/usr/bin/env bash
# Auto-generated schedule for lab-comp_gpu
set -euo pipefail

source .venv/bin/activate

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 0 | Run 6"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 0 --run 6 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 0 | Run 7"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 0 --run 7 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 0 | Run 8"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 0 --run 8 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 0 | Run 9"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 0 --run 9 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 0 | Run 10"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 0 --run 10 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 1 | Run 6"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 1 --run 6 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 1 | Run 7"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 1 --run 7 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 1 | Run 8"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 1 --run 8 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 1 | Run 9"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 1 --run 9 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 1 | Run 10"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 1 --run 10 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 2 | Run 6"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 2 --run 6 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 2 | Run 7"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 2 --run 7 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 2 | Run 8"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 2 --run 8 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 2 | Run 9"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 2 --run 9 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 2 | Run 10"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 2 --run 10 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 3 | Run 6"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 3 --run 6 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 3 | Run 7"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 3 --run 7 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 3 | Run 8"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 3 --run 8 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 3 | Run 9"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 3 --run 9 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 3 | Run 10"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 3 --run 10 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 5 | Run 6"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 5 --run 6 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 5 | Run 7"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 5 --run 7 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 5 | Run 8"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 5 --run 8 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 5 | Run 9"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 5 --run 9 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on minigrid | Ablation 5 | Run 10"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name minigrid --ablation 5 --run 10 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on minigrid | Ablation 3 | Run 6"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name minigrid --ablation 3 --run 6 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on minigrid | Ablation 3 | Run 7"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name minigrid --ablation 3 --run 7 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on minigrid | Ablation 3 | Run 8"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name minigrid --ablation 3 --run 8 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on minigrid | Ablation 5 | Run 6"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name minigrid --ablation 5 --run 6 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on minigrid | Ablation 5 | Run 7"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name minigrid --ablation 5 --run 7 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on minigrid | Ablation 5 | Run 8"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name minigrid --ablation 5 --run 8 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on minigrid | Ablation 5 | Run 9"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name minigrid --ablation 5 --run 9 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on minigrid | Ablation 5 | Run 10"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name minigrid --ablation 5 --run 10 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on cartpole | Ablation 5 | Run 6"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name cartpole --ablation 5 --run 6 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on cartpole | Ablation 5 | Run 8"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name cartpole --ablation 5 --run 8 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 0 | Run 6"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 0 --run 6 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 0 | Run 7"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 0 --run 7 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 0 | Run 8"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 0 --run 8 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 0 | Run 9"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 0 --run 9 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 0 | Run 10"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 0 --run 10 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 1 | Run 6"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 1 --run 6 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 1 | Run 7"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 1 --run 7 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 1 | Run 8"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 1 --run 8 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 1 | Run 9"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 1 --run 9 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 1 | Run 10"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 1 --run 10 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 2 | Run 6"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 2 --run 6 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 2 | Run 7"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 2 --run 7 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 2 | Run 8"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 2 --run 8 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 2 | Run 9"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 2 --run 9 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 2 | Run 10"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 2 --run 10 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 3 | Run 6"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 3 --run 6 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 3 | Run 7"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 3 --run 7 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 3 | Run 8"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 3 --run 8 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 3 | Run 9"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 3 --run 9 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 3 | Run 10"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 3 --run 10 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 5 | Run 6"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 5 --run 6 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 5 | Run 7"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 5 --run 7 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 5 | Run 8"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 5 --run 8 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 5 | Run 9"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 5 --run 9 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running dqn on mujoco | Ablation 5 | Run 10"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name mujoco --ablation 5 --run 10 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 0 | Run 6"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 0 --run 6 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 0 | Run 7"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 0 --run 7 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 0 | Run 8"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 0 --run 8 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 0 | Run 9"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 0 --run 9 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 0 | Run 10"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 0 --run 10 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 1 | Run 6"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 1 --run 6 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 1 | Run 7"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 1 --run 7 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 1 | Run 8"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 1 --run 8 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 1 | Run 9"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 1 --run 9 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 1 | Run 10"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 1 --run 10 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 2 | Run 6"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 2 --run 6 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 2 | Run 7"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 2 --run 7 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 2 | Run 8"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 2 --run 8 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 2 | Run 9"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 2 --run 9 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 2 | Run 10"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 2 --run 10 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 3 | Run 6"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 3 --run 6 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 3 | Run 7"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 3 --run 7 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 3 | Run 8"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 3 --run 8 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 3 | Run 9"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 3 --run 9 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 3 | Run 10"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 3 --run 10 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 5 | Run 6"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 5 --run 6 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 5 | Run 7"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 5 --run 7 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 5 | Run 8"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 5 --run 8 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 5 | Run 9"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 5 --run 9 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running sac on mujoco | Ablation 5 | Run 10"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo sac --env_name mujoco --ablation 5 --run 10 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 0 | Run 6"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 0 --run 6 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 0 | Run 7"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 0 --run 7 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 0 | Run 8"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 0 --run 8 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 0 | Run 9"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 0 --run 9 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 0 | Run 10"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 0 --run 10 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 1 | Run 6"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 1 --run 6 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 1 | Run 7"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 1 --run 7 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 1 | Run 8"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 1 --run 8 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 1 | Run 9"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 1 --run 9 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 1 | Run 10"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 1 --run 10 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 2 | Run 6"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 2 --run 6 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 2 | Run 7"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 2 --run 7 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 2 | Run 8"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 2 --run 8 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 2 | Run 9"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 2 --run 9 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 2 | Run 10"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 2 --run 10 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 3 | Run 6"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 3 --run 6 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 3 | Run 7"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 3 --run 7 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 3 | Run 8"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 3 --run 8 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 3 | Run 9"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 3 --run 9 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 3 | Run 10"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 3 --run 10 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 5 | Run 6"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 5 --run 6 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 5 | Run 7"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 5 --run 7 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 5 | Run 8"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 5 --run 8 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 5 | Run 9"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 5 --run 9 --device_name lab-comp_gpu

echo "[lab-comp_gpu] Running ppo on mujoco | Ablation 5 | Run 10"
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo ppo --env_name mujoco --ablation 5 --run 10 --device_name lab-comp_gpu

