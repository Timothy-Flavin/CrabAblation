#!/usr/bin/env bash
# Auto-generated multi-process schedule for lab-comp_cpu
# Capacity: 1
set -euo pipefail

source .venv/bin/activate

# --- Concurrency Queue 0 ---
(
  echo "[lab-comp_cpu - Q0] Running dqn on minigrid | Abl 4 | Run 11"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name minigrid --ablation 4 --run 11 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on minigrid | Abl 4 | Run 12"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name minigrid --ablation 4 --run 12 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on minigrid | Abl 4 | Run 13"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name minigrid --ablation 4 --run 13 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on minigrid | Abl 4 | Run 14"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name minigrid --ablation 4 --run 14 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on minigrid | Abl 4 | Run 15"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name minigrid --ablation 4 --run 15 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on minigrid | Abl 6 | Run 11"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name minigrid --ablation 6 --run 11 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on minigrid | Abl 6 | Run 12"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name minigrid --ablation 6 --run 12 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on minigrid | Abl 6 | Run 13"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name minigrid --ablation 6 --run 13 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on minigrid | Abl 6 | Run 14"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name minigrid --ablation 6 --run 14 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on minigrid | Abl 6 | Run 15"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name minigrid --ablation 6 --run 15 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running sac on minigrid | Abl 4 | Run 11"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo sac --env_name minigrid --ablation 4 --run 11 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running sac on minigrid | Abl 4 | Run 12"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo sac --env_name minigrid --ablation 4 --run 12 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running sac on minigrid | Abl 4 | Run 13"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo sac --env_name minigrid --ablation 4 --run 13 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running sac on minigrid | Abl 4 | Run 14"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo sac --env_name minigrid --ablation 4 --run 14 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running sac on minigrid | Abl 4 | Run 15"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo sac --env_name minigrid --ablation 4 --run 15 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running sac on minigrid | Abl 6 | Run 11"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo sac --env_name minigrid --ablation 6 --run 11 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running sac on minigrid | Abl 6 | Run 12"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo sac --env_name minigrid --ablation 6 --run 12 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running sac on minigrid | Abl 6 | Run 13"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo sac --env_name minigrid --ablation 6 --run 13 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running sac on minigrid | Abl 6 | Run 14"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo sac --env_name minigrid --ablation 6 --run 14 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running sac on minigrid | Abl 6 | Run 15"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo sac --env_name minigrid --ablation 6 --run 15 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on minigrid | Abl 4 | Run 11"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name minigrid --ablation 4 --run 11 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on minigrid | Abl 4 | Run 12"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name minigrid --ablation 4 --run 12 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on minigrid | Abl 4 | Run 13"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name minigrid --ablation 4 --run 13 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on minigrid | Abl 4 | Run 14"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name minigrid --ablation 4 --run 14 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on minigrid | Abl 4 | Run 15"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name minigrid --ablation 4 --run 15 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on minigrid | Abl 6 | Run 11"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name minigrid --ablation 6 --run 11 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on minigrid | Abl 6 | Run 12"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name minigrid --ablation 6 --run 12 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on minigrid | Abl 6 | Run 13"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name minigrid --ablation 6 --run 13 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on minigrid | Abl 6 | Run 14"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name minigrid --ablation 6 --run 14 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on minigrid | Abl 6 | Run 15"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name minigrid --ablation 6 --run 15 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 0 | Run 11"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 0 --run 11 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 0 | Run 12"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 0 --run 12 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 0 | Run 13"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 0 --run 13 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 0 | Run 14"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 0 --run 14 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 0 | Run 15"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 0 --run 15 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 1 | Run 11"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 1 --run 11 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 1 | Run 15"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 1 --run 15 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 2 | Run 12"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 2 --run 12 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 2 | Run 13"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 2 --run 13 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 2 | Run 14"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 2 --run 14 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 2 | Run 15"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 2 --run 15 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 3 | Run 11"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 3 --run 11 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 3 | Run 12"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 3 --run 12 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 3 | Run 13"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 3 --run 13 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 3 | Run 14"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 3 --run 14 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 3 | Run 15"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 3 --run 15 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 4 | Run 11"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 4 --run 11 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 4 | Run 12"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 4 --run 12 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 4 | Run 13"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 4 --run 13 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 4 | Run 14"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 4 --run 14 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 4 | Run 15"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 4 --run 15 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 5 | Run 11"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 5 --run 11 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 5 | Run 12"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 5 --run 12 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 5 | Run 13"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 5 --run 13 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 5 | Run 14"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 5 --run 14 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 5 | Run 15"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 5 --run 15 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 6 | Run 11"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 6 --run 11 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 6 | Run 12"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 6 --run 12 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 6 | Run 13"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 6 --run 13 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 6 | Run 14"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 6 --run 14 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on cartpole | Abl 6 | Run 15"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name cartpole --ablation 6 --run 15 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running sac on cartpole | Abl 4 | Run 11"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo sac --env_name cartpole --ablation 4 --run 11 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running sac on cartpole | Abl 4 | Run 12"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo sac --env_name cartpole --ablation 4 --run 12 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running sac on cartpole | Abl 4 | Run 13"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo sac --env_name cartpole --ablation 4 --run 13 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running sac on cartpole | Abl 4 | Run 14"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo sac --env_name cartpole --ablation 4 --run 14 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running sac on cartpole | Abl 4 | Run 15"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo sac --env_name cartpole --ablation 4 --run 15 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running sac on cartpole | Abl 6 | Run 11"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo sac --env_name cartpole --ablation 6 --run 11 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running sac on cartpole | Abl 6 | Run 12"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo sac --env_name cartpole --ablation 6 --run 12 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running sac on cartpole | Abl 6 | Run 13"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo sac --env_name cartpole --ablation 6 --run 13 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running sac on cartpole | Abl 6 | Run 14"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo sac --env_name cartpole --ablation 6 --run 14 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running sac on cartpole | Abl 6 | Run 15"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo sac --env_name cartpole --ablation 6 --run 15 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on cartpole | Abl 4 | Run 11"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name cartpole --ablation 4 --run 11 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on cartpole | Abl 4 | Run 12"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name cartpole --ablation 4 --run 12 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on cartpole | Abl 4 | Run 13"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name cartpole --ablation 4 --run 13 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on cartpole | Abl 4 | Run 14"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name cartpole --ablation 4 --run 14 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on cartpole | Abl 4 | Run 15"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name cartpole --ablation 4 --run 15 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on cartpole | Abl 6 | Run 11"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name cartpole --ablation 6 --run 11 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on cartpole | Abl 6 | Run 12"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name cartpole --ablation 6 --run 12 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on cartpole | Abl 6 | Run 13"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name cartpole --ablation 6 --run 13 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on cartpole | Abl 6 | Run 14"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name cartpole --ablation 6 --run 14 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on cartpole | Abl 6 | Run 15"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name cartpole --ablation 6 --run 15 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on mujoco | Abl 4 | Run 11"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name mujoco --ablation 4 --run 11 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on mujoco | Abl 4 | Run 12"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name mujoco --ablation 4 --run 12 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on mujoco | Abl 4 | Run 13"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name mujoco --ablation 4 --run 13 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on mujoco | Abl 4 | Run 14"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name mujoco --ablation 4 --run 14 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on mujoco | Abl 4 | Run 15"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name mujoco --ablation 4 --run 15 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on mujoco | Abl 6 | Run 11"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name mujoco --ablation 6 --run 11 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on mujoco | Abl 6 | Run 12"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name mujoco --ablation 6 --run 12 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on mujoco | Abl 6 | Run 13"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name mujoco --ablation 6 --run 13 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on mujoco | Abl 6 | Run 14"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name mujoco --ablation 6 --run 14 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running dqn on mujoco | Abl 6 | Run 15"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo dqn --env_name mujoco --ablation 6 --run 15 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running sac on mujoco | Abl 4 | Run 11"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo sac --env_name mujoco --ablation 4 --run 11 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running sac on mujoco | Abl 6 | Run 12"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo sac --env_name mujoco --ablation 6 --run 12 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running sac on mujoco | Abl 6 | Run 13"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo sac --env_name mujoco --ablation 6 --run 13 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running sac on mujoco | Abl 6 | Run 14"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo sac --env_name mujoco --ablation 6 --run 14 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running sac on mujoco | Abl 6 | Run 15"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo sac --env_name mujoco --ablation 6 --run 15 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on mujoco | Abl 4 | Run 11"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name mujoco --ablation 4 --run 11 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on mujoco | Abl 4 | Run 12"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name mujoco --ablation 4 --run 12 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on mujoco | Abl 4 | Run 13"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name mujoco --ablation 4 --run 13 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on mujoco | Abl 4 | Run 14"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name mujoco --ablation 4 --run 14 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on mujoco | Abl 4 | Run 15"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name mujoco --ablation 4 --run 15 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on mujoco | Abl 6 | Run 11"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name mujoco --ablation 6 --run 11 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on mujoco | Abl 6 | Run 12"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name mujoco --ablation 6 --run 12 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on mujoco | Abl 6 | Run 13"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name mujoco --ablation 6 --run 13 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on mujoco | Abl 6 | Run 14"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name mujoco --ablation 6 --run 14 --device_name lab-comp_cpu
  echo "[lab-comp_cpu - Q0] Running ppo on mujoco | Abl 6 | Run 15"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo ppo --env_name mujoco --ablation 6 --run 15 --device_name lab-comp_cpu
) &

echo "All queues launched. Waiting for completion..."
wait

echo "All tasks completed on this machine."
