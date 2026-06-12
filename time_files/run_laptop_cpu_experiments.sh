#!/usr/bin/env bash
# Auto-generated multi-process schedule for laptop_cpu
# Capacity: 1
set -euo pipefail

source .venv/bin/activate

# --- Concurrency Queue 0 ---
(
  echo "[laptop_cpu - Q0] Running ppo on minigrid | Abl 4 | Run 6"
  python runner.py --algo ppo --env_name minigrid --ablation 4 --run 6 --device_name laptop_cpu
  echo "[laptop_cpu - Q0] Running ppo on minigrid | Abl 4 | Run 7"
  python runner.py --algo ppo --env_name minigrid --ablation 4 --run 7 --device_name laptop_cpu
  echo "[laptop_cpu - Q0] Running dqn on cartpole | Abl 0 | Run 6"
  python runner.py --algo dqn --env_name cartpole --ablation 0 --run 6 --device_name laptop_cpu
  echo "[laptop_cpu - Q0] Running dqn on cartpole | Abl 0 | Run 7"
  python runner.py --algo dqn --env_name cartpole --ablation 0 --run 7 --device_name laptop_cpu
  echo "[laptop_cpu - Q0] Running dqn on cartpole | Abl 0 | Run 8"
  python runner.py --algo dqn --env_name cartpole --ablation 0 --run 8 --device_name laptop_cpu
  echo "[laptop_cpu - Q0] Running dqn on cartpole | Abl 0 | Run 9"
  python runner.py --algo dqn --env_name cartpole --ablation 0 --run 9 --device_name laptop_cpu
  echo "[laptop_cpu - Q0] Running dqn on cartpole | Abl 0 | Run 10"
  python runner.py --algo dqn --env_name cartpole --ablation 0 --run 10 --device_name laptop_cpu
  echo "[laptop_cpu - Q0] Running dqn on cartpole | Abl 1 | Run 6"
  python runner.py --algo dqn --env_name cartpole --ablation 1 --run 6 --device_name laptop_cpu
  echo "[laptop_cpu - Q0] Running dqn on cartpole | Abl 1 | Run 7"
  python runner.py --algo dqn --env_name cartpole --ablation 1 --run 7 --device_name laptop_cpu
  echo "[laptop_cpu - Q0] Running dqn on cartpole | Abl 1 | Run 8"
  python runner.py --algo dqn --env_name cartpole --ablation 1 --run 8 --device_name laptop_cpu
  echo "[laptop_cpu - Q0] Running dqn on cartpole | Abl 1 | Run 9"
  python runner.py --algo dqn --env_name cartpole --ablation 1 --run 9 --device_name laptop_cpu
  echo "[laptop_cpu - Q0] Running dqn on cartpole | Abl 1 | Run 10"
  python runner.py --algo dqn --env_name cartpole --ablation 1 --run 10 --device_name laptop_cpu
  echo "[laptop_cpu - Q0] Running dqn on cartpole | Abl 2 | Run 6"
  python runner.py --algo dqn --env_name cartpole --ablation 2 --run 6 --device_name laptop_cpu
  echo "[laptop_cpu - Q0] Running dqn on cartpole | Abl 2 | Run 7"
  python runner.py --algo dqn --env_name cartpole --ablation 2 --run 7 --device_name laptop_cpu
  echo "[laptop_cpu - Q0] Running dqn on cartpole | Abl 2 | Run 8"
  python runner.py --algo dqn --env_name cartpole --ablation 2 --run 8 --device_name laptop_cpu
  echo "[laptop_cpu - Q0] Running dqn on cartpole | Abl 2 | Run 9"
  python runner.py --algo dqn --env_name cartpole --ablation 2 --run 9 --device_name laptop_cpu
  echo "[laptop_cpu - Q0] Running dqn on cartpole | Abl 2 | Run 10"
  python runner.py --algo dqn --env_name cartpole --ablation 2 --run 10 --device_name laptop_cpu
  echo "[laptop_cpu - Q0] Running dqn on cartpole | Abl 4 | Run 6"
  python runner.py --algo dqn --env_name cartpole --ablation 4 --run 6 --device_name laptop_cpu
  echo "[laptop_cpu - Q0] Running dqn on cartpole | Abl 4 | Run 7"
  python runner.py --algo dqn --env_name cartpole --ablation 4 --run 7 --device_name laptop_cpu
  echo "[laptop_cpu - Q0] Running dqn on cartpole | Abl 4 | Run 8"
  python runner.py --algo dqn --env_name cartpole --ablation 4 --run 8 --device_name laptop_cpu
  echo "[laptop_cpu - Q0] Running dqn on cartpole | Abl 4 | Run 9"
  python runner.py --algo dqn --env_name cartpole --ablation 4 --run 9 --device_name laptop_cpu
  echo "[laptop_cpu - Q0] Running dqn on cartpole | Abl 4 | Run 10"
  python runner.py --algo dqn --env_name cartpole --ablation 4 --run 10 --device_name laptop_cpu
  echo "[laptop_cpu - Q0] Running dqn on cartpole | Abl 5 | Run 6"
  python runner.py --algo dqn --env_name cartpole --ablation 5 --run 6 --device_name laptop_cpu
  echo "[laptop_cpu - Q0] Running dqn on cartpole | Abl 5 | Run 7"
  python runner.py --algo dqn --env_name cartpole --ablation 5 --run 7 --device_name laptop_cpu
  echo "[laptop_cpu - Q0] Running dqn on cartpole | Abl 5 | Run 8"
  python runner.py --algo dqn --env_name cartpole --ablation 5 --run 8 --device_name laptop_cpu
  echo "[laptop_cpu - Q0] Running dqn on cartpole | Abl 5 | Run 9"
  python runner.py --algo dqn --env_name cartpole --ablation 5 --run 9 --device_name laptop_cpu
  echo "[laptop_cpu - Q0] Running dqn on cartpole | Abl 5 | Run 10"
  python runner.py --algo dqn --env_name cartpole --ablation 5 --run 10 --device_name laptop_cpu
) &

echo "All queues launched. Waiting for completion..."
wait

echo "All tasks completed on this machine."
