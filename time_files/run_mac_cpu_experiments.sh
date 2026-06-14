#!/usr/bin/env bash
# Auto-generated multi-process schedule for mac_cpu
# Capacity: 1
set -euo pipefail

source ../.venv/bin/activate

# --- Concurrency Queue 0 ---
(
  echo "[mac_cpu - Q0] Running dqn on minigrid | Abl 3 | Run 6"
  python runner.py --algo dqn --env_name minigrid --ablation 3 --run 6 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running dqn on minigrid | Abl 3 | Run 10"
  python runner.py --algo dqn --env_name minigrid --ablation 3 --run 10 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running dqn on minigrid | Abl 4 | Run 6"
  python runner.py --algo dqn --env_name minigrid --ablation 4 --run 6 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running dqn on minigrid | Abl 4 | Run 7"
  python runner.py --algo dqn --env_name minigrid --ablation 4 --run 7 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running dqn on minigrid | Abl 4 | Run 8"
  python runner.py --algo dqn --env_name minigrid --ablation 4 --run 8 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running dqn on minigrid | Abl 4 | Run 9"
  python runner.py --algo dqn --env_name minigrid --ablation 4 --run 9 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running dqn on minigrid | Abl 4 | Run 10"
  python runner.py --algo dqn --env_name minigrid --ablation 4 --run 10 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running ppo on minigrid | Abl 4 | Run 8"
  python runner.py --algo ppo --env_name minigrid --ablation 4 --run 8 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running ppo on minigrid | Abl 4 | Run 9"
  python runner.py --algo ppo --env_name minigrid --ablation 4 --run 9 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running ppo on minigrid | Abl 4 | Run 10"
  python runner.py --algo ppo --env_name minigrid --ablation 4 --run 10 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running dqn on cartpole | Abl 3 | Run 6"
  python runner.py --algo dqn --env_name cartpole --ablation 3 --run 6 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running dqn on cartpole | Abl 3 | Run 7"
  python runner.py --algo dqn --env_name cartpole --ablation 3 --run 7 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running dqn on cartpole | Abl 3 | Run 8"
  python runner.py --algo dqn --env_name cartpole --ablation 3 --run 8 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running dqn on cartpole | Abl 3 | Run 9"
  python runner.py --algo dqn --env_name cartpole --ablation 3 --run 9 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running dqn on cartpole | Abl 3 | Run 10"
  python runner.py --algo dqn --env_name cartpole --ablation 3 --run 10 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running sac on cartpole | Abl 4 | Run 6"
  python runner.py --algo sac --env_name cartpole --ablation 4 --run 6 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running sac on cartpole | Abl 4 | Run 7"
  python runner.py --algo sac --env_name cartpole --ablation 4 --run 7 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running sac on cartpole | Abl 4 | Run 8"
  python runner.py --algo sac --env_name cartpole --ablation 4 --run 8 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running sac on cartpole | Abl 4 | Run 9"
  python runner.py --algo sac --env_name cartpole --ablation 4 --run 9 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running sac on cartpole | Abl 4 | Run 10"
  python runner.py --algo sac --env_name cartpole --ablation 4 --run 10 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running ppo on cartpole | Abl 4 | Run 7"
  python runner.py --algo ppo --env_name cartpole --ablation 4 --run 7 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running ppo on cartpole | Abl 4 | Run 8"
  python runner.py --algo ppo --env_name cartpole --ablation 4 --run 8 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running ppo on cartpole | Abl 4 | Run 9"
  python runner.py --algo ppo --env_name cartpole --ablation 4 --run 9 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running ppo on cartpole | Abl 4 | Run 10"
  python runner.py --algo ppo --env_name cartpole --ablation 4 --run 10 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running dqn on mujoco | Abl 4 | Run 6"
  python runner.py --algo dqn --env_name mujoco --ablation 4 --run 6 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running dqn on mujoco | Abl 4 | Run 7"
  python runner.py --algo dqn --env_name mujoco --ablation 4 --run 7 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running dqn on mujoco | Abl 4 | Run 8"
  python runner.py --algo dqn --env_name mujoco --ablation 4 --run 8 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running dqn on mujoco | Abl 4 | Run 9"
  python runner.py --algo dqn --env_name mujoco --ablation 4 --run 9 --device_name mac_cpu
  echo "[mac_cpu - Q0] Running dqn on mujoco | Abl 4 | Run 10"
  python runner.py --algo dqn --env_name mujoco --ablation 4 --run 10 --device_name mac_cpu
) &

echo "All queues launched. Waiting for completion..."
wait

echo "All tasks completed on this machine."
