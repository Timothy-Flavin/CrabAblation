#!/usr/bin/env bash
# Auto-generated schedule for mac
set -euo pipefail

echo "[mac] Running dqn on minigrid | Ablation 4 | Run 1"
python dqn_runner.py --env_name minigrid --ablation 4 --run 1

echo "[mac] Running dqn on minigrid | Ablation 4 | Run 2"
python dqn_runner.py --env_name minigrid --ablation 4 --run 2

echo "[mac] Running dqn on minigrid | Ablation 4 | Run 3"
python dqn_runner.py --env_name minigrid --ablation 4 --run 3

echo "[mac] Running ppo on minigrid | Ablation 4 | Run 1"
python pg_runner.py --env_name minigrid --ablation 4 --run 1

echo "[mac] Running ppo on minigrid | Ablation 4 | Run 2"
python pg_runner.py --env_name minigrid --ablation 4 --run 2

echo "[mac] Running ppo on minigrid | Ablation 4 | Run 3"
python pg_runner.py --env_name minigrid --ablation 4 --run 3

echo "[mac] Running dqn on cartpole | Ablation 0 | Run 1"
python dqn_runner.py --env_name cartpole --ablation 0 --run 1

echo "[mac] Running dqn on cartpole | Ablation 0 | Run 2"
python dqn_runner.py --env_name cartpole --ablation 0 --run 2

echo "[mac] Running dqn on cartpole | Ablation 1 | Run 1"
python dqn_runner.py --env_name cartpole --ablation 1 --run 1

echo "[mac] Running dqn on cartpole | Ablation 1 | Run 2"
python dqn_runner.py --env_name cartpole --ablation 1 --run 2

echo "[mac] Running dqn on cartpole | Ablation 1 | Run 3"
python dqn_runner.py --env_name cartpole --ablation 1 --run 3

echo "[mac] Running dqn on cartpole | Ablation 2 | Run 2"
python dqn_runner.py --env_name cartpole --ablation 2 --run 2

echo "[mac] Running dqn on cartpole | Ablation 2 | Run 3"
python dqn_runner.py --env_name cartpole --ablation 2 --run 3

echo "[mac] Running dqn on cartpole | Ablation 3 | Run 1"
python dqn_runner.py --env_name cartpole --ablation 3 --run 1

echo "[mac] Running dqn on cartpole | Ablation 3 | Run 2"
python dqn_runner.py --env_name cartpole --ablation 3 --run 2

echo "[mac] Running dqn on cartpole | Ablation 3 | Run 3"
python dqn_runner.py --env_name cartpole --ablation 3 --run 3

echo "[mac] Running dqn on cartpole | Ablation 4 | Run 1"
python dqn_runner.py --env_name cartpole --ablation 4 --run 1

echo "[mac] Running dqn on cartpole | Ablation 4 | Run 2"
python dqn_runner.py --env_name cartpole --ablation 4 --run 2

echo "[mac] Running dqn on cartpole | Ablation 4 | Run 3"
python dqn_runner.py --env_name cartpole --ablation 4 --run 3

echo "[mac] Running dqn on cartpole | Ablation 5 | Run 1"
python dqn_runner.py --env_name cartpole --ablation 5 --run 1

echo "[mac] Running dqn on cartpole | Ablation 5 | Run 2"
python dqn_runner.py --env_name cartpole --ablation 5 --run 2

echo "[mac] Running dqn on cartpole | Ablation 5 | Run 3"
python dqn_runner.py --env_name cartpole --ablation 5 --run 3

echo "[mac] Running ppo on cartpole | Ablation 4 | Run 3"
python pg_runner.py --env_name cartpole --ablation 4 --run 3

echo "[mac] Running dqn on mujoco | Ablation 4 | Run 1"
python dqn_runner.py --env_name mujoco --ablation 4 --run 1

echo "[mac] Running dqn on mujoco | Ablation 4 | Run 2"
python dqn_runner.py --env_name mujoco --ablation 4 --run 2

echo "[mac] Running dqn on mujoco | Ablation 4 | Run 3"
python dqn_runner.py --env_name mujoco --ablation 4 --run 3

echo "[mac] Running ppo on mujoco | Ablation 4 | Run 1"
python pg_runner.py --env_name mujoco --ablation 4 --run 1

echo "[mac] Running ppo on mujoco | Ablation 4 | Run 2"
python pg_runner.py --env_name mujoco --ablation 4 --run 2

echo "[mac] Running ppo on mujoco | Ablation 4 | Run 3"
python pg_runner.py --env_name mujoco --ablation 4 --run 3

