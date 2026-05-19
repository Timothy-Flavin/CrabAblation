#!/usr/bin/env bash
# Auto-generated schedule for mac
set -euo pipefail

source ../.venv/bin/activate

echo "[mac] Running dqn on minigrid | Ablation 1 | Run 1"
python runner.py --algo dqn --env_name minigrid --ablation 1 --run 1 --device_name mac

echo "[mac] Running dqn on minigrid | Ablation 3 | Run 1"
python runner.py --algo dqn --env_name minigrid --ablation 3 --run 1 --device_name mac

echo "[mac] Running dqn on minigrid | Ablation 3 | Run 2"
python runner.py --algo dqn --env_name minigrid --ablation 3 --run 2 --device_name mac

echo "[mac] Running dqn on minigrid | Ablation 3 | Run 3"
python runner.py --algo dqn --env_name minigrid --ablation 3 --run 3 --device_name mac

echo "[mac] Running dqn on minigrid | Ablation 3 | Run 4"
python runner.py --algo dqn --env_name minigrid --ablation 3 --run 4 --device_name mac

echo "[mac] Running dqn on minigrid | Ablation 3 | Run 5"
python runner.py --algo dqn --env_name minigrid --ablation 3 --run 5 --device_name mac

echo "[mac] Running dqn on minigrid | Ablation 4 | Run 4"
python runner.py --algo dqn --env_name minigrid --ablation 4 --run 4 --device_name mac

echo "[mac] Running dqn on minigrid | Ablation 5 | Run 3"
python runner.py --algo dqn --env_name minigrid --ablation 5 --run 3 --device_name mac

echo "[mac] Running dqn on minigrid | Ablation 5 | Run 5"
python runner.py --algo dqn --env_name minigrid --ablation 5 --run 5 --device_name mac

echo "[mac] Running ppo on minigrid | Ablation 4 | Run 4"
python runner.py --algo ppo --env_name minigrid --ablation 4 --run 4 --device_name mac

echo "[mac] Running sac on minigrid | Ablation 3 | Run 3"
python runner.py --algo sac --env_name minigrid --ablation 3 --run 3 --device_name mac

echo "[mac] Running sac on minigrid | Ablation 4 | Run 1"
python runner.py --algo sac --env_name minigrid --ablation 4 --run 1 --device_name mac

echo "[mac] Running sac on minigrid | Ablation 4 | Run 2"
python runner.py --algo sac --env_name minigrid --ablation 4 --run 2 --device_name mac

echo "[mac] Running sac on minigrid | Ablation 4 | Run 3"
python runner.py --algo sac --env_name minigrid --ablation 4 --run 3 --device_name mac

echo "[mac] Running sac on minigrid | Ablation 4 | Run 4"
python runner.py --algo sac --env_name minigrid --ablation 4 --run 4 --device_name mac

echo "[mac] Running sac on minigrid | Ablation 4 | Run 5"
python runner.py --algo sac --env_name minigrid --ablation 4 --run 5 --device_name mac

