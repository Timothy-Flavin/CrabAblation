#!/usr/bin/env bash
# Auto-generated script to run ALL missing experiments on timpc
set -euo pipefail

# DQN Experiments
echo "# DQN Experiments"
python runner.py --algo dqn --env_name minigrid --ablation 4 --run 1
python runner.py --algo dqn --env_name minigrid --ablation 4 --run 2
python runner.py --algo dqn --env_name minigrid --ablation 4 --run 3
python runner.py --algo dqn --env_name minigrid --ablation 5 --run 1
python runner.py --algo dqn --env_name minigrid --ablation 5 --run 2
python runner.py --algo dqn --env_name minigrid --ablation 5 --run 3

# PPO Experiments
echo "# PPO Experiments"
python runner.py --algo ppo --env_name minigrid --ablation 4 --run 1
python runner.py --algo ppo --env_name minigrid --ablation 4 --run 2
python runner.py --algo ppo --env_name minigrid --ablation 4 --run 3

# Add more lines here for other missing experiments as needed