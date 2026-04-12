#!/usr/bin/env bash
# Auto-generated script to run ALL missing experiments on timpc
set -euo pipefail

# DQN Experiments
echo "# DQN Experiments"
python runner.py --algo dqn --env_name cartpole --ablation 5 --run 2
python runner.py --algo dqn --env_name cartpole --ablation 5 --run 3
python runner.py --algo dqn --env_name mujoco --ablation 0 --run 1
python runner.py --algo dqn --env_name mujoco --ablation 0 --run 2
python runner.py --algo dqn --env_name mujoco --ablation 0 --run 3
python runner.py --algo dqn --env_name mujoco --ablation 1 --run 1
python runner.py --algo dqn --env_name mujoco --ablation 1 --run 2
python runner.py --algo dqn --env_name mujoco --ablation 1 --run 3
python runner.py --algo dqn --env_name mujoco --ablation 2 --run 1
python runner.py --algo dqn --env_name mujoco --ablation 2 --run 2
python runner.py --algo dqn --env_name mujoco --ablation 2 --run 3
python runner.py --algo dqn --env_name mujoco --ablation 3 --run 1
python runner.py --algo dqn --env_name mujoco --ablation 3 --run 2
python runner.py --algo dqn --env_name mujoco --ablation 3 --run 3
python runner.py --algo dqn --env_name mujoco --ablation 4 --run 2
python runner.py --algo dqn --env_name mujoco --ablation 4 --run 3
python runner.py --algo dqn --env_name mujoco --ablation 5 --run 1
python runner.py --algo dqn --env_name mujoco --ablation 5 --run 2
python runner.py --algo dqn --env_name mujoco --ablation 5 --run 3

# PPO Experiments
echo "# PPO Experiments"
python runner.py --algo ppo --env_name cartpole --ablation 4 --run 3
python runner.py --algo ppo --env_name mujoco --ablation 4 --run 3

# SAC Experiments
echo "# SAC Experiments"
