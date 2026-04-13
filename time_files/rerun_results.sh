#!/usr/bin/env bash
# Auto-generated script to run ALL missing experiments on timpc
set -euo pipefail

# DQN Experiments
echo "# DQN Experiments"
# Anomaly 2: Rerun all dqn cartpole ablations on timpc with proper truncation
python runner.py --algo dqn --env_name cartpole --ablation 0 --run 1
python runner.py --algo dqn --env_name cartpole --ablation 0 --run 2
python runner.py --algo dqn --env_name cartpole --ablation 0 --run 3
python runner.py --algo dqn --env_name cartpole --ablation 1 --run 1
python runner.py --algo dqn --env_name cartpole --ablation 1 --run 2
python runner.py --algo dqn --env_name cartpole --ablation 1 --run 3
python runner.py --algo dqn --env_name cartpole --ablation 2 --run 1
python runner.py --algo dqn --env_name cartpole --ablation 2 --run 2
python runner.py --algo dqn --env_name cartpole --ablation 2 --run 3
python runner.py --algo dqn --env_name cartpole --ablation 3 --run 1
python runner.py --algo dqn --env_name cartpole --ablation 3 --run 2
python runner.py --algo dqn --env_name cartpole --ablation 3 --run 3
python runner.py --algo dqn --env_name cartpole --ablation 4 --run 1
python runner.py --algo dqn --env_name cartpole --ablation 4 --run 2
python runner.py --algo dqn --env_name cartpole --ablation 4 --run 3
python runner.py --algo dqn --env_name cartpole --ablation 5 --run 1
python runner.py --algo dqn --env_name cartpole --ablation 5 --run 2
python runner.py --algo dqn --env_name cartpole --ablation 5 --run 3

# PPO Experiments
echo "# PPO Experiments"
# Anomaly 4: ppo cartpole ablation 4 ran for far fewer episodes. Rerun likely needed.
python runner.py --algo ppo --env_name cartpole --ablation 4 --run 1
python runner.py --algo ppo --env_name cartpole --ablation 4 --run 2
python runner.py --algo ppo --env_name cartpole --ablation 4 --run 3

# SAC Experiments
echo "# SAC Experiments"
