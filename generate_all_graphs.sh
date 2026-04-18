#!/bin/bash
# Script to generate all graphs for all algorithms, environments, and x-axis options
# Usage: bash generate_all_graphs.sh

set -e

# Algorithms and environments to process
ALGORITHMS=(dqn sac ppo)
ENVIRONMENTS=(cartpole minigrid mujoco)
XAXES=(episodes steps time)

# Default runs and smoothing weight (edit as needed)
RUNS="1 2 3"
WEIGHT=0.95
# Set max_steps for 'steps' xaxis (edit as needed)
MAX_STEPS=1000000

for algo in "${ALGORITHMS[@]}"; do
  for env in "${ENVIRONMENTS[@]}"; do
    for xaxis in "${XAXES[@]}"; do
      # Only add --max_steps if xaxis is steps
      if [ "$xaxis" = "steps" ]; then
        echo "Generating: $algo $env $xaxis (with max_steps)"
        python3 graph.py --env "$env" --runs $RUNS --weight $WEIGHT --xaxis "$xaxis" --max_steps $MAX_STEPS
      else
        echo "Generating: $algo $env $xaxis"
        python3 graph.py --env "$env" --runs $RUNS --weight $WEIGHT --xaxis "$xaxis"
      fi
    done
  done
done
