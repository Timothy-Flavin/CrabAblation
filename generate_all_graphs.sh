#!/bin/bash
# Script to generate all graphs for all single agent environments and x-axis options
# Usage: bash generate_all_graphs.sh

set -e

# Environments to process
ENVIRONMENTS=(cartpole minigrid mujoco nchain)
XAXES=(episodes steps time)

# Default runs and smoothing weight (edit as needed)
RUNS="6 7 8 9 10 11 12 13 14 15"
WEIGHT=0.95

for env in "${ENVIRONMENTS[@]}"; do
  for xaxis in "${XAXES[@]}"; do
    echo "Generating: $env $xaxis"
    python3 graph.py --env "$env" --runs $RUNS --weight $WEIGHT --xaxis "$xaxis"
  done
done
