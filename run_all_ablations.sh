#!/usr/bin/env bash

# Run all ablation studies sequentially with CUDA device
# Usage: bash run_all_ablations.sh

set -euo pipefail

# Export DEVICE_ARG for scripts that respect the environment variable (e.g., cartpole)
export DEVICE_ARG="--device cuda"

echo "=================================================="
echo "Starting CartPole Ablations"
echo "=================================================="
bash run_cartpole_ablation.sh

echo "=================================================="
echo "Starting MiniGrid Ablations"
echo "=================================================="
bash run_minigrid_ablation.sh

echo "=================================================="
echo "Starting MuJoCo Ablations"
echo "=================================================="
bash run_mujoco_ablation.sh

echo "=================================================="
echo "All ablation studies completed successfully."
echo "=================================================="
