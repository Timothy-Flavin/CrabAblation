#!/usr/bin/env bash

# Run all PPO/PG ablation studies sequentially
# Usage: bash run_pg_ablations.sh [device]
# Example: bash run_pg_ablations.sh cuda
# Or just run: bash run_pg_ablations.sh (defaults to cpu)

set -euo pipefail

# Default device is cpu if not provided
DEVICE=${1:-cpu}
RUNS=1 # Change this to run multiple seeds/trials per ablation

echo "Starting PPO/PG Ablations on device: $DEVICE"

ENVIRONMENTS=("cartpole" "minigrid" "mujoco")
ABLATIONS=(0 1 2 3 4 5)

for env in "${ENVIRONMENTS[@]}"; do
    echo "=================================================="
    echo "Starting $env PPO/PG Ablations"
    echo "=================================================="
    
    for abl in "${ABLATIONS[@]}"; do
        for run in $(seq 1 $RUNS); do
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running $env | Ablation: $abl | Run: $run"
            python pg_runner.py --env_name "$env" --ablation "$abl" --run "$run" --device "$DEVICE"
        done
    done
done

echo "=================================================="
echo "All PPO/PG ablation studies completed successfully."
echo "=================================================="
