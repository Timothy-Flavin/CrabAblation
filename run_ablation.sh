#!/usr/bin/env bash

# Run 18 trials: 3 runs each for no ablation (0) and ablating pillars 1..5
# Usage: bash run_ablation.sh

set -euo pipefail

# Ensure results directory exists for plots
mkdir -p results

# Optionally set a device here (uncomment to force GPU if available)
# DEVICE_ARG="--device cuda"
DEVICE_ARG=${DEVICE_ARG:-}

for ablation in 0 1 2 3 4 5; do
	for run in 1 2 3; do
		echo "Running trial: ablation=${ablation}, run=${run}"
		python mujoco_dqn_runner.py --ablation ${ablation} --run ${run} ${DEVICE_ARG}
	done
done

echo "All 18 trials completed."
