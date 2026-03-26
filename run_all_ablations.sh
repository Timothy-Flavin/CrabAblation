#!/usr/bin/env bash

set -euo pipefail
# Ensure results directory exists for plots
mkdir -p results

for ablation in 0 1 2 3 4 5; do
	for run in 1 2 3; do
		echo "Running trial: ablation=${ablation}, run=${run}"
		python runner.py --algo dqn --ablation ${ablation} --run ${run} --env_name cartpole --device_name ${DEVICE_NAME}
	done
done

echo "All 18 trials completed."
