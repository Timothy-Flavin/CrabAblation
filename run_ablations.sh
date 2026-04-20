#!/usr/bin/env bash

set -euo pipefail

# Default parameters
ALGOS=("dqn")
ENVS=("cartpole") 
#"mujoco" "cartpole" "minigrid"
ABLATIONS=(6) #0 1 2 3 4 5 
RUNS=5
DEVICE="cuda"
DEVICE_NAME="timpc"

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --algos) IFS=' ' read -r -a ALGOS <<< "$2"; shift 2 ;;
    --envs) IFS=' ' read -r -a ENVS <<< "$2"; shift 2 ;;
    --ablations) IFS=' ' read -r -a ABLATIONS <<< "$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --device_name) DEVICE_NAME="$2"; shift 2 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

# Ensure results directory exists
mkdir -p results

echo "Starting Ablations on device: ${DEVICE} (${DEVICE_NAME})"
echo "=================================================="

for algo in "${ALGOS[@]}"; do
    for env in "${ENVS[@]}"; do
      for abl in "${ABLATIONS[@]}"; do
        for run in $(seq 1 $RUNS); do
          result_file="results/${algo}/${env}/train_scores_${run}_${abl}.npy"
          if [[ -f "$result_file" ]]; then
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] SKIP: ${result_file} exists. Algo: ${algo} | Env: ${env} | Ablation: ${abl} | Run: ${run}"
            continue
          fi
          echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running Algo: ${algo} | Env: ${env} | Ablation: ${abl} | Run: ${run}"
          python runner.py --algo "${algo}" --env_name "${env}" --ablation "${abl}" --run "${run}" --device "${DEVICE}" --device_name "${DEVICE_NAME}"
        done
      done
    done
done

echo "=================================================="
echo "All trials completed successfully."
echo "=================================================="
