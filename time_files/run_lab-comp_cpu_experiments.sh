#!/usr/bin/env bash
# Auto-generated multi-process schedule for lab-comp_cpu
# Capacity: 1
set -euo pipefail

source .venv/bin/activate

# --- Concurrency Queue 0 ---
(
  echo "[lab-comp_cpu - Q0] Running sac on nchain | Abl 0 | Run 6"
  CUDA_VISIBLE_DEVICES="" numactl --preferred=0 taskset -c 0-15,32-47 python runner.py --algo sac --env_name nchain --ablation 0 --run 6 --device_name lab-comp_cpu
) &

echo "All queues launched. Waiting for completion..."
wait

echo "All tasks completed on this machine."
