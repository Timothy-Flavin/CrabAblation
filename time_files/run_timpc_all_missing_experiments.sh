#!/usr/bin/env bash
# Auto-generated script to run ALL missing experiments on timpc
source .venv/bin/activate
set -euo pipefail

# ==========================================
# BATCH 1: Cartpole Experiments (GPU 0)
# ==========================================
(
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name cartpole --ablation 2 --run 1 white-machine_gpu0
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name cartpole --ablation 2 --run 2 white-machine_gpu0
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name cartpole --ablation 2 --run 3 white-machine_gpu0
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name cartpole --ablation 4 --run 1 white-machine_gpu0
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name cartpole --ablation 4 --run 2 white-machine_gpu0
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name cartpole --ablation 4 --run 3 white-machine_gpu0
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name cartpole --ablation 5 --run 1 white-machine_gpu0
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name cartpole --ablation 5 --run 2 white-machine_gpu0
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 taskset -c 0-3,8-11 python runner.py --algo dqn --env_name cartpole --ablation 5 --run 3 white-machine_gpu0
) &

# ==========================================
# BATCH 2: Mujoco Experiments (GPU 1)
# ==========================================
(
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name mujoco --ablation 2 --run 1 white-machine_gpu1
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo dqn --env_name mujoco --ablation 5 --run 2 white-machine_gpu1
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo ppo --env_name mujoco --ablation 4 --run 1 white-machine_gpu1
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo ppo --env_name mujoco --ablation 4 --run 2 white-machine_gpu1
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 taskset -c 4-7,12-15 python runner.py --algo ppo --env_name mujoco --ablation 4 --run 3 white-machine_gpu1
) &

# Wait for both background processes to finish before exiting
wait
echo "All experiments completed!"