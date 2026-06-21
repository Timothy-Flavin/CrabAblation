#!/bin/bash

# Multi-Agent Ablation Runner for Leduc Poker, RPS, and Tic-Tac-Toe
# Mirrors run_ma_ablations.ps1 logic (sequential execution for stability)

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

# Trap Ctrl+C: kill both job groups and let EXIT cleanup fire
trap "echo 'Interrupted by user. Exiting...'; kill \$PID_BASELINE \$PID_SHARED 2>/dev/null; exit 1" INT

# Revert GPU to default on any exit (clean finish or error)
cleanup() {
    echo quit | nvidia-cuda-mps-control
    sudo nvidia-smi -i 0 -c DEFAULT
    echo "GPU settings restored."
}
trap cleanup EXIT

# Portable python: prefer the Windows venv, fall back to the POSIX venv, then PATH.
if [ -x "./.venv/Scripts/python.exe" ]; then
    PY="./.venv/Scripts/python.exe"
elif [ -x "./.venv/bin/python" ]; then
    source .venv/bin/activate
    PY="python"
else
    PY="python"
fi

ENVS=("leduc" "tictactoe" "rps")
ALGOS=("sac" "dqn" "ppo")
ABLATIONS=(0 1 2 3 4 5 6)
RUNS=5
EVAL_EPISODES=100
# Discrete-uniform regularizer strength for SAC on the (discrete) MA games.
SAC_DISCRETE_ENTROPY=5.0

# First CPU half (CCD0 + HT): baseline runs
HALF_CPU1="0-7,16-23"
# Second CPU half (CCD1 + HT): shared-model runs
HALF_CPU2="8-15,24-31"

# ─── GPU: enable MPS at 50% per job ──────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d
sleep 2
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50
# ─────────────────────────────────────────────────────────────────────────────

echo "========================================="
echo " RUNNING BASELINE + SHARED IN PARALLEL"
echo " Baseline  → CPUs $HALF_CPU1 | 50% GPU"
echo " Shared    → CPUs $HALF_CPU2 | 50% GPU"
echo "========================================="

# ─── Baseline runs (non-shared) ──────────────────────────────────────────────
(
    for run in $(seq 1 $RUNS); do
        for env in "${ENVS[@]}"; do
            for algo in "${ALGOS[@]}"; do
                for ablation in "${ABLATIONS[@]}"; do
                    echo "[baseline] Env=$env, Algo=$algo, Ablation=$ablation, Run=$run"

                    EPISODES=50000
                    if [ "$env" == "rps" ]; then
                        EPISODES=20000
                    fi

                    EXTRA_FLAGS=""
                    if [ "$algo" == "ppo" ] && [ "$ablation" == "6" ]; then
                        EXTRA_FLAGS="--ent_coef_override 0.1"
                    fi
                    if [ "$algo" == "sac" ]; then
                        EXTRA_FLAGS="--discrete_entropy $SAC_DISCRETE_ENTROPY"
                    fi

                    "$PY" multiagent_runner.py \
                        --algo "$algo" \
                        --ma_env "$env" \
                        --ablation "$ablation" \
                        --run "$run" \
                        --total_episodes "$EPISODES" \
                        --eval_episodes "$EVAL_EPISODES" \
                        $EXTRA_FLAGS
                done
            done
        done
    done
) &
PID_BASELINE=$!
taskset -cp "$HALF_CPU1" "$PID_BASELINE" \
    && echo "[cpu] Baseline pinned to $HALF_CPU1" \
    || echo "[warn] taskset failed for baseline — running unpinned"

# ─── Shared-model runs ───────────────────────────────────────────────────────
(
    for run in $(seq 1 $RUNS); do
        for env in "${ENVS[@]}"; do
            for algo in "${ALGOS[@]}"; do
                for ablation in "${ABLATIONS[@]}"; do
                    echo "[shared]   Env=$env, Algo=$algo, Ablation=$ablation, Run=$run"

                    EPISODES=50000
                    if [ "$env" == "rps" ]; then
                        EPISODES=20000
                    fi

                    EXTRA_FLAGS="--shared_model"
                    if [ "$algo" == "ppo" ] && [ "$ablation" == "6" ]; then
                        EXTRA_FLAGS="$EXTRA_FLAGS --ent_coef_override 0.1"
                    fi
                    # For PPO on unit-reward games (RPS, TTT), ent_coef needs to be much
                    # larger than Leduc (~0.1 is calibrated for chip-scale rewards).
                    # Analysis: need ent_coef > max_reward / log(n_actions).
                    # RPS: need >0.91, TTT: need >0.45. Use 0.5 for both.
                    if [ "$algo" == "ppo" ] && [ "$env" != "leduc" ]; then
                        EXTRA_FLAGS="$EXTRA_FLAGS --ent_coef_override 0.5"
                    fi
                    if [ "$algo" == "sac" ]; then
                        EXTRA_FLAGS="$EXTRA_FLAGS --discrete_entropy $SAC_DISCRETE_ENTROPY"
                    fi

                    "$PY" multiagent_runner.py \
                        --algo "$algo" \
                        --ma_env "$env" \
                        --ablation "$ablation" \
                        --run "$run" \
                        --total_episodes "$EPISODES" \
                        --eval_episodes "$EVAL_EPISODES" \
                        $EXTRA_FLAGS
                done
            done
        done
    done
) &
PID_SHARED=$!
taskset -cp "$HALF_CPU2" "$PID_SHARED" \
    && echo "[cpu] Shared   pinned to $HALF_CPU2" \
    || echo "[warn] taskset failed for shared — running unpinned"

wait $PID_BASELINE $PID_SHARED
echo -e "\nAll ablation runs complete.\n"
