#!/bin/bash

# Multi-Agent Ablation Runner for Leduc Poker and Rock-Paper-Scissors and Tic-Tac-Toe
# Ablations 0-5 are the same as runner.py
# Ablation 6 is the "Base" algorithm (Default params, high entropy for PPO)

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

# Portable python: prefer the Windows venv, fall back to the POSIX venv, then PATH.
if [ -x "./.venv/Scripts/python.exe" ]; then
    PY="./.venv/Scripts/python.exe"
elif [ -x "./.venv/bin/python" ]; then
    PY="./.venv/bin/python"
else
    PY="python"
fi

ENVS=("rps" "leduc")
ALGOS=("sac" "dqn" "ppo")
ABLATIONS=(0 1 2 3 4 5 6)
RUNS=5
EVAL_EPISODES=50
# Discrete-uniform regularizer strength for SAC on the (discrete) MA games. Pulls the
# argmax policy toward uniform; auto-disabled for the entropy ablation and continuous envs.
SAC_DISCRETE_ENTROPY=1.0

for env in "${ENVS[@]}"; do
    for algo in "${ALGOS[@]}"; do
        for ablation in "${ABLATIONS[@]}"; do
            echo "Starting Parallel MA Ablation: Env=$env, Algo=$algo, Ablation=$ablation"

            # Default episodes
            EPISODES=20000
            if [ "$env" == "rps" ]; then
                EPISODES=10000
            fi

            for run in $(seq 1 $RUNS); do
                # Per-algo extra flags
                EXTRA_FLAGS=""
                # Ablation 6 PPO needs high entropy for Nash
                if [ "$algo" == "ppo" ] && [ "$ablation" == "6" ]; then
                    EXTRA_FLAGS="--ent_coef_override 0.1"
                fi
                # SAC uses the discrete-uniform regularizer on these discrete games
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
                    $EXTRA_FLAGS &
            done
            wait
        done
    done
done
