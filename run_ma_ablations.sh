#!/bin/bash

# Multi-Agent Ablation Runner for Tic-Tac-Toe and Leduc Poker
# Ablations 0-5 are the same as runner.py
# Ablation 6 is the "Base" algorithm (Default params, high entropy for PPO)

ENVS=("tictactoe" "leduc")
ALGOS=("dqn" "ppo" "sac")
ABLATIONS=(0 1 2 3 4 5 6)

for env in "${ENVS[@]}"; do
    for algo in "${ALGOS[@]}"; do
        for ablation in "${ABLATIONS[@]}"; do
            echo "Starting MA Ablation: Env=$env, Algo=$algo, Ablation=$ablation"
            
            # Default episodes
            EPISODES=10000
            if [ "$env" == "leduc" ]; then
                EPISODES=20000
            fi

            # Extra flags for Ablation 6 PPO to ensure high entropy for Nash
            EXTRA_FLAGS=""
            if [ "$algo" == "ppo" ] && [ "$ablation" == "6" ]; then
                EXTRA_FLAGS="--ent_coef_override 0.1"
            fi

            ./.venv/bin/python multiagent_runner.py \
                --algo "$algo" \
                --ma_env "$env" \
                --ablation "$ablation" \
                --total_episodes "$EPISODES" \
                $EXTRA_FLAGS
        done
    done
done
