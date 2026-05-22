#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

# Thread 1: CPU Tasks (Multi-Agent Ablations)
(
    ENVS=("tictactoe" "leduc" "rps")
    ALGOS=("sac" "dqn" "ppo") 
    ABLATIONS=(0 1 2 3 4 5 6)
    RUNS=5

    for env in "${ENVS[@]}"; do
        for algo in "${ALGOS[@]}"; do
            for ablation in "${ABLATIONS[@]}"; do
                for run in $(seq 1 $RUNS); do
                    echo "[lab-comp_cpu] Running MA: Env=$env | Algo=$algo | Ablation=$ablation | Run=$run"
                    
                    # Default episodes
                    EPISODES=10000
                    if [ "$env" == "leduc" ]; then
                        EPISODES=20000
                    elif [ "$env" == "rps" ]; then
                        EPISODES=5000
                    fi

                    EXTRA_FLAGS=""
                    if [ "$algo" == "ppo" ] && [ "$ablation" == "6" ]; then
                        EXTRA_FLAGS="--ent_coef_override 0.1"
                    fi
                    
                    OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES="" numactl --cpunodebind=0 --membind=0 python multiagent_runner.py \
                        --algo "$algo" \
                        --ma_env "$env" \
                        --ablation "$ablation" \
                        --run "$run" \
                        --total_episodes "$EPISODES" \
                        $EXTRA_FLAGS
                done
            done
        done
    done
) &

# Thread 2: GPU Tasks (DQN Cartpole)
(
    ABLATIONS=(0 1 2 3 4 5 6)
    RUNS=5

    for ablation in "${ABLATIONS[@]}"; do
        for run in $(seq 1 $RUNS); do
            echo "[lab-comp_gpu] Running dqn on cartpole | Ablation $ablation | Run $run"
            OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=1 --membind=1 python runner.py --algo dqn --env_name cartpole --ablation "$ablation" --run "$run" --device_name lab-comp_gpu
        done
    done
) &

# Wait for both background processes to finish
wait
echo "All tasks completed for lab-comp."
