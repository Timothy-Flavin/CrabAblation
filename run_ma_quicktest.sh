#!/bin/bash

# Short Leduc/TTT smoke test to validate three recent fixes:
#   1. DQN eval no longer injects eps=0.5 (greedy/stochastic-as-designed at eval).
#   2. Per-env soft-DQN target entropy (0.5 for TTT/Leduc instead of the RPS-tuned 0.9).
#   3. plot_ma_results seed averaging (mean + SEM band) instead of small-N IQM.
#
# Deliberately tiny so it finishes quickly: one ablation, 3 seeds (enough to exercise
# the averaging band), short horizons. NOT a full sweep. Sequential (no GPU MPS/taskset).
# Results land in ./results/<algo>/<env>/ and plots are written per algo/env.

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

# Portable python: prefer the Windows venv, fall back to the POSIX venv, then PATH.
if [ -x "./.venv/Scripts/python.exe" ]; then
    PY="./.venv/Scripts/python.exe"
elif [ -x "./.venv/bin/python" ]; then
    source .venv/bin/activate
    PY="python"
else
    PY="python"
fi

ENVS=("leduc")
ALGOS=("dqn" "ppo" "sac")
ABLATION=6          # "Base" algorithm (default params; high entropy for PPO)
RUNS=3              # seeds 0,1,2 -> exercises the mean+SEM averaging band
EVAL_EPISODES=50
SAC_DISCRETE_ENTROPY=5.0

for env in "${ENVS[@]}"; do
    # Short horizons (full sweep uses 10k+). Keeps the smoke test quick.
    EPISODES=3000
    if [ "$env" == "leduc" ]; then
        EPISODES=4000
    fi

    for algo in "${ALGOS[@]}"; do
        for run in $(seq 1 $RUNS); do
            echo "QuickTest: Env=$env, Algo=$algo, Ablation=$ABLATION, Run=$run (episodes=$EPISODES)"

            EXTRA_FLAGS=""
            # Ablation 6 PPO needs high entropy for Nash.
            if [ "$algo" == "ppo" ]; then
                EXTRA_FLAGS="--ent_coef_override 0.1"
            fi
            # SAC uses the discrete-uniform regularizer on these discrete games.
            if [ "$algo" == "sac" ]; then
                EXTRA_FLAGS="--discrete_entropy $SAC_DISCRETE_ENTROPY"
            fi

            "$PY" multiagent_runner.py \
                --algo "$algo" \
                --ma_env "$env" \
                --ablation "$ABLATION" \
                --run "$run" \
                --total_episodes "$EPISODES" \
                --eval_episodes "$EVAL_EPISODES" \
                $EXTRA_FLAGS
        done
    done
done

# Plot each algo/env (mean + SEM band across the 3 seeds). Runner writes to ./results.
for env in "${ENVS[@]}"; do
    for algo in "${ALGOS[@]}"; do
        echo "Plotting: Algo=$algo, Env=$env"
        "$PY" plot_ma_results.py --mode single --algo "$algo" --env_name "$env" --results_dir results
    done
done

echo "Quick test done. Inspect ./results/<algo>/<env>/summary_plot.png"
echo "Check: DQN vs-random now trends up (TTT) and DQN exploitability trends down (Leduc); seed bands are visible."
