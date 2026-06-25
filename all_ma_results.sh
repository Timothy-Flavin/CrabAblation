#!/bin/bash

algos=("dqn" "ppo" "sac")
envs=("rps" "leduc" "tictactoe")

for algo in "${algos[@]}"; do
    for env in "${envs[@]}"; do
        echo "Plotting $algo on $env (baseline)..."
        python plot_ma_results.py --algo "$algo" --env_name "$env" --results_dir results
    done
done

echo ""
echo "=== Shared-model results ==="
echo ""

for algo in "${algos[@]}"; do
    for env in "${envs[@]}"; do
        echo "Plotting $algo on $env (shared model)..."
        python plot_ma_results.py --algo "$algo" --env_name "$env" --results_dir results_shared
    done
done

echo ""
echo "=== Aggregate plots (baseline) ==="
echo ""
python plot_ma_results.py --mode aggregate \
    --algos "${algos[@]}" \
    --envs "${envs[@]}" \
    --results_dir results

echo ""
echo "=== Aggregate plots (shared model) ==="
echo ""
python plot_ma_results.py --mode aggregate \
    --algos "${algos[@]}" \
    --envs "${envs[@]}" \
    --results_dir results_shared