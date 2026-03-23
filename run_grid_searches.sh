#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <computer_name>"
    echo "Example: $0 laptop"
    exit 1
fi

COMPUTER_NAME=$1

echo "Running grid search for cartpole..."
python benchmark_dqn.py --grid_search --grid_name "$COMPUTER_NAME" --env_name cartpole
python benchmark_pg.py --grid_search --grid_name "$COMPUTER_NAME" --env_name cartpole

echo "Running grid search for minigrid..."
python benchmark_dqn.py --grid_search --grid_name "$COMPUTER_NAME" --env_name minigrid
python benchmark_pg.py --grid_search --grid_name "$COMPUTER_NAME" --env_name minigrid

echo "Running grid search for mujoco..."
python benchmark_dqn.py --grid_search --grid_name "$COMPUTER_NAME" --env_name mujoco
python benchmark_pg.py --grid_search --grid_name "$COMPUTER_NAME" --env_name mujoco

echo "All grid searches completed!"
