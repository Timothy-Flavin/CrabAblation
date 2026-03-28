echo "Running grid search for hide-and-seek-engine..."
python benchmark.py --algo dqn --grid_search --device_name "laptop" --env_name hide-and-seek-engine
python benchmark.py --algo ppo --grid_search --device_name "laptop" --env_name hide-and-seek-engine
python benchmark.py --algo sac --grid_search --device_name "laptop" --env_name hide-and-seek-engine

echo "Running grid search for cartpole..."
python benchmark.py --algo dqn --grid_search --device_name "laptop" --env_name cartpole
python benchmark.py --algo ppo --grid_search --device_name "laptop" --env_name cartpole
python benchmark.py --algo sac --grid_search --device_name "laptop" --env_name cartpole

echo "Running grid search for minigrid..."
python benchmark.py --algo dqn --grid_search --device_name "laptop" --env_name minigrid
python benchmark.py --algo ppo --grid_search --device_name "laptop" --env_name minigrid
python benchmark.py --algo sac --grid_search --device_name "laptop" --env_name minigrid

echo "Running grid search for mujoco..."
python benchmark.py --algo dqn --grid_search --device_name "laptop" --env_name mujoco
python benchmark.py --algo ppo --grid_search --device_name "laptop" --env_name mujoco
python benchmark.py --algo sac --grid_search --device_name "laptop" --env_name mujoco

echo "All grid searches completed!"
