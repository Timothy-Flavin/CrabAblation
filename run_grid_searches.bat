echo "Running grid search for cartpole..."
python benchmark_dqn.py --grid_search --device_name "laptop" --env_name cartpole
python benchmark_pg.py --grid_search --device_name "laptop" --env_name cartpole
python benchmark_sac.py --grid_search --device_name "laptop" --env_name cartpole

echo "Running grid search for minigrid..."
python benchmark_dqn.py --grid_search --device_name "laptop" --env_name minigrid
python benchmark_pg.py --grid_search --device_name "laptop" --env_name minigrid
python benchmark_sac.py --grid_search --device_name "laptop" --env_name minigrid

echo "Running grid search for mujoco..."
python benchmark_dqn.py --grid_search --device_name "laptop" --env_name mujoco
python benchmark_pg.py --grid_search --device_name "laptop" --env_name mujoco
python benchmark_sac.py --grid_search --device_name "laptop" --env_name mujoco

echo "All grid searches completed!"
