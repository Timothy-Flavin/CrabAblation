# Repository Structure

This document provides an overview of the directories and files within the CrabAblation repository. 

## Root Directory Files
- `agent_api.md`: Documentation defining the agent interface and API specifications used across different learning algorithms.
- `benchmark.py`: Script used for benchmarking the performance (such as speed and throughput) of the algorithms and environments.
- `check_missing.py`: A utility script to check which experimental runs or ablations are missing from the results.
- `device_name.txt`: A simple text file used to specify or record the name of the device running the experiments.
- `environment_utils.py`: Utilities for initializing and wrapping the reinforcement learning environments (e.g., CartPole, Minigrid, MuJoCo).
- `experiment_consolidation.sh`: A shell script to consolidate or merge experimental data from various sources or directories.
- `generate_all_graphs.bat` / `generate_all_graphs.sh`: Batch and shell scripts to automate the generation of evaluation and training graphs across all algorithms and environments.
- `graph.py`: The core Python script used to process numpy arrays of training/evaluation scores and plot them.
- `install_deps.sh`: A shell script outlining the required dependencies for setting up the environment.
- `LICENSE`: The repository's open-source license.
- `organize.py`: A script used to organize or move generated results files into structured directories.
- `report.md`: A tracker document used to log anomalies, unexplainable results, and steps to fix issues found during ablation testing.
- `run_ablations.bat` / `run_ablations.sh`: Scripts to trigger a batch of ablation experiments.
- `run_grid_searches.bat` / `run_grid_searches.sh`: Scripts to trigger hyperparameter grid searches.
- `runner.py`: The main entry-point script for training and evaluating reinforcement learning models.
- `runner_utils.py`: Contains helper functions for `runner.py`, including argument parsing, system configuration, and seeding.

## Directories

### `all_results/`
Contains the final, structured log and metric files across all algorithms (`dqn`, `ppo`, `sac`) and all tested environments (`cartpole`, `hide-and-seek-engine`, `minigrid`, `mujoco`). This is likely the consolidated output data directory.

### `learning_algorithms/`
This directory houses the neural network architectures, replay buffers, and implementations logic for the respective reinforcement learning agents:
- `agent.py`: Base reinforcement learning agent class.
- `cleanrl_buffers.py`: Implementations of replay buffers, usually adapted from CleanRL.
- `DQN_Rainbow.py`, `DQN_Rainbow_spec.py`: Deep Q-Network implementations with Rainbow extensions.
- `PG_Rainbow.py`: Policy Gradient iterations leveraging Rainbow mechanisms.
- `SAC_Rainbow.py`: Soft Actor-Critic leveraging Rainbow enhancements.
- `MixedObservationEncoder.py`: Neural network module for processing diverse states/observations (like in Minigrid).
- Various `PopArt*Layer.py`: Variations of PopArt layers used to normalize targets in value networks.
- `RainbowNetworks.py`: Implementations of network architectures designed for the Rainbow algorithms.
- `RandomDistilation.py`: Implementation of Random Network Distillation (RND) for exploration bonuses.

### `results/`
This directory resembles `all_results/` and acts as the initial dumping ground for logging and metrics recorded sequentially by Tensorboard or standard loggers during the execution of models.

### `test_level/`
Holds JSON artifacts determining structure for specific environments or agent test boundaries.
- `agents.json`, `survivors.json`, `tiles.json`: Describe the entities, environments, or tiles, primarily for testing custom 2D grid/Hide and Seek environments.

### `time_files/`
Scripts dedicated to scheduling and orchestrating large suites of multi-run experiments distributed or split across multiple hardware limits or machines.
- `rerun_results.sh`: Auto-generated shell scripts that re-queue anomalous or incomplete experiments.
- `run_laptop_experiments.*`, `run_mac_experiments.sh`, `run_timpc_experiments.sh`: System-specific batch/shell scripts for distinct hardware profiles (e.g., `mac`, `timpc`, `shadow`, `laptop`).
- `run_scheduler.py`: A python script orchestrating the creation and assignment of runs across the device subdirectories based on computational limits.
- `laptop/`, `mac/`, `shadow/`, `timpc/`: Subdirectories containing JSON chunk configurations detailing exact lists of experimental runs parameterized for that specific machine architecture to execute.