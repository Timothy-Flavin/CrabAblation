set -euo pipefail

echo "[laptop] Running dqn on minigrid | Ablation 0 | Run 1"
python runner.py --algo dqn --env_name minigrid --ablation 0 --run 1

echo "[laptop] Running dqn on minigrid | Ablation 3 | Run 2"
python runner.py --algo dqn --env_name minigrid --ablation 3 --run 2

echo "[laptop] Running dqn on minigrid | Ablation 3 | Run 3"
python runner.py --algo dqn --env_name minigrid --ablation 3 --run 3

echo "[laptop] Running ppo on minigrid | Ablation 0 | Run 1"
python runner.py --algo ppo --env_name minigrid --ablation 0 --run 1

echo "[laptop] Running ppo on minigrid | Ablation 0 | Run 2"
python runner.py --algo ppo --env_name minigrid --ablation 0 --run 2

echo "[laptop] Running ppo on minigrid | Ablation 0 | Run 3"
python runner.py --algo ppo --env_name minigrid --ablation 0 --run 3

echo "[laptop] Running ppo on minigrid | Ablation 1 | Run 1"
python runner.py --algo ppo --env_name minigrid --ablation 1 --run 1

echo "[laptop] Running ppo on minigrid | Ablation 1 | Run 2"
python runner.py --algo ppo --env_name minigrid --ablation 1 --run 2

echo "[laptop] Running ppo on minigrid | Ablation 1 | Run 3"
python runner.py --algo ppo --env_name minigrid --ablation 1 --run 3

echo "[laptop] Running ppo on minigrid | Ablation 2 | Run 1"
python runner.py --algo ppo --env_name minigrid --ablation 2 --run 1

echo "[laptop] Running ppo on minigrid | Ablation 2 | Run 2"
python runner.py --algo ppo --env_name minigrid --ablation 2 --run 2

echo "[laptop] Running ppo on minigrid | Ablation 3 | Run 1"
python runner.py --algo ppo --env_name minigrid --ablation 3 --run 1

echo "[laptop] Running ppo on minigrid | Ablation 3 | Run 2"
python runner.py --algo ppo --env_name minigrid --ablation 3 --run 2

echo "[laptop] Running ppo on minigrid | Ablation 3 | Run 3"
python runner.py --algo ppo --env_name minigrid --ablation 3 --run 3

echo "[laptop] Running ppo on minigrid | Ablation 4 | Run 1"
python runner.py --algo ppo --env_name minigrid --ablation 4 --run 1

echo "[laptop] Running ppo on minigrid | Ablation 4 | Run 2"
python runner.py --algo ppo --env_name minigrid --ablation 4 --run 2

echo "[laptop] Running ppo on minigrid | Ablation 4 | Run 3"
python runner.py --algo ppo --env_name minigrid --ablation 4 --run 3

echo "[laptop] Running dqn on cartpole | Ablation 4 | Run 1"
python runner.py --algo dqn --env_name cartpole --ablation 4 --run 1

echo "[laptop] Running dqn on cartpole | Ablation 4 | Run 2"
python runner.py --algo dqn --env_name cartpole --ablation 4 --run 2

echo "[laptop] Running dqn on cartpole | Ablation 4 | Run 3"
python runner.py --algo dqn --env_name cartpole --ablation 4 --run 3

echo "[laptop] Running ppo on cartpole | Ablation 2 | Run 1"
python runner.py --algo ppo --env_name cartpole --ablation 2 --run 1

echo "[laptop] Running ppo on cartpole | Ablation 2 | Run 2"
python runner.py --algo ppo --env_name cartpole --ablation 2 --run 2

echo "[laptop] Running ppo on cartpole | Ablation 2 | Run 3"
python runner.py --algo ppo --env_name cartpole --ablation 2 --run 3

echo "[laptop] Running ppo on cartpole | Ablation 4 | Run 1"
python runner.py --algo ppo --env_name cartpole --ablation 4 --run 1

echo "[laptop] Running ppo on cartpole | Ablation 4 | Run 2"
python runner.py --algo ppo --env_name cartpole --ablation 4 --run 2

echo "[laptop] Running ppo on cartpole | Ablation 5 | Run 1"
python runner.py --algo ppo --env_name cartpole --ablation 5 --run 1

echo "[laptop] Running ppo on cartpole | Ablation 5 | Run 2"
python runner.py --algo ppo --env_name cartpole --ablation 5 --run 2

echo "[laptop] Running ppo on cartpole | Ablation 5 | Run 3"
python runner.py --algo ppo --env_name cartpole --ablation 5 --run 3

echo "[laptop] Running dqn on mujoco | Ablation 4 | Run 1"
python runner.py --algo dqn --env_name mujoco --ablation 4 --run 1

echo "[laptop] Running dqn on mujoco | Ablation 4 | Run 2"
python runner.py --algo dqn --env_name mujoco --ablation 4 --run 2

echo "[laptop] Running dqn on mujoco | Ablation 4 | Run 3"
python runner.py --algo dqn --env_name mujoco --ablation 4 --run 3

echo "[laptop] Running ppo on mujoco | Ablation 3 | Run 2"
python runner.py --algo ppo --env_name mujoco --ablation 3 --run 2

echo "[laptop] Running ppo on mujoco | Ablation 4 | Run 1"
python runner.py --algo ppo --env_name mujoco --ablation 4 --run 1

echo "[laptop] Running ppo on mujoco | Ablation 4 | Run 2"
python runner.py --algo ppo --env_name mujoco --ablation 4 --run 2

echo "[laptop] Running ppo on mujoco | Ablation 4 | Run 3"
python runner.py --algo ppo --env_name mujoco --ablation 4 --run 3

echo "[laptop] Running ppo on mujoco | Ablation 5 | Run 1"
python runner.py --algo ppo --env_name mujoco --ablation 5 --run 1

echo "[laptop] Running ppo on mujoco | Ablation 5 | Run 2"
python runner.py --algo ppo --env_name mujoco --ablation 5 --run 2

echo "[laptop] Running ppo on mujoco | Ablation 5 | Run 3"
python runner.py --algo ppo --env_name mujoco --ablation 5 --run 3

