source .venv/bin/activate

echo "[mac] Running dqn mujoco | Ablation 1 | Run 1000"
python runner.py --algo dqn --env_name mujoco --ablation 1 --run 1000 --device_name mac

echo "[mac] Running sac mujoco | Ablation 1 | Run 1000"
python runner.py --algo sac --env_name mujoco --ablation 1 --run 1000 --device_name mac

echo "[mac] Running ppo mujoco | Ablation 1 | Run 1000"
python runner.py --algo ppo --env_name mujoco --ablation 1 --run 1000 --device_name mac
