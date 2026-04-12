import os
from pathlib import Path

envs = ["minigrid", "cartpole", "mujoco"]
models = ["dqn", "ppo", "sac"]
ablations = [0, 1, 2, 3, 4, 5]
runs = [1, 2, 3]

missing_script = os.path.join("time_files", "run_timpc_all_missing_experiments.sh")

lines = [
    "#!/usr/bin/env bash",
    "# Auto-generated script to run ALL missing experiments on timpc",
    "set -euo pipefail",
    ""
]

for model in models:
    lines.append(f"# {model.upper()} Experiments")
    lines.append(f"echo \"# {model.upper()} Experiments\"")
    for env in envs:
        for abl in ablations:
            for run in runs:
                expected_file = Path(f"all_results/{model}/{env}/train_scores_{run}_{abl}.npy")
                if not expected_file.exists():
                    lines.append(f"python runner.py --algo {model} --env_name {env} --ablation {abl} --run {run}")
    lines.append("")

with open(missing_script, "w") as f:
    f.write("\n".join(lines))
