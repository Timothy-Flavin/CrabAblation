#!/usr/bin/env python3
import os
import re
import shutil
from pathlib import Path

"""
Organize legacy result files into folderized structure.

Old patterns (examples):
- results/dqn_mujoco_train_scores_{run}_{ablation}[.npy|.png]
- results/dqn_minigrid_eval_scores_{run}_{ablation}[.npy|.png]
- results/dqn_mujoco_loss_hist_{run}_{ablation}.npy
- results/dqn_mujoco_smooth_train_scores_{run}_{ablation}.npy
- cartpole previously (bug) wrote with 'dqn_mujoco' prefix; we attempt to guess by runner context.

New layout:
- results/{runner_name}/train_scores_{run}_{ablation}.npy
- results/{runner_name}/eval_scores_{run}_{ablation}.npy
- results/{runner_name}/loss_hist_{run}_{ablation}.npy
- results/{runner_name}/smooth_train_scores_{run}_{ablation}.npy
- and corresponding PNGs for plots.
"""

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Map prefix tokens to (algo, env)
PREFIX_TO_RUNNER = {
    "dqn_mujoco": ("dqn", "mujoco"),
    "dqn_minigrid": ("dqn", "minigrid"),
    "dqn_cartpole": ("dqn", "cartpole"),
    "ppo_mujoco": ("ppo", "mujoco"),
    "ppo_minigrid": ("ppo", "minigrid"),
    "ppo_cartpole": ("ppo", "cartpole"),
    "sac_mujoco": ("sac", "mujoco"),
    "sac_minigrid": ("sac", "minigrid"),
    "sac_cartpole": ("sac", "cartpole"),
}

# Recognize file base names we care about
BASE_NAMES = {
    "train_scores",
    "eval_scores",
    "loss_hist",
    "smooth_train_scores",
    "train_time",
}

# Regex: (algo)_(env)_(basename)_(run)_(ablation)(.ext)?
LEGACY_RE = re.compile(r"^([a-zA-Z0-9]+)_([a-zA-Z0-9\-]+)_([a-z_]+)_(\d+)_(\d+)(\..+)?$")


def plan_move(file: Path):
    """Return (dst_path) if the file matches a legacy pattern, else None."""
    name = file.name
    m = LEGACY_RE.match(name)
    if not m:
        return None
    algo, env, base, run, ablation, ext = m.groups()
    
    # Try using prefix map if it matched old bugs like dqn_mujoco for cartpole
    prefix = f"{algo}_{env}"
    mapping = PREFIX_TO_RUNNER.get(prefix)
    
    if mapping:
        algo, env = mapping
    else:
        # Infer from base/name just in case there's something weird
        if "minigrid" in name:
            env = "minigrid"
        elif "mujoco" in name:
            env = "mujoco"
        elif "cartpole" in name:
            env = "cartpole"
    
    if base not in BASE_NAMES:
        return None
    if not ext:
        # default to png if missing extension (matplotlib default)
        ext = ".png"
        
    dst_dir = RESULTS_DIR / algo / env
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_name = f"{base}_{run}_{ablation}{ext}"
    return dst_dir / dst_name


def main():
    moved = 0
    skipped = 0
    for p in RESULTS_DIR.glob("*"):
        if p.is_dir():
            # Skip new structured dirs
            if p.name in {"mujoco", "minigrid", "cartpole"}:
                continue
        if p.is_file():
            dst = plan_move(p)
            if dst is None:
                skipped += 1
                continue
            if dst.exists():
                # Avoid overwriting; keep the earliest or rename new with suffix
                base = dst.stem
                ext = dst.suffix
                i = 1
                new_dst = dst.with_name(f"{base}_{i}{ext}")
                while new_dst.exists():
                    i += 1
                    new_dst = dst.with_name(f"{base}_{i}{ext}")
                dst = new_dst
            print(f"Moving {p} -> {dst}")
            shutil.move(str(p), str(dst))
            moved += 1
    print(f"Done. Moved {moved} files, skipped {skipped}.")


if __name__ == "__main__":
    main()
