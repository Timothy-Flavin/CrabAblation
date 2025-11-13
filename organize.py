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

# Map prefix tokens to runner folder
PREFIX_TO_RUNNER = {
    "dqn_mujoco": "mujoco",
    "dqn_minigrid": "minigrid",
    "dqn_cartpole": "cartpole",
}

# Recognize file base names we care about
BASE_NAMES = {
    "train_scores",
    "eval_scores",
    "loss_hist",
    "smooth_train_scores",
}

# Regex: dqn_<runner>_<basename>_<run>_<ablation>(.ext)?
LEGACY_RE = re.compile(r"^(dqn_[a-zA-Z0-9]+)_([a-z_]+)_(\d+)_(\d+)(\..+)?$")


def plan_move(file: Path):
    """Return (dst_path) if the file matches a legacy pattern, else None."""
    name = file.name
    m = LEGACY_RE.match(name)
    if not m:
        return None
    prefix, base, run, ablation, ext = m.groups()
    runner = PREFIX_TO_RUNNER.get(prefix)
    if not runner:
        # Try to infer from base if prefix unknown
        if "minigrid" in name:
            runner = "minigrid"
        elif "mujoco" in name:
            runner = "mujoco"
        elif "cartpole" in name:
            runner = "cartpole"
        else:
            return None
    if base not in BASE_NAMES:
        return None
    if not ext:
        # default to png if missing extension (matplotlib default)
        ext = ".png"
    dst_dir = RESULTS_DIR / runner
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
