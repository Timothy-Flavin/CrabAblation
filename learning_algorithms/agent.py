from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch.utils.tensorboard import SummaryWriter

import os
import subprocess
import shutil


def auto_setup_msvc():
    if os.name != "nt":
        return  # Only for Windows

    # 1. Check if cl.exe is already visible
    if shutil.which("cl.exe"):
        return

    # 2. Use vswhere to find the VS installation path
    vswhere_path = os.path.expandvars(
        r"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    )
    if not os.path.exists(vswhere_path):
        return

    try:
        vs_path = (
            subprocess.check_output(
                [
                    vswhere_path,
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ]
            )
            .decode()
            .strip()
        )

        # 3. Dig into the MSVC folders to find the actual binaries
        msvc_root = os.path.join(vs_path, "VC", "Tools", "MSVC")
        versions = sorted(os.listdir(msvc_root), reverse=True)
        if versions:
            bin_path = os.path.join(msvc_root, versions[0], "bin", "Hostx64", "x64")
            if os.path.exists(bin_path):
                os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
                print(f"--- Auto-configured MSVC: {versions[0]} ---")
    except Exception as e:
        print(f"MSVC Auto-config failed: {e}")


# Run this before calling torch.compile()
auto_setup_msvc()


def initialize_msvc_env():
    if os.name != "nt":
        return

    # 1. Find vcvars64.bat using vswhere
    vswhere_path = os.path.expandvars(
        r"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    )
    if not os.path.exists(vswhere_path):
        print("❌ vswhere.exe not found. Is Visual Studio installed?")
        return

    vs_path = (
        subprocess.check_output(
            [vswhere_path, "-latest", "-products", "*", "-property", "installationPath"]
        )
        .decode()
        .strip()
    )

    vcvars_path = os.path.join(vs_path, "VC", "Auxiliary", "Build", "vcvars64.bat")

    if not os.path.exists(vcvars_path):
        print(f"❌ Could not find vcvars64.bat at {vcvars_path}")
        return

    # 2. Run the bat file and capture the environment variables it sets
    # We use 'set' to print all variables after the bat runs
    cmd = f'"{vcvars_path}" && set'
    output = subprocess.check_output(cmd, shell=True).decode(errors="ignore")

    # 3. Update os.environ with the new values (PATH, INCLUDE, LIB, etc.)
    for line in output.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            # Only update critical build variables to avoid bloat
            if key.upper() in ["PATH", "INCLUDE", "LIB", "LIBPATH"]:
                os.environ[key] = value

    print("✅ MSVC Environment (including OpenMP) fully initialized.")


# Call this at the VERY start of your benchmark.py
initialize_msvc_env()


class Agent(ABC):
    """Common runtime API for training/evaluation agents in this repository."""

    def __init__(self):
        self.tb_writer: SummaryWriter | None = None
        self.tb_prefix: str = "agent"
        self.last_losses: dict[str, Any] = {}
        self.timing: dict[str, float] = {}

    @abstractmethod
    def to(self, device) -> Any:
        """Move all model state and optimizer-linked modules to a target device."""
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        """Run a single update/training step and return a scalar loss/metric."""
        pass

    @abstractmethod
    def sample_action(self, *args, **kwargs) -> Any:
        """Return action(s) from observations under the current policy/value rule."""
        pass

    def attach_tensorboard(self, writer: SummaryWriter, prefix: str = "agent") -> Any:
        """Attach TensorBoard writer used by concrete agents for logging."""
        self.tb_writer = writer
        self.tb_prefix = prefix
        pass

    def update_target(self) -> Any:
        """Optional target-network update hook for value-based agents."""
        return None

    def update_running_stats(self, *args, **kwargs) -> Any:
        """Optional running-statistics update hook (obs/reward normalization, etc.)."""
        return None
