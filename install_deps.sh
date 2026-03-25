#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./install_deps.sh                 # CPU PyTorch (default)
#   ./install_deps.sh --cuda-cu124    # CUDA 12.4 PyTorch wheels
#   ./install_deps.sh --cuda-cu121    # CUDA 12.1 PyTorch wheels
#
# Note: user request listed "gemnasium[all]"; correct package name is "gymnasium[all]".

TORCH_MODE="cpu"

for arg in "$@"; do
  case "$arg" in
    --cuda-cu124)
      TORCH_MODE="cu124"
      ;;
    --cuda-cu121)
      TORCH_MODE="cu121"
      ;;
    --cpu)
      TORCH_MODE="cpu"
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Valid args: --cpu | --cuda-cu121 | --cuda-cu124"
      exit 1
      ;;
  esac
done

echo "==> Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "==> Installing PyTorch + torchvision (${TORCH_MODE})"
if [[ "$TORCH_MODE" == "cpu" ]]; then
  python -m pip install --upgrade torch torchvision
elif [[ "$TORCH_MODE" == "cu121" ]]; then
  python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121
elif [[ "$TORCH_MODE" == "cu124" ]]; then
  python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi

echo "==> Installing project dependencies"
python -m pip install --upgrade \
  numpy \
  matplotlib \
  pygame \
  tyro \
  tensorboard \
  "gymnasium[all]" \
  "pettingzoo[classic]" \
  minigrid \
  cleanrl \
  wandb \
  ortools

echo "==> Dependency installation complete"
echo "If using a venv, activate first: source .venv/bin/activate"
