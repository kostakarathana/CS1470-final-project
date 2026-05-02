#!/usr/bin/env bash
# =============================================================================
# Oscar environment setup for CS1470 Final Project
# Run this ONCE from your project root on Oscar before submitting any jobs.
#
# Usage:
#   cd ~/CS1470-final-project
#   bash scripts/oscar/setup_env.sh
# =============================================================================
set -euo pipefail

ENV_NAME="cs1470"
PYTHON_VERSION="3.11"

echo "=== CS1470 Oscar Environment Setup ==="
echo "This will create a conda environment named '${ENV_NAME}' with PyTorch + CUDA."
echo ""

# Load anaconda module (Oscar's module name — update if yours differs)
module load anaconda3/2023.09-0-aqbc

# Create env if it doesn't already exist
if conda env list | grep -q "^${ENV_NAME} "; then
  echo "Conda env '${ENV_NAME}' already exists — skipping creation."
else
  echo "Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
  conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
fi

# Activate
conda activate "${ENV_NAME}"

# Install PyTorch with CUDA 12.1 (matches Oscar's available CUDA toolkit)
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project + optional deps
echo "Installing project dependencies..."
pip install -e ".[dev,analysis]"
pip install docker

# Quick sanity check
echo "Verifying install..."
python - <<'PY'
import torch
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
PY

python -m madreamer.cli.train --config configs/ppo_smoke.yaml --steps 24
echo ""
echo "=== Setup complete! ==="
echo "Activate your env with:  conda activate ${ENV_NAME}"
echo "Then submit jobs with:   bash scripts/oscar/submit_all.sh"
