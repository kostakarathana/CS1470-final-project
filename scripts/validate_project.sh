#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m ruff check .
"$PYTHON_BIN" -m pytest
"$PYTHON_BIN" analyze_results.py

if [ -f artifacts/final/ppo-ffa/checkpoints/final-ppo-ffa_ppo_latest.pt ]; then
  "$PYTHON_BIN" visualize_game.py \
    --config configs/final/ppo_ffa.yaml \
    --checkpoint artifacts/final/ppo-ffa/checkpoints/final-ppo-ffa_ppo_latest.pt \
    --output artifacts/validation_gameplay.gif \
    --frames 4 \
    --episodes 1 \
    --fps 4 || echo "Skipping gameplay visualization: checkpoint is stale or incompatible."
fi

if [ -f artifacts/final/shared-h3-ffa/checkpoints/final-shared-h3-ffa_shared_latest.pt ]; then
  "$PYTHON_BIN" visualize_imagination.py \
    --config configs/final/shared_h3_ffa.yaml \
    --checkpoint artifacts/final/shared-h3-ffa/checkpoints/final-shared-h3-ffa_shared_latest.pt \
    --output artifacts/validation_imagination.gif \
    --frames 3 \
    --fps 3 || echo "Skipping imagination visualization: checkpoint is stale or incompatible."
fi

echo "Project validation complete."
