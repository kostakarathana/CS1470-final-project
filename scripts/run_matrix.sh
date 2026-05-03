#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

print_eval() {
  "$PYTHON_BIN" -c 'import json, sys
payload = json.load(sys.stdin)
print(json.dumps(payload.get("latest_eval_metrics", {}), indent=2, sort_keys=True))'
}

echo "=== FINAL EXPERIMENT MATRIX START ==="
echo "Started at $(date)"

# Baseline
echo -e "\n[1/7] Running ppo_ffa..."
"$PYTHON_BIN" -m madreamer.cli.train --config configs/final/ppo_ffa.yaml | print_eval

# Sharing ablation
echo -e "\n[2/7] Running independent_h3_ffa..."
"$PYTHON_BIN" -m madreamer.cli.train --config configs/final/independent_h3_ffa.yaml | print_eval

echo -e "\n[3/7] Running shared_h3_ffa..."
"$PYTHON_BIN" -m madreamer.cli.train --config configs/final/shared_h3_ffa.yaml | print_eval

echo -e "\n[4/7] Running opponent_aware_h3_ffa..."
"$PYTHON_BIN" -m madreamer.cli.train --config configs/final/opponent_aware_h3_ffa.yaml | print_eval

# Horizon ablation
echo -e "\n[5/7] Running shared_h1_ffa..."
"$PYTHON_BIN" -m madreamer.cli.train --config configs/final/shared_h1_ffa.yaml | print_eval

echo -e "\n[6/7] Running shared_h5_ffa..."
"$PYTHON_BIN" -m madreamer.cli.train --config configs/final/shared_h5_ffa.yaml | print_eval

# Stretch
echo -e "\n[7/7] Running team_shared_h3..."
"$PYTHON_BIN" -m madreamer.cli.train --config configs/final/team_shared_h3.yaml | print_eval

echo -e "\n=== FINAL EXPERIMENT MATRIX COMPLETE ==="
echo "Finished at $(date)"
echo -e "\nAll metrics written to artifacts/final/**/logs/metrics.jsonl"
"$PYTHON_BIN" analyze_results.py --log-root artifacts/final --output-dir results
