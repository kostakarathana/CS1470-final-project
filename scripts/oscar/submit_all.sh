#!/usr/bin/env bash
# =============================================================================
# Submit all Oscar training jobs for the CS1470 final experiment matrix.
#
# Prerequisites:
#   1. Run scripts/oscar/setup_env.sh once to create the conda environment.
#   2. Push / copy the repo to Oscar and cd into the project root.
#   3. Run this script from the project root:
#        bash scripts/oscar/submit_all.sh
#
# All 7 jobs are independent and run in parallel.
# Outputs go to artifacts/oscar/<experiment>/logs/ and checkpoints/.
# Monitor with: squeue -u $USER
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== CS1470 Oscar Job Submission ==="
echo "Submitting from: $(pwd)"
echo ""

submit() {
  local name="$1"
  local script="$SCRIPT_DIR/$2"
  local job_id
  job_id=$(sbatch --parsable "$script")
  printf "  %-30s job ID: %s\n" "$name" "$job_id"
}

# [1/7] PPO baseline
submit "ppo_ffa" "ppo_ffa.slurm"

# [2/7] Independent world models (sharing ablation)
submit "independent_h3_ffa" "independent_h3_ffa.slurm"

# [3/7] Shared world model, horizon 3 (main Dreamer result)
submit "shared_h3_ffa" "shared_h3_ffa.slurm"

# [4/7] Opponent-aware world model (sharing ablation)
submit "opponent_aware_h3_ffa" "opponent_aware_h3_ffa.slurm"

# [5/7] Shared world model, horizon 1 (horizon ablation)
submit "shared_h1_ffa" "shared_h1_ffa.slurm"

# [6/7] Shared world model, horizon 5 (horizon ablation)
submit "shared_h5_ffa" "shared_h5_ffa.slurm"

# [7/7] Team cooperative setting (stretch goal)
submit "team_shared_h3" "team_shared_h3.slurm"

echo ""
echo "All 7 jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Once all jobs finish, generate results with:"
echo "  python3 analyze_results.py --log-root artifacts/oscar --output-dir results"
