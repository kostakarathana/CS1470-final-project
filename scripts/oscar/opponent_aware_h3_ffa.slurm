#!/bin/bash
#SBATCH --job-name=opp-aware-h3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --output=artifacts/oscar/opponent-aware-h3-ffa/logs/slurm_%j.out
#SBATCH --error=artifacts/oscar/opponent-aware-h3-ffa/logs/slurm_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

module load anaconda3/2023.09-0-aqbc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cs1470

mkdir -p artifacts/oscar/opponent-aware-h3-ffa/logs artifacts/oscar/opponent-aware-h3-ffa/checkpoints

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Started at : $(date)"

# Opponent-aware conditions on other agents' actions — slightly heavier than shared
python -m madreamer.cli.train --config configs/oscar/opponent_aware_h3_ffa.yaml

echo "Finished at: $(date)"
