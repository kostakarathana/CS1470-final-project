#!/bin/bash
#SBATCH --job-name=team-shared-h3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --output=artifacts/oscar/team-shared-h3/logs/slurm_%j.out
#SBATCH --error=artifacts/oscar/team-shared-h3/logs/slurm_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

module load anaconda3/2023.09-0-aqbc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cs1470

mkdir -p artifacts/oscar/team-shared-h3/logs artifacts/oscar/team-shared-h3/checkpoints

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Started at : $(date)"

# Team / cooperative stretch goal — partial observability adds complexity
python -m madreamer.cli.train --config configs/oscar/team_shared_h3.yaml

echo "Finished at: $(date)"
