#!/bin/bash
#SBATCH --job-name=ppo-long-ffa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/oscar/ppo-long-ffa/logs/slurm_%j.out
#SBATCH --error=artifacts/oscar/ppo-long-ffa/logs/slurm_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

module load anaconda3/2023.09-0-aqbc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cs1470

mkdir -p artifacts/oscar/ppo-long-ffa/logs artifacts/oscar/ppo-long-ffa/checkpoints

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Started at : $(date)"

# PPO long run — 262k steps with shorter rollouts for more frequent updates
python -m madreamer.cli.train --config configs/oscar/ppo_long_ffa.yaml

echo "Finished at: $(date)"
