#!/bin/bash
#SBATCH --job-name=h3-800
#SBATCH --partition=ailab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/gpfs/CHIJ/milkkarten/LLM-Economist/slurm-h3-800-%j.out

# Resume H3 REINFORCE++ training from iteration 100 -> 800
# Stage 1 of extended training (stays in faster QOS queue with <72h limit)

SEED=${1:-42}

echo "================================================================"
echo "H3 REINFORCE++ Stage 1 - Seed $SEED (100 -> 800 iters)"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Target: 800 iterations (700 more to go, ~70 hours)"
echo "Stage 2 will continue 800 -> 1000"
echo "================================================================"

# Environment setup
source /scratch/gpfs/CHIJ/milkkarten/.bashrc
conda activate llm

# CRITICAL: Disable offline mode for HuggingFace Hub
unset HF_HUB_OFFLINE
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0

cd /scratch/gpfs/CHIJ/milkkarten/LLM-Economist

# Resume from iteration 100 checkpoint
python experiments/run_reinforce_h3_with_training_offline.py \
    --seed $SEED \
    --num-iterations 800 \
    --rollouts-per-iter 16 \
    --num-agents 100 \
    --output /scratch/gpfs/CHIJ/milkkarten/LLM-Economist/results/h3_extended_seed${SEED} \
    --resume /scratch/gpfs/CHIJ/milkkarten/LLM-Economist/results/h3_with_training_seed${SEED}/iter_100

echo "================================================================"
echo "Stage 1 complete at $(date)"
echo "Next: Submit stage 2 to continue 800 -> 1000 iterations"
echo "================================================================"
