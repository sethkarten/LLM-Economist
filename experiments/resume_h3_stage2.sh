#!/bin/bash
#SBATCH --job-name=h3-1k
#SBATCH --partition=ailab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gpfs/CHIJ/milkkarten/LLM-Economist/slurm-h3-1k-seed%a-%j.out

# H3 REINFORCE++ Stage 2: Continue from iteration 800 -> 1000
# Final stage of extended training

SEED=${1:-42}

echo "================================================================"
echo "H3 REINFORCE++ Stage 2 - Seed $SEED (800 -> 1000 iters)"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Target: 1000 total iterations (200 more to go, ~20 hours)"
echo "================================================================"

# Environment setup
source /scratch/gpfs/CHIJ/milkkarten/.bashrc
conda activate llm

cd /scratch/gpfs/CHIJ/milkkarten/LLM-Economist

# Resume from iteration 800 checkpoint (from stage 1)
python experiments/run_reinforce_h3_with_training_offline.py \
    --seed $SEED \
    --num-iterations 1000 \
    --rollouts-per-iter 16 \
    --num-agents 100 \
    --output /scratch/gpfs/CHIJ/milkkarten/LLM-Economist/results/h3_extended_seed${SEED} \
    --resume /scratch/gpfs/CHIJ/milkkarten/LLM-Economist/results/h3_extended_seed${SEED}/iter_800

echo "================================================================"
echo "Training complete at $(date)"
echo "Final checkpoint: results/h3_extended_seed${SEED}/best/"
echo "================================================================"
