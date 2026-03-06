#!/bin/bash
#SBATCH --job-name=reinforce-v2
#SBATCH --partition=ailab
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=/scratch/gpfs/CHIJ/milkkarten/LLM-Economist/logs/reinforce_v2_%A.out
#SBATCH --error=/scratch/gpfs/CHIJ/milkkarten/LLM-Economist/logs/reinforce_v2_%A.err

# REINFORCE++ v2 - Uses AsyncLLMEconomist as rollout env (fixes SWF/prompt/population bugs)
# Usage: sbatch experiments/jobs/launch_reinforce_v2.sh [SEED]

SEED=${1:-42}

echo "=================================================="
echo "REINFORCE++ v2 Training"
echo "Seed: ${SEED}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=================================================="

# Environment setup
module purge
module load anaconda3/2024.10
conda activate /home/sk9014/anaconda3/envs/llm

# Offline mode for compute nodes (no internet)
export HF_HOME=/scratch/gpfs/CHIJ/milkkarten/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR=/scratch/gpfs/CHIJ/milkkarten/LLM-Economist/wandb

# Disable torch compile to avoid hangs
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export VLLM_USE_V1=0

cd /scratch/gpfs/CHIJ/milkkarten/LLM-Economist

mkdir -p logs

python experiments/run_reinforce_training.py \
    --seed ${SEED} \
    --num-iterations 500 \
    --num-rollouts 16 \
    --num-agents 32 \
    --tax-year-length 64 \
    --num-tax-years 4 \
    --output-dir results/reinforce_v2_seed${SEED}

echo "=================================================="
echo "Job completed: $(date)"
echo "=================================================="
