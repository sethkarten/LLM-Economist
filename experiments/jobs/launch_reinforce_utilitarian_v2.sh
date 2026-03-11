#!/bin/bash
#SBATCH --job-name=reinforce-util-v2
#SBATCH --partition=ailab
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/gpfs/CHIJ/milkkarten/LLM-Economist/logs/reinforce_util_v2_%A.out
#SBATCH --error=/scratch/gpfs/CHIJ/milkkarten/LLM-Economist/logs/reinforce_util_v2_%A.err

# REINFORCE++ utilitarian v2 - Fixes entropy collapse and KL anchoring issues
# Changes vs v1: kl-coef 0.05→0.01, entropy-coef 0.01→0.05
# Usage: sbatch experiments/jobs/launch_reinforce_utilitarian_v2.sh [SEED]

SEED=${1:-42}

echo "=================================================="
echo "REINFORCE++ Utilitarian v2 Training"
echo "Seed: ${SEED}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=================================================="

# Environment setup — use uv venv, not conda
export PATH="/home/sk9014/.local/bin:$PATH"

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

uv run python experiments/run_reinforce_training.py \
    --seed ${SEED} \
    --num-iterations 200 \
    --num-rollouts 8 \
    --num-agents 32 \
    --tax-year-length 16 \
    --num-tax-years 1 \
    --bracket-setting three \
    --swf-weighting utilitarian \
    --kl-coef 0.01 \
    --entropy-coef 0.05 \
    --output-dir results/reinforce_utilitarian_v2

echo "=================================================="
echo "Job completed: $(date)"
echo "=================================================="
