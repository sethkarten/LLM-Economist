#!/bin/bash
#SBATCH --job-name=icrl-v2
#SBATCH --partition=ailab
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/gpfs/CHIJ/milkkarten/LLM-Economist/logs/icrl_v2_%A.out
#SBATCH --error=/scratch/gpfs/CHIJ/milkkarten/LLM-Economist/logs/icrl_v2_%A.err

# ICRL v2 - Uses ROLE_MESSAGES personas with 0-labor fix
# Usage: sbatch experiments/jobs/launch_icrl_v2.sh [SEED]

SEED=${1:-42}

echo "=================================================="
echo "ICRL v2 Evaluation"
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

# Disable torch compile to avoid hangs
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export VLLM_USE_V1=0

cd /scratch/gpfs/CHIJ/milkkarten/LLM-Economist

mkdir -p logs results/icrl_v2

uv run python experiments/run_icrl_single.py \
    --seed ${SEED} \
    --num-agents 100 \
    --max-timesteps 2000 \
    --tax-year-length 128 \
    --history-len 50 \
    --model Qwen/Qwen3-8B-AWQ \
    --bracket-setting three \
    --max-model-len 8192 \
    --output results/icrl_v2/seed${SEED}.json

echo "=================================================="
echo "Job completed: $(date)"
echo "=================================================="
