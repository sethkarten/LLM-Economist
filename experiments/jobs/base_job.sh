#!/bin/bash
#SBATCH --job-name=llm_econ
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# Base job script for LLM Economist experiments on B200 cluster
# This script is sourced by other job scripts

# Exit on error
set -e

# Print job info
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=============================================="

# Load modules (adjust for your cluster)
module load cuda/12.4 || true
module load anaconda || true

# Activate environment
source ~/.bashrc
conda activate llmecon || source /opt/conda/etc/profile.d/conda.sh && conda activate llmecon

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/scratch/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/$USER/.cache/transformers
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Create log directory
mkdir -p logs

# Print GPU info
nvidia-smi

echo "Environment ready. Starting experiment..."
