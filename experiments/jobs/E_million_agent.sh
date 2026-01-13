#!/bin/bash
#SBATCH --job-name=llm_econ_1M
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=ailab
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1400G

# 1 Million Agent Bounded Rationality Experiment
# Uses full H200 node (8 GPUs × 141GB = 1.1TB GPU memory)
# Estimated runtime: ~24-48 hours per model depending on throughput
#
# Usage:
#   sbatch experiments/jobs/E_million_agent.sh
#   sbatch --export=MODEL=llama-3.1-8b experiments/jobs/E_million_agent.sh

set -e

# Configuration (can be overridden via --export)
MODEL=${MODEL:-"gemma3-4b"}
SEED=${SEED:-42}
NUM_AGENTS=${NUM_AGENTS:-1000000}
MAX_TIMESTEPS=${MAX_TIMESTEPS:-2000}
TAX_YEAR_LENGTH=${TAX_YEAR_LENGTH:-128}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-4}  # TP=4 for balance of throughput and efficiency
BATCH_SIZE=${BATCH_SIZE:-10000}  # Large batch for 1M agents

# Print job info
echo "=============================================="
echo "1 Million Agent Experiment"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Model: $MODEL"
echo "Seed: $SEED"
echo "Agents: $NUM_AGENTS"
echo "Timesteps: $MAX_TIMESTEPS"
echo "Tensor Parallel: $TENSOR_PARALLEL"
echo "Start time: $(date)"
echo "=============================================="

# Load modules
module load cuda/12.4 || true
module load anaconda || true

# Activate environment
source ~/.bashrc
conda activate llmecon || source /opt/conda/etc/profile.d/conda.sh && conda activate llmecon

# Environment variables
export HF_HOME=/scratch/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/$USER/.cache/transformers
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Use FP8 on H200s (native support)
export VLLM_USE_FP8=1

# Create directories
mkdir -p logs
mkdir -p results/million_agent

# Print GPU info
nvidia-smi

# Output path
OUTPUT_PATH="results/million_agent/${MODEL}_seed${SEED}_${NUM_AGENTS}agents.json"

echo ""
echo "Starting simulation..."
echo "Output: $OUTPUT_PATH"
echo ""

# Run the simulation
python -m llm_economist.main_async \
    --num-agents $NUM_AGENTS \
    --max-timesteps $MAX_TIMESTEPS \
    --model $MODEL \
    --tensor-parallel $TENSOR_PARALLEL \
    --quantization fp8 \
    --tax-year-length $TAX_YEAR_LENGTH \
    --scenario bounded \
    --batch-size $BATCH_SIZE \
    --seed $SEED \
    --output $OUTPUT_PATH

echo ""
echo "=============================================="
echo "Experiment completed!"
echo "Output: $OUTPUT_PATH"
echo "End time: $(date)"
echo "=============================================="
