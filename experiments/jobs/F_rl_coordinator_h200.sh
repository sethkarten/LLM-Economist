#!/bin/bash
#SBATCH --job-name=rl_coord
#SBATCH --output=logs/rl_coord_%j.out
#SBATCH --error=logs/rl_coord_%j.err
#SBATCH --partition=ailab
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

# RL Training Coordinator for H200 Cluster
# Manages distributed REINFORCE++ training across H200 workers
#
# The coordinator:
# - Loads and updates the planner model
# - Distributes policy weights to workers
# - Collects rollouts and computes policy gradients
# - Saves checkpoints

set -e

# Print job info
echo "=============================================="
echo "RL Training Coordinator (H200)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
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
export VLLM_USE_FP8=1  # Native FP8 on H200

# Create directories
mkdir -p logs

nvidia-smi

# Configuration (can be overridden via --export)
NUM_WORKERS=${NUM_WORKERS:-8}
SHARED_DIR=${SHARED_DIR:-"/scratch/$USER/llm_economist_rl"}
OUTPUT_DIR=${OUTPUT_DIR:-"models/planner_rl_$(date +%Y%m%d_%H%M%S)"}
PLANNER_MODEL=${PLANNER_MODEL:-"Qwen/Qwen3-4B-Instruct"}
NUM_AGENTS=${NUM_AGENTS:-1000}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_ITERATIONS=${NUM_ITERATIONS:-100}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
MAX_TIMESTEPS=${MAX_TIMESTEPS:-500}

echo ""
echo "Configuration:"
echo "  Expected Workers: $NUM_WORKERS"
echo "  Shared Dir: $SHARED_DIR"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Planner Model: $PLANNER_MODEL"
echo "  Agents per rollout: $NUM_AGENTS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Iterations: $NUM_ITERATIONS"
echo "  Learning Rate: $LEARNING_RATE"
echo ""

# Create directories
mkdir -p $SHARED_DIR
mkdir -p $OUTPUT_DIR

# Start coordinator
python -m llm_economist.training.distributed \
    --mode coordinator \
    --num-workers $NUM_WORKERS \
    --shared-dir $SHARED_DIR \
    --output $OUTPUT_DIR \
    --planner-model $PLANNER_MODEL \
    --num-agents $NUM_AGENTS \
    --batch-size $BATCH_SIZE \
    --num-iterations $NUM_ITERATIONS \
    --max-timesteps $MAX_TIMESTEPS \
    --lr $LEARNING_RATE \
    --fp8

echo ""
echo "=============================================="
echo "RL Training Complete"
echo "Model saved to: $OUTPUT_DIR"
echo "End time: $(date)"
echo "=============================================="
