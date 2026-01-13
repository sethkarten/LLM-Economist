#!/bin/bash
#SBATCH --job-name=rl_rollout
#SBATCH --output=logs/rl_rollout_%j.out
#SBATCH --error=logs/rl_rollout_%j.err
#SBATCH --partition=ailab
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

# RL Rollout Collection Worker for H200 Cluster
# Runs simulations to collect (state, action, reward) trajectories
#
# Each worker:
# - Receives policy weights from coordinator
# - Runs economic simulations with bounded rationality agents
# - Collects rollouts with SWF rewards
# - Sends trajectories back to coordinator

set -e

# Print job info
echo "=============================================="
echo "RL Rollout Worker (H200)"
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

# Configuration
WORKER_ID=${WORKER_ID:-"worker_$SLURM_JOB_ID"}
SHARED_DIR=${SHARED_DIR:-"/scratch/$USER/llm_economist_rl"}
WORKER_MODEL=${WORKER_MODEL:-"google/gemma-3-4b-it"}  # Fast model for workers
NUM_AGENTS=${NUM_AGENTS:-1000}
MAX_TIMESTEPS=${MAX_TIMESTEPS:-500}

echo ""
echo "Configuration:"
echo "  Worker ID: $WORKER_ID"
echo "  Shared Dir: $SHARED_DIR"
echo "  Worker Model: $WORKER_MODEL"
echo "  Agents: $NUM_AGENTS"
echo "  Max Timesteps: $MAX_TIMESTEPS"
echo ""

# Create shared directory if needed
mkdir -p $SHARED_DIR

# Start worker
python -m llm_economist.training.distributed \
    --mode worker \
    --worker-id $WORKER_ID \
    --shared-dir $SHARED_DIR \
    --worker-model $WORKER_MODEL \
    --num-agents $NUM_AGENTS \
    --max-timesteps $MAX_TIMESTEPS \
    --fp8

echo ""
echo "=============================================="
echo "Worker $WORKER_ID finished"
echo "End time: $(date)"
echo "=============================================="
