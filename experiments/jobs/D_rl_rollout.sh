#!/bin/bash
#SBATCH --job-name=D_rollout
#SBATCH --output=logs/D_rollout_%j.out
#SBATCH --error=logs/D_rollout_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# D-Series: RL Rollout Collection Worker
# Runs as part of distributed RL training

source experiments/jobs/base_job.sh

# Configuration
WORKER_ID=${WORKER_ID:-"worker_$SLURM_JOB_ID"}
SHARED_DIR=${SHARED_DIR:-"/scratch/shared/llm_economist_rl"}
PLANNER_MODEL=${PLANNER_MODEL:-"Qwen/Qwen3-4B-Instruct"}
WORKER_MODEL=${WORKER_MODEL:-"Qwen/Qwen3-30B-A3B-Instruct"}
NUM_AGENTS=${NUM_AGENTS:-500}

echo "Starting RL Rollout Worker"
echo "Worker ID: $WORKER_ID"
echo "Shared Dir: $SHARED_DIR"
echo "Planner Model: $PLANNER_MODEL"
echo "Worker Model: $WORKER_MODEL"
echo "Agents: $NUM_AGENTS"

# Create shared directory if needed
mkdir -p $SHARED_DIR

# Start worker
python -m llm_economist.training.distributed \
    --mode worker \
    --worker-id $WORKER_ID \
    --shared-dir $SHARED_DIR \
    --planner-model $PLANNER_MODEL \
    --worker-model $WORKER_MODEL \
    --num-agents $NUM_AGENTS

echo ""
echo "=============================================="
echo "Worker $WORKER_ID finished"
echo "End time: $(date)"
echo "=============================================="
