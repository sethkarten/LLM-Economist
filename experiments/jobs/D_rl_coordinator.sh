#!/bin/bash
#SBATCH --job-name=D_coord
#SBATCH --output=logs/D_coord_%j.out
#SBATCH --error=logs/D_coord_%j.err
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# D-Series: RL Training Coordinator
# Manages distributed RL training across multiple B200 workers

source experiments/jobs/base_job.sh

# Configuration
NUM_WORKERS=${NUM_WORKERS:-8}
SHARED_DIR=${SHARED_DIR:-"/scratch/shared/llm_economist_rl"}
OUTPUT_DIR=${OUTPUT_DIR:-"models/planner_rl_$(date +%Y%m%d_%H%M%S)"}
PLANNER_MODEL=${PLANNER_MODEL:-"Qwen/Qwen3-4B-Instruct"}
WORKER_MODEL=${WORKER_MODEL:-"Qwen/Qwen3-30B-A3B-Instruct"}
NUM_AGENTS=${NUM_AGENTS:-500}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_ITERATIONS=${NUM_ITERATIONS:-100}
LEARNING_RATE=${LEARNING_RATE:-1e-5}

echo "Starting RL Training Coordinator"
echo "Expected Workers: $NUM_WORKERS"
echo "Shared Dir: $SHARED_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "Planner Model: $PLANNER_MODEL"
echo "Batch Size: $BATCH_SIZE"
echo "Iterations: $NUM_ITERATIONS"

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
    --worker-model $WORKER_MODEL \
    --num-agents $NUM_AGENTS \
    --batch-size $BATCH_SIZE \
    --num-iterations $NUM_ITERATIONS \
    --lr $LEARNING_RATE

echo ""
echo "=============================================="
echo "RL Training Complete"
echo "Model saved to: $OUTPUT_DIR"
echo "End time: $(date)"
echo "=============================================="
