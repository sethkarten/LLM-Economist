#!/bin/bash
# Launch distributed RL training on H200 cluster
#
# Usage:
#   ./experiments/jobs/launch_rl_training_h200.sh [num_workers]
#
# Example:
#   ./experiments/jobs/launch_rl_training_h200.sh 8    # 8 rollout workers
#   ./experiments/jobs/launch_rl_training_h200.sh 16   # 16 rollout workers

set -e

NUM_WORKERS=${1:-8}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SHARED_DIR=${SHARED_DIR:-"/scratch/$USER/llm_economist_rl_$TIMESTAMP"}
OUTPUT_DIR=${OUTPUT_DIR:-"models/planner_rl_$TIMESTAMP"}

# Training configuration
PLANNER_MODEL=${PLANNER_MODEL:-"Qwen/Qwen3-4B-Instruct"}
WORKER_MODEL=${WORKER_MODEL:-"google/gemma-3-4b-it"}
NUM_AGENTS=${NUM_AGENTS:-1000}
MAX_TIMESTEPS=${MAX_TIMESTEPS:-500}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_ITERATIONS=${NUM_ITERATIONS:-100}
LEARNING_RATE=${LEARNING_RATE:-1e-5}

echo "=============================================="
echo "Launching Distributed RL Training on H200s"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Workers: $NUM_WORKERS"
echo "  Planner: $PLANNER_MODEL"
echo "  Worker Model: $WORKER_MODEL"
echo "  Agents per rollout: $NUM_AGENTS"
echo "  Timesteps per rollout: $MAX_TIMESTEPS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Iterations: $NUM_ITERATIONS"
echo "  Learning Rate: $LEARNING_RATE"
echo ""
echo "  Shared Dir: $SHARED_DIR"
echo "  Output Dir: $OUTPUT_DIR"
echo ""

# Create directories
mkdir -p $SHARED_DIR
mkdir -p $OUTPUT_DIR
mkdir -p logs

# Launch coordinator first
echo "Launching coordinator..."
COORD_JOB=$(sbatch \
    --export=ALL,NUM_WORKERS=$NUM_WORKERS,SHARED_DIR=$SHARED_DIR,OUTPUT_DIR=$OUTPUT_DIR,PLANNER_MODEL=$PLANNER_MODEL,NUM_AGENTS=$NUM_AGENTS,MAX_TIMESTEPS=$MAX_TIMESTEPS,BATCH_SIZE=$BATCH_SIZE,NUM_ITERATIONS=$NUM_ITERATIONS,LEARNING_RATE=$LEARNING_RATE \
    experiments/jobs/F_rl_coordinator_h200.sh | awk '{print $4}')
echo "  Coordinator job: $COORD_JOB"

# Wait for coordinator to initialize
echo "Waiting 30s for coordinator to initialize..."
sleep 30

# Launch workers
echo ""
echo "Launching $NUM_WORKERS rollout workers..."
WORKER_JOBS=()
for i in $(seq 1 $NUM_WORKERS); do
    WORKER_ID="worker_$i"
    JOB_ID=$(sbatch \
        --export=ALL,WORKER_ID=$WORKER_ID,SHARED_DIR=$SHARED_DIR,WORKER_MODEL=$WORKER_MODEL,NUM_AGENTS=$NUM_AGENTS,MAX_TIMESTEPS=$MAX_TIMESTEPS \
        --dependency=after:$COORD_JOB \
        experiments/jobs/F_rl_rollout_h200.sh | awk '{print $4}')
    WORKER_JOBS+=($JOB_ID)
    echo "  Worker $i: Job $JOB_ID"
    sleep 1
done

echo ""
echo "=============================================="
echo "All jobs submitted!"
echo "=============================================="
echo ""
echo "Total H200 GPUs requested: $((NUM_WORKERS + 1))"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER -p ailab"
echo "  watch -n 30 'squeue -u \$USER -p ailab'"
echo ""
echo "Check coordinator logs:"
echo "  tail -f logs/rl_coord_${COORD_JOB}.out"
echo ""
echo "Check worker logs:"
echo "  tail -f logs/rl_rollout_*.out"
echo ""
echo "Job IDs:"
echo "  Coordinator: $COORD_JOB"
echo "  Workers: ${WORKER_JOBS[*]}"
echo ""
echo "Output will be saved to:"
echo "  $OUTPUT_DIR"
