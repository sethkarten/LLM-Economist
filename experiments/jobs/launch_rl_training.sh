#!/bin/bash
# Launch distributed RL training with coordinator + workers
#
# Usage:
#   ./experiments/jobs/launch_rl_training.sh [num_workers]
#
# Example:
#   ./experiments/jobs/launch_rl_training.sh 8

set -e

NUM_WORKERS=${1:-8}
SHARED_DIR=${SHARED_DIR:-"/scratch/shared/llm_economist_rl_$(date +%Y%m%d_%H%M%S)"}

echo "=============================================="
echo "Launching Distributed RL Training"
echo "=============================================="
echo "Workers: $NUM_WORKERS"
echo "Shared Dir: $SHARED_DIR"
echo ""

# Create shared directory
mkdir -p $SHARED_DIR
mkdir -p logs

# Launch coordinator first
echo "Launching coordinator..."
COORD_JOB=$(sbatch \
    --export=ALL,NUM_WORKERS=$NUM_WORKERS,SHARED_DIR=$SHARED_DIR \
    experiments/jobs/D_rl_coordinator.sh | awk '{print $4}')
echo "  Coordinator job: $COORD_JOB"

# Wait a bit for coordinator to initialize
sleep 10

# Launch workers
echo ""
echo "Launching $NUM_WORKERS workers..."
WORKER_JOBS=()
for i in $(seq 1 $NUM_WORKERS); do
    WORKER_ID="worker_$i"
    JOB_ID=$(sbatch \
        --export=ALL,WORKER_ID=$WORKER_ID,SHARED_DIR=$SHARED_DIR \
        --dependency=after:$COORD_JOB \
        experiments/jobs/D_rl_rollout.sh | awk '{print $4}')
    WORKER_JOBS+=($JOB_ID)
    echo "  Worker $i job: $JOB_ID"
done

echo ""
echo "=============================================="
echo "All jobs submitted!"
echo "=============================================="
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Check coordinator logs:"
echo "  tail -f logs/D_coord_${COORD_JOB}.out"
echo ""
echo "Shared directory:"
echo "  $SHARED_DIR"
echo ""
echo "Job IDs:"
echo "  Coordinator: $COORD_JOB"
echo "  Workers: ${WORKER_JOBS[*]}"
