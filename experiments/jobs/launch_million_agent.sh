#!/bin/bash
# Launch all 1M agent experiments (5 models × 3 seeds = 15 jobs)
#
# Usage:
#   ./experiments/jobs/launch_million_agent.sh          # Launch all
#   ./experiments/jobs/launch_million_agent.sh gemma3-4b  # Launch specific model

set -e

# Models to run (same as local experiments)
MODELS=("gemma3-4b" "mistral-7b-v0.3" "llama-3.1-8b" "qwen3-8b" "olmo3-7b")
SEEDS=(42 123 456)

# Parse args
FILTER_MODEL=${1:-""}

echo "=============================================="
echo "Launching 1M Agent Experiments on H200 Cluster"
echo "=============================================="
echo ""

# Create directories
mkdir -p logs
mkdir -p results/million_agent

# Track job IDs
JOB_IDS=()
JOB_COUNT=0

for MODEL in "${MODELS[@]}"; do
    # Skip if filtering
    if [[ -n "$FILTER_MODEL" && "$MODEL" != "$FILTER_MODEL" ]]; then
        continue
    fi

    for SEED in "${SEEDS[@]}"; do
        OUTPUT_FILE="results/million_agent/${MODEL}_seed${SEED}_1000000agents.json"

        # Skip if already completed
        if [[ -f "$OUTPUT_FILE" ]]; then
            echo "Skipping $MODEL seed=$SEED (already completed)"
            continue
        fi

        echo "Submitting: $MODEL seed=$SEED"

        JOB_ID=$(sbatch \
            --export=ALL,MODEL=$MODEL,SEED=$SEED \
            experiments/jobs/E_million_agent.sh | awk '{print $4}')

        JOB_IDS+=($JOB_ID)
        JOB_COUNT=$((JOB_COUNT + 1))

        echo "  Job ID: $JOB_ID"

        # Small delay between submissions
        sleep 2
    done
done

echo ""
echo "=============================================="
echo "Submitted $JOB_COUNT jobs"
echo "=============================================="
echo ""
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 30 'squeue -u \$USER'"
echo ""
echo "Check logs:"
echo "  tail -f logs/llm_econ_1M_<job_id>.out"
echo ""
echo "Results will be saved to:"
echo "  results/million_agent/"
