#!/bin/bash
# Wrapper script for running REINFORCE++ on della-ailab compute nodes (OFFLINE)

set -e

# Configure offline mode
export HF_HOME="/scratch/gpfs/CHIJ/milkkarten/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR="/scratch/gpfs/CHIJ/milkkarten/LLM-Economist/wandb_offline"

# Create wandb offline directory
mkdir -p "$WANDB_DIR"

echo "=== Running in OFFLINE mode ==="
echo "HF_HOME: $HF_HOME"
echo "WANDB_MODE: $WANDB_MODE"
echo "Working directory: $(pwd)"
echo ""

# Activate pre-installed venv
source .venv/bin/activate

# Run experiment
python experiments/run_reinforce_h1h2.py "$@"
