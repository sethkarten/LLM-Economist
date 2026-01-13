#!/bin/bash
#SBATCH --job-name=A1_model_comp
#SBATCH --output=logs/A1_%j.out
#SBATCH --error=logs/A1_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# A1: Model comparison experiment
# Tests: qwen3-30b-a3b, olmo3-32b, nemotron-30b, gemma3-27b
# Config: 1000 agents x 3000 steps x 3 seeds

source experiments/jobs/base_job.sh

# Configuration
NUM_AGENTS=1000
MAX_STEPS=3000
SEEDS="42 43 44"
OUTPUT_DIR="results/A1_model_comparison_$(date +%Y%m%d_%H%M%S)"

mkdir -p $OUTPUT_DIR

# Model to test (set via environment or default)
MODEL=${MODEL:-"qwen3-30b-a3b"}

echo "Running A1 Model Comparison"
echo "Model: $MODEL"
echo "Agents: $NUM_AGENTS, Steps: $MAX_STEPS"
echo "Output: $OUTPUT_DIR"

for SEED in $SEEDS; do
    echo ""
    echo "=== Running with seed $SEED ==="

    python -m llm_economist.main_async \
        --num-agents $NUM_AGENTS \
        --max-timesteps $MAX_STEPS \
        --model $MODEL \
        --quantization awq \
        --scenario bounded \
        --seed $SEED \
        --output "$OUTPUT_DIR/${MODEL}_seed${SEED}.json" \
        --collect-trajectories "$OUTPUT_DIR/trajectories/"

    echo "Seed $SEED complete."
done

echo ""
echo "=============================================="
echo "A1 Model Comparison Complete"
echo "Results saved to: $OUTPUT_DIR"
echo "End time: $(date)"
echo "=============================================="
