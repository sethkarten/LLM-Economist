#!/bin/bash
#SBATCH --job-name=B_scale
#SBATCH --output=logs/B_scale_%j.out
#SBATCH --error=logs/B_scale_%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# B-Series: Scale tests
# Tests different agent counts to show scalability

source experiments/jobs/base_job.sh

# Configuration (set via environment)
NUM_AGENTS=${NUM_AGENTS:-5000}
MAX_STEPS=${MAX_STEPS:-1000}
MODEL=${MODEL:-"qwen3-30b-a3b"}
SEED=${SEED:-42}

OUTPUT_DIR="results/B_scale_${NUM_AGENTS}agents_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "Running B-Series Scale Test"
echo "Agents: $NUM_AGENTS, Steps: $MAX_STEPS"
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"

python -m llm_economist.main_async \
    --num-agents $NUM_AGENTS \
    --max-timesteps $MAX_STEPS \
    --model $MODEL \
    --quantization awq \
    --scenario bounded \
    --seed $SEED \
    --output "$OUTPUT_DIR/results.json" \
    --collect-trajectories "$OUTPUT_DIR/trajectories/"

echo ""
echo "=============================================="
echo "Scale Test Complete"
echo "Agents: $NUM_AGENTS"
echo "Results saved to: $OUTPUT_DIR"
echo "End time: $(date)"
echo "=============================================="
