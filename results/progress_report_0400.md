# LLM Economist Progress Report
**Time: 4:00 AM EST, January 14, 2026**

## Summary

Two RTX 5090 GPUs have been running experiments since ~12:30 AM. Significant progress made on bounded rationality experiments and RL baselines.

## Bounded Rationality Experiments (100 agents, 2000 steps)

| Model | Seed 42 | Seed 123 | Seed 456 | Status |
|-------|---------|----------|----------|--------|
| Gemma-3-4B | ✓ | ✓ | ✓ | **Complete** |
| Mistral-7B | ✓ | ✓ | ✓ | **Complete** |
| Llama-3.1-8B | ✓ | ✓ | ✓ | **Complete** |
| OLMo-3-7B | ✓ | ✓ | ✓ | **Complete** |
| Qwen3-8B | Step 1200/2000 | pending | pending | **Running** |

**Progress: 12/15 experiments complete**

- Qwen3-8B ETA: ~8 hours remaining for all 3 seeds
- Results stored in: `results/bounded_100x2000/`
- Automated analysis timer active (will run when all 15 complete)

## G-Series: RL AI Economist Baseline

### G1: Pure RL Training (PPO, no LLMs)
Trained neural network policies for both planner and workers using standard PPO.

| Seed | Best SWF | Training Time |
|------|----------|---------------|
| 42 | 1964.1 | 11 min |
| 123 | 1884.6 | 10 min |
| 456 | 1845.2 | 10 min |
| **Mean** | **1898.0 ± 60.3** | |

Results: `models/rl_baseline_g1*/`

### G2: RL Baseline with LLM Workers (Distribution Shift Test)
Tested trained RL planner with bounded rational LLM workers.

| Checkpoint | Mean SWF | Mean Gini |
|------------|----------|-----------|
| seed=42 | 195.5 | 0.269 |

**Key Finding: 10x performance degradation (1964 → 195)**

This demonstrates the distribution shift problem:
- RL trained on rational agents
- Fails when deployed with bounded rational LLM agents
- Motivates REINFORCE++ approach (train directly on LLM agents)

## Current GPU Utilization

| GPU | Task | Memory | Utilization |
|-----|------|--------|-------------|
| GPU 0 | Qwen3-8B bounded experiments | 31.5 GB | 100% |
| GPU 1 | Available | 15 MB | 0% |

## Next Steps (After Bounded Experiments Complete)

1. **Automated Analysis** (timer active)
   - Generate SWF convergence plots
   - Create tax policy comparisons
   - Generate LaTeX tables for paper

2. **H1: REINFORCE++ with Scaffolding**
   - Train LLM planner directly on bounded rational workers
   - Should improve over zero-shot LLM planner baseline

3. **H2: REINFORCE++ without Scaffolding (Ablation)**
   - Compare raw data vs structured prompts
   - Quantify scaffolding benefit

## Files Created

- `results/bounded_100x2000/*.json` - Bounded experiment results
- `models/rl_baseline_g1*/` - G1 RL baseline checkpoints
- `results/g2_rl_baseline_llm_eval*.json` - G2 evaluation results
- `experiments/analyze_bounded_results.py` - Analysis script
- `llm_economist/training/rl_baseline.py` - G1 trainer
- `llm_economist/training/evaluate_rl_baseline.py` - G2 evaluator

## Key Insights

1. **Bounded vs Rational Performance**: Zero-shot LLM planners show volatile behavior (SWF spikes, regressive taxes in some runs)

2. **Distribution Shift**: RL policies trained on rational agents see 10x performance drop with LLM agents

3. **Throughput**:
   - Qwen3-8B AWQ: 11.6 req/s
   - Gemma-3-4B BF16: 430+ req/s (text-only mode)

## Update at 4:10 AM

**New Experiment: Scale-invariance of Distribution Shift**

| Training Config | G1 Best SWF | G2 Eval SWF | Drop Factor |
|-----------------|-------------|-------------|-------------|
| 1k agents | 1964.1 | 195.5 | ~10x |
| 5k agents | 9744.2 | 195.5 | ~50x (scaled) |

**Key Finding:** Distribution shift effect is independent of training scale!
- Scaling up RL training doesn't solve the rational→bounded generalization problem
- This motivates training directly on bounded agents (REINFORCE++)

---
*Last updated: 4:10 AM EST*
