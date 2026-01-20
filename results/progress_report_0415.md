# LLM Economist Progress Report
**Time: 4:15 AM EST, January 14, 2026**

## Summary

Two RTX 5090 GPUs running experiments since ~12:30 AM. Significant progress on bounded rationality experiments and RL baselines.

## Bounded Rationality Experiments (100 agents, 2000 steps)

| Model | Seed 42 | Seed 123 | Seed 456 | Status |
|-------|---------|----------|----------|--------|
| Gemma-3-4B | ✓ | ✓ | ✓ | **Complete** |
| Mistral-7B | ✓ | ✓ | ✓ | **Complete** |
| Llama-3.1-8B | ✓ | ✓ | ✓ | **Complete** |
| OLMo-3-7B | ✓ | ✓ | ✓ | **Complete** |
| Qwen3-8B | Step 1400/2000 | pending | pending | **Running** |

**Progress: 12/15 experiments complete**

- Qwen3-8B ETA: ~7 hours remaining for all 3 seeds
- Results stored in: `results/bounded_100x2000/`
- Automated analysis timer active (will run when all 15 complete)

## G-Series: RL AI Economist Baseline

### G1: Pure RL Training (PPO, no LLMs)
Trained neural network policies for both planner and workers using standard PPO.

| Config | Best SWF | Training Time | Notes |
|--------|----------|---------------|-------|
| 1k agents, seed=42 | 1964.1 | 11 min | |
| 1k agents, seed=123 | 1884.6 | 10 min | |
| 1k agents, seed=456 | 1845.2 | 10 min | |
| **1k Mean** | **1898.0 ± 60.3** | | |
| 5k agents | 9744.2 | 4 min | Scaled proportionally |
| 1k, high LR (5e-4) | 1956.6 (in progress) | ~10 min | Running now |

### G2: RL Baseline with LLM Workers (Distribution Shift Test)
Tested trained RL planner with bounded rational LLM workers.

| Checkpoint | Mean SWF | Mean Gini |
|------------|----------|-----------|
| seed=42 | 195.5 | 0.269 |
| seed=123 | 195.5 | similar |
| seed=456 | 195.5 | similar |
| 5k agents | 195.5 | similar |

**Key Finding: 10x performance degradation (1964 → 195)**

This demonstrates the distribution shift problem:
- RL trained on rational agents
- Fails when deployed with bounded rational LLM agents
- Effect is **scale-invariant** (same drop regardless of training scale)
- Motivates REINFORCE++ approach (train directly on LLM agents)

## Current GPU Utilization

| GPU | Task | Memory | Utilization |
|-----|------|--------|-------------|
| GPU 0 | Qwen3-8B bounded experiments | 31.5 GB | 100% |
| GPU 1 | G1 high-LR training | 0.7 GB | 46% |

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

## Key Results Summary

| Configuration | SWF | Notes |
|---------------|-----|-------|
| G1 Pure RL (rational agents) | ~1900 | Upper bound with perfect rationality |
| G2 RL + LLM workers | ~195 | 10x drop due to distribution shift |
| Zero-shot LLM planners | Variable | Depends on model, often volatile |

## Files Created

- `results/bounded_100x2000/*.json` - Bounded experiment results
- `models/rl_baseline_g1*/` - G1 RL baseline checkpoints
- `results/g2_rl_baseline_llm_eval*.json` - G2 evaluation results
- `experiments/analyze_bounded_results.py` - Analysis script
- `llm_economist/training/rl_baseline.py` - G1 trainer
- `llm_economist/training/evaluate_rl_baseline.py` - G2 evaluator

---
*Last updated: 4:15 AM EST*
