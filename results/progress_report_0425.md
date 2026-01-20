# LLM Economist Progress Report
**Time: 4:25 AM EST, January 14, 2026**

## Summary

Two RTX 5090 GPUs running experiments since ~12:30 AM. Both GPUs currently saturated.

## Bounded Rationality Experiments (100 agents, 2000 steps)

| Model | Seed 42 | Seed 123 | Seed 456 | Status |
|-------|---------|----------|----------|--------|
| Gemma-3-4B | ✓ | ✓ | ✓ | **Complete** |
| Mistral-7B | ✓ | ✓ | ✓ | **Complete** |
| Llama-3.1-8B | ✓ | ✓ | ✓ | **Complete** |
| OLMo-3-7B | ✓ | ✓ | ✓ | **Complete** |
| Qwen3-8B | Step 1400/2000 | pending | pending | **Running** |

**Progress: 12/15 experiments complete**

- Qwen3-8B ETA: ~5.5 hours remaining for all 3 seeds
- Results stored in: `results/bounded_100x2000/`

## G-Series: RL AI Economist Baseline (Complete)

### G1: Pure RL Training (PPO, no LLMs)

| Config | Best SWF | Training Time | Notes |
|--------|----------|---------------|-------|
| 1k agents, seed=42 | 1964.1 | 11 min | |
| 1k agents, seed=123 | 1884.6 | 10 min | |
| 1k agents, seed=456 | 1845.2 | 10 min | |
| **1k Mean** | **1898.0 ± 60.3** | | |
| 5k agents | 9744.2 | 4 min | Scaled proportionally |
| 1k, high LR (5e-4) | 1960.5 | 10 min | **Just completed** |

### G1 Variants (Now Running on GPU 1)
Job queue started with 5 experiments:
1. `g1_lowlr` - lr=1e-4 (running)
2. `g1_verylowlr` - lr=5e-5
3. `g1_2k` - 2000 agents
4. `g1_longrun` - 500k steps, batch 4096
5. `g1_highlr_seed123` - seed=123 for high-LR

### G2: RL Baseline with LLM Workers (Distribution Shift Test)

| Checkpoint | Mean SWF | Mean Gini |
|------------|----------|-----------|
| All G1 variants | ~195.5 | ~0.269 |

**Key Finding: 10x performance degradation, scale-invariant**

## Current GPU Utilization

| GPU | Task | Memory | Utilization |
|-----|------|--------|-------------|
| GPU 0 | Qwen3-8B bounded experiments | 31.5 GB | 95% |
| GPU 1 | G1 variant job queue | Running | Active |

## Key Experimental Results

### Distribution Shift Discovery
- G1 (rational agents): SWF ~1900
- G2 (LLM workers): SWF ~195
- **10x performance drop** regardless of:
  - Training scale (1k vs 5k agents)
  - Training duration (200k vs 500k steps)
  - Learning rate variations

This strongly motivates REINFORCE++ training directly on bounded rational LLM agents.

## Next Steps

1. **Complete Qwen3-8B** (~5.5 hours)
   - Running seed=42 at Step 1400/2000
   - Then seeds 123 and 456

2. **Automated Analysis** (when all 15 complete)
   - Generate SWF convergence plots
   - Create tax policy comparisons
   - Generate LaTeX tables for paper

3. **H1: REINFORCE++ with Scaffolding**
   - After bounded experiments complete
   - Train LLM planner directly on bounded rational workers

4. **H2: REINFORCE++ without Scaffolding**
   - Ablation study

## Files Created This Session

- `models/rl_baseline_g1*/` - Various G1 checkpoints
- `results/g2_rl_baseline_llm_eval*.json` - G2 evaluations
- `experiments/gpu1_job_queue.py` - Automated job queue
- `llm_economist/training/rl_baseline.py` - G1 trainer
- `llm_economist/training/evaluate_rl_baseline.py` - G2 evaluator

---
*Last updated: 4:25 AM EST*
