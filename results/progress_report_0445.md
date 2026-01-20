# LLM Economist Progress Report
**Time: 4:45 AM EST, January 14, 2026**

## Summary

Two RTX 5090 GPUs running experiments. Both GPUs saturated and progressing well.

## Bounded Rationality Experiments (100 agents, 2000 steps)

| Model | Seed 42 | Seed 123 | Seed 456 | Status |
|-------|---------|----------|----------|--------|
| Gemma-3-4B | ✓ | ✓ | ✓ | **Complete** |
| Mistral-7B | ✓ | ✓ | ✓ | **Complete** |
| Llama-3.1-8B | ✓ | ✓ | ✓ | **Complete** |
| OLMo-3-7B | ✓ | ✓ | ✓ | **Complete** |
| Qwen3-8B | Step 1600/2000 | pending | pending | **Running** |

**Progress: 12/15 experiments complete**

- Qwen3-8B seed=42 ETA: ~60 minutes
- Then seeds 123 and 456: ~3 hours each

## G-Series: RL AI Economist Baseline

### G1: Pure RL Training Summary

| Config | Best SWF | Notes |
|--------|----------|-------|
| 1k agents, seed=42 | 1964.1 | |
| 1k agents, seed=123 | 1884.6 | |
| 1k agents, seed=456 | 1845.2 | |
| **1k Mean** | **1898.0 ± 60.3** | |
| 5k agents | 9744.2 | Scales proportionally |
| high LR (5e-4) | 1960.5 | |
| low LR (1e-4) | 1959.9 | |
| very low LR (5e-5) | 1959.9 | |
| 2000 agents | 3891.8 | Scales proportionally |
| 500k steps (running) | In progress | |

**Key Finding**: SWF converges to ~1960 regardless of learning rate (within reasonable range). Network is hitting optimization ceiling around 1960 SWF.

### G2: Distribution Shift Test

| Checkpoint | Mean SWF | Notes |
|------------|----------|-------|
| All G1 variants | ~195.5 | **10x drop** |

## Current GPU Utilization

| GPU | Task | Status |
|-----|------|--------|
| GPU 0 | Qwen3-8B bounded | Step 1600/2000 |
| GPU 1 | G1 job queue | Job 4/5 (500k steps) |

## Job Queue Progress

| Job | Config | Best SWF | Status |
|-----|--------|----------|--------|
| 1 | lr=1e-4 | 1959.9 | ✓ |
| 2 | lr=5e-5 | 1959.9 | ✓ |
| 3 | 2000 agents | 3891.8 | ✓ |
| 4 | 500k steps | In progress | Running |
| 5 | high-LR seed=123 | Pending | Queued |

## Key Insights

1. **G1 Optimization Ceiling**: SWF converges to ~1960 regardless of hyperparameters
   - This appears to be the optimal policy for rational agents in this environment

2. **Linear Scaling**: SWF scales linearly with number of agents
   - 1k agents: ~1960
   - 2k agents: ~3890
   - 5k agents: ~9740

3. **Distribution Shift**: Consistent 10x performance drop (1960 → 195) with LLM workers
   - Independent of training configuration
   - Motivates REINFORCE++ approach

## Next Actions

1. Complete Qwen3-8B bounded experiments (~5 hours)
2. Complete G1 job queue (~15 minutes)
3. Run automated analysis when all 15 bounded experiments done
4. Start H1: REINFORCE++ with scaffolding

---
*Last updated: 4:45 AM EST*
