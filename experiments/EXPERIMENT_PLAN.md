# LLM Economist Experiment Plan

## Hardware Setup

### Local Machine: RTX 5090 (Development & Small Tests)
- **GPU**: RTX 5090 32GB (Blackwell architecture, SM 100)
- **vLLM Config**: TRITON_ATTN backend, enforce_eager=True (FLASH_ATTN incompatible)
- Quick iteration on code changes
- Small-scale validation (10-100 agents, 100-2000 steps)
- Debug RL training loops

#### Measured Local Throughput (RTX 5090)
| Model | VRAM | Quantization | Throughput | Notes |
|-------|------|--------------|------------|-------|
| unsloth/Qwen3-8B-bnb-4bit | ~6GB | BNB 4-bit | **10 req/s** | Recommended for local |
| unsloth/Qwen3-14B-bnb-4bit | ~8GB | BNB 4-bit | ~6 req/s | Higher quality |
| qwen3-30b-a3b | Too large | AWQ | N/A | Requires B200 |

### B200 Server (Flagship Experiments)
- **1x B200 per job** (80GB HBM3, ~2000 TFLOPS FP8)
- **Up to 16 parallel jobs**
- **24 hours max per job**
- Use for: large-scale simulations (1000+ agents), 30B+ models, RL training rollouts

#### Required Hardware per Model
| Model | Min VRAM | Recommended Hardware |
|-------|----------|---------------------|
| qwen3-30b-a3b (AWQ) | ~15GB | B200 (for batching) |
| olmo3-32b (AWQ) | ~16GB | B200 |
| nemotron-30b (AWQ) | ~16GB | B200 |
| gemma3-27b (AWQ) | ~14GB | B200 |
| unsloth-8b (BNB) | ~6GB | RTX 5090 local |

---

## Experiment Categories

### Phase 1: Local Validation (Days 1-2)
Small tests to validate infrastructure before B200 runs.

| Test | Agents | Steps | Purpose | Time |
|------|--------|-------|---------|------|
| L1: Smoke test | 10 | 100 | Verify async engine works | ~5 min |
| L2: RL loop test | 10 | 256 | Verify REINFORCE++ gradient flow | ~15 min |
| L3: Multi-seed | 20 | 500 | Verify reproducibility | ~30 min |
| L4: Persona test | 50 | 256 | Verify population-aligned personas | ~20 min |

### Phase 2: B200 Flagship Experiments (Days 3-7)

#### A-Series: Model Comparison
| Job | Experiment | Agents | Steps | Model | Seeds | Time Est |
|-----|------------|--------|-------|-------|-------|----------|
| A1 | Model comparison | 1000 | 3000 | qwen3-30b-a3b | 3 | ~8h |
| A2 | Model comparison | 1000 | 3000 | olmo3-32b | 3 | ~12h |
| A3 | Model comparison | 1000 | 3000 | nemotron-30b | 3 | ~10h |
| A4 | Model comparison | 1000 | 3000 | gemma3-27b | 3 | ~12h |

#### B-Series: Scale Tests
| Job | Experiment | Agents | Steps | Model | Time Est |
|-----|------------|--------|-------|-------|----------|
| B1 | Medium scale | 5000 | 1000 | qwen3-30b-a3b | ~8h |
| B2 | Large scale | 10000 | 500 | qwen3-30b-a3b | ~8h |
| B3 | Massive scale | 20000 | 250 | qwen3-30b-a3b | ~8h |
| B4 | Extreme scale | 50000 | 100 | qwen3-30b-a3b | ~8h |

#### C-Series: Scenario Comparison
| Job | Experiment | Agents | Steps | Scenario | Seeds | Time Est |
|-----|------------|--------|-------|----------|-------|----------|
| C1 | Rational | 1000 | 2000 | rational | 3 | ~6h |
| C2 | Bounded | 1000 | 2000 | bounded | 3 | ~6h |
| C3 | Democratic | 1000 | 2000 | democratic | 3 | ~6h |

#### D-Series: RL Planner Finetuning (MAIN CONTRIBUTION)
| Job | Experiment | Description | Time Est |
|-----|------------|-------------|----------|
| D1-D8 | Rollout collection | 8 parallel B200s collecting planner rollouts | ~12h |
| D9-D12 | RL training | REINFORCE++ training on collected data | ~8h |
| D13 | Evaluation | Compare finetuned vs base planner | ~4h |

#### E-Series: OG AI Economist Baseline Comparison
**Purpose**: Compare LLM-Economist against original AI Economist (Zheng et al. 2020)

Key differences from OG AI Economist:
1. **Utility function**: Our isoelastic utility is more realistic than OG's linear utility
2. **Sample complexity**: Pure RL (tabula rasa) needs orders of magnitude more samples
3. **Interpretability**: LLM agents provide reasoning; RL agents are black boxes
4. **Generalization**: LLMs leverage pretraining; RL must learn from scratch

| Job | Experiment | Agents | Steps | Method | Seeds | Time Est |
|-----|------------|--------|-------|--------|-------|----------|
| E1 | OG-RL baseline | 100 | 50000 | PPO (tabula rasa) | 3 | ~24h |
| E2 | OG-RL extended | 100 | 200000 | PPO (tabula rasa) | 3 | ~24h |
| E3 | LLM-Economist | 100 | 2000 | LLM workers + RL planner | 3 | ~6h |
| E4 | Hybrid | 100 | 10000 | RL workers + LLM planner | 3 | ~12h |

**Hypothesis**: LLM-Economist achieves comparable or better SWF with 10-100x fewer steps

---

## RL Finetuning Architecture

### Algorithm: REINFORCE++ with Group Relative Baseline

```
For each training iteration:
    1. Sample N rollouts in parallel across B200 jobs
    2. Each rollout: planner sets tax policy → workers respond → compute SWF
    3. Compute advantages using group-relative baseline:
       A_i = R_i - mean(R_group)
    4. Update policy: θ += α * ∇_θ log π(a|s) * A
    5. Optional: KL penalty to prevent drift from base model
```

### Reward Signal
```python
reward = (next_swf - prev_swf) / abs(prev_swf)  # Normalized SWF improvement
# Bonus terms (optional):
reward += 0.1 * max(0, -gini_change)  # Reward inequality reduction
reward -= 0.1 * max(0, -labor_change/total_labor)  # Penalize labor reduction
```

### Distributed Rollout Collection

```
┌─────────────────────────────────────────────────────────────────┐
│                     Coordinator (Local/Head Node)               │
│  - Manages rollout assignments                                  │
│  - Aggregates (state, action, reward) tuples                   │
│  - Computes policy gradients                                   │
│  - Checkpoints model weights                                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Redis/HTTP
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   ┌────▼────┐      ┌────▼────┐      ┌────▼────┐
   │ B200 #1 │      │ B200 #2 │      │ B200 #N │
   │         │      │         │      │         │
   │ vLLM:   │      │ vLLM:   │      │ vLLM:   │
   │ Workers │      │ Workers │      │ Workers │
   │ (30B)   │      │ (30B)   │      │ (30B)   │
   │         │      │         │      │         │
   │ Planner │      │ Planner │      │ Planner │
   │ (4B)    │      │ (4B)    │      │ (4B)    │
   └─────────┘      └─────────┘      └─────────┘
```

### Training Hyperparameters

```yaml
# REINFORCE++ Config
learning_rate: 1e-5
batch_size: 64  # rollouts per gradient update
rollouts_per_job: 8  # rollouts collected per B200 job
num_parallel_jobs: 8  # B200s collecting in parallel
total_rollouts: 512  # per training iteration
kl_coef: 0.1  # KL penalty coefficient
entropy_coef: 0.01  # Entropy bonus
max_grad_norm: 1.0  # Gradient clipping

# Environment Config
num_agents: 500  # workers per rollout
steps_per_rollout: 256  # ~2 tax years
tax_year_length: 128

# Model Config
planner_model: "qwen3-4b"  # Model to finetune
worker_model: "qwen3-30b-a3b"  # Fixed worker model
lora_r: 64
lora_alpha: 128
```

---

## Job Scheduling Strategy

### Batch 1: Model Comparison (4 jobs, ~12h)
```bash
# Submit all 4 model comparison jobs in parallel
sbatch jobs/A1_qwen3.sh
sbatch jobs/A2_olmo3.sh
sbatch jobs/A3_nemotron.sh
sbatch jobs/A4_gemma3.sh
```

### Batch 2: Scale Tests (4 jobs, ~8h)
```bash
# After Batch 1 completes
sbatch jobs/B1_5k.sh
sbatch jobs/B2_10k.sh
sbatch jobs/B3_20k.sh
sbatch jobs/B4_50k.sh
```

### Batch 3: RL Rollout Collection (8 jobs, ~12h)
```bash
# 8 parallel B200s collecting planner rollouts
for i in {1..8}; do
    sbatch jobs/D_rollout_$i.sh
done
```

### Batch 4: RL Training (4 jobs, ~8h)
```bash
# Train on collected rollouts with different seeds
for seed in 42 43 44 45; do
    sbatch jobs/D_train_$seed.sh
done
```

### Batch 5: Evaluation & Scenarios (5 jobs, ~6h)
```bash
sbatch jobs/D_eval.sh
sbatch jobs/C1_rational.sh
sbatch jobs/C2_bounded.sh
sbatch jobs/C3_democratic.sh
```

---

## Expected Outputs

### Tables for Paper
1. **Model Comparison**: SWF convergence across Qwen3/OLMo3/Nemotron/Gemma3
2. **Scale Results**: 1K → 50K agents performance
3. **RL Finetuning**: Base 4B vs Finetuned 4B vs 30B reference
4. **Scenario Comparison**: Rational vs Bounded vs Democratic

### Figures for Paper
1. SWF convergence curves (all models)
2. Scale vs final SWF
3. RL training curve (reward over iterations)
4. Policy comparison (tax rates from base vs finetuned)
5. Income distribution before/after optimization

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| B200 job fails mid-run | Checkpoint every 100 steps, resume capability |
| RL training unstable | Start with small LR, add KL penalty |
| Reward hacking | Multiple reward components, human inspection |
| OOM on B200 | Use AWQ int4, limit batch size |
| Scheduling delays | Submit jobs with dependencies, use --dependency flag |

---

## File Structure for Experiments

```
experiments/
├── EXPERIMENT_PLAN.md          # This file
├── jobs/                       # SLURM job scripts
│   ├── A1_qwen3.sh
│   ├── B1_5k.sh
│   ├── D_rollout_1.sh
│   └── ...
├── configs/                    # Experiment configs
│   ├── model_comparison.yaml
│   ├── scale_test.yaml
│   └── rl_training.yaml
├── results/                    # Output directory
│   ├── A1_qwen3_*/
│   ├── B1_5k_*/
│   └── D_rl_training_*/
└── analysis/                   # Analysis scripts
    ├── plot_convergence.py
    ├── compare_models.py
    └── generate_tables.py
```
