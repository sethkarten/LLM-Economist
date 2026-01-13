# LLM Economist Experiment Plan

## Hardware Setup

### Local Machine: RTX 5090 (Development & Validation)
- **GPU**: RTX 5090 32GB (Blackwell architecture, SM 100)
- **vLLM Config**: TRITON_ATTN backend, enforce_eager=True (FLASH_ATTN incompatible with Blackwell)
- Quick iteration on code changes
- Small-scale validation (100 agents, 2000 steps)
- 5 models × 3 seeds experiments

#### Measured Local Throughput (RTX 5090)
| Model | VRAM | Quantization | Throughput | Notes |
|-------|------|--------------|------------|-------|
| Gemma-3-4B | ~8.6GB | BF16 (text-only) | **92.0 req/s** | FASTEST - requires `limit_mm_per_prompt={'image': 0}` |
| Mistral-7B-v0.3 | ~4GB | AWQ | 67.1 req/s | Good balance |
| Llama-3.1-8B | ~4GB | AWQ | 65.9 req/s | Meta's flagship |
| Qwen3-8B | ~4GB | AWQ | 60.5 req/s | Strong reasoning |
| OLMo3-7B | ~7GB | FP8 | 31.5 req/s | Fully open weights & data |

### H200 Cluster (AI Lab - Flagship Experiments)
- **Partition**: `ailab`
- **18 nodes × 8 GPUs** (H200 PCIe, 141GB each)
- **64 CPU cores, 1.5TB RAM per node**
- **FP8 native support**
- Use for: 1M agent simulations, RL planner training

---

## Local Experiments (RTX 5090)

### Bounded Rationality: 5 Models × 3 Seeds
Run with `experiments/run_bounded_experiments.py`

| Model | Config | Quantization | Est. Time/Seed | Total (3 seeds) |
|-------|--------|--------------|----------------|-----------------|
| gemma3-4b | google/gemma-3-4b-it | BF16 text-only | 43.5 min | 2.2 hours |
| mistral-7b-v0.3 | mistralai/Mistral-7B-Instruct-v0.3 | AWQ | 59.6 min | 3.0 hours |
| llama-3.1-8b | meta-llama/Llama-3.1-8B-Instruct | AWQ | 60.7 min | 3.0 hours |
| qwen3-8b | Qwen/Qwen3-8B-Instruct | AWQ | 66.1 min | 3.3 hours |
| olmo3-7b | allenai/OLMo-3-7B-Instruct | FP8 | 127.0 min | 6.4 hours |
| **TOTAL** | | | | **~17.8 hours** |

**Configuration:**
- Agents: 100
- Timesteps: 2000
- Tax year length: 128 steps (~15 planner updates per run)
- Seeds: 42, 123, 456

**Commands:**
```bash
# List experiments and status
python experiments/run_bounded_experiments.py --list

# Run all experiments
python experiments/run_bounded_experiments.py --all

# Run specific model
python experiments/run_bounded_experiments.py --model gemma3-4b
```

---

## H200 Cluster Experiments

### E-Series: 1 Million Agent Experiments
Run with `experiments/jobs/launch_million_agent.sh`

| Model | Agents | Steps | TP | Est. Time/Seed |
|-------|--------|-------|-----|----------------|
| gemma3-4b | 1,000,000 | 2000 | 4 | ~24h |
| mistral-7b-v0.3 | 1,000,000 | 2000 | 4 | ~36h |
| llama-3.1-8b | 1,000,000 | 2000 | 4 | ~36h |
| qwen3-8b | 1,000,000 | 2000 | 4 | ~40h |
| olmo3-7b | 1,000,000 | 2000 | 4 | ~72h |

**Slurm Jobs:**
```bash
# Single job
sbatch experiments/jobs/E_million_agent.sh

# All 5 models × 3 seeds (15 jobs)
./experiments/jobs/launch_million_agent.sh

# Specific model
./experiments/jobs/launch_million_agent.sh gemma3-4b
```

### F-Series: RL Planner Training (REINFORCE++)
Run with `experiments/jobs/launch_rl_training_h200.sh`

**Architecture:**
- Coordinator: 1 H200 GPU (policy updates, gradient aggregation)
- Rollout Workers: N H200 GPUs (simulation, trajectory collection)

**Training Config:**
```yaml
planner_model: Qwen/Qwen3-4B-Instruct  # Model to finetune
worker_model: google/gemma-3-4b-it     # Fast model for agents
num_agents: 1000                        # Agents per rollout
max_timesteps: 500                      # Steps per rollout
batch_size: 64                          # Rollouts per gradient update
num_iterations: 100                     # Training iterations
learning_rate: 1e-5
```

**Launch Commands:**
```bash
# 8 rollout workers + 1 coordinator (9 H200 GPUs total)
./experiments/jobs/launch_rl_training_h200.sh 8

# 16 rollout workers (faster collection)
./experiments/jobs/launch_rl_training_h200.sh 16
```

---

## RL Finetuning Details

### Algorithm: REINFORCE++ with Group Relative Baseline

```
For each training iteration:
    1. Sample N rollouts in parallel across H200 workers
    2. Each rollout: planner sets tax policy → workers respond → compute SWF
    3. Compute advantages using group-relative baseline:
       A_i = R_i - mean(R_group)
    4. Update policy: θ += α * ∇_θ log π(a|s) * A
    5. KL penalty to prevent drift from base model
```

### Reward Signal
```python
reward = (next_swf - prev_swf) / abs(prev_swf)  # Normalized SWF improvement
# Bonus terms:
reward += 0.1 * max(0, -gini_change)           # Reward inequality reduction
reward -= 0.1 * max(0, -labor_change/total)    # Penalize labor reduction
```

### Distributed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Coordinator (H200 #0)                        │
│  - Manages rollout assignments                                   │
│  - Aggregates (state, action, reward) tuples                    │
│  - Computes policy gradients                                    │
│  - Checkpoints model weights                                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Shared filesystem
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   ┌────▼────┐      ┌────▼────┐      ┌────▼────┐
   │ H200 #1 │      │ H200 #2 │      │ H200 #N │
   │         │      │         │      │         │
   │ vLLM:   │      │ vLLM:   │      │ vLLM:   │
   │ Workers │      │ Workers │      │ Workers │
   │(Gemma4B)│      │(Gemma4B)│      │(Gemma4B)│
   │         │      │         │      │         │
   │ Planner │      │ Planner │      │ Planner │
   │ (Qwen4B)│      │ (Qwen4B)│      │ (Qwen4B)│
   └─────────┘      └─────────┘      └─────────┘
```

---

## Expected Outputs

### Tables for Paper
1. **Model Comparison**: SWF convergence across 5 models (local 100 agents)
2. **Scale Results**: 100 → 1M agents performance (H200 cluster)
3. **RL Finetuning**: Base 4B vs Finetuned 4B planner
4. **Scenario Comparison**: Rational vs Bounded vs Democratic

### Figures for Paper
1. SWF convergence curves (all 5 models)
2. Scale vs final SWF (100 to 1M agents)
3. RL training curve (reward over iterations)
4. Policy comparison (tax rates from base vs finetuned)
5. Income distribution before/after optimization

---

## Job File Reference

| File | Purpose | Hardware |
|------|---------|----------|
| `run_bounded_experiments.py` | Local 5×3 experiments | RTX 5090 |
| `E_million_agent.sh` | 1M agent single job | H200 (8 GPUs) |
| `launch_million_agent.sh` | Submit all 1M experiments | H200 cluster |
| `F_rl_coordinator_h200.sh` | RL coordinator job | H200 (1 GPU) |
| `F_rl_rollout_h200.sh` | RL rollout worker job | H200 (1 GPU) |
| `launch_rl_training_h200.sh` | Launch distributed RL | H200 cluster |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| H200 job fails mid-run | Checkpoint every 100 steps, resume capability |
| RL training unstable | Start with small LR, add KL penalty |
| Reward hacking | Multiple reward components, human inspection |
| OOM on H200 | Use FP8, limit batch size |
| Planner returns bad format | Robust parsing (handles '10.0%', strings, etc.) |
