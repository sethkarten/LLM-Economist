# H3 REINFORCE++ Poor Performance Diagnosis

## Problem Statement
Current H3 experiments show only 0.13-0.17% SWF improvement (192.5 → 192.7) instead of expected 25% (192 → 240).

## Root Causes Identified

### 1. **Double-Centering of Advantages** (CRITICAL)
**Location:** `run_reinforce_h3_with_training_offline.py:727-733`

```python
# Step 1: Center advantages
baseline = rewards.mean()
advantages = rewards - baseline

# Step 2: Normalize (centers AGAIN!)
if advantages.std() > 1e-8:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

**Issue:** Subtracting `advantages.mean()` is redundant since advantages are already centered. This washes out the signal.

**Fix:** Use either raw advantages OR normalize without re-centering:
```python
# Option A: Raw advantages (simpler)
advantages = rewards - baseline

# Option B: Normalize without re-centering
advantages = (rewards - baseline) / (rewards.std() + 1e-8)
```

### 2. **Weak Worker Model** (MAJOR)
**Location:** `RLConfig.worker_model = "meta-llama/Llama-3.2-1B"`

**Issue:**
- Baseline SWF with Llama-3.2-1B workers: ~192.5
- ICRL with Gemma-3-4B workers: 207.5
- ICRL with OLMo-3-7B workers: 236.6

The baseline is artificially low because workers are using a weak 1B model. Even if the planner learns perfectly, it's limited by worker quality.

**Fix:** Use `google/gemma-2-2b-it` or `meta-llama/Llama-3.1-8B-Instruct` for workers.

### 3. **KL Penalty Too Strong** (MODERATE)
**Location:** `RLConfig.kl_coef = 0.05`, line 754

**Issue:**
```python
kl_loss = (log_ratio ** 2).mean()  # Simplified KL
total_loss = pg_loss + 0.05 * kl_loss - 0.01 * entropy_loss
```

With small reward magnitudes (~0.2), the KL term dominates and prevents exploration.

**Fix:** Reduce `kl_coef` to 0.01 or 0.005.

### 4. **Learning Rate Too Conservative** (MINOR)
**Location:** `RLConfig.learning_rate = 1e-5`

**Issue:** With LoRA, we can use higher learning rates safely.

**Fix:** Increase to `3e-5` or `5e-5`.

### 5. **Small Reward Magnitudes** (MODERATE)
**Current:** Rewards are absolute SWF differences: 0.1, 0.2, 0.3
**Issue:** After normalization, magnitude information is lost.

**Fix:** Scale rewards by expected improvement:
```python
# Expected: baseline ~192, target ~240, range = 48
reward_scaled = (final_swf - baseline_swf) / 48.0
```

## Recommended Fixes (Priority Order)

### 🔴 CRITICAL - Fix advantage computation
```python
# Before (lines 727-733)
baseline = rewards.mean()
advantages = rewards - baseline
if advantages.std() > 1e-8:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# After
baseline = rewards.mean()
advantages = rewards - baseline
# Normalize without re-centering
if rewards.std() > 1e-8:
    advantages = advantages / (rewards.std() + 1e-8)
```

### 🔴 CRITICAL - Use better worker model
```python
# Before
worker_model: str = "meta-llama/Llama-3.2-1B"

# After (choose one based on VRAM)
worker_model: str = "google/gemma-2-2b-it"  # 2B, fits easily
# OR
worker_model: str = "meta-llama/Llama-3.1-8B-Instruct"  # Best quality
```

### 🟡 HIGH - Reduce KL penalty
```python
# Before
kl_coef: float = 0.05

# After
kl_coef: float = 0.01
```

### 🟡 HIGH - Increase learning rate
```python
# Before
learning_rate: float = 1e-5

# After
learning_rate: float = 5e-5
```

### 🟢 MEDIUM - Scale rewards
```python
def compute_reward(final_swf: float, baseline_swf: float) -> float:
    """
    Compute reward with scaling for better learning.

    Expected improvement: 48 SWF (192 → 240)
    """
    raw_improvement = final_swf - baseline_swf
    # Scale to roughly [0, 1] range
    scaled_reward = raw_improvement / 48.0
    return scaled_reward
```

## Expected Results After Fixes

**Current:**
- Baseline SWF: 192.5 (Llama-3.2-1B workers)
- Best SWF: 192.7 (0.13% improvement)
- Reward range: 0.1-0.3

**After fixes:**
- Baseline SWF: ~207 (Gemma-2-2B workers, based on ICRL)
- Target SWF: ~230-240 (10-15% improvement over better baseline)
- Reward range: 0.5-1.0 (scaled)

## Testing Plan

1. **Quick test (local):** 10 iterations, 8 rollouts/iter
   - Verify: advantages have clear signal
   - Verify: rewards show increasing trend
   - Runtime: ~1 hour on RTX 5090

2. **Full test (della-gpu):** 100 iterations, 32 rollouts/iter
   - Target: SWF > 220 by iteration 50
   - Target: SWF > 235 by iteration 100
   - Runtime: ~8 hours on A6000

3. **Production run:** 1000 iterations, 32 rollouts/iter, 3 seeds
   - Only launch if test shows > 5% improvement
   - Runtime: 48 hours × 3 seeds
