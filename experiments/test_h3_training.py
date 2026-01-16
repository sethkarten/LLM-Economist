#!/usr/bin/env python3
"""
Quick test to verify H3 training script components work.
"""
import os
os.environ['VLLM_USE_V1'] = '0'

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        import torch.nn.functional as F
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_planner_policy():
    """Test PlannerPolicy can be instantiated and load model."""
    print("\nTesting PlannerPolicy...")
    try:
        from run_reinforce_h3_with_training import PlannerPolicy, RLConfig

        config = RLConfig(
            planner_model="google/gemma-3-4b-it",
            use_lora=True,
            lora_r=8,  # Small for testing
            lora_alpha=16,
        )

        policy = PlannerPolicy(
            model_name=config.planner_model,
            config=config,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        print("  Loading model with LoRA...")
        policy.load_model()

        print(f"  Model device: {policy.device}")
        print(f"  Trainable parameters: {sum(p.numel() for p in policy.model.parameters() if p.requires_grad):,}")

        print("✓ PlannerPolicy works")
        return True
    except Exception as e:
        print(f"✗ PlannerPolicy failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_action_sampling():
    """Test action sampling with log prob."""
    print("\nTesting action sampling...")
    try:
        from run_reinforce_h3_with_training import PlannerPolicy, RLConfig

        config = RLConfig(
            planner_model="google/gemma-3-4b-it",
            use_lora=True,
            lora_r=8,
            lora_alpha=16,
        )

        policy = PlannerPolicy(
            model_name=config.planner_model,
            config=config,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        policy.load_model()

        system_prompt = "You are a tax policy planner. Output tax policy as JSON."
        user_prompt = 'Set tax rates. Output: {"tax_rates": [0.1, 0.2, 0.3]}'

        print("  Sampling action...")
        tax_rates, log_prob = policy.sample_action(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,
            max_new_tokens=50,
        )

        print(f"  Tax rates: {tax_rates}")
        print(f"  Log prob: {log_prob:.3f}")

        if tax_rates is None:
            print("  Warning: Failed to parse tax rates (this is okay for test)")

        print("✓ Action sampling works")
        return True
    except Exception as e:
        print(f"✗ Action sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_train_step():
    """Test that train_step can be called."""
    print("\nTesting train_step...")
    try:
        from run_reinforce_h3_with_training import REINFORCEExperiment, RLConfig

        config = RLConfig(
            num_agents=10,  # Small for test
            num_iterations=2,
            rollouts_per_iter=4,
            use_lora=True,
            lora_r=8,
            lora_alpha=16,
        )

        experiment = REINFORCEExperiment(config, "test_output")

        # Create dummy rollouts
        dummy_rollouts = [
            (
                {
                    "initial_state": {"mean_income": 50000, "gini": 0.3},
                    "action": {"tax_rates": [0.1, 0.2, 0.3], "brackets": [10000, 50000]},
                    "reward": 0.5,
                    "system_prompt": "Test",
                    "user_prompt": "Test",
                },
                0.5,  # reward
                -5.0,  # log_prob
            )
            for _ in range(4)
        ]

        # Load planner (needed for train_step)
        from run_reinforce_h3_with_training import PlannerPolicy
        experiment.planner_policy = PlannerPolicy(
            model_name=config.planner_model,
            config=config,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        experiment.planner_policy.load_model()
        experiment.setup_optimizer()

        print("  Running train_step...")
        metrics = experiment.train_step(dummy_rollouts, iter_num=0)

        print(f"  Metrics: {metrics}")
        print(f"  Loss: {metrics['loss/total']:.3f}")

        print("✓ train_step works")
        return True
    except Exception as e:
        print(f"✗ train_step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("H3 Training Script Component Tests")
    print("="*60)

    results = []

    results.append(("Imports", test_imports()))

    if torch.cuda.is_available():
        print(f"\nGPU available: {torch.cuda.get_device_name(0)}")
        results.append(("PlannerPolicy", test_planner_policy()))
        results.append(("Action sampling", test_action_sampling()))
        results.append(("Train step", test_train_step()))
    else:
        print("\nNo GPU available, skipping model tests")

    print("\n" + "="*60)
    print("Test Results:")
    print("="*60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s}: {status}")

    all_passed = all(r[1] for r in results)
    print("\nOverall: " + ("✓ ALL TESTS PASSED" if all_passed else "✗ SOME TESTS FAILED"))

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
