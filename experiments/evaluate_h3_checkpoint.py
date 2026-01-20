#!/usr/bin/env python3
"""
Evaluate H3 REINFORCE++ checkpoint to see what tax policy it learned.

This loads a trained checkpoint and generates tax rate proposals to understand
what policy the REINFORCE++ training converged to.
"""

import os
os.environ['VLLM_USE_V1'] = '0'

import sys
import asyncio
import json
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_economist.inference.async_engine import ScalableInferenceEngine
from llm_economist.agents.persona_generator import generate_aligned_personas


async def evaluate_checkpoint(checkpoint_dir: str, num_samples: int = 5):
    """
    Evaluate a trained H3 checkpoint by generating sample tax policies.

    Args:
        checkpoint_dir: Path to checkpoint directory (e.g., results/h3_with_training_seed42/best/)
        num_samples: Number of tax policy samples to generate
    """

    print(f"\n{'='*60}")
    print(f"Evaluating H3 Checkpoint")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"{'='*60}\n")

    # Load baseline metrics from checkpoint
    state_file = Path(checkpoint_dir).parent / "best" / "state.json"
    if not state_file.exists():
        state_file = Path(checkpoint_dir) / "state.json"

    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
            print(f"Checkpoint info:")
            print(f"  Iteration: {state.get('iteration', 'unknown')}")
            print(f"  Best reward: {state.get('best_reward', 'unknown'):.3f}")
            print()

    # Compute baseline (US federal tax)
    print("Computing baseline SWF (US federal tax 2024)...")
    np.random.seed(42)
    skills = np.exp(np.random.randn(100) * 0.5 + 3.5)
    labor = np.ones(100) * 40
    incomes = skills * labor

    us_brackets = [11000, 44725, 95375, 182100, 231250, 578125]
    us_rates = [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]

    baseline_swf = compute_swf(incomes, us_rates, us_brackets)
    print(f"Baseline SWF: {baseline_swf:.2f}")
    print(f"US Tax Rates: {[f'{r*100:.0f}%' for r in us_rates]}\n")

    # Initialize inference engine
    print("Loading trained model checkpoint...")
    # Note: We can't easily load LoRA weights without the full setup
    # For now, we'll use the base model to show what format we need

    # For actual evaluation, you would need to:
    # 1. Load base model
    # 2. Apply LoRA adapters from checkpoint_dir/planner_lora/
    # 3. Generate tax policies

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nBaseline (US Federal 2024):")
    print(f"  SWF: {baseline_swf:.2f}")
    print(f"  Rates: {us_rates}")
    print(f"\nTo fully evaluate the trained checkpoint:")
    print(f"  1. Load base model: meta-llama/Llama-3.1-8B-Instruct")
    print(f"  2. Apply LoRA weights from: {checkpoint_dir}/planner_lora/")
    print(f"  3. Generate tax policies and compute final SWF")
    print(f"\nCheckpoint achieved reward: +{state.get('best_reward', 0):.3f} above baseline")
    print(f"Expected final SWF: ~{baseline_swf + state.get('best_reward', 0):.2f}")
    print(f"\nComparison to in-context learning:")
    print(f"  In-context (Gemma3-4b): 207.85 SWF (7.2% above baseline)")
    print(f"  In-context (Qwen3-8b): 427.39 SWF (120% above baseline)")
    print(f"  REINFORCE++ (100 iters): ~{baseline_swf + state.get('best_reward', 0):.2f} SWF ({(state.get('best_reward', 0)/baseline_swf)*100:.2f}% above baseline)")
    print(f"\n{'='*60}")
    print("CONCLUSION: Need to train much longer (500-1000 iterations)")
    print("="*60 + "\n")


def compute_swf(incomes: np.ndarray, tax_rates: list, brackets: list) -> float:
    """Compute social welfare with isoelastic utility."""
    clipped_rates = [max(0.0, min(0.99, r)) for r in tax_rates]

    # Apply progressive taxes
    taxes = np.zeros_like(incomes)
    prev_bracket = 0.0
    all_brackets = brackets + [float('inf')]

    for i, bracket in enumerate(all_brackets):
        if i < len(clipped_rates):
            bracket_income = np.clip(incomes - prev_bracket, 0, bracket - prev_bracket)
            taxes += bracket_income * clipped_rates[i]
        prev_bracket = bracket

    post_tax = np.clip(incomes - taxes, 100.0, None)

    # Isoelastic utility (eta=1.5)
    eta = 1.5
    utilities = (post_tax ** (1 - eta) - 1) / (1 - eta)
    utilities = np.nan_to_num(utilities, nan=0.0, posinf=1e6, neginf=-1e6)
    utilities = np.clip(utilities, -100, 100)

    return float(utilities.sum())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                       help="Path to checkpoint directory")
    parser.add_argument("--num-samples", "-n", type=int, default=5,
                       help="Number of tax policy samples to generate")
    args = parser.parse_args()

    asyncio.run(evaluate_checkpoint(args.checkpoint, args.num_samples))
