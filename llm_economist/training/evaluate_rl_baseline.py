#!/usr/bin/env python3
"""
G2: Evaluate RL Baseline with Bounded Rational LLM Workers.

Tests the trained RL planner policy with LLM-based workers to measure
the distribution shift effect (trained on rational, tested on bounded).

Usage:
    python -m llm_economist.training.evaluate_rl_baseline \
        --checkpoint models/rl_baseline_g1/best/checkpoint.pt \
        --worker-model gemma-3-4b \
        --num-agents 100 \
        --output results/rl_baseline_llm_eval.json
"""

import argparse
import json
import os
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

import torch
import numpy as np

from .rl_baseline import PlannerNetwork, RLBaselineConfig


@dataclass
class EvalConfig:
    """Configuration for G2 evaluation."""
    checkpoint_path: str = "models/rl_baseline_g1/best/checkpoint.pt"
    worker_model: str = "gemma-3-4b"
    num_agents: int = 100
    max_timesteps: int = 2000
    tax_year_length: int = 128
    num_eval_episodes: int = 3
    output_path: str = "results/rl_baseline_llm_eval.json"
    device: str = "cuda"
    seed: int = 42


class RLPlannerWrapper:
    """
    Wraps the trained RL planner for use in the LLM simulation.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.planner_net = None
        self.config = None

    def load(self):
        """Load the trained planner network."""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # Get config
        self.config = RLBaselineConfig(**checkpoint["config"])

        # Create and load network
        planner_input_dim = 7  # aggregate stats
        self.planner_net = PlannerNetwork(
            planner_input_dim,
            self.config.num_brackets,
            self.config.planner_hidden_dim
        ).to(self.device)

        self.planner_net.load_state_dict(checkpoint["planner_net"])
        self.planner_net.eval()

        print(f"Loaded RL planner from {self.checkpoint_path}")
        print(f"  Best SWF during training: {checkpoint['best_swf']:.1f}")
        print(f"  Total training steps: {checkpoint['total_steps']:,}")

    def get_tax_rates(self, state: Dict[str, Any]) -> List[float]:
        """Get tax rates from the RL planner given economic state."""
        # Convert state to tensor
        planner_state = torch.tensor([
            state.get("mean_income", 50000),
            state.get("std_income", 30000),
            state.get("median_income", 45000),
            state.get("gini", 0.35),
            state.get("mean_labor", 40),
            state.get("std_labor", 10),
            state.get("swf", 200),
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            mean, _, _ = self.planner_net(planner_state)
            tax_rates = mean.squeeze().cpu().numpy().tolist()

        return tax_rates


async def run_evaluation(config: EvalConfig) -> Dict[str, Any]:
    """
    Run G2 evaluation: RL planner with LLM workers.
    """
    from llm_economist.inference.async_engine import ScalableInferenceEngine
    from llm_economist.inference.config import get_model_config

    print(f"\n{'='*60}")
    print("G2: RL Baseline Evaluation with LLM Workers")
    print(f"{'='*60}")
    print(f"RL Planner: {config.checkpoint_path}")
    print(f"LLM Workers: {config.worker_model}")
    print(f"Agents: {config.num_agents}")
    print(f"Episodes: {config.num_eval_episodes}")
    print(f"{'='*60}\n")

    # Load RL planner
    planner = RLPlannerWrapper(config.checkpoint_path, config.device)
    planner.load()

    # Initialize LLM engine for workers
    try:
        model_config = get_model_config(config.worker_model)
    except ValueError as e:
        raise ValueError(f"Unknown worker model: {config.worker_model}") from e

    # Get quantization - use None if NONE/none (full precision)
    quant = model_config.recommended_quantization
    if hasattr(quant, 'value'):
        quant_str = quant.value
    else:
        quant_str = str(quant)
    if quant_str.lower() == 'none':
        quant_str = None

    engine = ScalableInferenceEngine(
        model_name=model_config.hf_name,
        quantization=quant_str,
        tensor_parallel_size=1,
        max_model_len=4096,
        text_only_mode=model_config.text_only_mode,
        gpu_memory_utilization=0.85,  # Lower to avoid OOM during CUDA graph capture
    )

    results = {
        "config": asdict(config),
        "episodes": [],
        "summary": {},
    }

    all_swfs = []
    all_ginis = []

    for episode in range(config.num_eval_episodes):
        print(f"\nEpisode {episode + 1}/{config.num_eval_episodes}")

        # Run simulation with RL planner and LLM workers
        episode_result = await run_episode(
            planner=planner,
            engine=engine,
            num_agents=config.num_agents,
            max_timesteps=config.max_timesteps,
            tax_year_length=config.tax_year_length,
            seed=config.seed + episode,
        )

        results["episodes"].append(episode_result)
        all_swfs.append(episode_result["final_swf"])
        all_ginis.append(episode_result["final_gini"])

        print(f"  Final SWF: {episode_result['final_swf']:.1f}")
        print(f"  Final Gini: {episode_result['final_gini']:.3f}")

    # Compute summary statistics
    results["summary"] = {
        "mean_swf": float(np.mean(all_swfs)),
        "std_swf": float(np.std(all_swfs)),
        "mean_gini": float(np.mean(all_ginis)),
        "std_gini": float(np.std(all_ginis)),
    }

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Mean SWF: {results['summary']['mean_swf']:.1f} ± {results['summary']['std_swf']:.1f}")
    print(f"Mean Gini: {results['summary']['mean_gini']:.3f} ± {results['summary']['std_gini']:.3f}")

    # Save results
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    await engine.shutdown()

    return results


async def run_episode(
    planner: RLPlannerWrapper,
    engine,
    num_agents: int,
    max_timesteps: int,
    tax_year_length: int,
    seed: int,
) -> Dict[str, Any]:
    """Run a single evaluation episode."""
    from llm_economist.agents.persona_generator import generate_aligned_personas

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate worker personas (returns Dict[id, prompt_str])
    persona_prompts = generate_aligned_personas(n=num_agents, seed=seed)
    # Convert to list of prompts for easier indexing
    personas = list(persona_prompts.values())

    # Initialize workers with skills
    skills = np.exp(np.random.randn(num_agents) * 0.5 + 3.5)  # Log-normal
    labor = np.ones(num_agents) * 40

    # Tax brackets (US-style)
    brackets = [10000, 40000, 85000, 160000, 200000, 500000]
    tax_rates = [0.1, 0.15, 0.22, 0.24, 0.32, 0.35, 0.37]

    metrics_history = []

    for step in range(max_timesteps):
        # Compute current state
        incomes = skills * labor
        mean_income = incomes.mean()
        median_income = np.median(incomes)
        gini = compute_gini(incomes)
        swf = compute_swf(incomes, tax_rates, brackets)

        state = {
            "mean_income": mean_income,
            "std_income": incomes.std(),
            "median_income": median_income,
            "gini": gini,
            "mean_labor": labor.mean(),
            "std_labor": labor.std(),
            "swf": swf,
        }

        # Update tax rates from RL planner (every tax year)
        if step % tax_year_length == 0:
            tax_rates = planner.get_tax_rates(state)

        # Workers choose labor (LLM-based)
        labor = await get_llm_labor_choices(
            engine=engine,
            personas=personas,
            skills=skills,
            incomes=incomes,
            tax_rates=tax_rates,
            brackets=brackets,
        )

        metrics_history.append({
            "step": step,
            "swf": swf,
            "gini": gini,
            "mean_income": mean_income,
            "tax_rates": tax_rates.copy() if isinstance(tax_rates, list) else tax_rates.tolist(),
        })

        if step % 100 == 0:
            print(f"  Step {step}: SWF={swf:.1f}, Gini={gini:.3f}")

    return {
        "seed": seed,
        "final_swf": metrics_history[-1]["swf"],
        "final_gini": metrics_history[-1]["gini"],
        "metrics_history": metrics_history,
    }


async def get_llm_labor_choices(
    engine,
    personas: List[str],  # List of persona prompt strings
    skills: np.ndarray,
    incomes: np.ndarray,
    tax_rates: List[float],
    brackets: List[float],
) -> np.ndarray:
    """Get labor choices from LLM workers."""
    from llm_economist.inference.async_engine import BatchRequest
    import re

    prompts = []
    for i, persona_prompt in enumerate(personas):
        # Persona is already a prompt string, add economic context
        prompt = f"""{persona_prompt}

Your current economic situation:
- Skill level: {skills[i]:.1f}
- Current income: ${incomes[i]:,.0f}
- Tax rates: {tax_rates[0]*100:.0f}% (lowest bracket) to {tax_rates[-1]*100:.0f}% (highest bracket)

How many hours (0-100) will you work this week? Consider your personality, work-life balance, and economic situation.
Respond with ONLY a number."""

        prompts.append(prompt)

    # Create batch request
    batch = BatchRequest(
        request_ids=[f"worker_{i}" for i in range(len(prompts))],
        prompts=prompts,
        system_prompts=["You are a worker choosing how many hours to work."] * len(prompts),
        temperatures=[0.7] * len(prompts),
        max_tokens=10,
    )

    # Batch LLM calls
    response = await engine.generate_batch(batch)

    # Parse responses
    labor = np.ones(len(personas)) * 40  # Default

    for i, resp in enumerate(response.responses):
        try:
            # Extract number from response
            numbers = re.findall(r'\d+\.?\d*', resp)
            if numbers:
                labor[i] = min(100, max(0, float(numbers[0])))
        except:
            pass

    return labor


def compute_gini(values: np.ndarray) -> float:
    """Compute Gini coefficient."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return (2 * (index * sorted_vals).sum() / (n * sorted_vals.sum()) - (n + 1) / n)


def compute_swf(incomes: np.ndarray, tax_rates: List[float], brackets: List[float]) -> float:
    """Compute social welfare function with robust numerical handling."""
    # Clip tax rates to valid range [0, 0.99] to prevent negative post-tax income
    clipped_rates = [max(0.0, min(0.99, r)) for r in tax_rates]

    # Apply taxes
    post_tax = apply_taxes(incomes, clipped_rates, brackets)

    # Ensure post-tax income is positive (minimum $100 for numerical stability)
    post_tax = np.clip(post_tax, 100.0, None)

    # Isoelastic utility with numerical stability
    eta = 1.5
    utilities = (post_tax ** (1 - eta) - 1) / (1 - eta)

    # Handle any remaining NaN/inf values
    utilities = np.nan_to_num(utilities, nan=0.0, posinf=1e6, neginf=-1e6)

    # Clip extreme utility values to prevent outliers from dominating
    utilities = np.clip(utilities, -100, 100)

    return float(utilities.sum())


def apply_taxes(incomes: np.ndarray, tax_rates: List[float], brackets: List[float]) -> np.ndarray:
    """Apply progressive tax schedule."""
    taxes = np.zeros_like(incomes)
    prev_bracket = 0.0

    all_brackets = brackets + [float('inf')]

    for i, bracket in enumerate(all_brackets):
        if i < len(tax_rates):
            bracket_income = np.clip(incomes - prev_bracket, 0, bracket - prev_bracket)
            taxes += bracket_income * tax_rates[i]
        prev_bracket = bracket

    return incomes - taxes


def main():
    parser = argparse.ArgumentParser(description="G2: RL Baseline with LLM Workers")

    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                       help="Path to RL baseline checkpoint")
    parser.add_argument("--worker-model", "-w", type=str, default="gemma-3-4b",
                       help="LLM model for workers")
    parser.add_argument("--num-agents", type=int, default=100)
    parser.add_argument("--max-timesteps", type=int, default=2000)
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    config = EvalConfig(
        checkpoint_path=args.checkpoint,
        worker_model=args.worker_model,
        num_agents=args.num_agents,
        max_timesteps=args.max_timesteps,
        num_eval_episodes=args.num_episodes,
        output_path=args.output,
        device=args.device,
        seed=args.seed,
    )

    asyncio.run(run_evaluation(config))


if __name__ == "__main__":
    main()
