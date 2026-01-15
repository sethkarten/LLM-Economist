#!/usr/bin/env python3
"""
REINFORCE++ H3 Experiment: Baseline Comparison Approach

H3: Single-decision baseline comparison
- Run US federal progressive tax as baseline (2024 rates)
- Each rollout: Planner makes ONE decision for new tax rates
- Roll out full tax year with those rates
- Reward = final_swf - baseline_swf (direct comparison)

This addresses the H1 training instability by using a fixed baseline
instead of comparing step 0 vs step 64 within the same episode.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/run_reinforce_h3_baseline.py --seed 42
"""

# Use vLLM legacy API to avoid V1 compilation issues
import os
os.environ['VLLM_USE_V1'] = '0'

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch

# =============================================================================
# PROMPT CONFIGURATIONS
# =============================================================================

# H3: BASELINE COMPARISON PROMPT (beat US federal tax baseline)
H3_SYSTEM_PROMPT = """You are an AI tax policy planner optimizing social welfare in an economic simulation.

Your goal: Set tax rates that achieve HIGHER social welfare than the US federal progressive tax baseline.

Baseline Performance (US 2024 Federal Tax):
- Social Welfare: {baseline_swf:.1f}
- Gini: {baseline_gini:.3f}
- Labor: {baseline_labor:.1f} hours/week

Output tax policy as JSON to beat this baseline."""

H3_USER_TEMPLATE = """## Current Economic State

**Income Distribution:**
- Mean Income: ${mean_income:,.0f}
- Gini: {gini:.3f}

**US Federal Baseline to Beat:**
- SWF: {baseline_swf:.1f}
- Gini: {baseline_gini:.3f}
- Labor: {baseline_labor:.1f} hours/week

Set tax rates to maximize welfare above baseline.
Output: {{"tax_rates": [rate1, rate2, ...], "brackets": [threshold1, threshold2, ...]}}"""


@dataclass
class RLConfig:
    """Configuration for REINFORCE++ H3 training."""
    experiment: str = "h3"
    planner_model: str = "google/gemma-3-4b-it"
    worker_model: str = "google/gemma-3-4b-it"

    # Environment
    num_agents: int = 100
    tax_year_length: int = 64  # 1 tax year per rollout

    # Training
    num_iterations: int = 50
    rollouts_per_iter: int = 16
    learning_rate: float = 1e-5
    kl_coef: float = 0.05
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0

    # Reward: Direct comparison to baseline (no complex weighting)
    # reward = final_swf - baseline_swf

    # LoRA
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64

    # Checkpointing
    save_every: int = 10
    seed: int = 42

    def to_dict(self):
        return asdict(self)


def format_h3_system_prompt(baseline_metrics: Dict[str, float]) -> str:
    """Format H3 system prompt with baseline metrics."""
    return H3_SYSTEM_PROMPT.format(
        baseline_swf=baseline_metrics["swf"],
        baseline_gini=baseline_metrics["gini"],
        baseline_labor=baseline_metrics["mean_labor"],
    )


def format_h3_user_prompt(state: Dict[str, Any], baseline_metrics: Dict[str, float]) -> str:
    """Format H3 user prompt with current state and baseline."""
    return H3_USER_TEMPLATE.format(
        mean_income=state["mean_income"],
        gini=state["gini"],
        baseline_swf=baseline_metrics["swf"],
        baseline_gini=baseline_metrics["gini"],
        baseline_labor=baseline_metrics["mean_labor"],
    )


def compute_reward(final_swf: float, baseline_swf: float) -> float:
    """
    Compute reward for H3: Direct comparison to baseline.

    reward = final_swf - baseline_swf

    Positive reward means planner beat the US federal tax baseline.
    """
    return final_swf - baseline_swf


class REINFORCEExperiment:
    """
    Runs REINFORCE++ training for H3 experiment with baseline comparison.
    """

    def __init__(self, config: RLConfig, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.iteration = 0
        self.best_reward = float("-inf")
        self.metrics_history = []

        # Baseline metrics (computed once during setup)
        self.baseline_metrics = None

        # Models (loaded lazily)
        self.planner_engine = None
        self.worker_engine = None

    async def setup(self):
        """Initialize inference engines and compute baseline."""
        from llm_economist.inference.async_engine import ScalableInferenceEngine
        from llm_economist.inference.config import get_model_config

        print(f"\n{'='*60}")
        print(f"Setting up H3 Experiment (Baseline Comparison)")
        print(f"{'='*60}")
        print(f"Planner: {self.config.planner_model}")
        print(f"Workers: {self.config.worker_model}")
        print(f"Agents: {self.config.num_agents}")
        print(f"{'='*60}\n")

        # Get model config
        model_config = get_model_config(self.config.planner_model)

        # Use same model for planner and workers
        quant = model_config.recommended_quantization
        if hasattr(quant, 'value'):
            quant_str = quant.value
        else:
            quant_str = str(quant)
        if quant_str.lower() == 'none':
            quant_str = None

        print(f"Loading inference engine with {quant_str or 'no'} quantization...")

        self.planner_engine = ScalableInferenceEngine(
            model_name=model_config.hf_name,
            quantization=quant_str,
            tensor_parallel_size=1,
            max_model_len=4096,
            text_only_mode=model_config.text_only_mode,
            enforce_eager=True,
        )

        # Use same engine for workers (self-play)
        self.worker_engine = self.planner_engine

        print("Engines loaded.\n")

        # Compute baseline metrics using US federal tax (2024 rates)
        print("Computing baseline (US federal progressive tax 2024)...")
        self.baseline_metrics = await self._compute_baseline()
        print(f"Baseline SWF: {self.baseline_metrics['swf']:.2f}")
        print(f"Baseline Gini: {self.baseline_metrics['gini']:.3f}")
        print(f"Baseline Labor: {self.baseline_metrics['mean_labor']:.1f} hours/week\n")

    async def _compute_baseline(self) -> Dict[str, float]:
        """
        Compute baseline metrics using US federal progressive tax (2024).

        Returns baseline SWF, Gini, and labor to beat.
        """
        from llm_economist.agents.persona_generator import generate_aligned_personas
        from llm_economist.inference.async_engine import BatchRequest
        import re

        # Initialize with same seed for reproducibility
        np.random.seed(self.config.seed)

        skills = np.exp(np.random.randn(self.config.num_agents) * 0.5 + 3.5)
        labor = np.ones(self.config.num_agents) * 40

        # US federal tax 2024 brackets and rates
        us_brackets = [11000, 44725, 95375, 182100, 231250, 578125]
        us_rates = [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]

        # Generate personas
        personas = generate_aligned_personas(n=self.config.num_agents, seed=self.config.seed)
        persona_list = list(personas.values())

        # Run full tax year with US federal rates
        for step in range(self.config.tax_year_length):
            incomes = skills * labor

            # Workers choose labor given US tax rates
            prompts = []
            for i, persona in enumerate(persona_list):
                prompt = f"""{persona}

Your skill level: {skills[i]:.1f}
Current income: ${incomes[i]:,.0f}
Tax rates: US federal progressive (10%-37%)

Hours to work this week (0-100)? Number only:"""
                prompts.append(prompt)

            batch = BatchRequest(
                request_ids=[f"baseline_w_{i}" for i in range(len(prompts))],
                prompts=prompts,
                system_prompts=["You are a worker deciding hours to work."] * len(prompts),
                temperatures=[0.7] * len(prompts),
                max_tokens=10,
            )

            response = await self.worker_engine.generate_batch(batch)

            # Parse labor choices
            for i, resp in enumerate(response.responses):
                try:
                    numbers = re.findall(r'\d+\.?\d*', resp)
                    if numbers:
                        labor[i] = min(100, max(0, float(numbers[0])))
                except:
                    pass

        # Final metrics
        final_incomes = skills * labor
        baseline_swf = self._compute_swf(final_incomes, us_rates, us_brackets)
        baseline_gini = self._compute_gini(final_incomes)
        baseline_labor = labor.mean()

        return {
            "swf": baseline_swf,
            "gini": baseline_gini,
            "mean_labor": baseline_labor,
        }

    async def collect_rollout(
        self,
        rollout_id: int,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Collect a single rollout with H3 baseline comparison approach.

        1. Initialize economic state
        2. Planner makes ONE decision for new tax rates
        3. Roll out full tax year with those rates
        4. Reward = final_swf - baseline_swf

        Returns:
            Tuple of (rollout_data, reward)
        """
        from llm_economist.agents.persona_generator import generate_aligned_personas
        from llm_economist.inference.async_engine import BatchRequest
        import re

        # Initialize economic state
        np.random.seed(self.config.seed + rollout_id + self.iteration * 1000)

        skills = np.exp(np.random.randn(self.config.num_agents) * 0.5 + 3.5)
        labor = np.ones(self.config.num_agents) * 40

        # US-style brackets (planner can modify rates)
        brackets = [11000, 44725, 95375, 182100, 231250, 578125]

        # Generate personas
        personas = generate_aligned_personas(n=self.config.num_agents, seed=self.config.seed + rollout_id)
        persona_list = list(personas.values())

        # Initial state for planner decision
        incomes = skills * labor
        initial_state = {
            "mean_income": incomes.mean(),
            "gini": self._compute_gini(incomes),
        }

        # Get planner action (ONE decision for entire tax year)
        tax_rates = await self._get_planner_action(initial_state)
        if not tax_rates:
            # Fallback to US federal rates if parsing fails
            tax_rates = [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]

        # Run full tax year with planner's chosen rates
        for step in range(self.config.tax_year_length):
            incomes = skills * labor

            # Workers choose labor given planner's tax rates
            prompts = []
            for i, persona in enumerate(persona_list):
                prompt = f"""{persona}

Your skill level: {skills[i]:.1f}
Current income: ${incomes[i]:,.0f}
Tax rates: {tax_rates[0]*100:.0f}%-{tax_rates[-1]*100:.0f}%

Hours to work this week (0-100)? Number only:"""
                prompts.append(prompt)

            batch = BatchRequest(
                request_ids=[f"w_{i}" for i in range(len(prompts))],
                prompts=prompts,
                system_prompts=["You are a worker deciding hours to work."] * len(prompts),
                temperatures=[0.7] * len(prompts),
                max_tokens=10,
            )

            response = await self.worker_engine.generate_batch(batch)

            # Parse labor choices
            for i, resp in enumerate(response.responses):
                try:
                    numbers = re.findall(r'\d+\.?\d*', resp)
                    if numbers:
                        labor[i] = min(100, max(0, float(numbers[0])))
                except:
                    pass

        # Final state after full tax year
        final_incomes = skills * labor
        final_swf = self._compute_swf(final_incomes, tax_rates, brackets)

        # Compute reward: Direct comparison to baseline
        reward = compute_reward(final_swf, self.baseline_metrics["swf"])

        rollout_data = {
            "initial_state": initial_state,
            "action": {"tax_rates": tax_rates, "brackets": brackets},
            "final_swf": final_swf,
            "baseline_swf": self.baseline_metrics["swf"],
            "reward": reward,
        }

        return rollout_data, reward

    async def _get_planner_action(self, state: Dict[str, Any]) -> Optional[List[float]]:
        """Get tax policy from planner using H3 baseline comparison prompt."""
        from llm_economist.inference.async_engine import BatchRequest
        import re

        # Format prompts with baseline metrics
        system_prompt = format_h3_system_prompt(self.baseline_metrics)
        user_prompt = format_h3_user_prompt(state, self.baseline_metrics)

        batch = BatchRequest(
            request_ids=["planner"],
            prompts=[user_prompt],
            system_prompts=[system_prompt],
            temperatures=[0.7],
            max_tokens=128,
        )

        response = await self.planner_engine.generate_batch(batch)
        resp_text = response.responses[0]

        # Parse tax rates from response
        try:
            # Look for JSON
            if "{" in resp_text:
                start = resp_text.index("{")
                end = resp_text.rindex("}") + 1
                data = json.loads(resp_text[start:end])
                if "tax_rates" in data:
                    rates = [max(0.0, min(0.99, r)) for r in data["tax_rates"]]
                    return rates
        except:
            pass

        return None

    def _compute_gini(self, values: np.ndarray) -> float:
        """Compute Gini coefficient."""
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        index = np.arange(1, n + 1)
        return float((2 * (index * sorted_vals).sum() / (n * sorted_vals.sum()) - (n + 1) / n))

    def _compute_swf(self, incomes: np.ndarray, tax_rates: List[float], brackets: List[float]) -> float:
        """Compute social welfare with numerical stability."""
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

        # Isoelastic utility
        eta = 1.5
        utilities = (post_tax ** (1 - eta) - 1) / (1 - eta)
        utilities = np.nan_to_num(utilities, nan=0.0, posinf=1e6, neginf=-1e6)
        utilities = np.clip(utilities, -100, 100)

        return float(utilities.sum())

    async def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Starting REINFORCE++ Training ({self.config.experiment.upper()})")
        print(f"{'='*60}")
        print(f"Iterations: {self.config.num_iterations}")
        print(f"Rollouts/iter: {self.config.rollouts_per_iter}")
        print(f"{'='*60}\n")

        start_iter = self.iteration  # Supports resume from checkpoint
        for iteration in range(start_iter, self.config.num_iterations):
            self.iteration = iteration
            iter_start = time.time()

            print(f"\nIteration {iteration+1}/{self.config.num_iterations}")

            # Collect rollouts
            rewards = []
            for r in range(self.config.rollouts_per_iter):
                rollout_data, reward = await self.collect_rollout(r)
                rewards.append(reward)

                if (r + 1) % 8 == 0:
                    print(f"  Rollouts: {r+1}/{self.config.rollouts_per_iter}")

            rewards = np.array(rewards)
            iter_time = time.time() - iter_start

            metrics = {
                "iteration": iteration,
                "reward_mean": float(rewards.mean()),
                "reward_std": float(rewards.std()),
                "reward_max": float(rewards.max()),
                "reward_min": float(rewards.min()),
                "time": iter_time,
            }
            self.metrics_history.append(metrics)

            print(f"  Reward: {metrics['reward_mean']:.3f} ± {metrics['reward_std']:.3f}")
            print(f"  Time: {iter_time:.1f}s")

            # Track best
            if metrics["reward_mean"] > self.best_reward:
                self.best_reward = metrics["reward_mean"]
                self.save_checkpoint("best")

            # Periodic save
            if (iteration + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"iter_{iteration+1}")

        # Final save
        self.save_checkpoint("final")
        print(f"\nTraining complete! Best reward: {self.best_reward:.3f}")

    def save_checkpoint(self, name: str):
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "config": self.config.to_dict(),
            "iteration": self.iteration,
            "best_reward": self.best_reward,
            "metrics_history": self.metrics_history,
        }

        with open(checkpoint_dir / "state.json", "w") as f:
            json.dump(state, f, indent=2)

        print(f"  Saved: {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load training state from checkpoint. Returns True if successful."""
        state_file = Path(checkpoint_path) / "state.json"
        if not state_file.exists():
            print(f"No checkpoint found at {checkpoint_path}")
            return False

        with open(state_file) as f:
            state = json.load(f)

        self.iteration = state.get("iteration", 0) + 1  # Resume from next iteration
        self.best_reward = state.get("best_reward", float("-inf"))
        self.metrics_history = state.get("metrics_history", [])

        print(f"Resumed from checkpoint: {checkpoint_path}")
        print(f"  Starting at iteration {self.iteration}")
        print(f"  Best reward so far: {self.best_reward:.3f}")

        return True

    async def shutdown(self):
        """Cleanup."""
        if self.planner_engine:
            await self.planner_engine.shutdown()


async def main():
    parser = argparse.ArgumentParser(
        description="REINFORCE++ H3 Experiment: Baseline Comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-iterations", type=int, default=50,
                       help="Number of training iterations")
    parser.add_argument("--rollouts-per-iter", type=int, default=16,
                       help="Rollouts per iteration")
    parser.add_argument("--num-agents", type=int, default=100,
                       help="Number of worker agents")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output directory (default: models/reinforce_h3_seed{seed})")
    parser.add_argument("--resume", "-r", type=str, default=None,
                       help="Resume from checkpoint (e.g., models/reinforce_h3_seed42/iter_10)")

    args = parser.parse_args()

    # Set output directory
    output_dir = args.output or f"models/reinforce_h3_seed{args.seed}"

    config = RLConfig(
        num_iterations=args.num_iterations,
        rollouts_per_iter=args.rollouts_per_iter,
        num_agents=args.num_agents,
        seed=args.seed,
    )

    experiment = REINFORCEExperiment(config, output_dir)

    try:
        await experiment.setup()

        # Resume from checkpoint if specified
        if args.resume:
            experiment.load_checkpoint(args.resume)

        await experiment.train()
    finally:
        await experiment.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
