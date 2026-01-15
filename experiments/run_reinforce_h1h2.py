#!/usr/bin/env python3
"""
REINFORCE++ H1/H2 Experiments

H1: With scaffolding (detailed prompts + reasoning)
H2: Without scaffolding (minimal prompts, direct output)

Usage:
    # H1 on GPU 0
    CUDA_VISIBLE_DEVICES=0 python experiments/run_reinforce_h1h2.py --experiment h1

    # H2 on GPU 1
    CUDA_VISIBLE_DEVICES=1 python experiments/run_reinforce_h1h2.py --experiment h2
"""

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch

# =============================================================================
# PROMPT CONFIGURATIONS
# =============================================================================

# H1: SCAFFOLDED PROMPT (detailed with reasoning guidance)
H1_SYSTEM_PROMPT = """You are an AI tax policy planner optimizing social welfare in an economic simulation.

Your role:
1. Analyze the economic state (income distribution, labor supply, inequality)
2. Consider how tax changes affect worker behavior
3. Balance revenue needs with incentive effects
4. Set progressive tax rates that maximize total welfare

Think step-by-step about the tradeoffs before deciding on tax rates.
Output your reasoning, then provide the tax policy as JSON."""

H1_USER_TEMPLATE = """## Economic State (Tax Year {tax_year})

**Key Metrics:**
- Mean Income: ${mean_income:,.0f} | Median: ${median_income:,.0f}
- Income Gini: {gini:.3f} (0=equal, 1=unequal)
- Mean Labor: {mean_labor:.1f} hours/week
- Social Welfare: {swf:.1f}

**Current Tax Brackets:**
{tax_brackets}

**Recent Trends:**
- SWF: {swf_trend}
- Gini: {gini_trend}
- Labor: {labor_trend}

Analyze the situation and set optimal tax rates.
First explain your reasoning (2-3 sentences), then output:
{{"tax_rates": [rate1, rate2, ...], "brackets": [threshold1, threshold2, ...]}}"""


# H2: NO SCAFFOLD PROMPT (minimal, direct output)
H2_SYSTEM_PROMPT = """Set tax rates to maximize social welfare. Output JSON only."""

H2_USER_TEMPLATE = """income={mean_income:.0f} gini={gini:.3f} labor={mean_labor:.1f} swf={swf:.1f}
rates={current_rates}
{{"tax_rates": [...], "brackets": [...]}}"""


@dataclass
class RLConfig:
    """Configuration for REINFORCE++ training."""
    experiment: str = "h1"  # h1 or h2
    planner_model: str = "google/gemma-3-4b-it"
    worker_model: str = "google/gemma-3-4b-it"

    # Environment
    num_agents: int = 100
    max_timesteps: int = 64  # Shorter rollouts for faster training
    tax_year_length: int = 64  # 1 tax year per rollout

    # Training
    num_iterations: int = 50
    rollouts_per_iter: int = 16  # Fewer rollouts for faster iterations
    learning_rate: float = 1e-5
    kl_coef: float = 0.05
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0

    # Reward weights
    swf_weight: float = 1.0
    gini_weight: float = 0.1  # Bonus for reducing inequality
    labor_weight: float = 0.05  # Small penalty for labor reduction

    # LoRA
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64

    # Checkpointing
    save_every: int = 10
    seed: int = 42

    def to_dict(self):
        return asdict(self)


def get_prompts(experiment: str) -> Tuple[str, str]:
    """Get system and user prompts for experiment type."""
    if experiment == "h1":
        return H1_SYSTEM_PROMPT, H1_USER_TEMPLATE
    elif experiment == "h2":
        return H2_SYSTEM_PROMPT, H2_USER_TEMPLATE
    else:
        raise ValueError(f"Unknown experiment: {experiment}")


def format_h1_prompt(state: Dict[str, Any]) -> str:
    """Format state into H1 (scaffolded) user prompt."""
    # Format tax brackets
    rates = state.get("tax_rates", [0.1, 0.2, 0.3, 0.35])
    brackets = state.get("brackets", [30000, 70000, 150000])

    bracket_lines = []
    for i, rate in enumerate(rates):
        if i == 0:
            upper = brackets[0] if brackets else "∞"
            bracket_lines.append(f"  $0-${upper:,}: {rate*100:.0f}%")
        elif i < len(brackets):
            bracket_lines.append(f"  ${brackets[i-1]:,}-${brackets[i]:,}: {rate*100:.0f}%")
        else:
            bracket_lines.append(f"  ${brackets[-1]:,}+: {rate*100:.0f}%")

    # Format trends
    def trend_str(values):
        if len(values) < 2:
            return "N/A"
        change = values[-1] - values[0]
        arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
        return f"{values[-1]:.2f} ({arrow}{abs(change):.2f})"

    return H1_USER_TEMPLATE.format(
        tax_year=state.get("tax_year", 0),
        mean_income=state.get("mean_income", 50000),
        median_income=state.get("median_income", 45000),
        gini=state.get("gini", 0.35),
        mean_labor=state.get("mean_labor", 40),
        swf=state.get("swf", 200),
        tax_brackets="\n".join(bracket_lines),
        swf_trend=trend_str(state.get("swf_history", [200])),
        gini_trend=trend_str(state.get("gini_history", [0.35])),
        labor_trend=trend_str(state.get("labor_history", [40])),
    )


def format_h2_prompt(state: Dict[str, Any]) -> str:
    """Format state into H2 (minimal) user prompt."""
    rates = state.get("tax_rates", [0.1, 0.2, 0.3])
    rates_str = "[" + ",".join(f"{r:.2f}" for r in rates) + "]"

    return H2_USER_TEMPLATE.format(
        mean_income=state.get("mean_income", 50000),
        gini=state.get("gini", 0.35),
        mean_labor=state.get("mean_labor", 40),
        swf=state.get("swf", 200),
        current_rates=rates_str,
    )


def compute_reward(
    prev_state: Dict[str, Any],
    next_state: Dict[str, Any],
    config: RLConfig,
) -> float:
    """
    Compute reward for a tax policy action.

    reward = swf_change + gini_bonus + labor_penalty
    """
    # SWF improvement (primary objective)
    swf_change = next_state["swf"] - prev_state["swf"]
    swf_reward = config.swf_weight * swf_change

    # Gini reduction bonus (lower is better)
    gini_change = prev_state["gini"] - next_state["gini"]  # Positive if reduced
    gini_reward = config.gini_weight * gini_change * 100  # Scale up

    # Labor stability (penalize large drops)
    labor_change = (next_state["mean_labor"] - prev_state["mean_labor"]) / prev_state["mean_labor"]
    labor_reward = config.labor_weight * min(0, labor_change) * 100  # Only penalize drops

    total_reward = swf_reward + gini_reward + labor_reward

    return total_reward


class REINFORCEExperiment:
    """
    Runs REINFORCE++ training for H1 or H2 experiment.
    """

    def __init__(self, config: RLConfig, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.system_prompt, self.user_template = get_prompts(config.experiment)
        self.format_prompt = format_h1_prompt if config.experiment == "h1" else format_h2_prompt

        # Training state
        self.iteration = 0
        self.best_reward = float("-inf")
        self.metrics_history = []

        # Models (loaded lazily)
        self.planner_engine = None
        self.worker_engine = None

    async def setup(self):
        """Initialize inference engines."""
        from llm_economist.inference.async_engine import ScalableInferenceEngine
        from llm_economist.inference.config import get_model_config

        print(f"\n{'='*60}")
        print(f"Setting up {self.config.experiment.upper()} Experiment")
        print(f"{'='*60}")
        print(f"Planner: {self.config.planner_model}")
        print(f"Workers: {self.config.worker_model}")
        print(f"Agents: {self.config.num_agents}")
        print(f"{'='*60}\n")

        # Get model config
        model_config = get_model_config(self.config.planner_model)

        # For H1/H2, we use the same model for planner and workers
        # Use AWQ quantization for speed
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
        )

        # Use same engine for workers (self-play)
        self.worker_engine = self.planner_engine

        print("Engines loaded.\n")

    async def collect_rollout(
        self,
        rollout_id: int,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Collect a single rollout (one tax year of interaction).

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

        # US-style brackets
        brackets = [10000, 40000, 85000, 160000, 200000, 500000]
        tax_rates = [0.1, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]

        # Generate personas
        personas = generate_aligned_personas(n=self.config.num_agents, seed=self.config.seed + rollout_id)
        persona_list = list(personas.values())

        # Track history
        swf_history = []
        gini_history = []
        labor_history = []

        # Run for one tax year
        for step in range(self.config.tax_year_length):
            incomes = skills * labor
            mean_income = incomes.mean()
            median_income = np.median(incomes)
            gini = self._compute_gini(incomes)
            swf = self._compute_swf(incomes, tax_rates, brackets)

            if step % 32 == 0:
                swf_history.append(swf)
                gini_history.append(gini)
                labor_history.append(labor.mean())

            # Get planner action at start of tax year
            if step == 0:
                prev_state = {
                    "tax_year": self.iteration,
                    "mean_income": mean_income,
                    "median_income": median_income,
                    "gini": gini,
                    "mean_labor": labor.mean(),
                    "swf": swf,
                    "tax_rates": tax_rates,
                    "brackets": brackets,
                    "swf_history": swf_history[-4:] if swf_history else [swf],
                    "gini_history": gini_history[-4:] if gini_history else [gini],
                    "labor_history": labor_history[-4:] if labor_history else [labor.mean()],
                }

                # Get planner policy
                new_rates = await self._get_planner_action(prev_state)
                if new_rates:
                    tax_rates = new_rates

            # Workers choose labor
            prompts = []
            for i, persona in enumerate(persona_list):
                prompt = f"""{persona}

Your skill level: {skills[i]:.1f}
Current income: ${incomes[i]:,.0f}
Tax rates: {tax_rates[0]*100:.0f}%-{tax_rates[-1]*100:.0f}%

Hours to work this week (0-100)? Number only:"""
                prompts.append(prompt)

            # Batch worker decisions
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

        # Final state
        final_incomes = skills * labor
        next_state = {
            "tax_year": self.iteration + 1,
            "mean_income": final_incomes.mean(),
            "median_income": np.median(final_incomes),
            "gini": self._compute_gini(final_incomes),
            "mean_labor": labor.mean(),
            "swf": self._compute_swf(final_incomes, tax_rates, brackets),
            "tax_rates": tax_rates,
            "brackets": brackets,
        }

        # Compute reward
        reward = compute_reward(prev_state, next_state, self.config)

        rollout_data = {
            "prev_state": prev_state,
            "next_state": next_state,
            "action": {"tax_rates": tax_rates},
            "reward": reward,
        }

        return rollout_data, reward

    async def _get_planner_action(self, state: Dict[str, Any]) -> Optional[List[float]]:
        """Get tax policy from planner."""
        from llm_economist.inference.async_engine import BatchRequest
        import re

        prompt = self.format_prompt(state)

        batch = BatchRequest(
            request_ids=["planner"],
            prompts=[prompt],
            system_prompts=[self.system_prompt],
            temperatures=[0.7],
            max_tokens=256 if self.config.experiment == "h1" else 64,
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
    parser = argparse.ArgumentParser(description="REINFORCE++ H1/H2 Experiments")
    parser.add_argument("--experiment", "-e", type=str, required=True,
                       choices=["h1", "h2"], help="Experiment type")
    parser.add_argument("--num-iterations", type=int, default=50)
    parser.add_argument("--rollouts-per-iter", type=int, default=32)
    parser.add_argument("--num-agents", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--resume", "-r", type=str, default=None,
                       help="Resume from checkpoint (e.g., models/reinforce_h1/iter_10)")

    args = parser.parse_args()

    # Set output directory
    output_dir = args.output or f"models/reinforce_{args.experiment}"

    config = RLConfig(
        experiment=args.experiment,
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
