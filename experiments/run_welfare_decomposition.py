#!/usr/bin/env python3
"""
Welfare Gap Decomposition Experiment.

Decomposes the ~10x welfare gap between G1 (rational PPO, SWF ~1960) and
G2 (bounded LLM + PPO planner, SWF ~195) into:
  (a) Policy misalignment -- the PPO planner's tax policy is sub-optimal for LLM populations
  (b) Utility-function mismatch -- bounded utility with satisfaction penalty vs. isoelastic
  (c) Interaction effects

Conditions:
  G1       : Rational workers (PPO) + PPO planner      [existing, SWF ~1960]
  G2       : LLM workers + PPO planner + bounded utility [existing, SWF ~195]
  G2-iso   : LLM workers + PPO planner + isoelastic utility (no satisfaction penalty)
  G2-iso-ICRL : LLM workers + ICRL planner + isoelastic utility (no satisfaction penalty)

Decomposition logic:
  Total gap          = G1 - G2
  Utility-fn effect  = G2-iso - G2        (effect of removing satisfaction penalty)
  Behavioral effect  = G1 - G2-iso        (LLM vs rational workers, same utility fn)
  Policy effect      = G2-iso - G2-iso-ICRL (PPO planner vs ICRL planner with LLM workers)

Usage:
    python experiments/run_welfare_decomposition.py \
        --model llama-3.1-8b --num-agents 100 --seed 42

    python experiments/run_welfare_decomposition.py \
        --model mistral-7b-v0.3 --num-agents 100 --seeds 42 123 456

    python experiments/run_welfare_decomposition.py --analyze-only
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results" / "welfare_decomposition"
G1_CHECKPOINT = BASE_DIR / "models" / "rl_baseline_g1" / "best" / "checkpoint.pt"
G2_RESULTS_DIR = BASE_DIR / "results" / "g2_evals"

# Tax brackets matching the RL baseline (see rl_baseline.py)
TAX_BRACKETS = [10000, 40000, 85000, 160000, 200000, 500000]

# G1 reference SWF (from training metrics)
G1_REFERENCE_SWF = 1960.0


# ---------------------------------------------------------------------------
# SWF computation (matching rl_baseline.py exactly)
# ---------------------------------------------------------------------------
def compute_isoelastic_swf(incomes: np.ndarray, tax_rates: List[float],
                           brackets: List[float], eta: float = 1.5) -> float:
    """
    Compute SWF using isoelastic utility: sum((z_post^(1-eta) - 1) / (1-eta)).
    This exactly matches the G1 RL baseline SWF formula.
    """
    post_tax = apply_taxes(incomes, tax_rates, brackets)
    post_tax = np.clip(post_tax, 100.0, None)  # numerical stability
    utilities = (post_tax ** (1 - eta) - 1) / (1 - eta)
    utilities = np.nan_to_num(utilities, nan=0.0, posinf=1e6, neginf=-1e6)
    utilities = np.clip(utilities, -100, 100)
    return float(utilities.sum())


def compute_bounded_swf(incomes: np.ndarray, tax_rates: List[float],
                        brackets: List[float],
                        satisfaction_rates: np.ndarray,
                        eta: float = 1.5) -> float:
    """
    Compute bounded SWF: same as isoelastic but utilities multiplied by
    satisfaction factor r (1.0 if satisfied, 0.5 if dissatisfied).
    """
    post_tax = apply_taxes(incomes, tax_rates, brackets)
    post_tax = np.clip(post_tax, 100.0, None)
    utilities = (post_tax ** (1 - eta) - 1) / (1 - eta)
    utilities = np.nan_to_num(utilities, nan=0.0, posinf=1e6, neginf=-1e6)
    utilities = np.clip(utilities, -100, 100)
    # Apply satisfaction penalty
    adjusted = utilities * satisfaction_rates
    return float(adjusted.sum())


def apply_taxes(incomes: np.ndarray, tax_rates: List[float],
                brackets: List[float]) -> np.ndarray:
    """Apply progressive tax schedule. Returns post-tax incomes."""
    clipped_rates = [max(0.0, min(0.99, r)) for r in tax_rates]
    taxes = np.zeros_like(incomes)
    prev_bracket = 0.0
    all_brackets = brackets + [float('inf')]
    for i, bracket in enumerate(all_brackets):
        if i < len(clipped_rates):
            bracket_income = np.clip(incomes - prev_bracket, 0, bracket - prev_bracket)
            taxes += bracket_income * clipped_rates[i]
        prev_bracket = bracket
    return incomes - taxes


def compute_gini(values: np.ndarray) -> float:
    """Compute Gini coefficient."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    if n < 2 or sorted_vals.sum() < 1e-8:
        return 0.0
    index = np.arange(1, n + 1)
    return float(2 * (index * sorted_vals).sum() / (n * sorted_vals.sum()) - (n + 1) / n)


# ---------------------------------------------------------------------------
# PPO planner wrapper
# ---------------------------------------------------------------------------
class PPOPlannerWrapper:
    """Load and query the trained G1 PPO planner."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.planner_net = None
        self._best_swf = None

    def load(self):
        import torch
        from llm_economist.training.rl_baseline import PlannerNetwork, RLBaselineConfig

        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        config = RLBaselineConfig(**checkpoint["config"])
        self.planner_net = PlannerNetwork(
            7, config.num_brackets, config.planner_hidden_dim
        )
        self.planner_net.load_state_dict(checkpoint["planner_net"])
        self.planner_net.eval()
        self._best_swf = checkpoint["best_swf"]
        logger.info(f"Loaded PPO planner (best training SWF: {self._best_swf:.1f})")

    def get_tax_rates(self, state: Dict[str, float]) -> List[float]:
        import torch
        planner_state = torch.tensor([
            state.get("mean_income", 50000) / 1000,
            state.get("std_income", 30000) / 1000,
            state.get("median_income", 45000) / 1000,
            state.get("gini", 0.35),
            state.get("mean_labor", 40),
            state.get("std_labor", 10),
            state.get("swf", 200) / 100,  # normalize like training
        ], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            mean, _, _ = self.planner_net(planner_state)
            rates = mean.squeeze().cpu().numpy().tolist()
        return rates


# ---------------------------------------------------------------------------
# LLM Worker simulation
# ---------------------------------------------------------------------------
async def run_llm_workers(
    engine,
    personas: List[str],
    skills: np.ndarray,
    labor: np.ndarray,
    incomes: np.ndarray,
    tax_rates: List[float],
    brackets: List[float],
    batch_size: int = 100,
) -> np.ndarray:
    """Get labor choices from LLM workers via batch inference."""
    from llm_economist.inference.async_engine import BatchRequest

    num_agents = len(personas)
    all_labor = np.copy(labor)

    for batch_start in range(0, num_agents, batch_size):
        batch_end = min(batch_start + batch_size, num_agents)
        prompts = []
        for i in range(batch_start, batch_end):
            prompt = (
                f"{personas[i]}\n\n"
                f"Your current economic situation:\n"
                f"- Skill level: {skills[i]:.1f}\n"
                f"- Current income: ${incomes[i]:,.0f}\n"
                f"- Tax rates: {tax_rates[0]*100:.0f}% (lowest bracket) to "
                f"{tax_rates[-1]*100:.0f}% (highest bracket)\n\n"
                f"How many hours (0-100) will you work this week? "
                f"Consider your personality, work-life balance, and economic situation.\n"
                f"Respond with ONLY a number."
            )
            prompts.append(prompt)

        batch = BatchRequest(
            request_ids=[f"worker_{i}" for i in range(batch_start, batch_end)],
            prompts=prompts,
            system_prompts=["You are a worker choosing how many hours to work."]
            * (batch_end - batch_start),
            temperatures=[0.7] * (batch_end - batch_start),
            max_tokens=10,
        )

        response = await engine.generate_batch(batch)

        for j, resp in enumerate(response.responses):
            idx = batch_start + j
            try:
                numbers = re.findall(r"\d+\.?\d*", resp)
                if numbers:
                    all_labor[idx] = min(100, max(0, float(numbers[0])))
            except Exception:
                pass

    return all_labor


async def simulate_satisfaction(
    engine,
    personas: List[str],
    skills: np.ndarray,
    incomes: np.ndarray,
    post_tax_incomes: np.ndarray,
    tax_rates: List[float],
    batch_size: int = 100,
) -> np.ndarray:
    """Ask LLM workers if they are satisfied with tax policy (bounded utility)."""
    from llm_economist.inference.async_engine import BatchRequest

    num_agents = len(personas)
    satisfaction = np.ones(num_agents)

    for batch_start in range(0, num_agents, batch_size):
        batch_end = min(batch_start + batch_size, num_agents)
        prompts = []
        for i in range(batch_start, batch_end):
            tax_paid = incomes[i] - post_tax_incomes[i]
            prompt = (
                f"{personas[i]}\n\n"
                f"Based on this year's economic summary:\n"
                f"- Pre-tax income: ${incomes[i]:,.0f}\n"
                f"- Tax paid: ${tax_paid:,.0f}\n"
                f"- Post-tax income: ${post_tax_incomes[i]:,.0f}\n"
                f"- Tax rates: {[f'{r*100:.0f}%' for r in tax_rates]}\n\n"
                f"Are you satisfied with the overall tax policy? "
                f"Answer YES or NO only."
            )
            prompts.append(prompt)

        batch = BatchRequest(
            request_ids=[f"sat_{i}" for i in range(batch_start, batch_end)],
            prompts=prompts,
            system_prompts=["Answer YES or NO."] * (batch_end - batch_start),
            temperatures=[0.3] * (batch_end - batch_start),
            max_tokens=5,
        )

        response = await engine.generate_batch(batch)

        for j, resp in enumerate(response.responses):
            idx = batch_start + j
            resp_lower = resp.strip().lower()
            if "no" in resp_lower:
                satisfaction[idx] = 0.5
            else:
                satisfaction[idx] = 1.0

    return satisfaction


# ---------------------------------------------------------------------------
# ICRL zero-shot LLM planner
# ---------------------------------------------------------------------------
async def get_icrl_tax_rates(
    engine,
    state: Dict[str, float],
    current_rates: List[float],
    num_brackets: int = 7,
) -> List[float]:
    """Get tax rates from a zero-shot LLM planner (ICRL)."""

    system_prompt = (
        "You are a tax policy planner trying to maximize social welfare. "
        "Social welfare is the sum of isoelastic utilities across all workers. "
        "Set marginal tax rates for each of 7 income brackets."
    )

    user_prompt = (
        f"Current economic state:\n"
        f"- Mean income: ${state['mean_income']:,.0f}\n"
        f"- Median income: ${state['median_income']:,.0f}\n"
        f"- Gini coefficient: {state['gini']:.3f}\n"
        f"- Current SWF: {state['swf']:.1f}\n"
        f"- Current tax rates: {[f'{r*100:.1f}%' for r in current_rates]}\n\n"
        f"Propose new tax rates as a list of 7 values between 0 and 1. "
        f"Respond with JSON: {{\"tax_rates\": [r1, r2, r3, r4, r5, r6, r7]}}"
    )

    response, _ = await engine.generate_single(
        system_prompt, user_prompt, temperature=0.7, max_tokens=256, json_format=True
    )

    try:
        data = json.loads(response)
        new_rates = data.get("tax_rates", current_rates)
        if len(new_rates) == num_brackets:
            parsed = []
            for r in new_rates:
                if isinstance(r, str):
                    r = r.strip().rstrip("%")
                    val = float(r)
                    if val > 1:
                        val /= 100.0
                else:
                    val = float(r)
                parsed.append(max(0.0, min(0.99, val)))
            return parsed
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass

    return current_rates


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
@dataclass
class EpisodeResult:
    condition: str
    seed: int
    model: str
    num_agents: int
    max_timesteps: int
    final_swf: float
    mean_swf_last200: float
    final_gini: float
    mean_labor: float
    mean_income: float
    final_tax_rates: List[float]
    mean_satisfaction: Optional[float]  # only for G2 condition
    swf_history: List[float]


async def run_episode(
    condition: str,
    engine,
    ppo_planner: Optional[PPOPlannerWrapper],
    num_agents: int,
    max_timesteps: int,
    tax_year_length: int,
    seed: int,
    model_name: str,
) -> EpisodeResult:
    """
    Run a single episode for a given condition.

    Conditions:
      'G2'          : LLM workers + PPO planner + bounded (with satisfaction)
      'G2-iso'      : LLM workers + PPO planner + isoelastic (no satisfaction)
      'G2-iso-ICRL' : LLM workers + ICRL planner + isoelastic (no satisfaction)
    """
    from llm_economist.agents.persona_generator import generate_aligned_personas

    np.random.seed(seed)

    # Generate personas
    persona_prompts = generate_aligned_personas(n=num_agents, seed=seed)
    personas = list(persona_prompts.values())

    # Initialize workers with log-normal skills (matching rl_baseline.py)
    skills = np.exp(np.random.randn(num_agents) * 0.5 + 3.5)
    labor = np.ones(num_agents) * 40.0

    # Initial tax rates
    tax_rates = [0.10, 0.15, 0.22, 0.24, 0.32, 0.35, 0.37]
    brackets = TAX_BRACKETS

    swf_history = []
    gini_history = []
    labor_history = []
    income_history = []
    satisfaction_history = []

    use_ppo = condition in ("G2", "G2-iso")
    use_satisfaction = condition == "G2"
    use_icrl = condition == "G2-iso-ICRL"

    for step in range(max_timesteps):
        incomes = skills * labor

        # Compute current state
        state = {
            "mean_income": float(incomes.mean()),
            "std_income": float(incomes.std()),
            "median_income": float(np.median(incomes)),
            "gini": compute_gini(incomes),
            "mean_labor": float(labor.mean()),
            "std_labor": float(labor.std()),
            "swf": compute_isoelastic_swf(incomes, tax_rates, brackets),
        }

        # Update tax rates at start of each tax year
        if step % tax_year_length == 0:
            if use_ppo and ppo_planner is not None:
                tax_rates = ppo_planner.get_tax_rates(state)
            elif use_icrl:
                tax_rates = await get_icrl_tax_rates(
                    engine, state, tax_rates, num_brackets=7
                )

        # Workers choose labor
        labor = await run_llm_workers(
            engine, personas, skills, labor, incomes, tax_rates, brackets,
        )

        # Recompute incomes after labor update
        incomes = skills * labor

        # Compute SWF
        if use_satisfaction:
            post_tax = apply_taxes(incomes, tax_rates, brackets)
            satisfaction = await simulate_satisfaction(
                engine, personas, skills, incomes, post_tax, tax_rates,
            )
            swf = compute_bounded_swf(incomes, tax_rates, brackets, satisfaction)
            mean_sat = float(satisfaction.mean())
            satisfaction_history.append(mean_sat)
        else:
            swf = compute_isoelastic_swf(incomes, tax_rates, brackets)
            satisfaction_history.append(None)

        gini = compute_gini(incomes)
        swf_history.append(swf)
        gini_history.append(gini)
        labor_history.append(float(labor.mean()))
        income_history.append(float(incomes.mean()))

        if step % 100 == 0 or step == max_timesteps - 1:
            sat_str = ""
            if use_satisfaction and satisfaction_history[-1] is not None:
                sat_str = f", sat={satisfaction_history[-1]:.2f}"
            logger.info(
                f"[{condition}] Step {step}: SWF={swf:.1f}, "
                f"Gini={gini:.3f}, labor={labor.mean():.1f}{sat_str}"
            )

    # Compute summary
    last200 = swf_history[-200:] if len(swf_history) >= 200 else swf_history
    mean_sat = None
    if use_satisfaction:
        sat_vals = [s for s in satisfaction_history if s is not None]
        mean_sat = float(np.mean(sat_vals)) if sat_vals else None

    return EpisodeResult(
        condition=condition,
        seed=seed,
        model=model_name,
        num_agents=num_agents,
        max_timesteps=max_timesteps,
        final_swf=swf_history[-1],
        mean_swf_last200=float(np.mean(last200)),
        final_gini=gini_history[-1],
        mean_labor=float(np.mean(labor_history)),
        mean_income=float(np.mean(income_history)),
        final_tax_rates=tax_rates,
        mean_satisfaction=mean_sat,
        swf_history=swf_history,
    )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
async def run_decomposition(args) -> Dict[str, Any]:
    """Run the full welfare gap decomposition experiment."""
    from llm_economist.inference.async_engine import ScalableInferenceEngine
    from llm_economist.inference.config import get_model_config

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    seeds = args.seeds

    logger.info("=" * 70)
    logger.info("Welfare Gap Decomposition Experiment")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Agents: {args.num_agents}")
    logger.info(f"Timesteps: {args.max_timesteps}")
    logger.info(f"Tax year length: {args.tax_year_length}")
    logger.info(f"Seeds: {seeds}")

    # Load PPO planner
    ppo_planner = PPOPlannerWrapper(str(G1_CHECKPOINT))
    ppo_planner.load()

    # Initialize LLM engine
    model_config = get_model_config(args.model)
    quant = model_config.recommended_quantization
    if hasattr(quant, "value"):
        quant_str = quant.value
    else:
        quant_str = str(quant)
    if quant_str.lower() == "none":
        quant_str = None

    engine = ScalableInferenceEngine(
        model_name=model_config.hf_name,
        quantization=quant_str,
        tensor_parallel_size=1,
        max_model_len=4096,
        text_only_mode=getattr(model_config, "text_only_mode", False),
    )
    await engine.initialize()

    # Run conditions
    conditions = ["G2-iso", "G2-iso-ICRL"]
    if args.run_g2:
        conditions.insert(0, "G2")

    all_results = {}
    for condition in conditions:
        all_results[condition] = []
        for seed in seeds:
            logger.info(f"\n--- Running {condition} (seed={seed}) ---")
            start_t = time.time()

            result = await run_episode(
                condition=condition,
                engine=engine,
                ppo_planner=ppo_planner,
                num_agents=args.num_agents,
                max_timesteps=args.max_timesteps,
                tax_year_length=args.tax_year_length,
                seed=seed,
                model_name=args.model,
            )

            elapsed = time.time() - start_t
            logger.info(
                f"Completed {condition} (seed={seed}) in {elapsed/60:.1f} min: "
                f"SWF={result.mean_swf_last200:.1f}"
            )
            all_results[condition].append(result)

    await engine.shutdown()
    return all_results


def compute_decomposition(
    all_results: Dict[str, List[EpisodeResult]],
    g1_swf: float = G1_REFERENCE_SWF,
    g2_swf: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute the welfare gap decomposition.

    Decomposition:
      Total gap     = G1 - G2
      Behavioral    = G1 - G2-iso     (LLM workers respond differently, same utility fn)
      Utility fn    = G2-iso - G2     (satisfaction penalty effect)
      ICRL benefit  = G2-iso-ICRL - G2-iso (ICRL vs PPO with LLM workers)
    """

    def mean_swf(results: List[EpisodeResult]) -> float:
        return float(np.mean([r.mean_swf_last200 for r in results]))

    def std_swf(results: List[EpisodeResult]) -> float:
        return float(np.std([r.mean_swf_last200 for r in results]))

    g2_iso_swf = mean_swf(all_results["G2-iso"])
    g2_iso_std = std_swf(all_results["G2-iso"])

    g2_iso_icrl_swf = mean_swf(all_results["G2-iso-ICRL"])
    g2_iso_icrl_std = std_swf(all_results["G2-iso-ICRL"])

    # Use measured G2 if available, otherwise use provided reference
    if "G2" in all_results and all_results["G2"]:
        g2_measured = mean_swf(all_results["G2"])
        g2_std = std_swf(all_results["G2"])
    else:
        g2_measured = g2_swf if g2_swf is not None else 195.0
        g2_std = 0.0

    total_gap = g1_swf - g2_measured
    behavioral_gap = g1_swf - g2_iso_swf
    utility_fn_gap = g2_iso_swf - g2_measured
    icrl_effect = g2_iso_icrl_swf - g2_iso_swf

    decomposition = {
        "reference_values": {
            "G1_swf": g1_swf,
            "G2_swf": g2_measured,
            "G2_swf_std": g2_std,
            "G2-iso_swf": g2_iso_swf,
            "G2-iso_swf_std": g2_iso_std,
            "G2-iso-ICRL_swf": g2_iso_icrl_swf,
            "G2-iso-ICRL_swf_std": g2_iso_icrl_std,
        },
        "gaps": {
            "total_gap": total_gap,
            "behavioral_gap": behavioral_gap,
            "utility_fn_gap": utility_fn_gap,
            "icrl_effect": icrl_effect,
        },
        "percentages": {
            "pct_behavioral": (
                behavioral_gap / total_gap * 100 if total_gap != 0 else 0
            ),
            "pct_utility_fn": (
                utility_fn_gap / total_gap * 100 if total_gap != 0 else 0
            ),
            "pct_interaction": (
                (total_gap - behavioral_gap - utility_fn_gap)
                / total_gap
                * 100
                if total_gap != 0
                else 0
            ),
        },
        "interpretation": {},
    }

    # Add interpretation
    if abs(g2_iso_swf - g2_measured) < 50:
        decomposition["interpretation"]["utility_fn"] = (
            "Minimal utility-function effect: satisfaction penalty has little impact "
            "on aggregate SWF."
        )
    else:
        decomposition["interpretation"]["utility_fn"] = (
            f"Significant utility-function effect: satisfaction penalty reduces SWF "
            f"by {utility_fn_gap:.0f} ({decomposition['percentages']['pct_utility_fn']:.1f}%)."
        )

    if behavioral_gap / total_gap > 0.7:
        decomposition["interpretation"]["behavioral"] = (
            f"The gap is predominantly behavioral ({decomposition['percentages']['pct_behavioral']:.1f}%): "
            f"LLM workers respond to taxes fundamentally differently than rational agents."
        )
    elif behavioral_gap / total_gap > 0.3:
        decomposition["interpretation"]["behavioral"] = (
            f"Both behavioral and utility-function effects contribute "
            f"({decomposition['percentages']['pct_behavioral']:.1f}% behavioral, "
            f"{decomposition['percentages']['pct_utility_fn']:.1f}% utility fn)."
        )
    else:
        decomposition["interpretation"]["behavioral"] = (
            f"The gap is primarily due to utility-function mismatch "
            f"({decomposition['percentages']['pct_utility_fn']:.1f}%), not behavioral differences."
        )

    if icrl_effect > 50:
        decomposition["interpretation"]["icrl"] = (
            f"ICRL improves SWF by {icrl_effect:.0f} over PPO planner with LLM workers, "
            f"suggesting adaptive planning helps compensate for LLM behavior."
        )
    elif icrl_effect < -50:
        decomposition["interpretation"]["icrl"] = (
            f"ICRL reduces SWF by {abs(icrl_effect):.0f} vs PPO planner, "
            f"suggesting the PPO planner generalizes reasonably well."
        )
    else:
        decomposition["interpretation"]["icrl"] = (
            f"ICRL and PPO planner produce similar outcomes with LLM workers "
            f"(delta={icrl_effect:.0f})."
        )

    return decomposition


def print_results(
    all_results: Dict[str, List[EpisodeResult]],
    decomposition: Dict[str, Any],
):
    """Print a formatted results table."""
    print("\n" + "=" * 70)
    print("WELFARE GAP DECOMPOSITION RESULTS")
    print("=" * 70)

    ref = decomposition["reference_values"]
    gaps = decomposition["gaps"]
    pcts = decomposition["percentages"]

    print(f"\n{'Condition':<18} {'SWF (mean)':<14} {'SWF (std)':<12} {'Workers':<12} {'Planner':<10} {'Utility'}")
    print("-" * 80)
    print(f"{'G1 (reference)':<18} {ref['G1_swf']:<14.1f} {'N/A':<12} {'Rational':<12} {'PPO':<10} Isoelastic")
    print(f"{'G2 (reference)':<18} {ref['G2_swf']:<14.1f} {ref['G2_swf_std']:<12.1f} {'LLM':<12} {'PPO':<10} Bounded")
    print(f"{'G2-iso':<18} {ref['G2-iso_swf']:<14.1f} {ref['G2-iso_swf_std']:<12.1f} {'LLM':<12} {'PPO':<10} Isoelastic")
    print(f"{'G2-iso-ICRL':<18} {ref['G2-iso-ICRL_swf']:<14.1f} {ref['G2-iso-ICRL_swf_std']:<12.1f} {'LLM':<12} {'ICRL':<10} Isoelastic")

    print(f"\n{'Decomposition':}")
    print("-" * 50)
    print(f"  Total gap (G1 - G2):        {gaps['total_gap']:>8.1f}")
    print(f"  Behavioral (G1 - G2-iso):   {gaps['behavioral_gap']:>8.1f}  ({pcts['pct_behavioral']:>5.1f}%)")
    print(f"  Utility fn (G2-iso - G2):   {gaps['utility_fn_gap']:>8.1f}  ({pcts['pct_utility_fn']:>5.1f}%)")
    print(f"  Interaction:                {gaps['total_gap'] - gaps['behavioral_gap'] - gaps['utility_fn_gap']:>8.1f}  ({pcts['pct_interaction']:>5.1f}%)")
    print(f"  ICRL effect:                {gaps['icrl_effect']:>8.1f}")

    print(f"\nInterpretation:")
    for key, text in decomposition["interpretation"].items():
        print(f"  [{key}] {text}")

    # Per-seed details
    for condition, results in all_results.items():
        print(f"\n  {condition} per seed:")
        for r in results:
            sat_str = ""
            if r.mean_satisfaction is not None:
                sat_str = f", satisfaction={r.mean_satisfaction:.2f}"
            print(
                f"    seed={r.seed}: SWF={r.mean_swf_last200:.1f}, "
                f"Gini={r.final_gini:.3f}, labor={r.mean_labor:.1f}"
                f"{sat_str}"
            )

    print("\n" + "=" * 70)


def analyze_existing_results(results_dir: Path = RESULTS_DIR):
    """Analyze previously saved results."""
    if not results_dir.exists():
        print(f"No results directory found at {results_dir}")
        return

    all_files = sorted(results_dir.glob("*.json"))
    if not all_files:
        print("No result files found.")
        return

    print(f"\nFound {len(all_files)} result files:")
    for f in all_files:
        data = json.load(open(f))
        if "decomposition" in data:
            print(f"\n  {f.name}:")
            d = data["decomposition"]
            ref = d["reference_values"]
            print(f"    G1={ref['G1_swf']:.1f}  G2={ref['G2_swf']:.1f}  "
                  f"G2-iso={ref['G2-iso_swf']:.1f}  G2-iso-ICRL={ref['G2-iso-ICRL_swf']:.1f}")
            pcts = d["percentages"]
            print(f"    Behavioral: {pcts['pct_behavioral']:.1f}%  "
                  f"Utility fn: {pcts['pct_utility_fn']:.1f}%  "
                  f"Interaction: {pcts['pct_interaction']:.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Welfare Gap Decomposition Experiment"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.1-8b",
        help="Worker LLM model name",
    )
    parser.add_argument("--num-agents", type=int, default=100)
    parser.add_argument("--max-timesteps", type=int, default=1000)
    parser.add_argument("--tax-year-length", type=int, default=128)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds to run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single seed (shorthand for --seeds)",
    )
    parser.add_argument(
        "--g1-swf",
        type=float,
        default=G1_REFERENCE_SWF,
        help="G1 reference SWF value",
    )
    parser.add_argument(
        "--g2-swf",
        type=float,
        default=195.0,
        help="G2 reference SWF value (used if --run-g2 is not set)",
    )
    parser.add_argument(
        "--run-g2",
        action="store_true",
        help="Also run G2 condition (bounded utility with satisfaction)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze existing results, don't run experiments",
    )

    args = parser.parse_args()

    if args.analyze_only:
        analyze_existing_results()
        return

    # Handle seed argument
    if args.seed is not None:
        args.seeds = [args.seed]

    # Run experiment
    all_results = asyncio.run(run_decomposition(args))

    # Compute decomposition
    decomposition = compute_decomposition(
        all_results,
        g1_swf=args.g1_swf,
        g2_swf=args.g2_swf,
    )

    # Print results
    print_results(all_results, decomposition)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = args.output or str(
        RESULTS_DIR / f"decomposition_{args.model}_seeds{'_'.join(map(str, args.seeds))}.json"
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "config": {
            "model": args.model,
            "num_agents": args.num_agents,
            "max_timesteps": args.max_timesteps,
            "tax_year_length": args.tax_year_length,
            "seeds": args.seeds,
            "g1_swf_reference": args.g1_swf,
            "g2_swf_reference": args.g2_swf,
        },
        "conditions": {},
        "decomposition": decomposition,
    }

    for condition, results in all_results.items():
        save_data["conditions"][condition] = [
            {
                "seed": r.seed,
                "final_swf": r.final_swf,
                "mean_swf_last200": r.mean_swf_last200,
                "final_gini": r.final_gini,
                "mean_labor": r.mean_labor,
                "mean_income": r.mean_income,
                "final_tax_rates": r.final_tax_rates,
                "mean_satisfaction": r.mean_satisfaction,
                # Don't save full history to keep file size manageable
                "swf_history_summary": {
                    "first_100_mean": float(np.mean(r.swf_history[:100])),
                    "last_100_mean": float(np.mean(r.swf_history[-100:])),
                    "overall_mean": float(np.mean(r.swf_history)),
                    "overall_std": float(np.std(r.swf_history)),
                    "min": float(np.min(r.swf_history)),
                    "max": float(np.max(r.swf_history)),
                },
            }
            for r in results
        ]

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
