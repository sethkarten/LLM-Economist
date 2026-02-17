#!/usr/bin/env python3
"""
Dynamic Empirical Saez Baseline Experiment

This experiment demonstrates that estimating labor elasticity empirically from
LLM agent populations and plugging those estimates into the Saez optimal tax
formula produces WORSE outcomes than either static Saez baselines or ICRL.

The core problem: LLM agents are stochastic and heterogeneous. Elasticity
estimates from perturbing tax rates are noisy and high-variance because:
  1. Each agent responds differently based on persona, history, and LLM randomness
  2. The single-parameter-per-bracket assumption in Saez cannot capture this
  3. Finite perturbation samples amplify noise in the estimates

Algorithm:
  1. Initialize N=100 Census-calibrated workers with a starting tax policy
  2. Run baseline simulation for K timesteps to get baseline labor responses
  3. For each tax bracket j:
     a. Perturb rate by +delta, run K timesteps, record labor changes
     b. Perturb rate by -delta, run K timesteps, record labor changes
     c. Estimate elasticity: e_j = mean((dl_i/l_i) / (d(1-t_j)/(1-t_j)))
  4. Plug estimated elasticities into Saez piecewise-linear formula
  5. Apply the resulting "optimal" Saez tax rates
  6. Run full simulation and compute SWF
  7. Compare against static Saez, ICRL final policy, and US Federal rates

Usage:
    # Quick test (10 agents, 8 timesteps per phase)
    python experiments/run_empirical_saez.py --num-agents 10 --timesteps-per-phase 8

    # Full experiment (100 agents, default settings)
    python experiments/run_empirical_saez.py --num-agents 100 --model gemma3-4b

    # With wandb logging
    python experiments/run_empirical_saez.py --num-agents 100 --wandb
"""

# Disable V1 and compilation to avoid hangs
import os
os.environ['VLLM_USE_V1'] = '0'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
if not os.environ.get('HF_TOKEN'):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
if 'SLURM_JOB_ID' in os.environ:
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HOME'] = '/scratch/gpfs/CHIJ/milkkarten/huggingface'
    os.environ['TRANSFORMERS_CACHE'] = '/scratch/gpfs/CHIJ/milkkarten/huggingface/hub'
    os.environ['WANDB_MODE'] = 'offline'
    os.environ['WANDB_DIR'] = '/scratch/gpfs/CHIJ/milkkarten/LLM-Economist/wandb'

import argparse
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tax bracket definitions (matches llm_economist/utils/bracket.py)
# ---------------------------------------------------------------------------

BRACKET_CONFIGS = {
    "three": {
        "brackets": [0, 90000, 159100, 10_000_000],
        "num_brackets": 3,
    },
    "US_FED": {
        "brackets": [0, 11600, 47150, 100525, 191950, 243725, 609350, 10_000_000],
        "num_brackets": 7,
    },
}

US_FED_RATES_PCT = [10, 12, 22, 24, 32, 35, 37]  # percentages
FLAT_RATE_PCT = [20, 20, 20, 20, 20, 20, 20]  # 20% flat


# ---------------------------------------------------------------------------
# Helper: Saez optimal tax formula (mirrors llm_economist/utils/common.py)
# ---------------------------------------------------------------------------

def saez_optimal_tax_rates(
    skills: np.ndarray,
    brackets: List[float],
    elasticities: List[float],
) -> List[float]:
    """
    Compute Saez optimal marginal tax rates for income brackets.

    Uses the piecewise-linear Saez (2001) formula:
        tau_j = (1 - G(z_j)) / (1 - G(z_j) + a(z_j) * e_j)

    Parameters
    ----------
    skills : array-like
        Worker skill levels (income = skill * labor_hours).
    brackets : list of float
        Bracket boundaries [b0, b1, ..., bK].
    elasticities : list of float
        Per-bracket compensated labor supply elasticities.

    Returns
    -------
    list of float
        Optimal marginal rates in PERCENTAGE form (0-100).
    """
    from scipy import stats

    incomes = np.sort(np.array(skills) * 100.0)  # income at ~100 hrs reference
    n_brackets = len(brackets) - 1

    if len(elasticities) != n_brackets:
        raise ValueError(
            f"Need {n_brackets} elasticities, got {len(elasticities)}"
        )

    # Welfare weights: inverse-income Pareto weights
    welfare_weights = 1.0 / np.maximum(incomes, 1e-10)
    welfare_weights /= welfare_weights.sum()

    kde = stats.gaussian_kde(incomes)

    tax_rates = []
    for i in range(n_brackets):
        b_lo, b_hi = brackets[i], brackets[i + 1]

        # Representative income in bracket
        if i < n_brackets - 1:
            z = 0.5 * (b_lo + b_hi)
        else:
            z = b_lo + 0.1 * (b_hi - b_lo)

        F_z = float(np.mean(incomes <= z))
        f_z = float(kde(z)[0])

        # Pareto tail parameter a(z) = z*f(z)/(1-F(z))
        if F_z < 1.0:
            a_z = (z * f_z) / (1.0 - F_z)
        else:
            a_z = 10.0

        # Refine for top bracket
        incomes_above = incomes[incomes >= z]
        if i == n_brackets - 1 and incomes_above.size > 0:
            m = incomes_above.mean()
            denom = m - b_lo
            a_z = m / denom if denom > 0 else 10.0

        # Average welfare weight above z
        if incomes_above.size > 0 and F_z < 1.0:
            G_z = welfare_weights[incomes >= z].sum() / (1.0 - F_z)
        else:
            G_z = 0.0

        e = elasticities[i]

        # Saez formula: tau = (1-G)/(1-G + a*e)
        denom = (1.0 - G_z) + a_z * e
        if denom > 0:
            tau = (1.0 - G_z) / denom
        else:
            tau = 0.5  # fallback

        tau = max(0.0, min(1.0, tau))
        tax_rates.append(round(tau * 100, 2))

    return tax_rates


# ---------------------------------------------------------------------------
# Tax computation helpers
# ---------------------------------------------------------------------------

def compute_taxes(
    incomes: np.ndarray,
    tax_rates_pct: List[float],
    brackets: List[float],
) -> Tuple[np.ndarray, float]:
    """
    Apply marginal tax schedule. Returns (post_tax_incomes, total_tax).
    tax_rates_pct are in percentage (0-100) form.
    """
    n = len(incomes)
    taxes = np.zeros(n)
    rates = [r / 100.0 for r in tax_rates_pct]

    for j in range(len(brackets) - 1):
        lo = brackets[j]
        hi = brackets[j + 1]
        rate = rates[j] if j < len(rates) else rates[-1]
        bracket_income = np.clip(incomes - lo, 0, hi - lo)
        taxes += bracket_income * rate * (incomes > lo).astype(float)

    total_tax = float(taxes.sum())
    post_tax = incomes - taxes
    return post_tax, total_tax


def compute_swf(
    incomes: np.ndarray,
    labors: np.ndarray,
    tax_rates_pct: List[float],
    brackets: List[float],
    c: float = 0.0005,
    delta: float = 3.5,
) -> float:
    """
    Compute Social Welfare Function: sum_i u_i / z_i
    where u_i = z_tilde_i - c * l_i^delta  (isoelastic utility)
    and z_tilde_i = post_tax_income_i + rebate.
    """
    post_tax, total_tax = compute_taxes(incomes, tax_rates_pct, brackets)
    rebate = total_tax / len(incomes)
    z_tilde = post_tax + rebate
    utilities = z_tilde - c * np.power(labors, delta)
    # SWF = sum(u_i / max(z_i, 1))
    swf = float(np.sum(utilities / np.maximum(incomes, 1.0)))
    return swf


def compute_gini(values: np.ndarray) -> float:
    """Compute Gini coefficient."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    if n < 2 or sorted_vals.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * (index * sorted_vals).sum() / (n * sorted_vals.sum())) - (n + 1) / n)


# ---------------------------------------------------------------------------
# Worker state
# ---------------------------------------------------------------------------

@dataclass
class WorkerState:
    """Lightweight per-worker state."""
    id: int
    skill: float
    labor: float
    income: float
    persona: str


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

class EmpiricalSaezExperiment:
    """
    Estimates labor elasticity empirically from LLM agent populations,
    plugs the estimates into the Saez optimal tax formula, and evaluates
    the resulting policy against baselines.
    """

    def __init__(
        self,
        num_agents: int = 100,
        timesteps_per_phase: int = 32,
        perturbation_delta: float = 0.05,
        bracket_setting: str = "three",
        model_name: str = "gemma3-4b",
        tensor_parallel: int = 1,
        quantization: str = "none",
        batch_size: int = 100,
        seed: int = 42,
        output: str = "results/empirical_saez",
        use_wandb: bool = False,
        wandb_offline: bool = False,
        debug: bool = False,
        num_perturbation_repeats: int = 1,
    ):
        self.num_agents = num_agents
        self.timesteps_per_phase = timesteps_per_phase
        self.perturbation_delta = perturbation_delta
        self.bracket_setting = bracket_setting
        self.model_name = model_name
        self.tensor_parallel = tensor_parallel
        self.quantization = quantization
        self.batch_size = batch_size
        self.seed = seed
        self.output_dir = Path(output)
        self.use_wandb = use_wandb
        self.wandb_offline = wandb_offline
        self.debug = debug
        self.num_perturbation_repeats = num_perturbation_repeats

        # Bracket config
        cfg = BRACKET_CONFIGS[bracket_setting]
        self.brackets: List[float] = cfg["brackets"]
        self.num_brackets: int = cfg["num_brackets"]

        # Engine
        self.engine = None

        # Workers (initialized later)
        self.workers: List[WorkerState] = []
        self.skills: np.ndarray = np.array([])

        # Results accumulator
        self.results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def setup(self):
        """Initialize vLLM engine and worker population."""
        from llm_economist.inference.async_engine import ScalableInferenceEngine
        from llm_economist.inference.config import get_model_config
        from llm_economist.agents.persona_generator import generate_aligned_personas
        from llm_economist.utils.common import rGB2

        np.random.seed(self.seed)

        print(f"\n{'=' * 70}")
        print("DYNAMIC EMPIRICAL SAEZ BASELINE EXPERIMENT")
        print(f"{'=' * 70}")
        print(f"  Agents:             {self.num_agents}")
        print(f"  Timesteps/phase:    {self.timesteps_per_phase}")
        print(f"  Perturbation delta: {self.perturbation_delta}")
        print(f"  Brackets:           {self.bracket_setting} ({self.num_brackets})")
        print(f"  Model:              {self.model_name}")
        print(f"  Seed:               {self.seed}")
        print(f"  Repeats/perturb:    {self.num_perturbation_repeats}")
        print(f"{'=' * 70}\n")

        # Resolve model
        model_config = get_model_config(self.model_name)
        quant_str = None if self.quantization == "none" else self.quantization

        # Determine dtype
        dtype = None
        if (
            hasattr(model_config, "recommended_quantization")
            and model_config.recommended_quantization.value == "none"
        ):
            dtype = "bfloat16"

        self.engine = ScalableInferenceEngine(
            model_name=model_config.hf_name,
            tensor_parallel_size=self.tensor_parallel,
            quantization=quant_str,
            max_model_len=4096,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            kv_cache_dtype="auto",
            max_num_seqs=min(512, self.batch_size * 2),
            text_only_mode=getattr(model_config, "text_only_mode", False),
            dtype=dtype,
        )
        await self.engine.initialize()

        # Sample skills from GB2 (US income calibrated)
        incomes_raw = rGB2(self.num_agents)
        self.skills = np.array([float(x / 40.0) for x in incomes_raw])

        # Generate personas
        personas = generate_aligned_personas(
            n=self.num_agents,
            use_llm_narratives=False,
            seed=self.seed,
        )
        persona_list = list(personas.values())

        # Initialize workers
        self.workers = []
        for i in range(self.num_agents):
            ws = WorkerState(
                id=i,
                skill=self.skills[i],
                labor=40.0,
                income=self.skills[i] * 40.0,
                persona=persona_list[i % len(persona_list)],
            )
            self.workers.append(ws)

        print(f"Initialized {self.num_agents} workers")
        print(f"  Skill range: [{self.skills.min():.1f}, {self.skills.max():.1f}]")
        print(f"  Mean income at 40hrs: ${self.skills.mean() * 40:.0f}\n")

    # ------------------------------------------------------------------
    # Simulation primitives
    # ------------------------------------------------------------------

    async def run_simulation_phase(
        self,
        tax_rates_pct: List[float],
        num_timesteps: int,
        phase_label: str = "",
    ) -> Dict[str, Any]:
        """
        Run a simulation phase: workers choose labor given fixed tax rates.
        Returns dict with final labors, incomes, SWF, Gini, etc.
        """
        from llm_economist.inference.async_engine import BatchRequest

        labors = np.array([w.labor for w in self.workers])

        for step in range(num_timesteps):
            incomes = self.skills * labors

            # Build prompts
            prompts = []
            for i, w in enumerate(self.workers):
                post_tax, total_tax = compute_taxes(
                    incomes[i:i+1], tax_rates_pct, self.brackets
                )
                rebate = total_tax / self.num_agents  # approximate
                prompt = self._build_worker_prompt(
                    w, labors[i], incomes[i], tax_rates_pct, step
                )
                prompts.append(prompt)

            # Batch inference
            all_responses = []
            for batch_start in range(0, self.num_agents, self.batch_size):
                batch_end = min(batch_start + self.batch_size, self.num_agents)
                batch = BatchRequest(
                    request_ids=[
                        f"{phase_label}_s{step}_w{i}"
                        for i in range(batch_start, batch_end)
                    ],
                    prompts=prompts[batch_start:batch_end],
                    system_prompts=[
                        "You are a worker in an economic simulation choosing hours to work."
                    ] * (batch_end - batch_start),
                    temperatures=[0.7] * (batch_end - batch_start),
                    max_tokens=32,
                )
                response = await self.engine.generate_batch(batch)
                all_responses.extend(response.responses)

            # Parse labor choices
            for i, resp in enumerate(all_responses):
                try:
                    numbers = re.findall(r"\d+\.?\d*", resp)
                    if numbers:
                        labors[i] = float(np.clip(float(numbers[0]), 0, 100))
                except (ValueError, IndexError):
                    pass  # keep previous labor

            if self.debug or step == num_timesteps - 1:
                incomes = self.skills * labors
                swf = compute_swf(incomes, labors, tax_rates_pct, self.brackets)
                print(
                    f"  [{phase_label}] Step {step+1}/{num_timesteps}: "
                    f"mean_labor={labors.mean():.1f}, "
                    f"mean_income=${incomes.mean():.0f}, "
                    f"SWF={swf:.2f}"
                )

        # Final metrics
        final_incomes = self.skills * labors
        final_swf = compute_swf(final_incomes, labors, tax_rates_pct, self.brackets)
        final_gini = compute_gini(final_incomes)

        # Update worker states
        for i, w in enumerate(self.workers):
            w.labor = labors[i]
            w.income = final_incomes[i]

        return {
            "labors": labors.copy(),
            "incomes": final_incomes.copy(),
            "swf": final_swf,
            "gini": final_gini,
            "mean_labor": float(labors.mean()),
            "mean_income": float(final_incomes.mean()),
            "tax_rates_pct": tax_rates_pct,
        }

    def _build_worker_prompt(
        self,
        worker: WorkerState,
        current_labor: float,
        current_income: float,
        tax_rates_pct: List[float],
        step: int,
    ) -> str:
        """Build worker prompt for labor decision."""
        rates_str = ", ".join(f"{r:.1f}%" for r in tax_rates_pct)
        brackets_str = ", ".join(f"${b:,.0f}" for b in self.brackets[:-1])

        return f"""{worker.persona}

You are a worker in an economic simulation.
Your skill level (hourly wage): ${worker.skill:.2f}
Current labor hours: {current_labor:.0f}
Current pre-tax income: ${current_income:,.0f}
Tax brackets: [{brackets_str}]
Marginal tax rates: [{rates_str}]
Timestep: {step}

Choose how many hours to work this week (0-100).
Consider the tax rates and your personal preferences.
Respond with ONLY a number between 0 and 100."""

    # ------------------------------------------------------------------
    # Bracket assignment
    # ------------------------------------------------------------------

    def assign_workers_to_brackets(
        self, incomes: np.ndarray
    ) -> Dict[int, List[int]]:
        """Map each worker to the bracket their income falls in."""
        bracket_workers: Dict[int, List[int]] = {
            j: [] for j in range(self.num_brackets)
        }
        for i, inc in enumerate(incomes):
            for j in range(self.num_brackets):
                lo = self.brackets[j]
                hi = self.brackets[j + 1]
                if lo <= inc < hi:
                    bracket_workers[j].append(i)
                    break
            else:
                # Income above all brackets: assign to top bracket
                bracket_workers[self.num_brackets - 1].append(i)
        return bracket_workers

    # ------------------------------------------------------------------
    # Elasticity estimation
    # ------------------------------------------------------------------

    async def estimate_elasticities(
        self,
        base_rates_pct: List[float],
        base_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Estimate labor supply elasticity per bracket by perturbing tax rates.

        For each bracket j:
          - Perturb rate_j by +delta and -delta
          - Run simulation for each
          - Compute elasticity: e_j = mean( (dl/l) / (d(1-t)/(1-t)) )

        Returns dict with elasticity estimates and diagnostics.
        """
        base_labors = base_result["labors"]
        base_incomes = base_result["incomes"]
        bracket_workers = self.assign_workers_to_brackets(base_incomes)

        elasticity_results = {}
        total_perturbation_sims = 0

        for j in range(self.num_brackets):
            workers_in_bracket = bracket_workers[j]
            n_in_bracket = len(workers_in_bracket)
            current_rate = base_rates_pct[j]

            print(f"\n--- Bracket {j} [{self.brackets[j]:,.0f} - {self.brackets[j+1]:,.0f}] ---")
            print(f"  Workers in bracket: {n_in_bracket}")
            print(f"  Current rate: {current_rate:.1f}%")

            if n_in_bracket == 0:
                elasticity_results[j] = {
                    "elasticity_mean": float("nan"),
                    "elasticity_std": float("nan"),
                    "elasticity_individual": [],
                    "n_workers": 0,
                    "rate": current_rate,
                    "note": "no workers in bracket",
                }
                continue

            all_individual_elasticities = []

            for rep in range(self.num_perturbation_repeats):
                rep_label = f"_rep{rep}" if self.num_perturbation_repeats > 1 else ""

                # --- Positive perturbation ---
                rates_up = base_rates_pct.copy()
                rates_up[j] = min(99.0, current_rate + self.perturbation_delta * 100)

                # Reset worker labors to baseline
                for i, w in enumerate(self.workers):
                    w.labor = base_labors[i]

                result_up = await self.run_simulation_phase(
                    rates_up,
                    self.timesteps_per_phase,
                    phase_label=f"perturb_b{j}_up{rep_label}",
                )
                total_perturbation_sims += 1

                # --- Negative perturbation ---
                rates_down = base_rates_pct.copy()
                rates_down[j] = max(0.0, current_rate - self.perturbation_delta * 100)

                # Reset worker labors to baseline
                for i, w in enumerate(self.workers):
                    w.labor = base_labors[i]

                result_down = await self.run_simulation_phase(
                    rates_down,
                    self.timesteps_per_phase,
                    phase_label=f"perturb_b{j}_down{rep_label}",
                )
                total_perturbation_sims += 1

                # --- Compute individual elasticities for workers in this bracket ---
                # e_i = (dl_i / l_i) / (d(1-t) / (1-t))
                # where dl_i = l_up_i - l_down_i, d(1-t) = (1-t_down) - (1-t_up)
                t_up = rates_up[j] / 100.0
                t_down = rates_down[j] / 100.0
                # Change in net-of-tax rate
                d_net_of_tax = (1.0 - t_down) - (1.0 - t_up)  # positive
                avg_net_of_tax = 0.5 * ((1.0 - t_up) + (1.0 - t_down))

                for idx in workers_in_bracket:
                    l_up = result_up["labors"][idx]
                    l_down = result_down["labors"][idx]
                    l_base = base_labors[idx]

                    dl = l_down - l_up  # labor change when net-of-tax increases
                    avg_l = 0.5 * (l_up + l_down)

                    if avg_l > 1.0 and abs(d_net_of_tax) > 1e-8 and avg_net_of_tax > 1e-8:
                        pct_dl = dl / avg_l
                        pct_d_nettax = d_net_of_tax / avg_net_of_tax
                        e_i = pct_dl / pct_d_nettax
                    else:
                        e_i = float("nan")

                    all_individual_elasticities.append(e_i)

            # Filter out NaNs for summary statistics
            valid_elasticities = [
                e for e in all_individual_elasticities if not np.isnan(e)
            ]

            if valid_elasticities:
                e_mean = float(np.mean(valid_elasticities))
                e_std = float(np.std(valid_elasticities))
                e_median = float(np.median(valid_elasticities))
                e_min = float(np.min(valid_elasticities))
                e_max = float(np.max(valid_elasticities))
                e_iqr = float(
                    np.percentile(valid_elasticities, 75)
                    - np.percentile(valid_elasticities, 25)
                )
            else:
                e_mean = e_std = e_median = e_min = e_max = e_iqr = float("nan")

            print(f"  Elasticity estimate: {e_mean:.4f} +/- {e_std:.4f}")
            print(f"  Median: {e_median:.4f}, Range: [{e_min:.4f}, {e_max:.4f}]")
            print(f"  IQR: {e_iqr:.4f}, N_valid: {len(valid_elasticities)}/{len(all_individual_elasticities)}")

            elasticity_results[j] = {
                "elasticity_mean": e_mean,
                "elasticity_std": e_std,
                "elasticity_median": e_median,
                "elasticity_min": e_min,
                "elasticity_max": e_max,
                "elasticity_iqr": e_iqr,
                "elasticity_individual": [
                    float(e) if not np.isnan(e) else None
                    for e in all_individual_elasticities
                ],
                "n_workers": n_in_bracket,
                "n_valid": len(valid_elasticities),
                "rate": current_rate,
            }

        return {
            "per_bracket": elasticity_results,
            "total_perturbation_sims": total_perturbation_sims,
        }

    # ------------------------------------------------------------------
    # Main experiment flow
    # ------------------------------------------------------------------

    async def run(self):
        """Run the full empirical Saez experiment."""
        start_time = time.time()

        # Initialize wandb
        if self.use_wandb:
            import wandb
            if self.wandb_offline:
                os.environ["WANDB_MODE"] = "offline"
            wandb.init(
                project="llm-economist",
                name=f"empirical_saez_{self.model_name}_{self.bracket_setting}_{self.num_agents}a_seed{self.seed}",
                config={
                    "experiment": "empirical_saez",
                    "num_agents": self.num_agents,
                    "timesteps_per_phase": self.timesteps_per_phase,
                    "perturbation_delta": self.perturbation_delta,
                    "bracket_setting": self.bracket_setting,
                    "model_name": self.model_name,
                    "seed": self.seed,
                    "num_perturbation_repeats": self.num_perturbation_repeats,
                },
                tags=[
                    "empirical_saez",
                    self.bracket_setting,
                    f"agents_{self.num_agents}",
                    f"seed_{self.seed}",
                ],
            )

        # ========== PHASE 1: Baseline with initial tax rates ==========
        print("\n" + "=" * 70)
        print("PHASE 1: Baseline Simulation (Initial Tax Rates)")
        print("=" * 70)

        # Use US Federal rates as starting point for US_FED, flat 20% for three
        if self.bracket_setting == "US_FED":
            initial_rates_pct = US_FED_RATES_PCT.copy()
        else:
            initial_rates_pct = [20.0] * self.num_brackets

        base_result = await self.run_simulation_phase(
            initial_rates_pct,
            self.timesteps_per_phase,
            phase_label="baseline",
        )
        print(f"\nBaseline results:")
        print(f"  SWF:         {base_result['swf']:.2f}")
        print(f"  Gini:        {base_result['gini']:.4f}")
        print(f"  Mean labor:  {base_result['mean_labor']:.1f}")
        print(f"  Mean income: ${base_result['mean_income']:,.0f}")

        # ========== PHASE 2: Elasticity Estimation via Perturbation ==========
        print("\n" + "=" * 70)
        print("PHASE 2: Estimating Elasticities via Tax Perturbation")
        print("=" * 70)
        print(f"Delta: +/- {self.perturbation_delta * 100:.1f} percentage points")
        print(f"Repeats per bracket: {self.num_perturbation_repeats}")

        elasticity_data = await self.estimate_elasticities(
            initial_rates_pct, base_result
        )

        # Extract elasticity vector for Saez formula
        empirical_elasticities = []
        for j in range(self.num_brackets):
            e = elasticity_data["per_bracket"][j]["elasticity_mean"]
            if np.isnan(e) or e <= 0:
                # Fallback: use Saez (2002) default of 0.4 for invalid estimates
                e = 0.4
                print(f"  Bracket {j}: invalid estimate, using default e=0.4")
            empirical_elasticities.append(e)

        print(f"\nEmpirical elasticity vector: {empirical_elasticities}")

        # ========== PHASE 3: Compute Empirical Saez Rates ==========
        print("\n" + "=" * 70)
        print("PHASE 3: Computing Dynamic Empirical Saez Tax Rates")
        print("=" * 70)

        empirical_saez_rates = saez_optimal_tax_rates(
            self.skills, self.brackets, empirical_elasticities
        )
        print(f"  Empirical Saez rates (%): {empirical_saez_rates}")

        # ========== PHASE 4: Evaluate Empirical Saez Policy ==========
        print("\n" + "=" * 70)
        print("PHASE 4: Evaluating Empirical Saez Policy")
        print("=" * 70)

        # Reset workers to baseline state
        for i, w in enumerate(self.workers):
            w.labor = base_result["labors"][i]

        empirical_saez_result = await self.run_simulation_phase(
            empirical_saez_rates,
            self.timesteps_per_phase * 2,  # longer evaluation
            phase_label="empirical_saez_eval",
        )

        # ========== PHASE 5: Static Saez Baselines ==========
        print("\n" + "=" * 70)
        print("PHASE 5: Static Saez Baselines")
        print("=" * 70)

        # Static Saez with e=0.4 (Saez 2002 empirical estimate)
        static_flat_elasticity = [0.4] * self.num_brackets
        static_flat_rates = saez_optimal_tax_rates(
            self.skills, self.brackets, static_flat_elasticity
        )
        print(f"  Static Saez (e=0.4 flat) rates (%): {static_flat_rates}")

        for i, w in enumerate(self.workers):
            w.labor = base_result["labors"][i]
        static_flat_result = await self.run_simulation_phase(
            static_flat_rates,
            self.timesteps_per_phase * 2,
            phase_label="static_saez_flat",
        )

        # Static Saez with AI Economist default e=3.0
        static_ai_econ_elasticity = [3.0] * self.num_brackets
        static_ai_econ_rates = saez_optimal_tax_rates(
            self.skills, self.brackets, static_ai_econ_elasticity
        )
        print(f"  Static Saez (e=3.0, AI Econ) rates (%): {static_ai_econ_rates}")

        for i, w in enumerate(self.workers):
            w.labor = base_result["labors"][i]
        static_ai_econ_result = await self.run_simulation_phase(
            static_ai_econ_rates,
            self.timesteps_per_phase * 2,
            phase_label="static_saez_ai_econ",
        )

        # ========== PHASE 6: US Federal Baseline ==========
        print("\n" + "=" * 70)
        print("PHASE 6: US Federal Tax Baseline")
        print("=" * 70)

        if self.bracket_setting == "US_FED":
            us_fed_rates = US_FED_RATES_PCT.copy()
        else:
            # Map US federal effective rates to 3-bracket system
            us_fed_rates = [15.0, 25.0, 35.0]

        print(f"  US Federal rates (%): {us_fed_rates}")

        for i, w in enumerate(self.workers):
            w.labor = base_result["labors"][i]
        us_fed_result = await self.run_simulation_phase(
            us_fed_rates,
            self.timesteps_per_phase * 2,
            phase_label="us_fed_eval",
        )

        # ========== Results Summary ==========
        total_time = time.time() - start_time

        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"{'Policy':<30} {'SWF':>10} {'Gini':>8} {'Mean Labor':>12}")
        print("-" * 60)
        print(
            f"{'Empirical Saez (ours)':<30} "
            f"{empirical_saez_result['swf']:>10.2f} "
            f"{empirical_saez_result['gini']:>8.4f} "
            f"{empirical_saez_result['mean_labor']:>12.1f}"
        )
        print(
            f"{'Static Saez (e=0.4)':<30} "
            f"{static_flat_result['swf']:>10.2f} "
            f"{static_flat_result['gini']:>8.4f} "
            f"{static_flat_result['mean_labor']:>12.1f}"
        )
        print(
            f"{'Static Saez (e=3.0, AI Econ)':<30} "
            f"{static_ai_econ_result['swf']:>10.2f} "
            f"{static_ai_econ_result['gini']:>8.4f} "
            f"{static_ai_econ_result['mean_labor']:>12.1f}"
        )
        print(
            f"{'US Federal':<30} "
            f"{us_fed_result['swf']:>10.2f} "
            f"{us_fed_result['gini']:>8.4f} "
            f"{us_fed_result['mean_labor']:>12.1f}"
        )
        print("-" * 60)

        # Elasticity diagnostics
        print(f"\n{'=' * 70}")
        print("ELASTICITY DIAGNOSTICS (shows why empirical estimation fails)")
        print(f"{'=' * 70}")
        print(f"{'Bracket':<15} {'e_mean':>10} {'e_std':>10} {'e_median':>10} {'IQR':>10} {'CV':>10} {'N':>6}")
        print("-" * 70)
        for j in range(self.num_brackets):
            ed = elasticity_data["per_bracket"][j]
            cv = (
                ed["elasticity_std"] / abs(ed["elasticity_mean"])
                if abs(ed.get("elasticity_mean", 0)) > 1e-8
                else float("inf")
            )
            print(
                f"  {j}  [{self.brackets[j]:>9,.0f}]  "
                f"{ed['elasticity_mean']:>10.4f} "
                f"{ed['elasticity_std']:>10.4f} "
                f"{ed.get('elasticity_median', float('nan')):>10.4f} "
                f"{ed.get('elasticity_iqr', float('nan')):>10.4f} "
                f"{cv:>10.2f} "
                f"{ed['n_workers']:>6d}"
            )
        print("-" * 70)
        print(f"Total perturbation simulations: {elasticity_data['total_perturbation_sims']}")
        print(f"  (Each = {self.timesteps_per_phase} steps x {self.num_agents} agents = "
              f"{self.timesteps_per_phase * self.num_agents} LLM calls)")
        total_llm_calls = (
            elasticity_data["total_perturbation_sims"]
            * self.timesteps_per_phase
            * self.num_agents
        )
        print(f"Total LLM calls for elasticity estimation: {total_llm_calls:,}")
        print(f"Total experiment time: {total_time:.1f}s ({total_time/60:.1f} min)")

        # ========== Save Results ==========
        self.output_dir.mkdir(parents=True, exist_ok=True)
        results = {
            "config": {
                "num_agents": self.num_agents,
                "timesteps_per_phase": self.timesteps_per_phase,
                "perturbation_delta": self.perturbation_delta,
                "bracket_setting": self.bracket_setting,
                "model_name": self.model_name,
                "seed": self.seed,
                "num_perturbation_repeats": self.num_perturbation_repeats,
            },
            "baseline": {
                "tax_rates_pct": initial_rates_pct,
                "swf": base_result["swf"],
                "gini": base_result["gini"],
                "mean_labor": base_result["mean_labor"],
                "mean_income": base_result["mean_income"],
            },
            "empirical_saez": {
                "tax_rates_pct": empirical_saez_rates,
                "elasticities_used": empirical_elasticities,
                "swf": empirical_saez_result["swf"],
                "gini": empirical_saez_result["gini"],
                "mean_labor": empirical_saez_result["mean_labor"],
                "mean_income": empirical_saez_result["mean_income"],
            },
            "static_saez_flat": {
                "tax_rates_pct": static_flat_rates,
                "elasticities_used": static_flat_elasticity,
                "swf": static_flat_result["swf"],
                "gini": static_flat_result["gini"],
                "mean_labor": static_flat_result["mean_labor"],
                "mean_income": static_flat_result["mean_income"],
            },
            "static_saez_ai_econ": {
                "tax_rates_pct": static_ai_econ_rates,
                "elasticities_used": static_ai_econ_elasticity,
                "swf": static_ai_econ_result["swf"],
                "gini": static_ai_econ_result["gini"],
                "mean_labor": static_ai_econ_result["mean_labor"],
                "mean_income": static_ai_econ_result["mean_income"],
            },
            "us_federal": {
                "tax_rates_pct": us_fed_rates,
                "swf": us_fed_result["swf"],
                "gini": us_fed_result["gini"],
                "mean_labor": us_fed_result["mean_labor"],
                "mean_income": us_fed_result["mean_income"],
            },
            "elasticity_diagnostics": {
                str(j): {
                    k: v
                    for k, v in elasticity_data["per_bracket"][j].items()
                    if k != "elasticity_individual"  # skip raw array for summary
                }
                for j in range(self.num_brackets)
            },
            "elasticity_individual_values": {
                str(j): elasticity_data["per_bracket"][j]["elasticity_individual"]
                for j in range(self.num_brackets)
            },
            "compute_cost": {
                "total_perturbation_sims": elasticity_data["total_perturbation_sims"],
                "total_llm_calls_elasticity": total_llm_calls,
                "total_time_seconds": total_time,
            },
        }

        output_path = self.output_dir / f"empirical_saez_seed{self.seed}.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")

        # Log final summary to wandb
        if self.use_wandb:
            import wandb
            wandb.log({
                "swf/empirical_saez": empirical_saez_result["swf"],
                "swf/static_saez_flat": static_flat_result["swf"],
                "swf/static_saez_ai_econ": static_ai_econ_result["swf"],
                "swf/us_federal": us_fed_result["swf"],
                "gini/empirical_saez": empirical_saez_result["gini"],
                "gini/us_federal": us_fed_result["gini"],
                "elasticity/mean_cv": float(np.nanmean([
                    elasticity_data["per_bracket"][j]["elasticity_std"]
                    / max(abs(elasticity_data["per_bracket"][j]["elasticity_mean"]), 1e-8)
                    for j in range(self.num_brackets)
                    if elasticity_data["per_bracket"][j]["n_workers"] > 0
                ])),
                "compute/total_perturbation_sims": elasticity_data["total_perturbation_sims"],
                "compute/total_llm_calls": total_llm_calls,
                "compute/total_time_minutes": total_time / 60,
            })

            # Log elasticity table
            columns = ["bracket", "e_mean", "e_std", "e_median", "iqr", "cv", "n_workers"]
            table_data = []
            for j in range(self.num_brackets):
                ed = elasticity_data["per_bracket"][j]
                cv = (
                    ed["elasticity_std"] / abs(ed["elasticity_mean"])
                    if abs(ed.get("elasticity_mean", 0)) > 1e-8
                    else float("inf")
                )
                table_data.append([
                    f"[{self.brackets[j]:,.0f}-{self.brackets[j+1]:,.0f}]",
                    ed["elasticity_mean"],
                    ed["elasticity_std"],
                    ed.get("elasticity_median", float("nan")),
                    ed.get("elasticity_iqr", float("nan")),
                    cv,
                    ed["n_workers"],
                ])
            wandb.log({"elasticity_table": wandb.Table(columns=columns, data=table_data)})

            wandb.finish()

        return results

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self):
        """Clean up vLLM engine."""
        if self.engine is not None:
            await self.engine.shutdown()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dynamic Empirical Saez Baseline: demonstrates that empirically "
            "estimated elasticities from LLM agents are too noisy for the "
            "Saez optimal tax formula."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core parameters
    parser.add_argument(
        "--num-agents", type=int, default=100, help="Number of worker agents"
    )
    parser.add_argument(
        "--timesteps-per-phase",
        type=int,
        default=32,
        help="Timesteps for each simulation phase (baseline, perturbation, eval)",
    )
    parser.add_argument(
        "--perturbation-delta",
        type=float,
        default=0.05,
        help="Tax rate perturbation size (fraction, e.g. 0.05 = 5pp)",
    )
    parser.add_argument(
        "--bracket-setting",
        type=str,
        default="three",
        choices=["three", "US_FED"],
        help="Tax bracket configuration",
    )
    parser.add_argument(
        "--num-perturbation-repeats",
        type=int,
        default=1,
        help="Number of perturbation repeats per bracket (increases sample size)",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3-4b",
        help="Worker LLM model (from inference config)",
    )
    parser.add_argument(
        "--tensor-parallel", type=int, default=1, help="Tensor parallel size (GPUs)"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        choices=["awq", "gptq", "fp8", "bitsandbytes", "none"],
        help="Quantization method",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size for inference"
    )

    # Experiment parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="results/empirical_saez",
        help="Output directory",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Wandb
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Run wandb in offline mode (for SLURM)",
    )

    return parser


async def main():
    parser = create_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    experiment = EmpiricalSaezExperiment(
        num_agents=args.num_agents,
        timesteps_per_phase=args.timesteps_per_phase,
        perturbation_delta=args.perturbation_delta,
        bracket_setting=args.bracket_setting,
        model_name=args.model,
        tensor_parallel=args.tensor_parallel,
        quantization=args.quantization,
        batch_size=args.batch_size,
        seed=args.seed,
        output=args.output,
        use_wandb=args.wandb,
        wandb_offline=args.wandb_offline,
        debug=args.debug,
        num_perturbation_repeats=args.num_perturbation_repeats,
    )

    try:
        await experiment.setup()
        results = await experiment.run()
    finally:
        await experiment.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
