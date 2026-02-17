#!/usr/bin/env python3
"""
Post-Hoc Perturbation Refinement for ICRL Tax Policies.

Takes a converged ICRL tax policy and performs a local grid search / perturbation
sweep to test whether additional SWF can be squeezed out cheaply (no training,
just evaluation).  Also runs a random-search baseline for comparison.

Algorithm:
  1. Load ICRL result to get final converged tax rates.
  2. Coordinate-wise sweep: for each bracket, try perturbations of
     [-10, -5, -2, 0, +2, +5, +10] percentage points while holding the other
     brackets fixed.  Pick the best per-bracket perturbation.
  3. (Optional) Second pass with finer grid [-3, -1, 0, +1, +3] around the
     best from pass 1.
  4. Random-search baseline: sample M random tax policies and evaluate each.
  5. Report results as JSON + optional matplotlib plots.

Each "evaluation" runs N=100 LLM workers for K timesteps under a fixed tax
policy and computes the mean SWF over the last 20 timesteps.

Usage:
    # Evaluate the Mistral ICRL run with perturbation refinement
    python experiments/run_perturbation_refinement.py \
        --results-dir results/bounded_100x2000 \
        --result-file mistral-7b-v0.3_seed42.json \
        --model mistral-7b-v0.3 \
        --eval-timesteps 50 \
        --num-agents 100 \
        --output results/perturbation_refinement/mistral_seed42.json

    # Quick test (fewer steps)
    python experiments/run_perturbation_refinement.py \
        --results-dir results/bounded_100x2000 \
        --result-file mistral-7b-v0.3_seed42.json \
        --model mistral-7b-v0.3 \
        --eval-timesteps 20 \
        --num-agents 20 \
        --num-random 10 \
        --output results/perturbation_refinement/quick_test.json
"""

# Use vLLM legacy API to avoid V1 compilation issues
import os
os.environ['VLLM_USE_V1'] = '0'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
# Enable offline mode if HF_TOKEN not set (use cached models for gated repos like gemma)
if not os.environ.get('HF_TOKEN'):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_economist.inference.async_engine import (
    ScalableInferenceEngine,
    BatchRequest,
    BatchResponse,
)
from llm_economist.inference.config import (
    get_model_config,
    SUPPORTED_MODELS,
    MODEL_ALIASES,
    QuantizationType,
)
from llm_economist.agents.persona_generator import generate_aligned_personas
from llm_economist.utils.common import rGB2

logger = logging.getLogger(__name__)


def resolve_cached_model_path(hf_name: str) -> str:
    """Resolve HuggingFace model name to cached snapshot path for offline mode."""
    if os.environ.get('HF_HUB_OFFLINE') != '1':
        return hf_name

    hf_cache = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    model_dir_name = f"models--{hf_name.replace('/', '--')}"

    possible_paths = [
        os.path.join(hf_cache, model_dir_name),
        os.path.join(hf_cache, 'hub', model_dir_name),
        os.path.join('/data1/milkkarten/.cache/huggingface', model_dir_name),
    ]

    for model_cache_dir in possible_paths:
        if os.path.exists(model_cache_dir):
            snapshots_dir = os.path.join(model_cache_dir, 'snapshots')
            if os.path.exists(snapshots_dir):
                snapshots = os.listdir(snapshots_dir)
                if snapshots:
                    resolved = os.path.join(snapshots_dir, snapshots[0])
                    print(f"Offline mode: resolved {hf_name} -> {resolved}")
                    return resolved

    print(f"WARNING: Could not find cached model for {hf_name}, using name as-is")
    return hf_name


# US Federal 2024 brackets (same as in main_async.py)
US_FED_BRACKETS = [0, 23000, 47000, 94000, 192000, 244000, 500000, 1000000]


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Result of evaluating one candidate tax policy."""
    tax_rates: List[float]
    mean_swf: float           # mean SWF over last `avg_window` steps
    std_swf: float            # std  SWF over last `avg_window` steps
    all_swf: List[float]      # SWF at every evaluation timestep
    label: str                # human-readable label (e.g. "bracket_2_+5pp")
    eval_time: float          # wall-clock seconds


# ─────────────────────────────────────────────────────────────────────────────
# Loading ICRL results
# ─────────────────────────────────────────────────────────────────────────────

def load_icrl_tax_rates(results_dir: str, result_file: str) -> Tuple[List[float], float, dict]:
    """
    Load the converged tax rates from an ICRL result file.

    The ICRL async simulator stores tax_rates as fractions in [0, 1].

    Returns:
        (tax_rates, final_swf_avg, full_config_dict)
    """
    filepath = Path(results_dir) / result_file
    if not filepath.exists():
        raise FileNotFoundError(f"ICRL result file not found: {filepath}")

    with open(filepath) as f:
        data = json.load(f)

    config = data["config"]
    final_state = data["final_state"]
    metrics = data["metrics_history"]

    # Extract the final converged tax rates
    tax_rates = final_state["tax_rates"]

    # Compute average SWF over the last 20 timesteps for stability
    last_n = min(20, len(metrics))
    avg_swf = np.mean([m["swf"] for m in metrics[-last_n:]])

    print(f"Loaded ICRL result: {result_file}")
    print(f"  Model: {config.get('model_name', 'unknown')}")
    print(f"  Scenario: {config.get('scenario', 'unknown')}")
    print(f"  Seed: {config.get('seed', 'unknown')}")
    print(f"  Final tax rates: {[f'{r:.3f}' for r in tax_rates]}")
    print(f"  Final SWF (last-{last_n} avg): {avg_swf:.2f}")
    print(f"  Final SWF (point): {final_state['swf']:.2f}")

    return tax_rates, avg_swf, config


# ─────────────────────────────────────────────────────────────────────────────
# Fixed-policy evaluator
# ─────────────────────────────────────────────────────────────────────────────

class FixedPolicyEvaluator:
    """
    Evaluate a fixed tax policy by running N LLM workers for K timesteps.

    The planner is NOT an LLM here -- we fix the tax rates and only run
    worker decisions.  This is cheap: one vLLM batch per timestep.
    """

    def __init__(
        self,
        engine: ScalableInferenceEngine,
        num_agents: int,
        eval_timesteps: int,
        avg_window: int = 20,
        batch_size: int = 100,
        seed: int = 42,
    ):
        self.engine = engine
        self.num_agents = num_agents
        self.eval_timesteps = eval_timesteps
        self.avg_window = avg_window
        self.batch_size = batch_size
        self.seed = seed

        # Pre-generate personas and skills (reused across evaluations)
        np.random.seed(seed)
        self.personas = generate_aligned_personas(
            n=num_agents, use_llm_narratives=False, seed=seed,
        )
        self.persona_list = list(self.personas.values())

        # Skills from GB2 distribution
        incomes = rGB2(num_agents)
        self.skills = np.array([float(x / 40.0) for x in incomes])

    async def evaluate(
        self,
        tax_rates: List[float],
        label: str = "",
    ) -> EvalResult:
        """
        Run a full evaluation of a fixed tax policy.

        Returns an EvalResult with SWF trajectory and summary statistics.
        """
        start = time.time()

        # Initialize worker state
        labor = np.ones(self.num_agents) * 40.0
        swf_trajectory: List[float] = []

        for step in range(self.eval_timesteps):
            # Compute current incomes
            incomes = self.skills * labor

            # Build worker prompts
            prompts = []
            sys_prompts = []
            for i in range(self.num_agents):
                persona = self.persona_list[i % len(self.persona_list)]
                tax_display = ", ".join(f"{r*100:.1f}%" for r in tax_rates)

                sys_prompt = (
                    f"You are an economic agent in a tax simulation.\n"
                    f"{persona}\n\n"
                    f"Your goal is to maximize your utility by choosing how many "
                    f"hours to work per week.\n"
                    f"Utility = post_tax_income + rebate - cost_of_labor\n"
                    f"where cost_of_labor increases with hours worked."
                )

                user_prompt = (
                    f"Timestep {step}:\n"
                    f"Your skill level (hourly wage): ${self.skills[i]:.2f}\n"
                    f"Your current labor hours: {labor[i]:.0f}\n"
                    f"Your current pre-tax income: ${incomes[i]:.2f}\n"
                    f"Tax rates per bracket: [{tax_display}]\n\n"
                    f"Choose your labor hours for this period (0-100).\n"
                    f'Respond with JSON: {{"labor_hours": <number>}}'
                )

                sys_prompts.append(sys_prompt)
                prompts.append(user_prompt)

            # Batch inference
            for batch_start in range(0, self.num_agents, self.batch_size):
                batch_end = min(batch_start + self.batch_size, self.num_agents)
                batch = BatchRequest(
                    request_ids=[
                        f"eval_{label}_s{step}_w{i}"
                        for i in range(batch_start, batch_end)
                    ],
                    prompts=prompts[batch_start:batch_end],
                    system_prompts=sys_prompts[batch_start:batch_end],
                    temperatures=[0.7] * (batch_end - batch_start),
                    max_tokens=64,
                    json_format=True,
                )
                response = await self.engine.generate_batch(batch)

                # Parse labor decisions
                for idx_in_batch, resp_text in enumerate(response.responses):
                    global_idx = batch_start + idx_in_batch
                    parsed = self._parse_labor(resp_text)
                    if parsed is not None:
                        labor[global_idx] = parsed

            # Re-compute incomes after labor choices
            incomes = self.skills * labor

            # Apply taxes
            total_tax = 0.0
            post_tax = np.zeros(self.num_agents)
            for i in range(self.num_agents):
                tax = self._calculate_tax(incomes[i], tax_rates)
                post_tax[i] = incomes[i] - tax
                total_tax += tax

            rebate = total_tax / self.num_agents
            post_tax += rebate

            # Compute utilities and SWF
            c_labor = 0.01
            delta_labor = 2.0
            swf = 0.0
            for i in range(self.num_agents):
                cost = c_labor * (labor[i] ** delta_labor)
                utility = post_tax[i] - cost
                if incomes[i] > 0:
                    swf += utility / incomes[i]

            swf_trajectory.append(swf)

        # Summary
        window = min(self.avg_window, len(swf_trajectory))
        tail = swf_trajectory[-window:]
        elapsed = time.time() - start

        return EvalResult(
            tax_rates=list(tax_rates),
            mean_swf=float(np.mean(tail)),
            std_swf=float(np.std(tail)),
            all_swf=swf_trajectory,
            label=label,
            eval_time=elapsed,
        )

    @staticmethod
    def _calculate_tax(income: float, tax_rates: List[float]) -> float:
        """Apply marginal tax using US Federal brackets."""
        tax = 0.0
        prev = 0.0
        for bracket, rate in zip(US_FED_BRACKETS[1:], tax_rates):
            if income <= prev:
                break
            taxable = min(income, bracket) - prev
            tax += taxable * rate
            prev = bracket
        # Top bracket (income above last threshold)
        if income > US_FED_BRACKETS[-1]:
            tax += (income - US_FED_BRACKETS[-1]) * tax_rates[-1]
        return tax

    @staticmethod
    def _parse_labor(text: str) -> Optional[float]:
        """Parse labor hours from worker response."""
        try:
            data = json.loads(text)
            val = float(data.get("labor_hours", -1))
            if 0 <= val <= 100:
                return val
        except (json.JSONDecodeError, TypeError, ValueError, KeyError):
            pass
        # Fallback: extract any number
        nums = re.findall(r"\d+\.?\d*", text)
        if nums:
            val = float(nums[0])
            return min(100, max(0, val))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Perturbation sweep
# ─────────────────────────────────────────────────────────────────────────────

def generate_coordinate_perturbations(
    base_rates: List[float],
    perturbation_pps: List[float],
) -> List[Tuple[List[float], str]]:
    """
    Generate candidate tax policies by perturbing one bracket at a time.

    Args:
        base_rates: converged ICRL rates (fractions, e.g. 0.10)
        perturbation_pps: perturbation magnitudes in percentage points
                          (e.g. [-10, -5, 0, 5, 10])

    Returns:
        List of (candidate_rates, label) tuples.
    """
    candidates = []
    for b_idx in range(len(base_rates)):
        for pp in perturbation_pps:
            if pp == 0:
                continue  # skip the identity (evaluated separately as baseline)
            new_rates = list(base_rates)
            new_rate = base_rates[b_idx] + pp / 100.0
            new_rate = max(0.0, min(0.99, new_rate))
            # Skip if clamping made it identical to base
            if abs(new_rate - base_rates[b_idx]) < 1e-6:
                continue
            new_rates[b_idx] = new_rate
            sign = "+" if pp > 0 else ""
            label = f"bracket_{b_idx}_{sign}{pp}pp"
            candidates.append((new_rates, label))
    return candidates


def generate_random_policies(
    num_brackets: int,
    n_samples: int,
    seed: int = 123,
) -> List[Tuple[List[float], str]]:
    """
    Sample random tax policies (uniform in valid range per bracket).

    Rates are sampled uniformly from [0, 0.60] to stay in a reasonable range,
    with the constraint that higher brackets have weakly higher rates
    (progressive structure).
    """
    rng = np.random.RandomState(seed)
    candidates = []
    for i in range(n_samples):
        rates = sorted(rng.uniform(0.0, 0.60, num_brackets))
        label = f"random_{i}"
        candidates.append((list(rates), label))
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

async def run_perturbation_experiment(args):
    """Run the full perturbation refinement experiment."""

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # ── Load ICRL result ────────────────────────────────────────────────
    base_rates, icrl_swf, icrl_config = load_icrl_tax_rates(
        args.results_dir, args.result_file,
    )
    num_brackets = len(base_rates)

    # Parse perturbation range
    perturbation_pps = [float(x) for x in args.perturbation_range.split(",")]
    print(f"\nPerturbation range (pp): {perturbation_pps}")

    # ── Initialize inference engine ─────────────────────────────────────
    model_config = get_model_config(args.model)
    quant = args.quantization if args.quantization != "none" else None

    # Resolve model path (handle offline mode with cached gated models)
    resolved_model = resolve_cached_model_path(model_config.hf_name)

    engine = ScalableInferenceEngine(
        model_name=resolved_model,
        tensor_parallel_size=args.tensor_parallel,
        quantization=quant,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_seqs=min(512, args.batch_size * 2),
        text_only_mode=getattr(model_config, "text_only_mode", False),
        enforce_eager=True,  # safe default for Blackwell + non-Blackwell
    )
    await engine.initialize()

    evaluator = FixedPolicyEvaluator(
        engine=engine,
        num_agents=args.num_agents,
        eval_timesteps=args.eval_timesteps,
        avg_window=args.avg_window,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    all_results: List[Dict[str, Any]] = []
    total_evals = 0

    # ── 0. Evaluate ICRL baseline ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("Phase 0: Evaluating ICRL baseline policy")
    print(f"{'='*60}")

    baseline_result = await evaluator.evaluate(base_rates, label="icrl_baseline")
    total_evals += 1
    print(f"  ICRL baseline SWF: {baseline_result.mean_swf:.2f} "
          f"(+/- {baseline_result.std_swf:.2f})  [{baseline_result.eval_time:.1f}s]")

    all_results.append({
        "phase": "baseline",
        "label": baseline_result.label,
        "tax_rates": baseline_result.tax_rates,
        "mean_swf": baseline_result.mean_swf,
        "std_swf": baseline_result.std_swf,
        "eval_time": baseline_result.eval_time,
    })

    # ── 1. Coordinate-wise perturbation sweep (pass 1) ─────────────────
    print(f"\n{'='*60}")
    print("Phase 1: Coordinate-wise perturbation sweep")
    print(f"{'='*60}")

    candidates_pass1 = generate_coordinate_perturbations(base_rates, perturbation_pps)
    print(f"  {len(candidates_pass1)} candidate policies to evaluate")

    pass1_results: List[EvalResult] = []
    for idx, (candidate_rates, label) in enumerate(candidates_pass1):
        result = await evaluator.evaluate(candidate_rates, label=label)
        total_evals += 1
        pass1_results.append(result)

        delta = result.mean_swf - baseline_result.mean_swf
        marker = " ***" if delta > 0 else ""
        print(f"  [{idx+1}/{len(candidates_pass1)}] {label}: "
              f"SWF={result.mean_swf:.2f} (delta={delta:+.2f}){marker}  "
              f"[{result.eval_time:.1f}s]")

        all_results.append({
            "phase": "perturbation_pass1",
            "label": label,
            "tax_rates": result.tax_rates,
            "mean_swf": result.mean_swf,
            "std_swf": result.std_swf,
            "delta_swf": delta,
            "eval_time": result.eval_time,
        })

    # Find best per-bracket perturbation
    best_rates_pass1 = list(base_rates)  # start from baseline
    for b_idx in range(num_brackets):
        bracket_results = [
            r for r in pass1_results
            if r.label.startswith(f"bracket_{b_idx}_")
        ]
        if not bracket_results:
            continue
        # Include the baseline (no perturbation) for this bracket
        best_for_bracket = max(bracket_results, key=lambda r: r.mean_swf)
        if best_for_bracket.mean_swf > baseline_result.mean_swf:
            best_rates_pass1[b_idx] = best_for_bracket.tax_rates[b_idx]
            print(f"  Bracket {b_idx}: best perturbation = {best_for_bracket.label} "
                  f"(SWF {best_for_bracket.mean_swf:.2f})")

    # Evaluate the composite best from pass 1
    print(f"\n  Evaluating composite best from pass 1...")
    composite_pass1 = await evaluator.evaluate(best_rates_pass1, label="composite_pass1")
    total_evals += 1
    delta_composite = composite_pass1.mean_swf - baseline_result.mean_swf
    print(f"  Composite pass-1 SWF: {composite_pass1.mean_swf:.2f} "
          f"(delta={delta_composite:+.2f})  [{composite_pass1.eval_time:.1f}s]")

    all_results.append({
        "phase": "perturbation_pass1_composite",
        "label": "composite_pass1",
        "tax_rates": composite_pass1.tax_rates,
        "mean_swf": composite_pass1.mean_swf,
        "std_swf": composite_pass1.std_swf,
        "delta_swf": delta_composite,
        "eval_time": composite_pass1.eval_time,
    })

    # ── 2. Fine-grained pass 2 ─────────────────────────────────────────
    if args.fine_pass:
        print(f"\n{'='*60}")
        print("Phase 2: Fine-grained perturbation sweep")
        print(f"{'='*60}")

        fine_pps = [float(x) for x in args.fine_range.split(",")]
        candidates_pass2 = generate_coordinate_perturbations(
            best_rates_pass1, fine_pps,
        )
        print(f"  {len(candidates_pass2)} fine-grained candidates")

        pass2_results: List[EvalResult] = []
        for idx, (candidate_rates, label) in enumerate(candidates_pass2):
            label = f"fine_{label}"
            result = await evaluator.evaluate(candidate_rates, label=label)
            total_evals += 1
            pass2_results.append(result)

            delta = result.mean_swf - composite_pass1.mean_swf
            marker = " ***" if delta > 0 else ""
            print(f"  [{idx+1}/{len(candidates_pass2)}] {label}: "
                  f"SWF={result.mean_swf:.2f} (delta={delta:+.2f}){marker}  "
                  f"[{result.eval_time:.1f}s]")

            all_results.append({
                "phase": "perturbation_pass2",
                "label": label,
                "tax_rates": result.tax_rates,
                "mean_swf": result.mean_swf,
                "std_swf": result.std_swf,
                "delta_swf": delta,
                "eval_time": result.eval_time,
            })

        # Best from pass 2
        best_rates_pass2 = list(best_rates_pass1)
        for b_idx in range(num_brackets):
            bracket_results = [
                r for r in pass2_results
                if r.label.startswith(f"fine_bracket_{b_idx}_")
            ]
            if not bracket_results:
                continue
            best_for_bracket = max(bracket_results, key=lambda r: r.mean_swf)
            if best_for_bracket.mean_swf > composite_pass1.mean_swf:
                best_rates_pass2[b_idx] = best_for_bracket.tax_rates[b_idx]

        # Evaluate composite pass 2
        composite_pass2 = await evaluator.evaluate(
            best_rates_pass2, label="composite_pass2",
        )
        total_evals += 1
        delta_pass2 = composite_pass2.mean_swf - baseline_result.mean_swf
        print(f"  Composite pass-2 SWF: {composite_pass2.mean_swf:.2f} "
              f"(delta from ICRL={delta_pass2:+.2f})  "
              f"[{composite_pass2.eval_time:.1f}s]")

        all_results.append({
            "phase": "perturbation_pass2_composite",
            "label": "composite_pass2",
            "tax_rates": composite_pass2.tax_rates,
            "mean_swf": composite_pass2.mean_swf,
            "std_swf": composite_pass2.std_swf,
            "delta_swf": delta_pass2,
            "eval_time": composite_pass2.eval_time,
        })

        best_perturbation_result = composite_pass2
    else:
        best_perturbation_result = composite_pass1

    # ── 3. Random search baseline ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 3: Random search baseline ({args.num_random} samples)")
    print(f"{'='*60}")

    random_candidates = generate_random_policies(
        num_brackets, args.num_random, seed=args.seed + 1000,
    )

    random_results: List[EvalResult] = []
    for idx, (candidate_rates, label) in enumerate(random_candidates):
        result = await evaluator.evaluate(candidate_rates, label=label)
        total_evals += 1
        random_results.append(result)

        if (idx + 1) % 10 == 0 or idx == 0 or idx == len(random_candidates) - 1:
            best_random_so_far = max(random_results, key=lambda r: r.mean_swf)
            print(f"  [{idx+1}/{args.num_random}] Best random SWF so far: "
                  f"{best_random_so_far.mean_swf:.2f}  (current: {result.mean_swf:.2f})")

        all_results.append({
            "phase": "random_search",
            "label": label,
            "tax_rates": result.tax_rates,
            "mean_swf": result.mean_swf,
            "std_swf": result.std_swf,
            "eval_time": result.eval_time,
        })

    best_random = max(random_results, key=lambda r: r.mean_swf)

    # ── 4. Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  ICRL baseline SWF:        {baseline_result.mean_swf:.2f}")
    print(f"  Best perturbation SWF:    {best_perturbation_result.mean_swf:.2f} "
          f"(delta={best_perturbation_result.mean_swf - baseline_result.mean_swf:+.2f})")
    print(f"  Best perturbation rates:  "
          f"{[f'{r:.3f}' for r in best_perturbation_result.tax_rates]}")
    print(f"  Best random SWF:          {best_random.mean_swf:.2f} "
          f"(delta={best_random.mean_swf - baseline_result.mean_swf:+.2f})")
    print(f"  Best random rates:        "
          f"{[f'{r:.3f}' for r in best_random.tax_rates]}")
    print()

    n_perturb_evals = len(candidates_pass1) + 2  # +1 baseline, +1 composite
    if args.fine_pass:
        n_perturb_evals += len(candidates_pass2) + 1
    print(f"  Perturbation evaluations: {n_perturb_evals}")
    print(f"  Random evaluations:       {args.num_random}")
    print(f"  Total evaluations:        {total_evals}")

    icrl_is_local_opt = (best_perturbation_result.mean_swf
                         <= baseline_result.mean_swf + baseline_result.std_swf)
    print(f"\n  ICRL is at local optimum: {'YES' if icrl_is_local_opt else 'NO'}")
    print(f"  (within 1-sigma of baseline: +/- {baseline_result.std_swf:.2f})")

    # ── 5. Build output ────────────────────────────────────────────────
    output_data = {
        "experiment": "perturbation_refinement",
        "icrl_source": {
            "results_dir": args.results_dir,
            "result_file": args.result_file,
            "config": icrl_config,
            "reported_swf": icrl_swf,
        },
        "eval_config": {
            "model": args.model,
            "num_agents": args.num_agents,
            "eval_timesteps": args.eval_timesteps,
            "avg_window": args.avg_window,
            "seed": args.seed,
            "perturbation_range_pp": perturbation_pps,
            "fine_pass": args.fine_pass,
            "fine_range_pp": [float(x) for x in args.fine_range.split(",")]
                if args.fine_pass else None,
            "num_random": args.num_random,
        },
        "summary": {
            "icrl_baseline_swf": baseline_result.mean_swf,
            "icrl_baseline_std": baseline_result.std_swf,
            "best_perturbation_swf": best_perturbation_result.mean_swf,
            "best_perturbation_rates": best_perturbation_result.tax_rates,
            "best_perturbation_delta": (best_perturbation_result.mean_swf
                                        - baseline_result.mean_swf),
            "best_random_swf": best_random.mean_swf,
            "best_random_rates": best_random.tax_rates,
            "best_random_delta": best_random.mean_swf - baseline_result.mean_swf,
            "icrl_is_local_optimum": icrl_is_local_opt,
            "total_evaluations": total_evals,
            "perturbation_evaluations": n_perturb_evals,
            "random_evaluations": args.num_random,
        },
        "all_results": all_results,
        # Per-bracket heatmap data for visualization
        "heatmap_data": _build_heatmap_data(
            base_rates, pass1_results, baseline_result,
        ),
    }

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # ── 6. Generate plots (optional) ───────────────────────────────────
    if args.plot:
        _generate_plots(
            output_data,
            baseline_result,
            pass1_results,
            random_results,
            output_path,
        )

    # Shutdown engine
    await engine.shutdown()

    return output_data


def _build_heatmap_data(
    base_rates: List[float],
    pass1_results: List[EvalResult],
    baseline: EvalResult,
) -> Dict[str, Any]:
    """
    Build per-bracket heatmap data: SWF as a function of perturbation for
    each bracket index.  Suitable for a paper figure.
    """
    heatmap = {}
    num_brackets = len(base_rates)

    for b_idx in range(num_brackets):
        bracket_data = {
            "bracket_index": b_idx,
            "base_rate": base_rates[b_idx],
            "perturbations": [],
        }

        # Add the baseline (0 perturbation)
        bracket_data["perturbations"].append({
            "perturbation_pp": 0.0,
            "rate": base_rates[b_idx],
            "mean_swf": baseline.mean_swf,
            "std_swf": baseline.std_swf,
        })

        # Add each perturbation result for this bracket
        for r in pass1_results:
            if r.label.startswith(f"bracket_{b_idx}_"):
                # Extract perturbation magnitude from label
                parts = r.label.split("_")
                pp_str = parts[-1].replace("pp", "")
                pp = float(pp_str)
                bracket_data["perturbations"].append({
                    "perturbation_pp": pp,
                    "rate": r.tax_rates[b_idx],
                    "mean_swf": r.mean_swf,
                    "std_swf": r.std_swf,
                })

        # Sort by perturbation magnitude
        bracket_data["perturbations"].sort(key=lambda x: x["perturbation_pp"])
        heatmap[f"bracket_{b_idx}"] = bracket_data

    return heatmap


def _generate_plots(
    output_data: Dict,
    baseline: EvalResult,
    pass1_results: List[EvalResult],
    random_results: List[EvalResult],
    output_path: Path,
):
    """Generate matplotlib figures suitable for a paper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = output_path.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    num_brackets = len(baseline.tax_rates)

    # ── Figure 1: Per-bracket perturbation sensitivity ──────────────────
    n_cols = min(4, num_brackets)
    n_rows = (num_brackets + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if num_brackets == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    heatmap = output_data["heatmap_data"]
    for b_idx in range(num_brackets):
        ax = axes[b_idx]
        bdata = heatmap[f"bracket_{b_idx}"]
        pps = [p["perturbation_pp"] for p in bdata["perturbations"]]
        swfs = [p["mean_swf"] for p in bdata["perturbations"]]
        stds = [p["std_swf"] for p in bdata["perturbations"]]

        ax.errorbar(pps, swfs, yerr=stds, marker="o", capsize=3,
                    linewidth=1.5, markersize=5)
        ax.axhline(baseline.mean_swf, color="gray", linestyle="--",
                   alpha=0.6, label="ICRL baseline")
        ax.axvline(0, color="gray", linestyle=":", alpha=0.4)
        ax.set_xlabel("Perturbation (pp)")
        ax.set_ylabel("SWF")
        ax.set_title(f"Bracket {b_idx}\n(base rate {bdata['base_rate']*100:.1f}%)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # Hide unused axes
    for i in range(num_brackets, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("SWF Sensitivity to Per-Bracket Tax Rate Perturbations",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = fig_dir / "perturbation_sensitivity.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"  Saved: {fig_path}")
    plt.close(fig)

    # ── Figure 2: Perturbation vs Random vs ICRL comparison ────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    # Random search distribution
    random_swfs = sorted([r.mean_swf for r in random_results])
    ax.hist(random_swfs, bins=min(20, len(random_swfs)),
            alpha=0.5, color="steelblue", label="Random search", edgecolor="white")

    # ICRL baseline
    ax.axvline(baseline.mean_swf, color="red", linewidth=2,
               linestyle="--", label=f"ICRL baseline ({baseline.mean_swf:.1f})")

    # Best perturbation
    best_pert_swf = output_data["summary"]["best_perturbation_swf"]
    ax.axvline(best_pert_swf, color="green", linewidth=2,
               linestyle="-.", label=f"Best perturbation ({best_pert_swf:.1f})")

    # Best random
    best_rand_swf = output_data["summary"]["best_random_swf"]
    ax.axvline(best_rand_swf, color="orange", linewidth=2,
               linestyle=":", label=f"Best random ({best_rand_swf:.1f})")

    ax.set_xlabel("Social Welfare Function (SWF)")
    ax.set_ylabel("Count")
    ax.set_title("ICRL vs Perturbation Refinement vs Random Search")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig_path = fig_dir / "method_comparison.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"  Saved: {fig_path}")
    plt.close(fig)

    # ── Figure 3: SWF vs perturbation magnitude (aggregated) ───────────
    fig, ax = plt.subplots(figsize=(7, 5))

    # Collect all coordinate-wise perturbation results grouped by |delta|
    from collections import defaultdict
    magnitude_groups = defaultdict(list)
    for r in pass1_results:
        parts = r.label.split("_")
        pp_str = parts[-1].replace("pp", "")
        pp = float(pp_str)
        magnitude_groups[abs(pp)].append(r.mean_swf)

    magnitudes = sorted(magnitude_groups.keys())
    mean_swfs = [np.mean(magnitude_groups[m]) for m in magnitudes]
    std_swfs = [np.std(magnitude_groups[m]) for m in magnitudes]
    min_swfs = [np.min(magnitude_groups[m]) for m in magnitudes]
    max_swfs = [np.max(magnitude_groups[m]) for m in magnitudes]

    ax.errorbar(magnitudes, mean_swfs, yerr=std_swfs, marker="s", capsize=4,
                linewidth=2, markersize=6, color="navy", label="Mean +/- 1 std")
    ax.fill_between(magnitudes, min_swfs, max_swfs, alpha=0.15, color="navy",
                    label="Min-max range")
    ax.axhline(baseline.mean_swf, color="red", linestyle="--",
               linewidth=1.5, label=f"ICRL baseline ({baseline.mean_swf:.1f})")

    ax.set_xlabel("Perturbation Magnitude (percentage points)")
    ax.set_ylabel("Social Welfare Function (SWF)")
    ax.set_title("SWF vs Tax Rate Perturbation Magnitude")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig_path = fig_dir / "swf_vs_perturbation_magnitude.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"  Saved: {fig_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc perturbation refinement for ICRL tax policies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    parser.add_argument(
        "--results-dir", type=str, default="results/bounded_100x2000",
        help="Directory containing ICRL result JSON files",
    )
    parser.add_argument(
        "--result-file", type=str, required=True,
        help="Specific ICRL result file to refine (e.g. mistral-7b-v0.3_seed42.json)",
    )

    # Model / Inference
    parser.add_argument(
        "--model", type=str, default="mistral-7b-v0.3",
        help="Worker LLM model name (must match a key in SUPPORTED_MODELS or MODEL_ALIASES)",
    )
    parser.add_argument(
        "--quantization", type=str, default="awq",
        choices=["awq", "gptq", "fp8", "bitsandbytes", "none"],
        help="Quantization method",
    )
    parser.add_argument(
        "--tensor-parallel", type=int, default=1,
        help="Tensor parallel size (GPUs)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Batch size for inference",
    )

    # Evaluation
    parser.add_argument(
        "--num-agents", type=int, default=100,
        help="Number of worker agents per evaluation",
    )
    parser.add_argument(
        "--eval-timesteps", type=int, default=50,
        help="Timesteps per evaluation run",
    )
    parser.add_argument(
        "--avg-window", type=int, default=20,
        help="Number of final timesteps to average SWF over",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )

    # Perturbation config
    parser.add_argument(
        "--perturbation-range", type=str, default="-10,-5,-2,0,2,5,10",
        help="Comma-separated perturbation magnitudes in percentage points",
    )
    parser.add_argument(
        "--fine-pass", action="store_true",
        help="Enable second fine-grained perturbation pass",
    )
    parser.add_argument(
        "--fine-range", type=str, default="-3,-1,0,1,3",
        help="Comma-separated fine-grained perturbation magnitudes (pp)",
    )

    # Random search
    parser.add_argument(
        "--num-random", type=int, default=50,
        help="Number of random tax policies to evaluate",
    )

    # Output
    parser.add_argument(
        "--output", type=str,
        default="results/perturbation_refinement/result.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate matplotlib figures",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()
    asyncio.run(run_perturbation_experiment(args))


if __name__ == "__main__":
    main()
