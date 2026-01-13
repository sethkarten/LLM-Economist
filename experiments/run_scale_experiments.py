#!/usr/bin/env python3
"""
Run scale experiments for LLM Economist ICML/COLM 2026 submission.

Experiments are optimized for 6-hour budget on 2x RTX 5090.
Uses pareto-optimal configurations to maximize agent-steps product.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_economist.main_async import AsyncLLMEconomist
from llm_economist.inference.config import (
    calculate_pareto_configs,
    estimate_throughput,
    SUPPORTED_MODELS,
)


# Experiment configurations for 6-hour budget
EXPERIMENTS = {
    # Model comparison experiments (A-series)
    "A1_model_comparison_small": {
        "description": "Compare all models at small scale",
        "num_agents": 100,
        "max_timesteps": 1000,
        "models": ["qwen3-30b-a3b", "olmo3-32b-instruct", "nemotron-30b-a3b", "gemma3-27b"],
        "seeds": [42, 43, 44],
        "estimated_hours": 4.0,
    },
    "A2_model_comparison_medium": {
        "description": "Compare top models at medium scale",
        "num_agents": 1000,
        "max_timesteps": 500,
        "models": ["qwen3-30b-a3b", "nemotron-30b-a3b"],  # MoE models for speed
        "seeds": [42, 43, 44],
        "estimated_hours": 5.5,
    },
    "A3_scale_test_large": {
        "description": "Large scale with best MoE model",
        "num_agents": 5000,
        "max_timesteps": 200,
        "models": ["qwen3-30b-a3b"],
        "seeds": [42],
        "estimated_hours": 5.5,
    },
    "A4_scale_test_massive": {
        "description": "Massive scale headline number",
        "num_agents": 20000,
        "max_timesteps": 50,
        "models": ["qwen3-30b-a3b"],
        "seeds": [42],
        "estimated_hours": 5.5,
    },

    # Ablation experiments (B-series)
    "B1_quantization_ablation": {
        "description": "Compare quantization methods",
        "num_agents": 500,
        "max_timesteps": 200,
        "models": ["qwen3-30b-a3b"],
        "quantizations": ["awq", "gptq", "fp8"],
        "seeds": [42],
        "estimated_hours": 3.0,
    },
    "B2_tax_year_ablation": {
        "description": "Compare tax year lengths",
        "num_agents": 500,
        "max_timesteps": 1000,
        "models": ["qwen3-30b-a3b"],
        "tax_year_lengths": [32, 64, 128, 256],
        "seeds": [42],
        "estimated_hours": 4.0,
    },
    "B3_reasoning_comparison": {
        "description": "Compare standard vs thinking models",
        "num_agents": 500,
        "max_timesteps": 300,
        "models": ["olmo3-32b-instruct", "olmo3-32b-think"],
        "seeds": [42, 43],
        "estimated_hours": 5.0,
    },

    # Scenario experiments (C-series)
    "C1_scenario_comparison": {
        "description": "Compare rational vs bounded vs democratic",
        "num_agents": 500,
        "max_timesteps": 500,
        "models": ["qwen3-30b-a3b"],
        "scenarios": ["rational", "bounded", "democratic"],
        "seeds": [42, 43],
        "estimated_hours": 5.5,
    },
}


async def run_single_experiment(
    name: str,
    num_agents: int,
    max_timesteps: int,
    model: str,
    seed: int,
    output_dir: str,
    scenario: str = "bounded",
    quantization: str = "awq",
    tax_year_length: int = 128,
    tensor_parallel: int = 2,
) -> Dict[str, Any]:
    """Run a single experiment configuration."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"  Model: {model}")
    print(f"  Agents: {num_agents}, Steps: {max_timesteps}")
    print(f"  Scenario: {scenario}, Seed: {seed}")
    print(f"{'='*60}\n")

    simulator = AsyncLLMEconomist(
        num_agents=num_agents,
        max_timesteps=max_timesteps,
        model_name=model,
        tensor_parallel_size=tensor_parallel,
        tax_year_length=tax_year_length,
        scenario=scenario,
        quantization=quantization,
        seed=seed,
        debug=False,
    )

    await simulator.initialize()
    results = await simulator.run()

    # Save results
    output_file = os.path.join(output_dir, f"{name}_{model}_{seed}.json")
    simulator.save_results(output_file)

    return {
        "name": name,
        "model": model,
        "seed": seed,
        "final_swf": simulator.state.swf,
        "output_file": output_file,
    }


async def run_experiment_suite(
    experiment_name: str,
    output_dir: str,
    tensor_parallel: int = 2,
) -> List[Dict]:
    """Run a full experiment suite."""
    if experiment_name not in EXPERIMENTS:
        print(f"Unknown experiment: {experiment_name}")
        print(f"Available: {list(EXPERIMENTS.keys())}")
        return []

    config = EXPERIMENTS[experiment_name]
    print(f"\n{'#'*60}")
    print(f"# Experiment Suite: {experiment_name}")
    print(f"# {config['description']}")
    print(f"# Estimated time: {config['estimated_hours']} hours")
    print(f"{'#'*60}\n")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = os.path.join(output_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(suite_dir, exist_ok=True)

    results = []

    # Determine what to iterate over
    models = config.get("models", ["qwen3-30b-a3b"])
    seeds = config.get("seeds", [42])
    scenarios = config.get("scenarios", ["bounded"])
    quantizations = config.get("quantizations", ["awq"])
    tax_year_lengths = config.get("tax_year_lengths", [128])

    # Run all combinations
    for model in models:
        for seed in seeds:
            for scenario in scenarios:
                for quant in quantizations:
                    for tyl in tax_year_lengths:
                        run_name = f"{experiment_name}_m{model}_s{seed}_sc{scenario}_q{quant}_ty{tyl}"

                        try:
                            result = await run_single_experiment(
                                name=run_name,
                                num_agents=config["num_agents"],
                                max_timesteps=config["max_timesteps"],
                                model=model,
                                seed=seed,
                                output_dir=suite_dir,
                                scenario=scenario,
                                quantization=quant,
                                tax_year_length=tyl,
                                tensor_parallel=tensor_parallel,
                            )
                            results.append(result)
                        except Exception as e:
                            print(f"ERROR in {run_name}: {e}")
                            results.append({
                                "name": run_name,
                                "error": str(e),
                            })

    # Save summary
    summary_file = os.path.join(suite_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "experiment": experiment_name,
            "config": config,
            "results": results,
            "timestamp": timestamp,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Experiment suite complete!")
    print(f"Results saved to: {suite_dir}")
    print(f"Summary: {summary_file}")
    print(f"{'='*60}\n")

    return results


def show_pareto_analysis():
    """Display pareto-optimal configurations for 6-hour budget."""
    print("\n" + "="*70)
    print("PARETO-OPTIMAL CONFIGURATIONS FOR 6-HOUR BUDGET")
    print("="*70 + "\n")

    for model_name in ["qwen3-30b-a3b", "nemotron-30b-a3b", "olmo3-32b-instruct"]:
        try:
            print(f"\n--- {model_name} ---")

            # Get throughput estimate
            estimates = estimate_throughput(model_name, num_gpus=2)
            print(f"Throughput: {estimates['requests_per_second']:.1f} req/s")
            print(f"Architecture: {estimates['model_architecture']} ({estimates['model_active_params']}B active)")

            # Get pareto configs
            configs = calculate_pareto_configs(
                time_budget_hours=6.0,
                model_name=model_name,
                num_gpus=2,
            )

            print(f"\nTop configurations:")
            for i, cfg in enumerate(configs[:5]):
                print(f"  {i+1}. {cfg['num_agents']:>6,} agents x {cfg['max_steps']:>4,} steps "
                      f"= {cfg['agent_step_product']:>10,} agent-steps "
                      f"(~{cfg['estimated_time_hours']:.1f}h)")

        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "="*70)
    print("RECOMMENDED EXPERIMENTS FOR PAPER:")
    print("="*70)
    print("""
    1. MODEL COMPARISON (A1): 100 agents x 1000 steps x 4 models x 3 seeds
       - Establishes baseline across model families
       - ~4 hours total

    2. SCALE DEMONSTRATION (A4): 20,000 agents x 50 steps
       - Headline "massive scale" result
       - ~5.5 hours

    3. SCENARIO COMPARISON (C1): 500 agents x 500 steps x 3 scenarios
       - Shows rational vs bounded vs democratic
       - ~5.5 hours

    Pick 1-2 experiments per day to stay within 6-hour budget.
    """)


def main():
    parser = argparse.ArgumentParser(description="Run LLM Economist experiments")

    parser.add_argument("--experiment", "-e", type=str,
                       choices=list(EXPERIMENTS.keys()) + ["all", "pareto"],
                       help="Experiment to run (or 'pareto' for analysis)")
    parser.add_argument("--output-dir", "-o", type=str, default="results",
                       help="Output directory")
    parser.add_argument("--tensor-parallel", "-tp", type=int, default=2,
                       help="Tensor parallel size (GPUs)")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available experiments")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable experiments:")
        for name, config in EXPERIMENTS.items():
            print(f"\n  {name}:")
            print(f"    {config['description']}")
            print(f"    Agents: {config['num_agents']}, Steps: {config['max_timesteps']}")
            print(f"    Estimated time: {config['estimated_hours']} hours")
        return

    if args.experiment == "pareto":
        show_pareto_analysis()
        return

    if not args.experiment:
        parser.print_help()
        return

    # Run experiment
    asyncio.run(run_experiment_suite(
        args.experiment,
        args.output_dir,
        args.tensor_parallel,
    ))


if __name__ == "__main__":
    main()
