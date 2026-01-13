#!/usr/bin/env python3
"""
Run bounded rationality experiments with 6 models × 3 seeds.

Usage:
    python experiments/run_bounded_experiments.py --all          # Run all experiments
    python experiments/run_bounded_experiments.py --model gemma3-4b  # Run specific model
    python experiments/run_bounded_experiments.py --list         # List experiments
    python experiments/run_bounded_experiments.py --estimate     # Estimate runtime
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Model configurations for local RTX 5090 experiments
# Ordered by speed (fastest first)
# Model names must match keys in llm_economist/inference/config.py SUPPORTED_MODELS
MODELS = {
    "gemma3-4b": {
        "config_name": "gemma3-4b",  # Key in SUPPORTED_MODELS
        "quantization": None,  # BF16 (uses recommended_quantization from config)
        "req_per_sec": 92.0,
        "description": "Gemma-3-4B BF16 text-only (FASTEST)"
    },
    "mistral-7b-v0.3": {
        "config_name": "mistral-7b-v0.3",
        "quantization": "awq",
        "req_per_sec": 67.1,
        "description": "Mistral-7B AWQ"
    },
    "llama-3.1-8b": {
        "config_name": "llama-3.1-8b",
        "quantization": "awq",
        "req_per_sec": 65.9,
        "description": "Llama-3.1-8B AWQ"
    },
    "qwen3-8b": {
        "config_name": "qwen3-8b",
        "quantization": "awq",
        "req_per_sec": 60.5,
        "description": "Qwen3-8B AWQ"
    },
    "olmo3-7b": {
        "config_name": "olmo3-7b",
        "quantization": "fp8",
        "req_per_sec": 31.5,
        "description": "OLMo-3-7B FP8 (slowest, fully open)"
    },
}

# Experiment parameters
NUM_AGENTS = 100
MAX_TIMESTEPS = 2000
TAX_YEAR_LENGTH = 128  # Planner updates taxes every 128 steps (~15 updates per run)
SEEDS = [42, 123, 456]
SCENARIO = "bounded"

# Estimated requests per run: agents × timesteps
REQUESTS_PER_RUN = NUM_AGENTS * MAX_TIMESTEPS


def estimate_runtime(model_name: str) -> float:
    """Estimate runtime in minutes for one seed."""
    model = MODELS[model_name]
    req_per_sec = model["req_per_sec"]
    runtime_sec = REQUESTS_PER_RUN / req_per_sec
    # Add 20% overhead for model loading, tax planner, etc.
    runtime_sec *= 1.2
    return runtime_sec / 60


def get_output_path(model_name: str, seed: int) -> Path:
    """Get output path for experiment results."""
    output_dir = Path(__file__).parent.parent / "results" / "bounded_100x2000"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{model_name}_seed{seed}.json"


def check_completed(model_name: str, seed: int) -> bool:
    """Check if experiment already completed."""
    output_path = get_output_path(model_name, seed)
    if output_path.exists():
        try:
            with open(output_path) as f:
                data = json.load(f)
            # Check if simulation completed all steps
            if len(data.get("metrics_history", [])) >= MAX_TIMESTEPS:
                return True
        except (json.JSONDecodeError, KeyError):
            pass
    return False


def list_experiments():
    """List all experiments and their status."""
    print(f"\n{'='*70}")
    print(f"Bounded Rationality Experiments: {NUM_AGENTS} agents × {MAX_TIMESTEPS} steps")
    print(f"{'='*70}")
    print(f"\n{'Model':<15} {'Speed':<12} {'Est. Time':<12} {'Seeds Status'}")
    print(f"{'-'*70}")

    total_remaining = 0
    for model_name, model in MODELS.items():
        est_time = estimate_runtime(model_name)
        seeds_status = []
        for seed in SEEDS:
            if check_completed(model_name, seed):
                seeds_status.append(f"✓{seed}")
            else:
                seeds_status.append(f"○{seed}")
                total_remaining += est_time

        print(f"{model_name:<15} {model['req_per_sec']:<12.1f} {est_time:<12.1f} {' '.join(seeds_status)}")

    print(f"\n{'='*70}")
    print(f"Total remaining time: {total_remaining:.1f} min ({total_remaining/60:.1f} hours)")
    print(f"{'='*70}\n")


def estimate_all():
    """Print detailed runtime estimates."""
    print(f"\n{'='*70}")
    print(f"Runtime Estimates")
    print(f"{'='*70}")
    print(f"\nConfiguration:")
    print(f"  - Agents: {NUM_AGENTS}")
    print(f"  - Timesteps: {MAX_TIMESTEPS}")
    print(f"  - Requests per run: {REQUESTS_PER_RUN:,}")
    print(f"  - Seeds: {SEEDS}")
    print(f"  - Scenario: {SCENARIO}")

    print(f"\n{'Model':<15} {'req/s':<10} {'1 seed':<12} {'3 seeds':<12} {'Description'}")
    print(f"{'-'*80}")

    total_time = 0
    for model_name, model in MODELS.items():
        est_time = estimate_runtime(model_name)
        total_for_model = est_time * len(SEEDS)
        total_time += total_for_model
        print(f"{model_name:<15} {model['req_per_sec']:<10.1f} {est_time:<12.1f} {total_for_model:<12.1f} {model['description']}")

    print(f"{'-'*80}")
    print(f"{'TOTAL':<15} {'':<10} {'':<12} {total_time:<12.1f} minutes")
    print(f"{'TOTAL':<15} {'':<10} {'':<12} {total_time/60:<12.1f} hours")

    finish_time = datetime.now() + timedelta(minutes=total_time)
    print(f"\nEstimated completion: {finish_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}\n")


async def run_experiment(model_name: str, seed: int):
    """Run a single experiment."""
    import asyncio
    import gc
    from llm_economist.main_async import AsyncLLMEconomist

    model = MODELS[model_name]
    output_path = get_output_path(model_name, seed)

    print(f"\n{'='*60}")
    print(f"Running: {model_name} (seed={seed})")
    print(f"Config: {model['config_name']}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")

    start_time = time.time()

    # Get config name for model lookup
    config_model_name = model["config_name"]

    # Determine quantization
    quant = model["quantization"] if model["quantization"] else "none"

    # Create simulator with model name (uses config lookup internally)
    simulator = AsyncLLMEconomist(
        num_agents=NUM_AGENTS,
        max_timesteps=MAX_TIMESTEPS,
        model_name=config_model_name,
        tensor_parallel_size=1,
        tax_year_length=TAX_YEAR_LENGTH,
        scenario=SCENARIO,
        quantization=quant,
        seed=seed,
        debug=False,
    )

    # Run simulation
    await simulator.initialize()
    metrics = await simulator.run()

    # Save results
    simulator.save_results(str(output_path))

    elapsed = time.time() - start_time
    print(f"\nCompleted {model_name} (seed={seed}) in {elapsed/60:.1f} minutes")

    # Cleanup
    if simulator.engine:
        del simulator.engine
    del simulator

    # Force garbage collection
    gc.collect()

    try:
        import torch
        torch.cuda.empty_cache()
    except:
        pass

    return metrics


async def run_model(model_name: str):
    """Run all seeds for a single model."""
    import asyncio

    for seed in SEEDS:
        if check_completed(model_name, seed):
            print(f"Skipping {model_name} seed={seed} (already completed)")
            continue

        await run_experiment(model_name, seed)

        # Wait between runs for GPU to cool down
        print("Waiting 10s between runs...")
        await asyncio.sleep(10)


async def run_all():
    """Run all experiments."""

    for model_name in MODELS.keys():
        await run_model(model_name)

    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)
    list_experiments()


def main():
    parser = argparse.ArgumentParser(description="Run bounded rationality experiments")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--model", type=str, help="Run specific model")
    parser.add_argument("--list", action="store_true", help="List experiments and status")
    parser.add_argument("--estimate", action="store_true", help="Show runtime estimates")

    args = parser.parse_args()

    if args.list:
        list_experiments()
    elif args.estimate:
        estimate_all()
    elif args.model:
        if args.model not in MODELS:
            print(f"Unknown model: {args.model}")
            print(f"Available models: {list(MODELS.keys())}")
            sys.exit(1)
        import asyncio
        asyncio.run(run_model(args.model))
    elif args.all:
        import asyncio
        asyncio.run(run_all())
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python experiments/run_bounded_experiments.py --list")
        print("  python experiments/run_bounded_experiments.py --estimate")
        print("  python experiments/run_bounded_experiments.py --model gemma3-4b")
        print("  python experiments/run_bounded_experiments.py --all")


if __name__ == "__main__":
    main()
