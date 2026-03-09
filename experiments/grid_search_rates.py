#!/usr/bin/env python3
"""Grid search over 3-bracket tax rates to establish exploration baselines.

Tests a set of fixed tax rate schedules (including non-monotonic/U-shaped patterns)
on the same population to determine if the REINFORCE++ planner is missing better solutions.

Usage:
    uv run python experiments/grid_search_rates.py \
        --seed 42 --max-timesteps 512 --worker-gpu 1 \
        --output results/grid_search/seed42.json
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from itertools import product

import numpy as np


# Rate schedules to test (3-bracket: [low, mid, high])
# Focused set: 18 schedules covering progressive, regressive, flat, and U-shaped
RATE_SCHEDULES = {
    # Baselines
    "us_federal":        [0.12, 0.24, 0.35],
    "flat_30":           [0.30, 0.30, 0.30],

    # REINFORCE++ discovered rates
    "reinforce_seed42":  [0.20, 0.35, 0.50],
    "reinforce_seed123": [0.30, 0.60, 0.90],

    # Higher progressive (does more redistribution help?)
    "prog_high":         [0.20, 0.40, 0.60],
    "prog_extreme":      [0.40, 0.60, 0.80],
    "max_progressive":   [0.50, 0.75, 0.99],

    # U-shaped (does Saez-style work better?)
    "u_shape_med":       [0.40, 0.20, 0.50],
    "u_shape_strong":    [0.50, 0.25, 0.60],

    # Regressive / decreasing from high (does direction matter?)
    "regressive":        [0.40, 0.30, 0.20],
    "regressive_high":   [0.90, 0.75, 0.50],
    "regressive_med":    [0.80, 0.60, 0.40],
    "regressive_low":    [0.70, 0.50, 0.30],

    # Top-heavy (Saez-computed optimal for this SWF)
    "top_heavy":         [0.05, 0.10, 0.60],
    "top_heavy_2":       [0.10, 0.20, 0.70],

    # Nearly-max redistribution
    "high_flat":         [0.80, 0.80, 0.80],
    "near_max":          [0.90, 0.90, 0.90],
    "max_all":           [0.99, 0.99, 0.99],
}


async def run_fixed_rates(sim, rates, max_timesteps, tax_year_length):
    """Run simulation with fixed tax rates and return final SWF/Gini."""
    sim.reset_state()
    sim.set_tax_rates(rates)

    for step in range(max_timesteps):
        if step % tax_year_length == 0 and step > 0:
            sim.set_tax_rates(rates)
        await sim.step()

    final_swf = sim.state.swf
    final_gini = sim._calculate_gini([a.income for a in sim.state.agent_states])
    mean_income = float(np.mean([a.income for a in sim.state.agent_states]))

    return {
        'swf': final_swf,
        'gini': final_gini,
        'mean_income': mean_income,
    }


async def main_async(args):
    from llm_economist.main_async import AsyncLLMEconomist

    original_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.worker_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.worker_gpu

    sim = AsyncLLMEconomist(
        num_agents=args.num_agents,
        max_timesteps=args.max_timesteps,
        model_name=args.worker_model,
        tensor_parallel_size=1,
        tax_year_length=args.tax_year_length,
        scenario="bounded",
        quantization="awq",
        batch_size=args.num_agents,
        seed=args.seed,
        external_planner=True,
        bracket_setting="three",
        gpu_memory_utilization=0.95,
        history_len=args.history_len,
    )
    await sim.initialize()

    if args.worker_gpu is not None:
        if original_cuda is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    results = {}
    schedules = list(RATE_SCHEDULES.items())
    total = len(schedules)

    print(f"\n=== Grid search: {total} rate schedules, {args.max_timesteps} steps each ===\n")

    for i, (name, rates) in enumerate(schedules):
        t0 = time.time()
        result = await run_fixed_rates(sim, rates, args.max_timesteps, args.tax_year_length)
        elapsed = time.time() - t0

        results[name] = {
            'rates': rates,
            **result,
        }

        rates_str = str([f'{r*100:.0f}%' for r in rates])
        print(f"[{i+1}/{total}] {name:25s} rates={rates_str:30s} "
              f"SWF={result['swf']:8.2f}  Gini={result['gini']:.3f}  "
              f"income=${result['mean_income']:,.0f}  ({elapsed:.1f}s)")

    # Sort by SWF
    print("\n=== Results ranked by SWF ===\n")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['swf'], reverse=True)
    for rank, (name, r) in enumerate(sorted_results, 1):
        marker = " <-- REINFORCE++" if "reinforce" in name else ""
        marker = " <-- US Federal" if name == "us_federal" else marker
        print(f"  {rank:2d}. {name:25s} SWF={r['swf']:8.2f}  Gini={r['gini']:.3f}  "
              f"rates={[f'{x*100:.0f}%' for x in r['rates']]}{marker}")

    # Shutdown
    if sim.engine:
        await sim.engine.shutdown()

    return {
        'type': 'grid_search',
        'seed': args.seed,
        'num_agents': args.num_agents,
        'max_timesteps': args.max_timesteps,
        'tax_year_length': args.tax_year_length,
        'worker_model': args.worker_model,
        'num_schedules': total,
        'results': results,
        'ranking': [{'rank': i+1, 'name': name, **r} for i, (name, r) in enumerate(sorted_results)],
    }


def main():
    parser = argparse.ArgumentParser(description='Grid search over 3-bracket tax rates')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-agents', type=int, default=100)
    parser.add_argument('--max-timesteps', type=int, default=256,
                        help='Steps per schedule (256=2 tax years, ~15min each)')
    parser.add_argument('--tax-year-length', type=int, default=128)
    parser.add_argument('--worker-model', type=str, default='Qwen/Qwen3-8B-AWQ')
    parser.add_argument('--history-len', type=int, default=5,
                        help='Number of recent timesteps to include in worker prompts')
    parser.add_argument('--worker-gpu', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    result = asyncio.run(main_async(args))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
