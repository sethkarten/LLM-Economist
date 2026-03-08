#!/usr/bin/env python3
"""Run a single ICRL simulation (for GPU manager submission).

Usage:
    python experiments/run_icrl_single.py --seed 42 --output results/icrl_v2/seed42.json
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


async def main_async(args):
    from llm_economist.main_async import AsyncLLMEconomist

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    sim = AsyncLLMEconomist(
        num_agents=args.num_agents,
        max_timesteps=args.max_timesteps,
        model_name=args.model,
        tensor_parallel_size=1,
        tax_year_length=args.tax_year_length,
        scenario="bounded",
        quantization="awq",
        batch_size=args.num_agents,
        seed=args.seed,
        external_planner=False,  # Use LLM planner (ICRL)
        bracket_setting=args.bracket_setting,
        history_len=args.history_len,
        gpu_memory_utilization=0.95,
        max_model_len=args.max_model_len,
    )
    await sim.initialize()

    t0 = time.time()
    for step in range(args.max_timesteps):
        await sim.step()

    elapsed = time.time() - t0
    print(f"\nCompleted {args.max_timesteps} steps in {elapsed/60:.1f} min")
    print(f"Final SWF: {sim.state.swf:.4f}")

    if args.output:
        sim.save_results(args.output)

    if sim.engine:
        await sim.engine.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-agents', type=int, default=100)
    parser.add_argument('--max-timesteps', type=int, default=2000)
    parser.add_argument('--tax-year-length', type=int, default=25)
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-8B-AWQ')
    parser.add_argument('--bracket-setting', type=str, default='three')
    parser.add_argument('--history-len', type=int, default=64)
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
