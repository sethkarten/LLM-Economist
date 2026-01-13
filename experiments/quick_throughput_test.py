#!/usr/bin/env python3
"""
Quick throughput test - run this once vLLM is set up to get actual numbers.

Usage:
    # Start vLLM server first:
    vllm serve Qwen/Qwen3-30B-A3B-Instruct --tensor-parallel-size 2 --quantization awq

    # Then run this test:
    python experiments/quick_throughput_test.py --batch-sizes 1,10,50,100,200
"""

import argparse
import asyncio
import time
from typing import List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_batch_throughput(
    model_name: str,
    base_url: str,
    batch_sizes: List[int],
    requests_per_batch: int = 3
):
    """Test throughput at different batch sizes."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key="dummy",
        base_url=f"{base_url}/v1"
    )

    # Standard worker prompt
    system = "You are an economic agent choosing labor hours. Respond with JSON: {\"labor_hours\": <0-100>}"
    user = "Your skill=$50/hr, tax=22%. Current hours=40, income=$2000. Choose new labor hours."

    results = []

    print(f"\n{'='*70}")
    print(f"THROUGHPUT TEST: {model_name}")
    print(f"{'='*70}\n")

    for batch_size in batch_sizes:
        print(f"\nTesting batch_size={batch_size}...")

        latencies = []

        for trial in range(requests_per_batch):
            # Create batch of requests
            tasks = []
            start = time.time()

            for i in range(batch_size):
                task = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    max_tokens=64,
                    temperature=0.7,
                )
                tasks.append(task)

            # Wait for all to complete
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            elapsed = time.time() - start
            latencies.append(elapsed)

            successes = sum(1 for r in responses if not isinstance(r, Exception))
            print(f"  Trial {trial+1}: {successes}/{batch_size} succeeded in {elapsed:.2f}s")

        avg_batch_time = sum(latencies) / len(latencies)
        throughput = batch_size / avg_batch_time

        results.append({
            "batch_size": batch_size,
            "avg_batch_time_seconds": avg_batch_time,
            "throughput_requests_per_second": throughput,
        })

        print(f"  -> Avg batch time: {avg_batch_time:.2f}s, Throughput: {throughput:.1f} req/s")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Batch Size':>12} | {'Batch Time':>12} | {'Throughput':>15} | {'1000 agents/step':>18}")
    print("-" * 70)

    for r in results:
        time_for_1000 = 1000 / r["throughput_requests_per_second"]
        print(f"{r['batch_size']:>12} | {r['avg_batch_time_seconds']:>10.2f}s | {r['throughput_requests_per_second']:>12.1f} req/s | {time_for_1000:>16.1f}s")

    # Recommendation
    best = max(results, key=lambda x: x["throughput_requests_per_second"])
    print(f"\n{'='*70}")
    print(f"BEST: batch_size={best['batch_size']} -> {best['throughput_requests_per_second']:.1f} req/s")
    print(f"{'='*70}")

    # Project for 1000 agents
    throughput = best["throughput_requests_per_second"]
    time_per_1000_step = 1000 / throughput

    print(f"\nPROJECTIONS FOR 1000 AGENTS (at {throughput:.0f} req/s):")
    for steps in [1000, 2000, 3000, 5000]:
        total_time = (steps * time_per_1000_step) / 3600
        tax_years = steps / 128
        print(f"  {steps:,} steps ({tax_years:.0f} tax years): {total_time:.1f} hours")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--batch-sizes", default="1,10,50,100")
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    asyncio.run(test_batch_throughput(
        args.model,
        args.url,
        batch_sizes,
        args.trials
    ))


if __name__ == "__main__":
    main()
