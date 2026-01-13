#!/usr/bin/env python3
"""
Benchmark throughput for LLM Economist to determine realistic experiment configs.

Tests:
1. Single agent latency
2. Batch throughput at different sizes
3. Projected time for various agent counts

Run with:
    python experiments/benchmark_throughput.py --model qwen3-30b-a3b --gpu-count 2
"""

import argparse
import asyncio
import time
import sys
from pathlib import Path
from typing import List, Dict
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def benchmark_sync_vllm(model_name: str, num_requests: int = 100, port: int = 8000):
    """Benchmark synchronous vLLM calls."""
    from llm_economist.models.vllm_model import VLLMModel

    model = VLLMModel(
        model_name=model_name,
        base_url=f"http://localhost:{port}",
        max_tokens=256,
        temperature=0.7
    )

    # Test prompt similar to worker decision
    system_prompt = """You are an economic agent choosing labor hours.
Your skill level is $50/hour. Current tax rate is 22%."""

    user_prompt = """Given your situation, choose labor hours (0-100).
Respond with JSON: {"labor_hours": <number>, "reasoning": "<brief>"}"""

    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"Requests: {num_requests}")
    print(f"{'='*60}\n")

    # Warmup
    print("Warming up...")
    for _ in range(3):
        try:
            model.send_msg(system_prompt, user_prompt, json_format=True)
        except Exception as e:
            print(f"Warmup error: {e}")
            return None

    # Benchmark
    latencies = []
    print(f"Running {num_requests} requests...")

    start_total = time.time()
    for i in range(num_requests):
        start = time.time()
        try:
            response, _ = model.send_msg(system_prompt, user_prompt, json_format=True)
            latency = time.time() - start
            latencies.append(latency)

            if (i + 1) % 10 == 0:
                avg_so_far = sum(latencies) / len(latencies)
                print(f"  {i+1}/{num_requests} - avg latency: {avg_so_far:.3f}s")
        except Exception as e:
            print(f"  Request {i+1} failed: {e}")
            latencies.append(10.0)  # Penalty for failed request

    total_time = time.time() - start_total

    # Calculate stats
    avg_latency = sum(latencies) / len(latencies)
    throughput = num_requests / total_time

    results = {
        "model": model_name,
        "num_requests": num_requests,
        "total_time_seconds": total_time,
        "avg_latency_seconds": avg_latency,
        "min_latency": min(latencies),
        "max_latency": max(latencies),
        "throughput_requests_per_second": throughput,
    }

    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Avg latency: {avg_latency:.3f}s")
    print(f"  Min/Max latency: {min(latencies):.3f}s / {max(latencies):.3f}s")
    print(f"  Throughput: {throughput:.2f} req/s")
    print(f"{'='*60}\n")

    return results


def project_experiment_times(throughput: float, tax_year_length: int = 128):
    """Project experiment times based on measured throughput."""
    print(f"\n{'='*60}")
    print("PROJECTED EXPERIMENT TIMES")
    print(f"Based on throughput: {throughput:.2f} req/s")
    print(f"Tax year length: {tax_year_length} steps")
    print(f"{'='*60}\n")

    # Different configurations
    configs = [
        (100, 3000),    # 100 agents, 3000 steps (~23 tax years)
        (100, 5000),    # 100 agents, 5000 steps (~39 tax years)
        (500, 2000),    # 500 agents, 2000 steps (~15 tax years)
        (500, 3000),    # 500 agents, 3000 steps
        (1000, 1000),   # 1000 agents, 1000 steps (~8 tax years)
        (1000, 2000),   # 1000 agents, 2000 steps (~15 tax years)
        (1000, 3000),   # 1000 agents, 3000 steps (~23 tax years)
        (2000, 1000),   # 2000 agents, 1000 steps
        (2000, 2000),   # 2000 agents, 2000 steps
        (5000, 500),    # 5000 agents, 500 steps
        (5000, 1000),   # 5000 agents, 1000 steps
        (10000, 500),   # 10000 agents, 500 steps
        (20000, 200),   # 20000 agents, 200 steps
    ]

    print(f"{'Agents':>8} | {'Steps':>6} | {'Tax Years':>10} | {'Agent-Steps':>12} | {'Est. Time':>12} | {'Fits 6h?':>8}")
    print("-" * 80)

    results = []
    for agents, steps in configs:
        tax_years = steps / tax_year_length
        agent_steps = agents * steps
        # Time = agent_steps / throughput (each agent-step needs one LLM call)
        time_seconds = agent_steps / throughput
        time_hours = time_seconds / 3600
        fits_budget = "YES" if time_hours <= 6.0 else "NO"

        print(f"{agents:>8} | {steps:>6} | {tax_years:>10.1f} | {agent_steps:>12,} | {time_hours:>10.2f}h | {fits_budget:>8}")

        results.append({
            "agents": agents,
            "steps": steps,
            "tax_years": tax_years,
            "agent_steps": agent_steps,
            "estimated_hours": time_hours,
            "fits_6h_budget": time_hours <= 6.0
        })

    # Find best configs that fit 6h and have >= 15 tax years
    print(f"\n{'='*60}")
    print("RECOMMENDED CONFIGS (6h budget, >= 15 tax years)")
    print(f"{'='*60}\n")

    good_configs = [c for c in results if c["fits_6h_budget"] and c["tax_years"] >= 15]
    good_configs.sort(key=lambda x: x["agent_steps"], reverse=True)

    for i, c in enumerate(good_configs[:5]):
        print(f"{i+1}. {c['agents']:,} agents x {c['steps']:,} steps = {c['agent_steps']:,} agent-steps (~{c['estimated_hours']:.1f}h)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM Economist throughput")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Model name")
    parser.add_argument("--port", type=int, default=8000,
                       help="vLLM server port")
    parser.add_argument("--num-requests", type=int, default=50,
                       help="Number of requests for benchmark")
    parser.add_argument("--tax-year-length", type=int, default=128,
                       help="Tax year length in steps")
    parser.add_argument("--skip-benchmark", action="store_true",
                       help="Skip benchmark, use estimated throughput")
    parser.add_argument("--estimated-throughput", type=float, default=None,
                       help="Use this throughput instead of benchmarking")
    parser.add_argument("--output", type=str, default=None,
                       help="Save results to JSON file")

    args = parser.parse_args()

    if args.skip_benchmark or args.estimated_throughput:
        # Use estimated throughput
        if args.estimated_throughput:
            throughput = args.estimated_throughput
        else:
            # Default estimates based on model type
            if "30b" in args.model.lower() or "32b" in args.model.lower():
                throughput = 5.0  # Conservative estimate for 30B models
            elif "8b" in args.model.lower():
                throughput = 15.0  # 8B models are faster
            else:
                throughput = 3.0  # Very conservative

        print(f"Using estimated throughput: {throughput} req/s")
        results = {"throughput_requests_per_second": throughput}
    else:
        # Run actual benchmark
        results = benchmark_sync_vllm(
            model_name=args.model,
            num_requests=args.num_requests,
            port=args.port
        )

        if results is None:
            print("\nBenchmark failed. Make sure vLLM server is running:")
            print(f"  vllm serve {args.model} --port {args.port}")
            return

        throughput = results["throughput_requests_per_second"]

    # Project experiment times
    projections = project_experiment_times(throughput, args.tax_year_length)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "benchmark": results,
                "projections": projections
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
