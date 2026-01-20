#!/usr/bin/env python3
"""
GPU 1 Job Queue: Run G1 variants while Qwen experiments run on GPU 0.

This script runs multiple G1 RL baseline variants to explore:
1. Effect of learning rate
2. Effect of number of agents
3. Different seeds for statistical significance
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Ensure we're using the right Python
PYTHON = "/media/milkkarten/data/LLMEconomist/.venv/bin/python"
BASE_DIR = "/media/milkkarten/data/LLMEconomist/LLM-Economist"
LOG_DIR = "/tmp"

# Job configurations
JOBS = [
    # Job 1: Lower learning rate
    {
        "name": "g1_lowlr",
        "desc": "G1 with lr=1e-4",
        "args": {
            "--num-agents": "1000",
            "--training-steps": "300000",
            "--batch-size": "2048",
            "--lr": "1e-4",
            "--output": "models/rl_baseline_g1_lowlr",
            "--seed": "42",
        }
    },
    # Job 2: Very low learning rate
    {
        "name": "g1_verylowlr",
        "desc": "G1 with lr=5e-5",
        "args": {
            "--num-agents": "1000",
            "--training-steps": "300000",
            "--batch-size": "2048",
            "--lr": "5e-5",
            "--output": "models/rl_baseline_g1_verylowlr",
            "--seed": "42",
        }
    },
    # Job 3: More agents (2k)
    {
        "name": "g1_2k",
        "desc": "G1 with 2000 agents",
        "args": {
            "--num-agents": "2000",
            "--training-steps": "300000",
            "--batch-size": "4096",
            "--lr": "3e-4",
            "--output": "models/rl_baseline_g1_2k",
            "--seed": "42",
        }
    },
    # Job 4: Large batch, more steps
    {
        "name": "g1_longrun",
        "desc": "G1 with 500k steps, batch 4096",
        "args": {
            "--num-agents": "1000",
            "--training-steps": "500000",
            "--batch-size": "4096",
            "--lr": "3e-4",
            "--output": "models/rl_baseline_g1_longrun",
            "--seed": "42",
        }
    },
    # Job 5: Additional seed for high-LR config
    {
        "name": "g1_highlr_seed123",
        "desc": "G1 high-LR seed=123",
        "args": {
            "--num-agents": "1000",
            "--training-steps": "300000",
            "--batch-size": "2048",
            "--lr": "5e-4",
            "--output": "models/rl_baseline_g1_highlr_seed123",
            "--seed": "123",
        }
    },
]


def run_job(job: dict) -> dict:
    """Run a single job and return results."""
    name = job["name"]
    desc = job["desc"]
    args = job["args"]

    log_file = f"{LOG_DIR}/gpu1_{name}.log"

    print(f"\n{'='*60}")
    print(f"Starting: {desc}")
    print(f"Log: {log_file}")
    print(f"{'='*60}")

    # Build command
    cmd = [PYTHON, "-u", "-m", "llm_economist.training.rl_baseline"]
    for k, v in args.items():
        cmd.extend([k, v])

    # Run with CUDA_VISIBLE_DEVICES=1
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "1", "PYTHONUNBUFFERED": "1"}

    start_time = time.time()

    with open(log_file, "w") as f:
        f.write(f"# {desc}\n")
        f.write(f"# Started: {datetime.now().isoformat()}\n")
        f.write(f"# Command: {' '.join(cmd)}\n\n")
        f.flush()

        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=BASE_DIR,
            env=env
        )

        # Monitor until complete
        while process.poll() is None:
            time.sleep(30)
            # Print last line of log for monitoring
            try:
                with open(log_file, "r") as lf:
                    lines = lf.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        if "Step" in last_line:
                            print(f"  {last_line}")
            except:
                pass

        return_code = process.returncode

    elapsed = time.time() - start_time

    # Parse results from log
    result = {
        "name": name,
        "desc": desc,
        "args": args,
        "log_file": log_file,
        "elapsed_seconds": elapsed,
        "return_code": return_code,
        "best_swf": None,
    }

    # Try to extract best SWF from log
    try:
        with open(log_file, "r") as f:
            content = f.read()
            # Look for "Best SWF:" line
            for line in content.split("\n"):
                if "Best SWF:" in line:
                    parts = line.split("Best SWF:")
                    if len(parts) > 1:
                        result["best_swf"] = float(parts[1].strip().split()[0])
    except:
        pass

    print(f"\nCompleted: {desc}")
    print(f"  Return code: {return_code}")
    print(f"  Elapsed: {elapsed/60:.1f} min")
    if result["best_swf"]:
        print(f"  Best SWF: {result['best_swf']:.1f}")

    return result


def main():
    print("=" * 60)
    print("GPU 1 Job Queue: G1 Variants")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Jobs to run: {len(JOBS)}")

    results = []

    for job in JOBS:
        try:
            result = run_job(job)
            results.append(result)
        except Exception as e:
            print(f"Error running {job['name']}: {e}")
            results.append({
                "name": job["name"],
                "error": str(e)
            })

        # Brief pause between jobs
        time.sleep(5)

    # Save summary
    summary_file = f"{BASE_DIR}/results/g1_variants_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "completed": datetime.now().isoformat(),
            "results": results
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("All jobs complete!")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
