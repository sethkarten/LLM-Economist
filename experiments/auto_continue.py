#!/usr/bin/env python3
"""
Automated Continuation Script

Monitors experiment progress and triggers next phases automatically:
1. Monitor bounded experiments (GPU 0) and G2 evals (GPU 1)
2. Run final analysis when all 15 bounded experiments complete
3. Prepare H1/H2 REINFORCE++ after bounded experiments done

Usage:
    python experiments/auto_continue.py
"""

import subprocess
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import glob

PYTHON = "/media/milkkarten/data/LLMEconomist/.venv/bin/python"
BASE_DIR = "/media/milkkarten/data/LLMEconomist/LLM-Economist"
RESULTS_DIR = f"{BASE_DIR}/results/bounded_100x2000"

# Expected experiments
EXPECTED_EXPERIMENTS = [
    "gemma3-4b_seed42", "gemma3-4b_seed123", "gemma3-4b_seed456",
    "mistral-7b-v0.3_seed42", "mistral-7b-v0.3_seed123", "mistral-7b-v0.3_seed456",
    "llama-3.1-8b_seed42", "llama-3.1-8b_seed123", "llama-3.1-8b_seed456",
    "olmo3-7b_seed42", "olmo3-7b_seed123", "olmo3-7b_seed456",
    "qwen3-8b_seed42", "qwen3-8b_seed123", "qwen3-8b_seed456",
]


def check_bounded_experiments():
    """Check how many bounded experiments are complete."""
    completed = []
    for exp in EXPECTED_EXPERIMENTS:
        path = f"{RESULTS_DIR}/{exp}.json"
        if os.path.exists(path):
            completed.append(exp)
    return completed


def run_final_analysis():
    """Run comprehensive analysis on all bounded experiments."""
    print("\n" + "="*60)
    print("RUNNING FINAL ANALYSIS")
    print("="*60)

    # Run the analysis script
    cmd = [PYTHON, "experiments/analyze_bounded_results.py"]

    result = subprocess.run(
        cmd,
        cwd=BASE_DIR,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("Analysis complete!")
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    else:
        print(f"Analysis failed: {result.stderr}")

    return result.returncode == 0


def check_g2_queue_status():
    """Check if G2 evaluation queue is still running."""
    log_file = "/tmp/gpu1_g2_queue.log"
    if not os.path.exists(log_file):
        return "not_started"

    with open(log_file, "r") as f:
        content = f.read()

    if "All jobs complete" in content or "Summary saved to" in content:
        return "complete"
    return "running"


def get_gpu_status():
    """Get GPU memory and utilization."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True
        )
        lines = result.stdout.strip().split("\n")
        gpus = {}
        for line in lines:
            parts = line.split(", ")
            if len(parts) >= 3:
                idx = int(parts[0])
                gpus[idx] = {
                    "memory_used": int(parts[1]),
                    "utilization": int(parts[2])
                }
        return gpus
    except:
        return {}


def print_status(completed_bounded, g2_status, gpus):
    """Print current status."""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status Update")
    print(f"  Bounded experiments: {len(completed_bounded)}/15")
    print(f"  G2 evaluation queue: {g2_status}")
    for idx, info in gpus.items():
        print(f"  GPU {idx}: {info['memory_used']}MB, {info['utilization']}% util")


def main():
    print("="*60)
    print("AUTOMATED CONTINUATION MONITOR")
    print("="*60)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Monitoring bounded experiments in: {RESULTS_DIR}")
    print(f"Expected: {len(EXPECTED_EXPERIMENTS)} experiments")

    analysis_done = False
    last_status_time = 0

    while True:
        try:
            # Check progress
            completed_bounded = check_bounded_experiments()
            g2_status = check_g2_queue_status()
            gpus = get_gpu_status()

            # Print status every 5 minutes
            current_time = time.time()
            if current_time - last_status_time > 300:
                print_status(completed_bounded, g2_status, gpus)
                last_status_time = current_time

            # Check if all bounded experiments complete
            if len(completed_bounded) == 15 and not analysis_done:
                print("\n" + "!"*60)
                print("ALL 15 BOUNDED EXPERIMENTS COMPLETE!")
                print("!"*60)

                # Run final analysis
                success = run_final_analysis()
                analysis_done = True

                # Save completion marker
                with open(f"{BASE_DIR}/results/bounded_complete.marker", "w") as f:
                    json.dump({
                        "completed_at": datetime.now().isoformat(),
                        "experiments": completed_bounded,
                        "analysis_success": success
                    }, f, indent=2)

                print("\nBounded experiments phase complete!")
                print("Ready for H1/H2 REINFORCE++ training.")

            # Check if G2 queue is done
            if g2_status == "complete":
                print("\nG2 evaluation queue complete!")
                # Could trigger additional analysis here

            # Sleep before next check
            time.sleep(60)

        except KeyboardInterrupt:
            print("\nMonitor interrupted by user")
            break
        except Exception as e:
            print(f"Error in monitor loop: {e}")
            time.sleep(60)

    print("\nMonitor exiting.")


if __name__ == "__main__":
    main()
