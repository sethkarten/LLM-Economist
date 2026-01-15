#!/usr/bin/env python3
"""
G2 Evaluation Job Queue: Test distribution shift across different worker models.

This script runs G2 evaluations with various LLM worker models to:
1. Confirm distribution shift is consistent across models
2. Compare model-specific behavioral differences
3. Build comprehensive dataset for paper
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

PYTHON = "/media/milkkarten/data/LLMEconomist/.venv/bin/python"
BASE_DIR = "/media/milkkarten/data/LLMEconomist/LLM-Economist"
LOG_DIR = "/tmp"

# G2 evaluation configurations
# Each tests the same G1 checkpoint with different LLM workers
JOBS = [
    # Small/fast models (text-only or AWQ)
    {
        "name": "g2_gemma4b",
        "desc": "G2 with Gemma-3-4B workers",
        "worker_model": "gemma3-4b",
        "num_agents": 100,
        "max_timesteps": 1000,
        "num_episodes": 3,
    },
    {
        "name": "g2_mistral7b",
        "desc": "G2 with Mistral-7B workers",
        "worker_model": "mistral-7b-v0.3",
        "num_agents": 100,
        "max_timesteps": 1000,
        "num_episodes": 3,
    },
    {
        "name": "g2_llama8b",
        "desc": "G2 with Llama-3.1-8B workers",
        "worker_model": "llama-3.1-8b",
        "num_agents": 100,
        "max_timesteps": 1000,
        "num_episodes": 3,
    },
    {
        "name": "g2_qwen8b",
        "desc": "G2 with Qwen3-8B workers",
        "worker_model": "qwen3-8b",
        "num_agents": 100,
        "max_timesteps": 1000,
        "num_episodes": 3,
    },
    {
        "name": "g2_olmo7b",
        "desc": "G2 with OLMo-3-7B workers",
        "worker_model": "olmo3-7b",
        "num_agents": 100,
        "max_timesteps": 1000,
        "num_episodes": 3,
    },
]

# Different G1 checkpoints to test
G1_CHECKPOINTS = [
    ("best", "models/rl_baseline_g1/best/checkpoint.pt"),
    ("seed123", "models/rl_baseline_g1_seed123/best/checkpoint.pt"),
    ("seed456", "models/rl_baseline_g1_seed456/best/checkpoint.pt"),
]


def run_g2_eval(job: dict, checkpoint_name: str, checkpoint_path: str) -> dict:
    """Run a single G2 evaluation."""
    name = f"{job['name']}_{checkpoint_name}"
    desc = f"{job['desc']} (checkpoint: {checkpoint_name})"

    log_file = f"{LOG_DIR}/gpu1_{name}.log"
    output_file = f"results/g2_evals/{name}.json"

    print(f"\n{'='*60}")
    print(f"Starting: {desc}")
    print(f"Log: {log_file}")
    print(f"{'='*60}")

    # Build command
    cmd = [
        PYTHON, "-u", "-m", "llm_economist.training.evaluate_rl_baseline",
        "--checkpoint", checkpoint_path,
        "--worker-model", job["worker_model"],
        "--num-agents", str(job["num_agents"]),
        "--max-timesteps", str(job["max_timesteps"]),
        "--num-episodes", str(job["num_episodes"]),
        "--output", output_file,
        "--seed", "42",
    ]

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "1", "PYTHONUNBUFFERED": "1"}

    start_time = time.time()

    # Ensure output directory exists
    Path(f"{BASE_DIR}/results/g2_evals").mkdir(parents=True, exist_ok=True)

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

        while process.poll() is None:
            time.sleep(30)
            try:
                with open(log_file, "r") as lf:
                    lines = lf.readlines()
                    for line in reversed(lines[-10:]):
                        if "Step" in line or "SWF" in line:
                            print(f"  {line.strip()}")
                            break
            except:
                pass

        return_code = process.returncode

    elapsed = time.time() - start_time

    result = {
        "name": name,
        "desc": desc,
        "worker_model": job["worker_model"],
        "checkpoint": checkpoint_name,
        "log_file": log_file,
        "output_file": output_file,
        "elapsed_seconds": elapsed,
        "return_code": return_code,
        "mean_swf": None,
        "mean_gini": None,
    }

    # Parse results
    try:
        with open(f"{BASE_DIR}/{output_file}", "r") as f:
            data = json.load(f)
            result["mean_swf"] = data.get("summary", {}).get("mean_swf")
            result["mean_gini"] = data.get("summary", {}).get("mean_gini")
    except:
        pass

    print(f"\nCompleted: {desc}")
    print(f"  Return code: {return_code}")
    print(f"  Elapsed: {elapsed/60:.1f} min")
    if result["mean_swf"]:
        print(f"  Mean SWF: {result['mean_swf']:.1f}")
        print(f"  Mean Gini: {result['mean_gini']:.3f}")

    return result


def main():
    print("=" * 60)
    print("G2 Evaluation Job Queue: Distribution Shift Analysis")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Worker models to test: {len(JOBS)}")
    print(f"G1 checkpoints to test: {len(G1_CHECKPOINTS)}")
    print(f"Total evaluations: {len(JOBS) * len(G1_CHECKPOINTS)}")

    results = []

    # For efficiency, just test each model with the best checkpoint first
    # Then do additional checkpoints if time permits
    for job in JOBS:
        checkpoint_name, checkpoint_path = G1_CHECKPOINTS[0]  # "best"

        # Check if checkpoint exists
        full_path = f"{BASE_DIR}/{checkpoint_path}"
        if not os.path.exists(full_path):
            print(f"Skipping {job['name']}: checkpoint not found at {full_path}")
            continue

        try:
            result = run_g2_eval(job, checkpoint_name, checkpoint_path)
            results.append(result)
        except Exception as e:
            print(f"Error running {job['name']}: {e}")
            results.append({
                "name": job["name"],
                "error": str(e)
            })

        time.sleep(5)

    # Save summary
    summary_file = f"{BASE_DIR}/results/g2_evals_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "completed": datetime.now().isoformat(),
            "results": results
        }, f, indent=2)

    # Print summary table
    print(f"\n{'='*60}")
    print("G2 EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Worker Model':<20} {'Mean SWF':<12} {'Mean Gini':<12} {'Status'}")
    print("-" * 60)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<20} {'ERROR':<12} {'':<12} {r['error'][:20]}")
        else:
            swf = f"{r['mean_swf']:.1f}" if r['mean_swf'] else "N/A"
            gini = f"{r['mean_gini']:.3f}" if r['mean_gini'] else "N/A"
            print(f"{r['worker_model']:<20} {swf:<12} {gini:<12} OK")

    print(f"\n{'='*60}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
