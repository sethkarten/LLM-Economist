#!/usr/bin/env python3
"""
Wrapper script to run G2 evaluation via GPU manager.

Invokes llm_economist.training.evaluate_rl_baseline as a module.
This wrapper exists because evaluate_rl_baseline uses relative imports
and cannot be run as a standalone script.

Usage:
    python experiments/run_g2_eval.py \
        --worker-model llama-3.1-8b \
        --seed 123 \
        --checkpoint models/rl_baseline_g1/best/checkpoint.pt \
        --output results/g2_evals/g2_llama8b_seed123.json
"""

import subprocess
import sys
import os
import datetime


def main():
    # Pass all arguments through to the module
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [
        sys.executable, "-u", "-m",
        "llm_economist.training.evaluate_rl_baseline",
    ] + sys.argv[1:]

    # Create a log file for debugging
    log_dir = os.path.join(repo_root, "results", "g2_evals", "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Extract model name and seed from args for log filename
    args_str = "_".join(sys.argv[1:]).replace("/", "_").replace("-", "")[:80]
    log_file = os.path.join(log_dir, f"g2_eval_{timestamp}_{args_str}.log")

    print(f"Running: {' '.join(cmd)}", flush=True)
    print(f"CWD: {os.getcwd()}", flush=True)
    print(f"Repo root: {repo_root}", flush=True)
    print(f"Python: {sys.executable}", flush=True)
    print(f"Args: {sys.argv[1:]}", flush=True)
    print(f"Log file: {log_file}", flush=True)

    with open(log_file, "w") as lf:
        lf.write(f"Command: {' '.join(cmd)}\n")
        lf.write(f"CWD: {repo_root}\n")
        lf.write(f"Started: {datetime.datetime.now()}\n\n")
        lf.flush()

        result = subprocess.run(
            cmd,
            cwd=repo_root,
            stdout=lf,
            stderr=subprocess.STDOUT,
        )

        lf.write(f"\nExit code: {result.returncode}\n")
        lf.write(f"Finished: {datetime.datetime.now()}\n")

    print(f"Exit code: {result.returncode}", flush=True)
    print(f"Log saved to: {log_file}", flush=True)

    # Print last 50 lines of log to stdout
    with open(log_file) as lf:
        lines = lf.readlines()
        print("\n--- Last 50 lines of log ---", flush=True)
        for line in lines[-50:]:
            print(line.rstrip(), flush=True)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
