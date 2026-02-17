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

def main():
    # Pass all arguments through to the module
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [
        sys.executable, "-u", "-m",
        "llm_economist.training.evaluate_rl_baseline",
    ] + sys.argv[1:]

    print(f"Running: {' '.join(cmd)}", flush=True)
    print(f"CWD: {os.getcwd()}", flush=True)
    print(f"Repo root: {repo_root}", flush=True)
    print(f"Python: {sys.executable}", flush=True)
    print(f"Args: {sys.argv[1:]}", flush=True)

    result = subprocess.run(
        cmd,
        cwd=repo_root,
        stdout=sys.stdout,
        stderr=sys.stdout,  # Redirect stderr to stdout so GPU manager captures it
    )
    print(f"Exit code: {result.returncode}", flush=True)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
