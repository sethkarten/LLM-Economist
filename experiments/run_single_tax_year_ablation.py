#!/usr/bin/env python3
"""Wrapper for GPU manager: runs a single tax-year-length ablation experiment.

The GPU manager clones the repo and runs `python script.py args`.
This wrapper:
  1. Sets HF_HOME to the scratch cache so compute nodes find models offline.
  2. Installs the package with uv (or pip) if not already importable.
  3. Forwards all CLI args to llm_economist.main_async.

No --quantization flag is needed; google/gemma-3-4b-it runs in full precision.

Example invocation by the GPU manager:
    python experiments/run_single_tax_year_ablation.py \
        --scenario bounded \
        --num-agents 100 \
        --max-timesteps 2048 \
        --history-len 64 \
        --tax-year-length 64 \
        --model google/gemma-3-4b-it \
        --tensor-parallel 1 \
        --seed 3 \
        --output results/tax_year_ablation/TY64/seed_3/TY64_seed3.json
"""
import sys
import os
import subprocess
import traceback

LOG_FILE = "/tmp/tax_year_ablation_wrapper_debug.log"


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


try:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ------------------------------------------------------------------
    # HuggingFace cache: compute nodes have no internet, point at scratch
    # ------------------------------------------------------------------
    if "HF_HOME" not in os.environ:
        scratch_hf = "/scratch/gpfs/CHIJ/milkkarten/huggingface/"
        if os.path.isdir(scratch_hf):
            os.environ["HF_HOME"] = scratch_hf

    # Force offline mode so the library never attempts HTTP requests
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    log(f"Repo root: {repo_root}")
    log(f"Python: {sys.executable}")
    log(f"HF_HOME: {os.environ.get('HF_HOME', 'NOT SET')}")
    log(f"HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE')}")
    log(f"Args: {sys.argv[1:]}")
    log(f"CWD: {os.getcwd()}")

    # ------------------------------------------------------------------
    # Ensure the package is importable; install with uv if needed
    # ------------------------------------------------------------------
    try:
        import llm_economist
        log(f"Package already installed: {llm_economist.__file__}")
    except ImportError:
        log("Package not found, attempting install...")
        uv_path = os.path.expanduser("~/.local/bin/uv")
        if os.path.exists(uv_path):
            install_cmd = [uv_path, "pip", "install", "--no-deps", "-e", repo_root]
        else:
            install_cmd = [sys.executable, "-m", "pip", "install", "--no-deps", "-e", repo_root]
        log(f"Install cmd: {' '.join(install_cmd)}")
        result = subprocess.run(install_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log(f"Install FAILED (exit {result.returncode})")
            log(f"STDOUT: {result.stdout[-1000:]}")
            log(f"STDERR: {result.stderr[-1000:]}")
            sys.exit(1)
        log("Package installed OK")

    # ------------------------------------------------------------------
    # Run main_async, streaming output so SLURM logs show progress live
    # ------------------------------------------------------------------
    log("Launching main_async...")
    cmd = [sys.executable, "-m", "llm_economist.main_async"] + sys.argv[1:]
    log(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=repo_root)
    log(f"main_async exited with code {result.returncode}")
    sys.exit(result.returncode)

except Exception as e:
    log(f"WRAPPER ERROR: {e}")
    log(traceback.format_exc())
    sys.exit(1)
