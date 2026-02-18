#!/usr/bin/env python3
"""Wrapper for GPU manager: runs a single ICRL ablation experiment.

The GPU manager clones the repo and runs `python script.py args`.
This wrapper ensures the package is importable and forwards args to main_async.
"""
import sys
import os
import subprocess
import traceback

LOG_FILE = "/tmp/ablation_wrapper_debug.log"

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

try:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log(f"Repo root: {repo_root}")
    log(f"Python: {sys.executable}")
    log(f"Args: {sys.argv[1:]}")
    log(f"CWD: {os.getcwd()}")

    # Check if package is already importable; skip install on compute nodes (no internet)
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

    # Now run main_async as a module with all forwarded args
    # Stream output directly (no capture) so SLURM logs show progress in real-time
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
