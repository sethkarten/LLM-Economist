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

    # Install the package if needed
    log("Installing package...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", repo_root],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log(f"pip install FAILED (exit {result.returncode})")
        log(f"STDOUT: {result.stdout[-1000:]}")
        log(f"STDERR: {result.stderr[-1000:]}")
        sys.exit(1)
    log("Package installed OK")

    # Now run main_async as a module with all forwarded args
    log("Launching main_async...")
    result = subprocess.run(
        [sys.executable, "-m", "llm_economist.main_async"] + sys.argv[1:],
        cwd=repo_root,
    )
    log(f"main_async exited with code {result.returncode}")
    sys.exit(result.returncode)

except Exception as e:
    log(f"WRAPPER ERROR: {e}")
    log(traceback.format_exc())
    sys.exit(1)
