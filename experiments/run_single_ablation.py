#!/usr/bin/env python3
"""Wrapper for GPU manager: runs a single ICRL ablation experiment.

The GPU manager clones the repo and runs `python script.py args`.
This wrapper ensures the package is importable and forwards args to main_async.
"""
import sys
import os
import subprocess

# Ensure the repo root is on the path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

# Install the package if needed
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", repo_root],
    check=True,
    capture_output=True,
)

# Now run main_async as a module with all forwarded args
result = subprocess.run(
    [sys.executable, "-m", "llm_economist.main_async"] + sys.argv[1:],
    cwd=repo_root,
)
sys.exit(result.returncode)
