#!/usr/bin/env python3
"""
Wrapper to install package and run experiments on Pikachu.
"""
import subprocess
import sys

# Install the package in editable mode
print("Installing llm_economist package...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])

# Now run the actual experiment
print("Starting experiment...")
subprocess.check_call([sys.executable, "experiments/run_reinforce_h1h2.py"] + sys.argv[1:])
