#!/usr/bin/env python3
"""
Wrapper script for running REINFORCE++ on della-ailab compute nodes (OFFLINE)
Sets environment variables for offline mode and runs the experiment.
"""
import os
import sys
import subprocess

# Configure offline mode
os.environ['HF_HOME'] = '/scratch/gpfs/CHIJ/milkkarten/.cache/huggingface'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_DIR'] = '/scratch/gpfs/CHIJ/milkkarten/LLM-Economist/wandb_offline'

# Create wandb offline directory
os.makedirs(os.environ['WANDB_DIR'], exist_ok=True)

print("=== Running in OFFLINE mode ===")
print(f"HF_HOME: {os.environ['HF_HOME']}")
print(f"WANDB_MODE: {os.environ['WANDB_MODE']}")
print(f"Working directory: {os.getcwd()}")
print()

# Run experiment with arguments passed through
sys.path.insert(0, os.getcwd())

# Import and run the experiment directly
from experiments.run_reinforce_h1h2 import main
import asyncio

if __name__ == '__main__':
    asyncio.run(main())
