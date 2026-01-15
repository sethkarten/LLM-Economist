#!/usr/bin/env python3
"""
Offline wrapper for H3 experiment on della-ailab.

Sets offline environment variables before running H3 experiment.
"""

import os
import sys

# Set offline mode (no internet on compute nodes)
os.environ['HF_HOME'] = '/scratch/gpfs/CHIJ/milkkarten/.cache/huggingface'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['WANDB_MODE'] = 'offline'
os.environ['VLLM_USE_V1'] = '0'

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import and run main
from experiments.run_reinforce_h3_baseline import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())
