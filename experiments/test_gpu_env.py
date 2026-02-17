#!/usr/bin/env python3
"""Minimal test script for GPU manager environment debugging."""
import sys
import os

print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")
print(f"CWD: {os.getcwd()}")
print(f"Script: {os.path.abspath(__file__)}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

# Check if package is importable
try:
    import llm_economist
    print(f"llm_economist importable: {llm_economist.__file__}")
except ImportError as e:
    print(f"llm_economist NOT importable: {e}")

# Check if vllm is available
try:
    import vllm
    print(f"vllm version: {vllm.__version__}")
except ImportError as e:
    print(f"vllm NOT importable: {e}")

# Check GPU
try:
    import torch
    print(f"torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
except ImportError as e:
    print(f"torch NOT importable: {e}")

print("\nDone!")
