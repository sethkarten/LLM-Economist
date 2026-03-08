#!/usr/bin/env python3
"""Diagnostic script to debug GPU manager import issues."""
import sys
import os
from pathlib import Path

print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")
print(f"CWD: {os.getcwd()}")
print(f"Script: {__file__}")
print(f"sys.path: {sys.path[:5]}")

# Add project root
root = str(Path(__file__).parent.parent)
sys.path.insert(0, root)
print(f"Added to path: {root}")
print(f"Contents: {os.listdir(root)[:20]}")

# Check if llm_economist exists
pkg_path = os.path.join(root, 'llm_economist')
print(f"Package exists: {os.path.isdir(pkg_path)}")
if os.path.isdir(pkg_path):
    print(f"Package contents: {os.listdir(pkg_path)[:10]}")

# Try imports
try:
    import numpy as np
    print(f"numpy OK: {np.__version__}")
except ImportError as e:
    print(f"numpy FAIL: {e}")

try:
    import torch
    print(f"torch OK: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"torch FAIL: {e}")

try:
    import vllm
    print(f"vllm OK: {vllm.__version__}")
except ImportError as e:
    print(f"vllm FAIL: {e}")

try:
    from llm_economist.main_async import AsyncLLMEconomist
    print("llm_economist.main_async OK")
except Exception as e:
    print(f"llm_economist.main_async FAIL: {type(e).__name__}: {e}")

try:
    from llm_economist.inference import ScalableInferenceEngine
    print("llm_economist.inference OK")
except Exception as e:
    print(f"llm_economist.inference FAIL: {type(e).__name__}: {e}")

print("\nDiagnostic complete.")
