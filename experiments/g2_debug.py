#!/usr/bin/env python3
"""Debug script to diagnose G2 eval failures on Cynthia."""
import sys
import os
import traceback

print(f"Python: {sys.executable}", flush=True)
print(f"CWD: {os.getcwd()}", flush=True)
print(f"PATH entries:", flush=True)
for p in sys.path[:5]:
    print(f"  {p}", flush=True)

# Check checkpoint
ckpt = "models/rl_baseline_g1/best/checkpoint.pt"
print(f"\nCheckpoint exists: {os.path.exists(ckpt)}", flush=True)

# Check if llm_economist is importable
try:
    import llm_economist
    print(f"llm_economist imported from: {llm_economist.__file__}", flush=True)
except Exception as e:
    print(f"ERROR importing llm_economist: {e}", flush=True)

# Check if training module works
try:
    from llm_economist.training.evaluate_rl_baseline import EvalConfig
    print("EvalConfig imported successfully", flush=True)
except Exception as e:
    print(f"ERROR importing evaluate_rl_baseline: {e}", flush=True)
    traceback.print_exc()

# Check if vllm is available
try:
    import vllm
    print(f"vLLM version: {vllm.__version__}", flush=True)
except Exception as e:
    print(f"ERROR importing vllm: {e}", flush=True)

# Check torch + CUDA
try:
    import torch
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
except Exception as e:
    print(f"ERROR with torch: {e}", flush=True)

# Try loading the checkpoint
try:
    import torch
    if os.path.exists(ckpt):
        data = torch.load(ckpt, map_location="cpu", weights_only=False)
        print(f"Checkpoint keys: {list(data.keys())}", flush=True)
        print(f"Best SWF: {data.get('best_swf', 'N/A')}", flush=True)
except Exception as e:
    print(f"ERROR loading checkpoint: {e}", flush=True)

# Try the model config lookup
try:
    from llm_economist.inference.config import get_model_config
    for name in ["mistral-7b-v0.3", "gemma3-4b", "llama-3.1-8b", "qwen3-8b", "olmo3-32b-instruct"]:
        try:
            cfg = get_model_config(name)
            print(f"Model {name} -> {cfg.hf_name} (quant: {cfg.recommended_quantization})", flush=True)
        except Exception as e:
            print(f"Model {name} FAILED: {e}", flush=True)
except Exception as e:
    print(f"ERROR with config: {e}", flush=True)

print("\nDiagnostics complete.", flush=True)
