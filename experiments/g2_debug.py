#!/usr/bin/env python3
"""Debug script to diagnose G2 eval failures on Cynthia.
Writes output to a file since GPU manager log streaming seems broken."""
import sys
import os
import traceback

# Write all output to a file
log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "g2_debug_output.txt")

with open(log_file, "w") as f:
    def log(msg):
        print(msg, flush=True)
        f.write(msg + "\n")
        f.flush()

    log(f"Python: {sys.executable}")
    log(f"CWD: {os.getcwd()}")
    log(f"PATH entries:")
    for p in sys.path[:5]:
        log(f"  {p}")

    # Check checkpoint
    ckpt = "models/rl_baseline_g1/best/checkpoint.pt"
    log(f"\nCheckpoint exists: {os.path.exists(ckpt)}")

    # Check if llm_economist is importable
    try:
        import llm_economist
        log(f"llm_economist imported from: {llm_economist.__file__}")
    except Exception as e:
        log(f"ERROR importing llm_economist: {e}")

    # Check if training module works
    try:
        from llm_economist.training.evaluate_rl_baseline import EvalConfig
        log("EvalConfig imported successfully")
    except Exception as e:
        log(f"ERROR importing evaluate_rl_baseline: {e}")
        log(traceback.format_exc())

    # Check if vllm is available
    try:
        import vllm
        log(f"vLLM version: {vllm.__version__}")
    except Exception as e:
        log(f"ERROR importing vllm: {e}")

    # Check torch + CUDA
    try:
        import torch
        log(f"PyTorch: {torch.__version__}")
        log(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        log(f"ERROR with torch: {e}")

    # Try loading the checkpoint
    try:
        import torch
        if os.path.exists(ckpt):
            data = torch.load(ckpt, map_location="cpu", weights_only=False)
            log(f"Checkpoint keys: {list(data.keys())}")
            log(f"Best SWF: {data.get('best_swf', 'N/A')}")
    except Exception as e:
        log(f"ERROR loading checkpoint: {e}")

    # Try the model config lookup
    try:
        from llm_economist.inference.config import get_model_config
        for name in ["mistral-7b-v0.3", "gemma3-4b", "llama-3.1-8b", "qwen3-8b", "olmo3-32b-instruct"]:
            try:
                cfg = get_model_config(name)
                log(f"Model {name} -> {cfg.hf_name} (quant: {cfg.recommended_quantization})")
            except Exception as e:
                log(f"Model {name} FAILED: {e}")
    except Exception as e:
        log(f"ERROR with config: {e}")

    # Now try to actually run the eval with a mini config
    log("\n--- Attempting mini evaluation run ---")
    try:
        from llm_economist.training.evaluate_rl_baseline import run_evaluation, EvalConfig
        import asyncio

        config = EvalConfig(
            checkpoint_path=ckpt,
            worker_model="mistral-7b-v0.3",
            num_agents=5,
            max_timesteps=2,
            num_eval_episodes=1,
            output_path="results/g2_evals/g2_debug_test.json",
            seed=42,
        )
        log(f"Config created: {config}")
        log("Starting async evaluation...")
        asyncio.run(run_evaluation(config))
        log("Mini evaluation completed!")
    except Exception as e:
        log(f"ERROR in evaluation: {e}")
        log(traceback.format_exc())

    log(f"\nDiagnostics complete. Log saved to: {log_file}")
