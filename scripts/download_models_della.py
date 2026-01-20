#!/usr/bin/env python3
"""
Download models to della scratch storage for offline use.

Run on della login node (has internet):
    python scripts/download_models_della.py

This will download:
- Qwen/Qwen3-4B-Instruct-2507 (planner)
- google/gemma-3-4b-it (workers)
"""

import os
import sys

# Set cache directories
HF_HOME = "/scratch/gpfs/CHIJ/milkkarten/huggingface"
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = f"{HF_HOME}/hub"
os.environ["HF_HUB_CACHE"] = f"{HF_HOME}/hub"

# Create directories
os.makedirs(HF_HOME, exist_ok=True)
os.makedirs(f"{HF_HOME}/hub", exist_ok=True)

print(f"Downloading models to: {HF_HOME}")
print("=" * 60)

# Requires HF_TOKEN for gated models
if not os.environ.get("HF_TOKEN"):
    print("WARNING: HF_TOKEN not set. Gemma-3 is gated and requires authentication.")
    print("Set HF_TOKEN environment variable or run: huggingface-cli login")
    sys.exit(1)

from huggingface_hub import snapshot_download

models = [
    ("Qwen/Qwen3-4B-Instruct-2507", "Planner model"),
    ("google/gemma-3-4b-it", "Worker model (gated - requires HF_TOKEN)"),
]

for model_id, desc in models:
    print(f"\nDownloading: {model_id}")
    print(f"Description: {desc}")
    print("-" * 40)

    try:
        path = snapshot_download(
            model_id,
            cache_dir=f"{HF_HOME}/hub",
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"✓ Downloaded to: {path}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        sys.exit(1)

print("\n" + "=" * 60)
print("All models downloaded successfully!")
print(f"Cache location: {HF_HOME}")
print("=" * 60)
