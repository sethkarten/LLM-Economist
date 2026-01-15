#!/bin/bash
# Setup script to run on della-ailab LOGIN NODE (has internet access)
# This pre-downloads models and installs all packages for offline compute nodes

set -e

WORK_DIR="/scratch/gpfs/CHIJ/milkkarten/LLM-Economist"
HF_CACHE="/scratch/gpfs/CHIJ/milkkarten/.cache/huggingface"

echo "=== Setting up offline environment for della-ailab ==="

# Create directories
mkdir -p "$WORK_DIR"
mkdir -p "$HF_CACHE"

# Clone/update repo
if [ ! -d "$WORK_DIR/.git" ]; then
    echo "Cloning repository..."
    git clone git@github.com:sethkarten/LLM-Economist.git "$WORK_DIR"
else
    echo "Updating repository..."
    cd "$WORK_DIR"
    git fetch origin
    git checkout agents
    git pull origin agents
fi

cd "$WORK_DIR"

# Create venv with uv
echo "Creating virtual environment with uv..."
uv venv .venv --python 3.10

# Install dependencies
echo "Installing dependencies..."
uv pip install -e .

# Pre-download Gemma-3-4B model to HF cache
echo "Pre-downloading Gemma-3-4B model..."
uv run python -c "
from huggingface_hub import snapshot_download
import os
os.environ['HF_HOME'] = '$HF_CACHE'
snapshot_download('google/gemma-3-4b-it', cache_dir='$HF_CACHE/hub')
print('Model downloaded successfully!')
"

echo ""
echo "=== Setup complete! ==="
echo "HF cache location: $HF_CACHE"
echo "Virtual environment: $WORK_DIR/.venv"
echo ""
echo "Compute nodes will use these offline resources."
