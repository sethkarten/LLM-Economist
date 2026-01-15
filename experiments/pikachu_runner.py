#!/usr/bin/env python3
"""
Wrapper to run H1/H2 experiments on Pikachu with automatic setup.
"""
import subprocess
import sys
import os
from pathlib import Path

def setup_environment():
    """Setup uv and install dependencies."""
    print("Setting up Pikachu environment...")
    
    # Check if uv is installed
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("uv already installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing uv...")
        subprocess.run(
            ["curl", "-LsSf", "https://astral.sh/uv/install.sh"],
            check=True,
            stdout=subprocess.PIPE
        )
        
    # Set cache directory
    os.environ["UV_CACHE_DIR"] = "/data3/milkkarten/.uv-cache"
    Path("/data3/milkkarten/.uv-cache").mkdir(parents=True, exist_ok=True)
    
    # Install package
    print("Installing dependencies...")
    subprocess.run(
        ["python", "-m", "pip", "install", "-e", "."],
        check=True
    )
    
    print("Setup complete!")

def main():
    """Setup and run experiment."""
    setup_environment()
    
    # Import and run the actual experiment
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from experiments.run_reinforce_h1h2 import main as run_experiment
    import asyncio
    asyncio.run(run_experiment())

if __name__ == "__main__":
    main()
