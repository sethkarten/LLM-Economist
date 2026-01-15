#!/usr/bin/env python3
"""Install uv on Pikachu."""
import subprocess
import os
from pathlib import Path

# Set cache directory
cache_dir = Path("/data3/milkkarten/.uv-cache")
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["UV_CACHE_DIR"] = str(cache_dir)

# Install uv
print("Installing uv...")
subprocess.run(
    "curl -LsSf https://astral.sh/uv/install.sh | sh",
    shell=True,
    check=True
)

print("uv installed successfully!")
print(f"Cache directory: {cache_dir}")
