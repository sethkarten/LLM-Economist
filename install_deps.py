#!/usr/bin/env python3
"""Install dependencies using pip."""
import subprocess
import sys

print("Installing dependencies with pip...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "-e", "."])
print("Installation complete!")
