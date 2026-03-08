#!/usr/bin/env python3
"""Check if result files exist on this machine."""
import os
import json
from pathlib import Path

root = Path(__file__).parent.parent
checks = [
    "results/grid_search_v2/seed42.json",
    "results/grid_search_v2/seed123.json",
    "results/grid_search_v2/seed456.json",
    "results/icrl_v2/seed42.json",
    "results/icrl_v2/seed123.json",
    "results/icrl_v2/seed456.json",
]

for path in checks:
    full = root / path
    if full.exists():
        size = full.stat().st_size
        try:
            data = json.loads(full.read_text())
            if 'results' in data:
                n = len(data['results'])
                print(f"  FOUND: {path} ({size} bytes, {n} schedules)")
            elif 'metrics_history' in data:
                n = len(data['metrics_history'])
                print(f"  FOUND: {path} ({size} bytes, {n} steps)")
            else:
                print(f"  FOUND: {path} ({size} bytes, keys={list(data.keys())[:5]})")
        except:
            print(f"  FOUND: {path} ({size} bytes, parse error)")
    else:
        print(f"  MISSING: {path}")

print("\nDone.")
