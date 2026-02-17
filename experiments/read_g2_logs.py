#!/usr/bin/env python3
"""Read G2 evaluation logs and results from Cynthia."""
import os
import json
import glob

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(repo_root, "results", "g2_evals", "logs")
results_dir = os.path.join(repo_root, "results", "g2_evals")

print("=== G2 Eval Results ===", flush=True)
for f in sorted(glob.glob(os.path.join(results_dir, "g2_*.json"))):
    try:
        with open(f) as fh:
            data = json.load(fh)
        summary = data.get("summary", {})
        config = data.get("config", {})
        print(f"\n{os.path.basename(f)}:", flush=True)
        print(f"  Model: {config.get('worker_model', 'N/A')}", flush=True)
        print(f"  Seed: {config.get('seed', 'N/A')}", flush=True)
        print(f"  Mean SWF: {summary.get('mean_swf', 'N/A')}", flush=True)
        print(f"  Std SWF: {summary.get('std_swf', 'N/A')}", flush=True)
        print(f"  Mean Gini: {summary.get('mean_gini', 'N/A')}", flush=True)
    except Exception as e:
        print(f"\n{os.path.basename(f)}: ERROR reading: {e}", flush=True)

print("\n=== G2 Eval Logs ===", flush=True)
if os.path.exists(log_dir):
    for f in sorted(glob.glob(os.path.join(log_dir, "*.log")))[-10:]:
        print(f"\n--- {os.path.basename(f)} ---", flush=True)
        with open(f) as fh:
            lines = fh.readlines()
        # Print first 5 and last 30 lines
        for line in lines[:5]:
            print(line.rstrip(), flush=True)
        if len(lines) > 35:
            print(f"  ... ({len(lines) - 35} lines omitted) ...", flush=True)
        for line in lines[-30:]:
            print(line.rstrip(), flush=True)
else:
    print(f"No log directory at {log_dir}", flush=True)

# Also check the debug output
debug_file = os.path.join(repo_root, "g2_debug_output.txt")
if os.path.exists(debug_file):
    print("\n=== Debug Output ===", flush=True)
    with open(debug_file) as f:
        print(f.read(), flush=True)
