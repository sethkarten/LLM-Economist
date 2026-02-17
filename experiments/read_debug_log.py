#!/usr/bin/env python3
"""Read the debug log file from Cynthia."""
import os
log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "g2_debug_output.txt")
print(f"Reading: {log_file}", flush=True)
if os.path.exists(log_file):
    with open(log_file) as f:
        print(f.read(), flush=True)
else:
    print(f"File not found: {log_file}", flush=True)
    # Also check a few other locations
    for alt in ["/data1/milkkarten/Research/LLM-Economist/g2_debug_output.txt",
                "g2_debug_output.txt"]:
        if os.path.exists(alt):
            print(f"Found at: {alt}", flush=True)
            with open(alt) as f:
                print(f.read(), flush=True)
            break
