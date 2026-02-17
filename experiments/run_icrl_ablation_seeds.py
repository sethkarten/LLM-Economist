#!/usr/bin/env python3
"""
Run ICRL ablation experiments with multiple seeds for error bars.

Ablates two axes:
  1. K (context window size): K=8, 16, 64, 128, 256
  2. Prompt cues at K=64: no exploration, no exploitation

Total: 7 conditions x 3 seeds = 21 runs.

Usage:
    python experiments/run_icrl_ablation_seeds.py --dry-run              # Print commands
    python experiments/run_icrl_ablation_seeds.py --all                  # Run all conditions
    python experiments/run_icrl_ablation_seeds.py --conditions K8 K16    # Run specific conditions
    python experiments/run_icrl_ablation_seeds.py --parallel 2           # Run 2 at a time
    python experiments/run_icrl_ablation_seeds.py --summary              # Print summary table
"""

import os
import sys
import re
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Experiment conditions
# ---------------------------------------------------------------------------

# Each condition is a dict with:
#   name: short identifier (used in CLI --conditions filter and directory name)
#   K: history-len value
#   disable_exploration: bool
#   disable_exploitation: bool
#   description: human-readable label

CONDITIONS = [
    # K ablation (both exploration and exploitation enabled)
    {
        "name": "K8",
        "K": 8,
        "disable_exploration": False,
        "disable_exploitation": False,
        "description": "K=8 (full ICRL)",
    },
    {
        "name": "K16",
        "K": 16,
        "disable_exploration": False,
        "disable_exploitation": False,
        "description": "K=16 (full ICRL)",
    },
    {
        "name": "K64",
        "K": 64,
        "disable_exploration": False,
        "disable_exploitation": False,
        "description": "K=64 (full ICRL)",
    },
    {
        "name": "K128",
        "K": 128,
        "disable_exploration": False,
        "disable_exploitation": False,
        "description": "K=128 (full ICRL)",
    },
    {
        "name": "K256",
        "K": 256,
        "disable_exploration": False,
        "disable_exploitation": False,
        "description": "K=256 (full ICRL)",
    },
    # Prompt ablation at K=64
    {
        "name": "K64_no_explore",
        "K": 64,
        "disable_exploration": True,
        "disable_exploitation": False,
        "description": "K=64, no exploration (exploitation only)",
    },
    {
        "name": "K64_no_exploit",
        "K": 64,
        "disable_exploration": False,
        "disable_exploitation": True,
        "description": "K=64, no exploitation (exploration only)",
    },
]

# Simulation defaults
NUM_AGENTS = 100
MAX_TIMESTEPS = 2000
TWO_TIMESCALE = 25
SCENARIO = "bounded"
BRACKET_SETTING = "US_FED"
PROMPT_ALGO = "io"
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_SERVICE = "vllm"
DEFAULT_PORT = 8009

RESULTS_ROOT = PROJECT_ROOT / "results" / "icrl_ablation"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_output_dir(condition_name: str, seed: int) -> Path:
    """Return the output directory for a given condition and seed."""
    return RESULTS_ROOT / condition_name / f"seed_{seed}"


def build_command(
    condition: Dict,
    seed: int,
    model: str,
    service: str,
    port: int,
    wandb: bool,
) -> List[str]:
    """Build the subprocess command for a single run."""
    output_dir = get_output_dir(condition["name"], seed)

    cmd = [
        sys.executable, "-m", "llm_economist.main",
        "--scenario", SCENARIO,
        "--num-agents", str(NUM_AGENTS),
        "--max-timesteps", str(MAX_TIMESTEPS),
        "--history-len", str(condition["K"]),
        "--two-timescale", str(TWO_TIMESCALE),
        "--bracket-setting", BRACKET_SETTING,
        "--prompt-algo", PROMPT_ALGO,
        "--worker-type", "LLM",
        "--planner-type", "LLM",
        "--llm", model,
        "--service", service,
        "--port", str(port),
        "--seed", str(seed),
        "--log-dir", str(output_dir),
        "--name", f"{condition['name']}_seed{seed}",
    ]

    if condition["disable_exploration"]:
        cmd.append("--disable-exploration")
    if condition["disable_exploitation"]:
        cmd.append("--disable-exploitation")
    if wandb:
        cmd.append("--wandb")

    return cmd


def check_completed(condition_name: str, seed: int) -> bool:
    """Check whether a run already completed by looking for its log file."""
    output_dir = get_output_dir(condition_name, seed)
    log_file = output_dir / f"{condition_name}_seed{seed}.log"
    if not log_file.exists():
        return False
    # Check if the log contains the completion marker
    try:
        text = log_file.read_text()
        return "Simulation completed successfully" in text
    except Exception:
        return False


def parse_swf_from_log(condition_name: str, seed: int) -> Optional[List[float]]:
    """Parse SWF values from a completed log file.

    Returns a list of SWF values (one per timestep where the planner logs).
    """
    output_dir = get_output_dir(condition_name, seed)
    log_file = output_dir / f"{condition_name}_seed{seed}.log"
    if not log_file.exists():
        return None
    try:
        text = log_file.read_text()
        # Pattern: swf=<number> in planner log lines
        swf_values = [float(m) for m in re.findall(r"swf=([-\d.]+(?:e[+-]?\d+)?)", text)]
        return swf_values if swf_values else None
    except Exception:
        return None


def run_single(
    condition: Dict,
    seed: int,
    model: str,
    service: str,
    port: int,
    wandb: bool,
    dry_run: bool = False,
    label: str = "",
) -> Tuple[str, int, bool, Optional[str]]:
    """Run (or dry-run) a single experiment.

    Returns (condition_name, seed, success, error_msg).
    """
    cond_name = condition["name"]
    output_dir = get_output_dir(cond_name, seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(condition, seed, model, service, port, wandb)

    if dry_run:
        print(f"  [{label}] {' '.join(cmd)}")
        return (cond_name, seed, True, None)

    print(f"  [{label}] Starting {cond_name} seed={seed} ...")
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=3600 * 12,  # 12-hour hard timeout
        )
        elapsed = time.time() - start
        if result.returncode != 0:
            err_msg = result.stderr[-500:] if result.stderr else "unknown error"
            print(f"  [{label}] FAILED {cond_name} seed={seed} "
                  f"(exit {result.returncode}, {elapsed/60:.1f}min)")
            # Save stderr for debugging
            (output_dir / "stderr.txt").write_text(result.stderr or "")
            return (cond_name, seed, False, err_msg)
        else:
            print(f"  [{label}] Done {cond_name} seed={seed} in {elapsed/60:.1f}min")
            return (cond_name, seed, True, None)
    except subprocess.TimeoutExpired:
        print(f"  [{label}] TIMEOUT {cond_name} seed={seed}")
        return (cond_name, seed, False, "timeout")
    except Exception as e:
        print(f"  [{label}] ERROR {cond_name} seed={seed}: {e}")
        return (cond_name, seed, False, str(e))


# ---------------------------------------------------------------------------
# Summary / reporting
# ---------------------------------------------------------------------------

def print_summary(seeds: List[int]):
    """Parse logs and print mean +/- std SWF for each condition."""
    import numpy as np

    print(f"\n{'='*80}")
    print(f"ICRL Ablation Summary  ({NUM_AGENTS} agents x {MAX_TIMESTEPS} steps, "
          f"seeds={seeds})")
    print(f"{'='*80}")
    header = (f"{'Condition':<20} {'K':>4}  {'Explore':>7}  {'Exploit':>7}  "
              f"{'Final SWF (mean +/- std)':>28}  {'Status':>10}")
    print(header)
    print("-" * 80)

    for cond in CONDITIONS:
        final_swfs = []
        completed = 0
        for seed in seeds:
            swf_vals = parse_swf_from_log(cond["name"], seed)
            if swf_vals:
                final_swfs.append(swf_vals[-1])
                completed += 1

        explore_str = "OFF" if cond["disable_exploration"] else "ON"
        exploit_str = "OFF" if cond["disable_exploitation"] else "ON"

        if final_swfs:
            mean = np.mean(final_swfs)
            std = np.std(final_swfs) if len(final_swfs) > 1 else 0.0
            swf_str = f"{mean:>10.2f} +/- {std:>6.2f}"
        else:
            swf_str = f"{'N/A':>20}"

        status_str = f"{completed}/{len(seeds)}"
        print(f"{cond['name']:<20} {cond['K']:>4}  {explore_str:>7}  {exploit_str:>7}  "
              f"{swf_str:>28}  {status_str:>10}")

    print(f"{'='*80}\n")


def list_conditions(seeds: List[int]):
    """List all conditions and per-seed completion status."""
    print(f"\n{'='*80}")
    print(f"ICRL Ablation Conditions  ({NUM_AGENTS} agents x {MAX_TIMESTEPS} steps)")
    print(f"{'='*80}")
    print(f"{'#':>2}  {'Name':<20} {'K':>4}  {'Explore':>7}  {'Exploit':>7}  {'Seeds'}")
    print("-" * 80)

    total_remaining = 0
    for i, cond in enumerate(CONDITIONS):
        explore_str = "OFF" if cond["disable_exploration"] else "ON"
        exploit_str = "OFF" if cond["disable_exploitation"] else "ON"

        seed_status = []
        for seed in seeds:
            if check_completed(cond["name"], seed):
                seed_status.append(f"[done]{seed}")
            else:
                seed_status.append(f"[todo]{seed}")
                total_remaining += 1

        print(f"{i+1:>2}  {cond['name']:<20} {cond['K']:>4}  {explore_str:>7}  "
              f"{exploit_str:>7}  {' '.join(seed_status)}")

    total_runs = len(CONDITIONS) * len(seeds)
    done = total_runs - total_remaining
    print(f"\nProgress: {done}/{total_runs} runs completed, "
          f"{total_remaining} remaining")
    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run ICRL ablation experiments (K and prompt cue ablations) "
                    "with multiple seeds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                          # Show conditions and status
  %(prog)s --dry-run                       # Print commands without running
  %(prog)s --all                           # Run all conditions sequentially
  %(prog)s --all --parallel 3              # Run 3 experiments in parallel
  %(prog)s --conditions K8 K16 K64         # Run specific conditions only
  %(prog)s --summary                       # Print results summary table
""",
    )

    # Action flags
    parser.add_argument("--all", action="store_true",
                        help="Run all experiment conditions")
    parser.add_argument("--conditions", nargs="+", type=str, default=None,
                        help="Run only these conditions (e.g., K8 K64 K64_no_explore)")
    parser.add_argument("--list", action="store_true",
                        help="List conditions and per-seed completion status")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary table with mean +/- std SWF")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")

    # Execution parameters
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of experiments to run simultaneously (default: 1)")
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="Comma-separated seed list (default: 0,1,2)")
    parser.add_argument("--skip-completed", action="store_true", default=True,
                        help="Skip runs that already completed (default: True)")
    parser.add_argument("--no-skip-completed", action="store_true",
                        help="Re-run even if already completed")

    # Model / inference parameters
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"LLM model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--service", type=str, default=DEFAULT_SERVICE,
                        choices=["vllm", "ollama"],
                        help=f"Inference service (default: {DEFAULT_SERVICE})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Service port (default: {DEFAULT_PORT})")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging for each run")

    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    skip_completed = not args.no_skip_completed

    # Validate --conditions names
    valid_names = {c["name"] for c in CONDITIONS}
    if args.conditions:
        for name in args.conditions:
            if name not in valid_names:
                print(f"Error: unknown condition '{name}'")
                print(f"Valid conditions: {sorted(valid_names)}")
                sys.exit(1)

    # --list
    if args.list:
        list_conditions(seeds)
        return

    # --summary
    if args.summary:
        print_summary(seeds)
        return

    # Determine which conditions to run
    if args.all:
        selected = CONDITIONS
    elif args.conditions:
        selected = [c for c in CONDITIONS if c["name"] in args.conditions]
    elif args.dry_run:
        selected = CONDITIONS
    else:
        parser.print_help()
        return

    # Build the run list: (condition, seed) pairs
    run_list: List[Tuple[Dict, int]] = []
    for cond in selected:
        for seed in seeds:
            if skip_completed and check_completed(cond["name"], seed):
                print(f"Skipping {cond['name']} seed={seed} (already completed)")
                continue
            run_list.append((cond, seed))

    if not run_list:
        print("All selected runs are already completed. Use --no-skip-completed to re-run.")
        print_summary(seeds)
        return

    total = len(run_list)
    print(f"\n{total} runs to execute "
          f"({'DRY RUN' if args.dry_run else f'parallel={args.parallel}'})")
    print(f"Results directory: {RESULTS_ROOT}\n")

    start_time = time.time()

    if args.parallel <= 1 or args.dry_run:
        # Sequential execution
        results = []
        for idx, (cond, seed) in enumerate(run_list):
            label = f"{idx+1}/{total}"
            res = run_single(
                cond, seed, args.model, args.service, args.port,
                args.wandb, dry_run=args.dry_run, label=label,
            )
            results.append(res)
    else:
        # Parallel execution with ProcessPoolExecutor
        results = []
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {}
            for idx, (cond, seed) in enumerate(run_list):
                label = f"{idx+1}/{total}"
                fut = executor.submit(
                    run_single,
                    cond, seed, args.model, args.service, args.port,
                    args.wandb, dry_run=args.dry_run, label=label,
                )
                futures[fut] = (cond["name"], seed)

            for fut in as_completed(futures):
                results.append(fut.result())

    elapsed = time.time() - start_time

    if not args.dry_run:
        # Report
        succeeded = sum(1 for _, _, ok, _ in results if ok)
        failed = total - succeeded
        print(f"\n{'='*60}")
        print(f"Finished {total} runs in {elapsed/60:.1f} min "
              f"({succeeded} succeeded, {failed} failed)")
        print(f"{'='*60}")

        if failed:
            print("\nFailed runs:")
            for cname, seed, ok, err in results:
                if not ok:
                    print(f"  {cname} seed={seed}: {err}")

        # Print summary table
        print_summary(seeds)


if __name__ == "__main__":
    main()
