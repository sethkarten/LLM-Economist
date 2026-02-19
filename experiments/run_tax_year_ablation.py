#!/usr/bin/env python3
"""
Run tax-year-length ablation experiments with multiple seeds for error bars.

Ablation variable: tax_year_length (number of worker timesteps per planner update).
All conditions use history_len=64, 100 agents, bounded scenario, and 32 tax years total.
Model: gemma3-4b (google/gemma-3-4b-it) in BF16 (no quantization).

7 conditions x 10 seeds = 70 total runs.

Tax year length conditions (5):
  TY8   : tax_year_length=8,   max_timesteps=256   (32 tax years)
  TY16  : tax_year_length=16,  max_timesteps=512
  TY32  : tax_year_length=32,  max_timesteps=1024
  TY64  : tax_year_length=64,  max_timesteps=2048
  TY128 : tax_year_length=128, max_timesteps=4096

Prompt ablation conditions (2, at TY64):
  TY64_no_explore : TY=64, exploration disabled
  TY64_no_exploit : TY=64, exploitation disabled

Usage:
    python experiments/run_tax_year_ablation.py --dry-run              # Print commands
    python experiments/run_tax_year_ablation.py --all                  # Run all conditions
    python experiments/run_tax_year_ablation.py --conditions TY8 TY16  # Run specific conditions
    python experiments/run_tax_year_ablation.py --parallel 2           # Run 2 at a time
    python experiments/run_tax_year_ablation.py --summary              # Print summary table
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Experiment conditions
# ---------------------------------------------------------------------------

NUM_TAX_YEARS = 32

CONDITIONS = [
    # Tax year length ablation (both exploration and exploitation enabled)
    {
        "name": "TY8",
        "tax_year_length": 8,
        "max_timesteps": 8 * NUM_TAX_YEARS,   # 256
        "disable_exploration": False,
        "disable_exploitation": False,
        "description": "tax_year_length=8  (32 tax years, 256 total steps)",
    },
    {
        "name": "TY16",
        "tax_year_length": 16,
        "max_timesteps": 16 * NUM_TAX_YEARS,  # 512
        "disable_exploration": False,
        "disable_exploitation": False,
        "description": "tax_year_length=16 (32 tax years, 512 total steps)",
    },
    {
        "name": "TY32",
        "tax_year_length": 32,
        "max_timesteps": 32 * NUM_TAX_YEARS,  # 1024
        "disable_exploration": False,
        "disable_exploitation": False,
        "description": "tax_year_length=32 (32 tax years, 1024 total steps)",
    },
    {
        "name": "TY64",
        "tax_year_length": 64,
        "max_timesteps": 64 * NUM_TAX_YEARS,  # 2048
        "disable_exploration": False,
        "disable_exploitation": False,
        "description": "tax_year_length=64 (32 tax years, 2048 total steps)",
    },
    {
        "name": "TY128",
        "tax_year_length": 128,
        "max_timesteps": 128 * NUM_TAX_YEARS, # 4096
        "disable_exploration": False,
        "disable_exploitation": False,
        "description": "tax_year_length=128 (32 tax years, 4096 total steps)",
    },
    # Prompt ablation at TY64
    {
        "name": "TY64_no_explore",
        "tax_year_length": 64,
        "max_timesteps": 64 * NUM_TAX_YEARS,  # 2048
        "disable_exploration": True,
        "disable_exploitation": False,
        "description": "TY=64, no exploration (exploitation only)",
    },
    {
        "name": "TY64_no_exploit",
        "tax_year_length": 64,
        "max_timesteps": 64 * NUM_TAX_YEARS,  # 2048
        "disable_exploration": False,
        "disable_exploitation": True,
        "description": "TY=64, no exploitation (exploration only)",
    },
]

# Fixed simulation parameters
NUM_AGENTS = 100
HISTORY_LEN = 64       # Fixed for all conditions
SCENARIO = "bounded"
DEFAULT_SEEDS = list(range(10))  # seeds 0-9
DEFAULT_MODEL = "gemma3-4b"
DEFAULT_QUANTIZATION = "none"  # BF16, no quantization

RESULTS_ROOT = PROJECT_ROOT / "results" / "tax_year_ablation"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_output_dir(condition_name: str, seed: int) -> Path:
    return RESULTS_ROOT / condition_name / f"seed_{seed}"


def build_command(
    condition: Dict,
    seed: int,
    model: str,
    quantization: str,
    wandb: bool,
) -> List[str]:
    output_dir = get_output_dir(condition["name"], seed)
    cond_name = condition["name"]

    cmd = [
        sys.executable, "-m", "llm_economist.main_async",
        "--scenario", SCENARIO,
        "--num-agents", str(NUM_AGENTS),
        "--max-timesteps", str(condition["max_timesteps"]),
        "--history-len", str(HISTORY_LEN),
        "--tax-year-length", str(condition["tax_year_length"]),
        "--model", model,
        "--quantization", quantization,
        "--tensor-parallel", "1",
        "--seed", str(seed),
        "--output", str(output_dir / f"{cond_name}_seed{seed}.json"),
    ]

    if condition["disable_exploration"]:
        cmd.append("--disable-exploration")
    if condition["disable_exploitation"]:
        cmd.append("--disable-exploitation")
    if wandb:
        cmd.append("--wandb")

    return cmd


def check_completed(condition_name: str, seed: int) -> bool:
    output_dir = get_output_dir(condition_name, seed)
    json_file = output_dir / f"{condition_name}_seed{seed}.json"
    if json_file.exists():
        try:
            data = json.loads(json_file.read_text())
            return "metrics_history" in data and len(data["metrics_history"]) > 0
        except Exception:
            pass
    return False


def parse_swf_from_output(condition_name: str, seed: int) -> Optional[List[float]]:
    output_dir = get_output_dir(condition_name, seed)
    json_file = output_dir / f"{condition_name}_seed{seed}.json"
    if json_file.exists():
        try:
            data = json.loads(json_file.read_text())
            metrics = data.get("metrics_history", [])
            swf_values = [m["swf"] for m in metrics if "swf" in m]
            return swf_values if swf_values else None
        except Exception:
            pass
    return None


def run_single(
    condition: Dict,
    seed: int,
    model: str,
    quantization: str,
    wandb: bool,
    dry_run: bool = False,
    label: str = "",
) -> Tuple[str, int, bool, Optional[str]]:
    cond_name = condition["name"]
    output_dir = get_output_dir(cond_name, seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(condition, seed, model, quantization, wandb)

    if dry_run:
        print(f"  [{label}] {' '.join(cmd)}")
        return (cond_name, seed, True, None)

    print(f"  [{label}] Starting {cond_name} seed={seed} "
          f"(TY={condition['tax_year_length']}, steps={condition['max_timesteps']}) ...")
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=3600 * 24,  # 24-hour hard timeout
        )
        elapsed = time.time() - start
        if result.returncode != 0:
            err_msg = result.stderr[-500:] if result.stderr else "unknown error"
            print(f"  [{label}] FAILED {cond_name} seed={seed} "
                  f"(exit {result.returncode}, {elapsed/60:.1f}min)")
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
    try:
        import numpy as np
    except ImportError:
        print("numpy not available; install it to display summary statistics.")
        return

    print(f"\n{'='*90}")
    print(f"Tax Year Ablation Summary  ({NUM_AGENTS} agents, history_len={HISTORY_LEN}, "
          f"seeds={seeds})")
    print(f"{'='*90}")
    header = (f"{'Condition':<20} {'TY':>4}  {'Steps':>5}  {'Explore':>7}  {'Exploit':>7}  "
              f"{'Final SWF (mean +/- std)':>28}  {'Status':>10}")
    print(header)
    print("-" * 90)

    for cond in CONDITIONS:
        final_swfs = []
        completed = 0
        for seed in seeds:
            swf_vals = parse_swf_from_output(cond["name"], seed)
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
        print(f"{cond['name']:<20} {cond['tax_year_length']:>4}  "
              f"{cond['max_timesteps']:>5}  {explore_str:>7}  {exploit_str:>7}  "
              f"{swf_str:>28}  {status_str:>10}")

    print(f"{'='*90}\n")


def list_conditions(seeds: List[int]):
    print(f"\n{'='*90}")
    print(f"Tax Year Ablation Conditions  ({NUM_AGENTS} agents, "
          f"history_len={HISTORY_LEN}, model={DEFAULT_MODEL})")
    print(f"{'='*90}")
    print(f"{'#':>2}  {'Name':<20} {'TY':>4}  {'Steps':>5}  {'Explore':>7}  "
          f"{'Exploit':>7}  {'Seeds'}")
    print("-" * 90)

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

        print(f"{i+1:>2}  {cond['name']:<20} {cond['tax_year_length']:>4}  "
              f"{cond['max_timesteps']:>5}  {explore_str:>7}  {exploit_str:>7}  "
              f"{'  '.join(seed_status)}")

    total_runs = len(CONDITIONS) * len(seeds)
    done = total_runs - total_remaining
    print(f"\nProgress: {done}/{total_runs} runs completed, "
          f"{total_remaining} remaining")
    print(f"{'='*90}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run tax year length ablation experiments with multiple seeds. "
                    "Each condition produces exactly 32 tax years.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                              # Show conditions and status
  %(prog)s --dry-run                           # Print commands without running
  %(prog)s --all                               # Run all conditions sequentially
  %(prog)s --all --parallel 3                  # Run 3 experiments in parallel
  %(prog)s --conditions TY8 TY16 TY64         # Run specific conditions only
  %(prog)s --summary                           # Print results summary table
""",
    )

    parser.add_argument("--all", action="store_true",
                        help="Run all experiment conditions")
    parser.add_argument("--conditions", nargs="+", type=str, default=None,
                        help="Run only these conditions (e.g., TY8 TY64 TY64_no_explore)")
    parser.add_argument("--list", action="store_true",
                        help="List conditions and per-seed completion status")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary table with mean +/- std SWF")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")

    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of experiments to run simultaneously (default: 1)")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS),
                        help="Comma-separated seed list (default: 0-9)")
    parser.add_argument("--skip-completed", action="store_true", default=True,
                        help="Skip runs that already completed (default: True)")
    parser.add_argument("--no-skip-completed", action="store_true",
                        help="Re-run even if already completed")

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"LLM model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--quantization", type=str, default=DEFAULT_QUANTIZATION,
                        choices=["awq", "gptq", "fp8", "bitsandbytes", "none"],
                        help=f"Quantization method (default: {DEFAULT_QUANTIZATION})")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging for each run")

    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    skip_completed = not args.no_skip_completed

    valid_names = {c["name"] for c in CONDITIONS}
    if args.conditions:
        for name in args.conditions:
            if name not in valid_names:
                print(f"Error: unknown condition '{name}'")
                print(f"Valid conditions: {sorted(valid_names)}")
                sys.exit(1)

    if args.list:
        list_conditions(seeds)
        return

    if args.summary:
        print_summary(seeds)
        return

    if args.all:
        selected = CONDITIONS
    elif args.conditions:
        selected = [c for c in CONDITIONS if c["name"] in args.conditions]
    elif args.dry_run:
        selected = CONDITIONS
    else:
        parser.print_help()
        return

    run_list: List[Tuple[Dict, int]] = []
    for cond in selected:
        for seed in seeds:
            if skip_completed and check_completed(cond["name"], seed):
                print(f"Skipping {cond['name']} seed={seed} (already completed)")
                continue
            run_list.append((cond, seed))

    if not run_list:
        print("All selected runs are already completed. "
              "Use --no-skip-completed to re-run.")
        print_summary(seeds)
        return

    total = len(run_list)
    print(f"\n{total} runs to execute "
          f"({'DRY RUN' if args.dry_run else f'parallel={args.parallel}'})")
    print(f"Model: {args.model}, quantization: {args.quantization}")
    print(f"Fixed params: history_len={HISTORY_LEN}, num_agents={NUM_AGENTS}, "
          f"scenario={SCENARIO}")
    print(f"Results directory: {RESULTS_ROOT}\n")

    start_time = time.time()

    if args.parallel <= 1 or args.dry_run:
        results = []
        for idx, (cond, seed) in enumerate(run_list):
            label = f"{idx+1}/{total}"
            res = run_single(
                cond, seed, args.model, args.quantization,
                args.wandb, dry_run=args.dry_run, label=label,
            )
            results.append(res)
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {}
            for idx, (cond, seed) in enumerate(run_list):
                label = f"{idx+1}/{total}"
                fut = executor.submit(
                    run_single,
                    cond, seed, args.model, args.quantization,
                    args.wandb, dry_run=args.dry_run, label=label,
                )
                futures[fut] = (cond["name"], seed)

            for fut in as_completed(futures):
                results.append(fut.result())

    elapsed = time.time() - start_time

    if not args.dry_run:
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

        print_summary(seeds)


if __name__ == "__main__":
    main()
