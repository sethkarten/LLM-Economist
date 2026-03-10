#!/usr/bin/env python3
"""Evaluate a REINFORCE++-trained planner on the standard 100-agent, 2000-step setup.

Loads the best LoRA checkpoint, runs AsyncLLMEconomist with external_planner=True,
and reports SWF/Gini comparable to the paper's evaluation tables.

Usage:
    uv run python experiments/eval_reinforce_planner.py \
        --checkpoint results/reinforce_v2_seed42/best \
        --seed 42 --output results/reinforce_eval/seed42.json

    # US Federal baseline (no planner)
    uv run python experiments/eval_reinforce_planner.py \
        --baseline-only --seed 42 --output results/reinforce_eval/baseline_seed42.json
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model  # noqa: F401


# Reuse prompt templates from training
PLANNER_SYSTEM_PROMPT = """You are a tax policy planner for an economy with {num_agents} workers.
Your goal is to set {num_brackets} marginal tax bracket rates (each 0-100%) to maximize social welfare.
Workers respond to tax rates by choosing how many hours to work (0-100).
Utility = post_tax_income + rebate - 0.0005 * labor^3.5
Higher taxes fund redistribution but may reduce work incentives.
Respond ONLY with a JSON object: {{"tax_rates": [rate1, rate2, ...]}} where rates are percentages (0-100).
Tax brackets: {brackets}"""

PLANNER_USER_TEMPLATE = """Economic State:
- Number of workers: {num_agents}
- Mean pre-tax income: ${mean_income:,.0f}
- Median pre-tax income: ${median_income:,.0f}
- Income Gini: {gini:.3f}
- Current tax rates: {current_rates}
- Previous SWF: {prev_swf:.1f}
- US Federal baseline SWF: {baseline_swf:.1f}

Set {num_brackets} marginal tax rates (0-100%) to maximize social welfare.
Respond with JSON only: {{"tax_rates": [rate1, rate2, ...]}}"""

JSON_PREFIX = '{"tax_rates": ['


def load_planner(checkpoint_path: str, base_model: str = "google/gemma-3-4b-it", device: str = "cuda:0"):
    """Load the base model + LoRA checkpoint for inference."""
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    checkpoint = Path(checkpoint_path)
    if checkpoint.exists() and (checkpoint / "adapter_config.json").exists():
        # Load base model then apply saved LoRA adapter
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
        )
        model = PeftModel.from_pretrained(model, str(checkpoint))
        print(f"Loaded LoRA weights from {checkpoint}")
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint}, using base model with fresh LoRA")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
        )
        lora_config = LoraConfig(
            r=8, lora_alpha=8, lora_dropout=0.0,
            target_modules=["gate_proj", "up_proj", "down_proj"],
            bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.eval()
    return model, tokenizer


def generate_tax_rates(model, tokenizer, system_prompt, user_prompt, device="cuda:0"):
    """Generate tax rates using the planner model with forced JSON prefix."""
    # Build chat prompt
    messages = [
        {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"},
    ]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_with_prefix = full_prompt + JSON_PREFIX

    inputs = tokenizer(prompt_with_prefix, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.3,  # Lower temp for eval (more deterministic)
            do_sample=True,
            top_p=0.9,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    raw_suffix = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if '}' in raw_suffix:
        raw_suffix = raw_suffix[:raw_suffix.index('}') + 1]

    full_response = JSON_PREFIX + raw_suffix

    try:
        parsed = json.loads(full_response)
        rates = parsed.get("tax_rates", [])
        # Convert percentages to fractions if needed
        if any(r > 1 for r in rates):
            rates = [r / 100.0 for r in rates]
        rates = [max(0.0, min(1.0, r)) for r in rates]
        return rates
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def get_observation(sim):
    """Get current economic state for planner observation."""
    agents = sim.state.agent_states
    incomes = [a.income for a in agents]
    return {
        'num_agents': len(agents),
        'mean_income': float(np.mean(incomes)) if incomes else 0,
        'median_income': float(np.median(incomes)) if incomes else 0,
        'gini': sim._calculate_gini(incomes),
        'current_rates': list(sim.state.tax_rates),
        'prev_swf': sim.state.swf,
        'tax_brackets': list(sim.state.tax_brackets),
    }


def format_observation(obs, baseline_swf, num_brackets):
    """Format observation into prompts."""
    system_prompt = PLANNER_SYSTEM_PROMPT.format(
        num_agents=obs['num_agents'],
        num_brackets=num_brackets,
        brackets=obs['tax_brackets'],
    )
    user_prompt = PLANNER_USER_TEMPLATE.format(
        num_agents=obs['num_agents'],
        mean_income=obs['mean_income'],
        median_income=obs['median_income'],
        gini=obs['gini'],
        current_rates=[f"{r * 100:.1f}%" for r in obs['current_rates']],
        prev_swf=obs['prev_swf'],
        baseline_swf=baseline_swf,
        num_brackets=num_brackets,
    )
    return system_prompt, user_prompt


async def run_evaluation(
    checkpoint_path: str,
    seed: int = 42,
    num_agents: int = 100,
    max_timesteps: int = 2000,
    tax_year_length: int = 128,
    bracket_setting: str = "three",
    swf_weighting: str = "rawlsian",
    worker_model: str = "Qwen/Qwen3-8B-AWQ",
    baseline_only: bool = False,
    planner_device: str = "cuda:0",
    worker_gpu: str = None,
    gpu_memory_util: float = 0.95,
):
    """Run full evaluation of trained planner."""
    from llm_economist.main_async import AsyncLLMEconomist
    from llm_economist.utils.bracket import get_num_brackets

    num_brackets = get_num_brackets(bracket_setting)

    # US Federal rates for baseline
    if num_brackets == 3:
        us_rates = [0.12, 0.24, 0.35]
    elif num_brackets == 7:
        us_rates = [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]
    else:
        us_rates = [0.22]

    # Set up vLLM on worker GPU
    original_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    if worker_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = worker_gpu

    sim = AsyncLLMEconomist(
        num_agents=num_agents,
        max_timesteps=max_timesteps,
        model_name=worker_model,
        tensor_parallel_size=1,
        tax_year_length=tax_year_length,
        scenario="bounded",
        quantization="awq",
        batch_size=num_agents,
        seed=seed,
        external_planner=True,
        bracket_setting=bracket_setting,
        swf_weighting=swf_weighting,
        gpu_memory_utilization=gpu_memory_util,
    )
    await sim.initialize()

    # Restore CUDA visibility
    if worker_gpu is not None:
        if original_cuda is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda
        else:
            del os.environ["CUDA_VISIBLE_DEVICES"]

    # Load planner (skip if baseline-only)
    planner_model = None
    planner_tokenizer = None
    if not baseline_only:
        planner_model, planner_tokenizer = load_planner(checkpoint_path, device=planner_device)

    # Run baseline first
    print("\n=== Running US Federal baseline ===")
    sim.set_tax_rates(us_rates)
    for step in range(max_timesteps):
        if step % tax_year_length == 0 and step > 0:
            sim.set_tax_rates(us_rates)  # Keep constant
        await sim.step()
        if step % 200 == 0:
            print(f"  Baseline step {step}/{max_timesteps}: SWF={sim.state.swf:.2f}")

    baseline_swf = sim.state.swf
    baseline_gini = sim._calculate_gini([a.income for a in sim.state.agent_states])
    print(f"Baseline SWF: {baseline_swf:.2f}, Gini: {baseline_gini:.3f}")

    if baseline_only:
        return {
            'type': 'baseline',
            'seed': seed,
            'swf': baseline_swf,
            'gini': baseline_gini,
            'tax_rates': us_rates,
        }

    # Run REINFORCE++ planner evaluation
    print("\n=== Running REINFORCE++ planner ===")
    sim.reset_state()

    tax_rate_history = []
    swf_history = []
    format_failures = 0
    total_tax_years = 0

    for step in range(max_timesteps):
        is_tax_year_start = step % tax_year_length == 0

        if is_tax_year_start:
            total_tax_years += 1
            obs = get_observation(sim)
            sys_prompt, user_prompt = format_observation(obs, baseline_swf, num_brackets)
            rates = generate_tax_rates(planner_model, planner_tokenizer, sys_prompt, user_prompt, device=planner_device)

            if rates is None or len(rates) != num_brackets:
                rates = us_rates  # Fallback
                format_failures += 1
                print(f"  Tax year {total_tax_years}: FALLBACK to US Federal (parse failed)")
            else:
                print(f"  Tax year {total_tax_years}: rates={[f'{r*100:.1f}%' for r in rates]}")

            sim.set_tax_rates(rates)
            tax_rate_history.append(rates)

        await sim.step()

        if step % 200 == 0:
            swf_history.append({'step': step, 'swf': sim.state.swf})
            print(f"  Step {step}/{max_timesteps}: SWF={sim.state.swf:.2f}")

    final_swf = sim.state.swf
    final_gini = sim._calculate_gini([a.income for a in sim.state.agent_states])
    format_rate = 1.0 - (format_failures / max(total_tax_years, 1))

    print(f"\n=== Results ===")
    print(f"REINFORCE++ SWF: {final_swf:.2f} (baseline: {baseline_swf:.2f})")
    print(f"Improvement: {(final_swf - baseline_swf) / baseline_swf * 100:.1f}%")
    print(f"Gini: {final_gini:.3f}")
    print(f"Format success: {format_rate*100:.0f}%")

    # Shutdown
    if sim.engine:
        await sim.engine.shutdown()

    return {
        'type': 'reinforce_eval',
        'seed': seed,
        'checkpoint': checkpoint_path,
        'swf': final_swf,
        'gini': final_gini,
        'baseline_swf': baseline_swf,
        'baseline_gini': baseline_gini,
        'improvement_pct': (final_swf - baseline_swf) / baseline_swf * 100,
        'format_success_rate': format_rate,
        'total_tax_years': total_tax_years,
        'format_failures': format_failures,
        'tax_rate_history': [[float(r) for r in rates] for rates in tax_rate_history],
        'swf_history': swf_history,
        'config': {
            'num_agents': num_agents,
            'max_timesteps': max_timesteps,
            'tax_year_length': tax_year_length,
            'bracket_setting': bracket_setting,
            'swf_weighting': swf_weighting,
            'worker_model': worker_model,
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate REINFORCE++ trained planner')
    parser.add_argument('--checkpoint', type=str, default='results/reinforce_v2_seed42/best',
                        help='Path to LoRA checkpoint directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-agents', type=int, default=100)
    parser.add_argument('--max-timesteps', type=int, default=2000)
    parser.add_argument('--tax-year-length', type=int, default=128)
    parser.add_argument('--bracket-setting', type=str, default='three')
    parser.add_argument('--swf-weighting', type=str, default='rawlsian',
                        choices=['rawlsian', 'utilitarian'],
                        help='Social welfare function weighting')
    parser.add_argument('--worker-model', type=str, default='Qwen/Qwen3-8B-AWQ')
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--worker-gpu', type=str, default=None,
                        help='Physical GPU ID for vLLM workers (e.g. "1")')
    parser.add_argument('--gpu-memory-util', type=float, default=0.95)
    args = parser.parse_args()

    result = asyncio.run(run_evaluation(
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        num_agents=args.num_agents,
        max_timesteps=args.max_timesteps,
        tax_year_length=args.tax_year_length,
        bracket_setting=args.bracket_setting,
        swf_weighting=args.swf_weighting,
        worker_model=args.worker_model,
        baseline_only=args.baseline_only,
        worker_gpu=args.worker_gpu,
        gpu_memory_util=args.gpu_memory_util,
    ))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
