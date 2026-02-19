"""
Debug script for diagnosing why Gemma-3-4B-IT never changes tax rates
in the ICRL tax planner.

Replicates the EXACT planner prompt that _build_planner_prompt() would
build at the start of tax year 2 (timestep=64, tax_year_length=64),
calls the model 5 times, and inspects the raw responses.

Usage:
    uv run python experiments/debug_planner_response.py
"""

import asyncio
import json
import logging
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_economist.inference.async_engine import ScalableInferenceEngine
from llm_economist.inference.config import get_model_config, QuantizationType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants matching AsyncLLMEconomist defaults
# ---------------------------------------------------------------------------
US_FED_TAX_RATES = [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]
US_FED_TAX_BRACKETS = [0, 23000, 47000, 94000, 192000, 244000, 500000, 1000000]
NUM_AGENTS = 100
TAX_YEAR_LENGTH = 64
SWF_WEIGHTING = "rawlsian"  # default in AsyncLLMEconomist


def build_planner_prompt(
    timestep: int,
    current_tax_rates: list[float],
    planner_history: list[dict],
    worker_stats: list[dict],
    swf_weighting: str = SWF_WEIGHTING,
    disable_exploration: bool = False,
    disable_exploitation: bool = False,
    history_len: int = 50,
    total_tax_collected: float = 0.0,
    current_swf: float = 0.0,
) -> tuple[str, str]:
    """
    Exact replica of AsyncLLMEconomist._build_planner_prompt().
    """
    if swf_weighting == "utilitarian":
        swf_formula = "Social welfare = sum of utilities across all agents."
    else:
        swf_formula = "Social welfare = sum of (utility / pre_tax_income) across all agents."

    exploration_cue = ""
    if not disable_exploration:
        exploration_cue = "\n4. Explore diverse tax rates to find the best ones"

    exploitation_cue = ""
    if not disable_exploitation:
        exploitation_cue = "\n5. Use historically successful rates to guide decisions"

    system_prompt = f"""You are a tax policy planner trying to maximize social welfare.
{swf_formula}

You can adjust marginal tax rates for each bracket.
Your goal is to find tax rates that:
1. Raise sufficient revenue for government services
2. Redistribute to improve overall welfare
3. Don't discourage work too much{exploration_cue}{exploitation_cue}
"""

    # Summarize worker statistics
    incomes = [s["income"] for s in worker_stats]
    utilities = [s["utility"] for s in worker_stats]

    # Build historical context for ICRL
    history_text = ""
    if planner_history:
        K = min(history_len, len(planner_history))
        recent = planner_history[-K:]
        history_text += "Historical data:\n"
        for entry in recent:
            rates_str = [f"{r*100:.1f}%" for r in entry["tax_rates"]]
            if swf_weighting == "utilitarian":
                swf_label = f"swf = u_1 + ... + u_N = {entry['swf']:.4f}"
            else:
                swf_label = f"swf = u_1/z_1 + ... + u_N/z_N = {entry['swf']:.4f}"
            history_text += (
                f"Timestep {entry['timestep']}: "
                f"tax_rates={rates_str}, "
                f"social welfare: {swf_label}, "
                f"total_tax=${entry['total_tax']:.2f}, "
                f"mean_income=${entry['mean_income']:.2f}, "
                f"mean_utility={entry['mean_utility']:.2f}, "
                f"gini={entry['gini']:.3f}\n"
            )

        # Top-5 best timesteps by SWF
        sorted_by_swf = sorted(planner_history, key=lambda x: x["swf"], reverse=True)
        top_n = min(5, len(sorted_by_swf))
        history_text += f"\nBest {top_n} timesteps:\n"
        for entry in sorted_by_swf[:top_n]:
            rates_str = [f"{r*100:.1f}%" for r in entry["tax_rates"]]
            history_text += (
                f"Timestep {entry['timestep']}: "
                f"tax_rates={rates_str}, "
                f"SWF={entry['swf']:.4f}\n"
            )
        history_text += "\n"

    # Exploration / exploitation cues (gated by flags)
    cue_text = ""
    if not disable_exploration and planner_history:
        cue_text += "Use the historical data to influence your answer in order to maximize SWF, while balancing exploration and exploitation by choosing varying rates of TAX. "
        cue_text += "Try different rates of TAX before picking the one that corresponds to the highest SWF. "
    if not disable_exploitation and planner_history:
        K = min(history_len, len(planner_history))
        recent = planner_history[-K:]
        best_entry = max(recent, key=lambda x: x["swf"])
        best_rates = [f"{r*100:.1f}%" for r in best_entry["tax_rates"]]
        cue_text += f"The best marginal tax rate historically was TAX={best_rates} corresponding to SWF={best_entry['swf']:.4f}. "

    def _calculate_gini(values):
        if len(values) < 2:
            return 0.0
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        return (2 * np.sum((np.arange(1, n + 1) * sorted_values)) / (n * cumsum[-1])) - (n + 1) / n

    user_prompt = f"""{history_text}Timestep {timestep}:
Current tax rates: {[f"{r*100:.1f}%" for r in current_tax_rates]}
Total tax collected: ${total_tax_collected:.2f}
Current SWF: {current_swf:.4f}

Worker statistics (N={len(worker_stats)}):
- Mean income: ${np.mean(incomes):.2f}
- Median income: ${np.median(incomes):.2f}
- Mean utility: {np.mean(utilities):.2f}
- Income Gini: {_calculate_gini(np.array(incomes)):.3f}

{cue_text}
Propose new tax rates.
Respond with JSON: {{"tax_rates": [rate1, rate2, ...], "reasoning": "<explanation>"}}
"""
    return system_prompt, user_prompt


def generate_fake_history(num_timesteps: int = 64) -> list[dict]:
    """
    Generate a realistic planner_history list for 64 timesteps of constant
    US Federal tax rates with SWF around 200.
    """
    np.random.seed(42)
    history = []
    for t in range(num_timesteps):
        swf_noise = np.random.normal(0, 5)
        mean_income = 5000.0 + np.random.normal(0, 200)
        mean_utility = 4800.0 + np.random.normal(0, 200)
        gini = 0.50 + np.random.normal(0, 0.02)
        total_tax = mean_income * NUM_AGENTS * 0.18 + np.random.normal(0, 1000)

        history.append({
            "timestep": t,
            "tax_rates": US_FED_TAX_RATES.copy(),
            "swf": 200.0 + swf_noise,
            "total_tax": total_tax,
            "mean_income": mean_income,
            "mean_utility": mean_utility,
            "gini": max(0.0, min(1.0, gini)),
        })
    return history


def generate_fake_worker_stats(num_agents: int = 100) -> list[dict]:
    """
    Generate realistic worker stats matching the format used in _update_tax_rates().
    Skills sampled from a rough GB2-like lognormal for simplicity.
    """
    np.random.seed(123)
    skills = np.random.lognormal(mean=3.5, sigma=1.0, size=num_agents)
    labors = np.clip(np.random.normal(40, 10, size=num_agents), 5, 80)
    incomes = skills * labors
    utilities = incomes * 0.82 - 0.01 * (labors ** 2)

    return [
        {"income": float(incomes[i]), "utility": float(utilities[i])}
        for i in range(num_agents)
    ]


def parse_tax_rates_like_main(response_str: str, current_rates: list[float]) -> tuple[list[float] | None, str]:
    """
    Replicate the exact parsing logic from AsyncLLMEconomist._update_tax_rates().
    Returns (parsed_rates_or_None, diagnostic_message).
    """
    try:
        data = json.loads(response_str)
    except (json.JSONDecodeError, TypeError) as e:
        return None, f"JSON parse failed: {e}"

    new_rates = data.get("tax_rates", current_rates)

    if len(new_rates) != len(current_rates):
        return None, f"Wrong number of rates: got {len(new_rates)}, expected {len(current_rates)}"

    parsed_rates = []
    for r in new_rates:
        if isinstance(r, str):
            r = r.strip().rstrip("%")
            val = float(r)
            if val > 1:
                val = val / 100.0
        else:
            val = float(r)
        parsed_rates.append(max(0.0, min(0.99, val)))

    return parsed_rates, "OK"


def rates_are_identical(rates_a: list[float], rates_b: list[float], tol: float = 1e-6) -> bool:
    """Check if two rate vectors are effectively identical."""
    if len(rates_a) != len(rates_b):
        return False
    return all(abs(a - b) < tol for a, b in zip(rates_a, rates_b))


async def main():
    # -----------------------------------------------------------------------
    # 1. Load model via the inference config system
    # -----------------------------------------------------------------------
    model_key = os.environ.get("DEBUG_MODEL", "qwen3-8b-fp8")
    model_config = get_model_config(model_key)
    print(f"Model key: {model_key}")
    print(f"HuggingFace name: {model_config.hf_name}")
    print(f"Recommended quantization: {model_config.recommended_quantization}")
    print(f"Text-only mode: {model_config.text_only_mode}")
    print()

    # Determine quantization and dtype from config
    quant_value = None
    dtype = None
    if model_config.recommended_quantization == QuantizationType.NONE:
        quant_value = None
        dtype = "bfloat16"
    elif model_config.recommended_quantization == QuantizationType.FP8:
        quant_value = "fp8"
    elif model_config.recommended_quantization == QuantizationType.AWQ:
        quant_value = "awq"

    engine = ScalableInferenceEngine(
        model_name=model_config.hf_name,
        tensor_parallel_size=1,
        quantization=quant_value,
        max_model_len=32768,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        kv_cache_dtype="auto",
        max_num_seqs=256,
        text_only_mode=model_config.text_only_mode,
        dtype=dtype,
    )
    await engine.initialize()

    # -----------------------------------------------------------------------
    # 2. Build the EXACT planner prompt for tax year 2 (timestep=64)
    # -----------------------------------------------------------------------
    planner_history = generate_fake_history(num_timesteps=TAX_YEAR_LENGTH)
    worker_stats = generate_fake_worker_stats(NUM_AGENTS)

    # Compute aggregate values as the simulation would have at t=64
    incomes = [s["income"] for s in worker_stats]
    total_tax_collected = sum(i * 0.18 for i in incomes)  # rough estimate
    current_swf = planner_history[-1]["swf"]

    sys_prompt, user_prompt = build_planner_prompt(
        timestep=TAX_YEAR_LENGTH,
        current_tax_rates=US_FED_TAX_RATES,
        planner_history=planner_history,
        worker_stats=worker_stats,
        swf_weighting=SWF_WEIGHTING,
        disable_exploration=False,
        disable_exploitation=False,
        history_len=50,
        total_tax_collected=total_tax_collected,
        current_swf=current_swf,
    )

    print("=" * 80)
    print("SYSTEM PROMPT:")
    print("=" * 80)
    print(sys_prompt)
    print()
    print("=" * 80)
    print("USER PROMPT (first 2000 chars):")
    print("=" * 80)
    print(user_prompt[:2000])
    if len(user_prompt) > 2000:
        print(f"... ({len(user_prompt)} chars total, truncated)")
    print()

    # Also show the raw formatted prompt that vLLM actually sees
    raw_formatted = engine._format_prompt(sys_prompt, user_prompt)
    print("=" * 80)
    print(f"RAW FORMATTED PROMPT LENGTH: {len(raw_formatted)} chars")
    print("=" * 80)
    print()

    # -----------------------------------------------------------------------
    # 3. Call the model 5 times and inspect responses
    # -----------------------------------------------------------------------
    NUM_TRIALS = 5
    changed_count = 0

    for trial in range(NUM_TRIALS):
        print(f"{'=' * 80}")
        print(f"TRIAL {trial + 1}/{NUM_TRIALS}")
        print(f"{'=' * 80}")

        response_text, is_json_valid = await engine.generate_single(
            sys_prompt,
            user_prompt,
            temperature=0.7,
            max_tokens=512,
            json_format=True,
            request_id=f"planner_debug_{trial}",
        )

        print(f"Raw response ({len(response_text)} chars):")
        print(repr(response_text))
        print()
        print(f"Is JSON valid (after _extract_json): {is_json_valid}")
        print()

        # Also try to generate WITHOUT json_format to see what the model
        # actually outputs before _extract_json processing
        response_raw, _ = await engine.generate_single(
            sys_prompt,
            user_prompt,
            temperature=0.7,
            max_tokens=512,
            json_format=False,
            request_id=f"planner_debug_raw_{trial}",
        )
        print(f"Raw response WITHOUT json_format ({len(response_raw)} chars):")
        print(repr(response_raw[:1500]))
        if len(response_raw) > 1500:
            print(f"... ({len(response_raw)} chars total, truncated)")
        print()

        # Attempt to parse as the simulation does
        parsed_rates, diag = parse_tax_rates_like_main(response_text, US_FED_TAX_RATES)
        print(f"Parse result: {diag}")

        if parsed_rates is not None:
            print(f"Parsed tax rates: {parsed_rates}")
            print(f"As percentages:   {[f'{r*100:.1f}%' for r in parsed_rates]}")
            print(f"Original US Fed:  {US_FED_TAX_RATES}")
            print(f"As percentages:   {[f'{r*100:.1f}%' for r in US_FED_TAX_RATES]}")

            identical = rates_are_identical(parsed_rates, US_FED_TAX_RATES)
            if identical:
                print(">>> RATES ARE IDENTICAL TO US FED DEFAULT (no change)")
            else:
                print(">>> RATES CHANGED!")
                changed_count += 1

            # Check if model output percentages vs decimals
            try:
                data = json.loads(response_text)
                raw_values = data.get("tax_rates", [])
                any_above_1 = any(
                    (float(str(v).strip().rstrip("%")) > 1) if isinstance(v, str)
                    else (float(v) > 1)
                    for v in raw_values
                )
                print(f"Raw values in JSON: {raw_values}")
                print(f"Values appear to be {'PERCENTAGES (>1)' if any_above_1 else 'DECIMALS (0-1)'}")
            except Exception:
                pass
        else:
            print(f">>> PARSE FAILED: {diag}")
            # Try to show what the model actually returned
            try:
                data = json.loads(response_text)
                print(f"JSON keys present: {list(data.keys())}")
                if "tax_rates" in data:
                    print(f"tax_rates value: {data['tax_rates']}")
                    print(f"tax_rates length: {len(data['tax_rates'])}")
            except Exception:
                pass

        print()

    # -----------------------------------------------------------------------
    # 4. Summary
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Model: {model_config.hf_name}")
    print(f"Trials: {NUM_TRIALS}")
    print(f"Rates changed: {changed_count}/{NUM_TRIALS}")
    if changed_count == 0:
        print()
        print("DIAGNOSIS: The model NEVER proposed different tax rates.")
        print("Possible causes:")
        print("  1. Model copies rates from the prompt (system/user prompt shows current rates)")
        print("  2. JSON parsing succeeds but data.get('tax_rates') returns defaults")
        print("  3. Model outputs rates as percentages that get clipped to 0.99")
        print("  4. The stop=['}'] in json_format truncates the response before tax_rates is complete")
        print("  5. Model does not follow JSON instruction format and _extract_json fails silently")
    print()

    # -----------------------------------------------------------------------
    # 5. Additional diagnostic: test without json_format stop token
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("EXTRA DIAGNOSTIC: What does the model output with NO stop token and NO json_format?")
    print("=" * 80)
    response_full, _ = await engine.generate_single(
        sys_prompt,
        user_prompt,
        temperature=0.7,
        max_tokens=1024,
        json_format=False,
        request_id="planner_debug_full",
    )
    print(f"Full response ({len(response_full)} chars):")
    print(response_full[:3000])
    if len(response_full) > 3000:
        print(f"... ({len(response_full)} chars total)")
    print()

    # Try manual JSON extraction on the full response
    json_start = response_full.find("{")
    json_end = response_full.rfind("}") + 1
    if json_start != -1 and json_end > 0:
        candidate = response_full[json_start:json_end]
        print(f"Manual JSON extraction: {candidate[:500]}")
        try:
            data = json.loads(candidate)
            print(f"Valid JSON! Keys: {list(data.keys())}")
            if "tax_rates" in data:
                print(f"tax_rates: {data['tax_rates']}")
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
    else:
        print("No JSON braces found in full response")

    # -----------------------------------------------------------------------
    # 6. Test with a MINIMAL prompt to confirm model can output different rates
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("SANITY CHECK: Minimal prompt asking for different tax rates")
    print("=" * 80)
    minimal_sys = "You are a tax policy expert. Respond only with JSON."
    minimal_user = (
        "The current US Federal marginal tax rates are [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]. "
        "Propose a new set of 7 marginal tax rates that would improve income equality. "
        "The rates should be DIFFERENT from the current ones. "
        'Respond ONLY with: {"tax_rates": [r1, r2, r3, r4, r5, r6, r7], "reasoning": "..."}'
    )
    for i in range(3):
        resp, valid = await engine.generate_single(
            minimal_sys,
            minimal_user,
            temperature=0.7,
            max_tokens=256,
            json_format=True,
            request_id=f"sanity_{i}",
        )
        parsed, diag = parse_tax_rates_like_main(resp, US_FED_TAX_RATES)
        identical = rates_are_identical(parsed, US_FED_TAX_RATES) if parsed else "N/A"
        print(f"  Sanity trial {i+1}: valid={valid}, identical_to_default={identical}")
        print(f"    Raw: {repr(resp[:300])}")
        if parsed:
            print(f"    Parsed rates: {[f'{r*100:.1f}%' for r in parsed]}")
        print()

    # Shutdown
    await engine.shutdown()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
