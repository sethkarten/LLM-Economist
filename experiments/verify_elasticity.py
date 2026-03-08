#!/usr/bin/env python3
"""Quick verification: do workers respond to different tax rates?

Runs 10 agents for 256 steps under low-tax and high-tax schedules,
then compares mean income, Gini, and labor distributions.

Pass criteria:
- Mean income differs by >10% between schedules
- Gini differs by >0.03
- Workers show varied labor choices (std > 5)
- Mean labor is lower under high-tax schedule
"""

import asyncio
import json
import os
import sys
import time
import numpy as np


async def main():
    from llm_economist.main_async import AsyncLLMEconomist

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    print(f"Using GPU: {gpu}")

    num_agents = 10
    max_timesteps = 256
    tax_year_length = 128

    sim = AsyncLLMEconomist(
        num_agents=num_agents,
        max_timesteps=max_timesteps,
        model_name="Qwen/Qwen3-8B-AWQ",
        tensor_parallel_size=1,
        tax_year_length=tax_year_length,
        scenario="bounded",
        quantization="awq",
        batch_size=num_agents,
        seed=42,
        external_planner=True,
        bracket_setting="three",
        gpu_memory_utilization=0.95,
        debug=True,
    )
    await sim.initialize()

    schedules = {
        "low_tax":  [0.12, 0.24, 0.35],
        "high_tax": [0.90, 0.90, 0.90],
    }

    results = {}
    for name, rates in schedules.items():
        print(f"\n{'='*60}")
        print(f"Running schedule: {name} = {rates}")
        print(f"{'='*60}")

        sim.reset_state()
        sim.set_tax_rates(rates)

        for step in range(max_timesteps):
            if step % tax_year_length == 0 and step > 0:
                sim.set_tax_rates(rates)
            await sim.step()

        # Collect final-step stats
        incomes = [a.income for a in sim.state.agent_states]
        labors = [a.labor for a in sim.state.agent_states]
        gini = sim._calculate_gini(incomes)

        results[name] = {
            "rates": rates,
            "mean_income": float(np.mean(incomes)),
            "std_income": float(np.std(incomes)),
            "mean_labor": float(np.mean(labors)),
            "std_labor": float(np.std(labors)),
            "gini": gini,
            "swf": sim.state.swf,
            "labors": [float(l) for l in labors],
        }
        print(f"\n  Results for {name}:")
        print(f"    Mean income: ${np.mean(incomes):,.0f}")
        print(f"    Mean labor:  {np.mean(labors):.1f} hrs")
        print(f"    Std labor:   {np.std(labors):.1f}")
        print(f"    Gini:        {gini:.3f}")
        print(f"    SWF:         {sim.state.swf:.4f}")
        print(f"    Labor dist:  {sorted([int(l) for l in labors])}")

    # --- Verification checks ---
    low = results["low_tax"]
    high = results["high_tax"]

    print(f"\n{'='*60}")
    print("VERIFICATION CHECKS")
    print(f"{'='*60}")

    income_diff_pct = abs(low["mean_income"] - high["mean_income"]) / max(low["mean_income"], 1) * 100
    gini_diff = abs(low["gini"] - high["gini"])
    labor_std_ok = low["std_labor"] > 5 or high["std_labor"] > 5
    labor_direction = high["mean_labor"] < low["mean_labor"]

    checks = {
        f"Income diff > 10% (got {income_diff_pct:.1f}%)": income_diff_pct > 10,
        f"Gini diff > 0.03 (got {gini_diff:.3f})": gini_diff > 0.03,
        f"Labor variety (std > 5): low={low['std_labor']:.1f}, high={high['std_labor']:.1f}": labor_std_ok,
        f"Labor lower under high tax: {high['mean_labor']:.1f} < {low['mean_labor']:.1f}": labor_direction,
    }

    all_pass = True
    for desc, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {desc}")

    print(f"\n{'='*60}")
    if all_pass:
        print("ALL CHECKS PASSED — elasticity fix is working!")
    else:
        print("SOME CHECKS FAILED — may need further prompt tuning")
    print(f"{'='*60}")

    # Shutdown
    if sim.engine:
        await sim.engine.shutdown()

    return results


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    asyncio.run(main())
