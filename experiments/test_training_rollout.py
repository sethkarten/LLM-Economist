#!/usr/bin/env python3
"""Validation tests for REINFORCE++ v2 rollout environment."""

import os
os.environ['VLLM_USE_V1'] = '0'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

import asyncio
import sys
import numpy as np
import random
import time
import traceback

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llm_economist.main_async import AsyncLLMEconomist
from llm_economist.utils.common import rGB2
from llm_economist.agents.persona_generator import generate_aligned_personas


# =============================================================================
# Shared simulator creation
# =============================================================================

def create_fixed_population(num_agents: int = 32, seed: int = 42):
    """Generate a fixed population (skills + personas) for reproducible rollouts."""
    np.random.seed(seed)
    random.seed(seed)
    incomes = rGB2(num_agents)
    fixed_skills = [float(inc / 40.0) for inc in incomes]
    fixed_personas = generate_aligned_personas(n=num_agents, use_llm_narratives=False, seed=seed)
    return fixed_skills, fixed_personas


async def create_shared_simulator(
    num_agents: int = 32,
    max_timesteps: int = 64,
    seed: int = 42,
):
    """Create a single AsyncLLMEconomist instance shared across tests.

    This avoids reinitializing the vLLM engine for every test.
    """
    fixed_skills, fixed_personas = create_fixed_population(num_agents=num_agents, seed=seed)

    sim = AsyncLLMEconomist(
        num_agents=num_agents,
        max_timesteps=max_timesteps,
        model_name="qwen3-8b",
        tensor_parallel_size=1,
        tax_year_length=64,
        quantization="awq",
        batch_size=num_agents,
        seed=seed,
        external_planner=True,
        fixed_skills=fixed_skills,
        fixed_personas=fixed_personas,
        bracket_setting="three",
    )
    await sim.initialize()
    return sim, fixed_skills


# =============================================================================
# Test 1: Population Stability
# =============================================================================

async def test_population_stability(sim: AsyncLLMEconomist, fixed_skills: list):
    """Test 1: Fixed population produces identical initial conditions across rollouts."""
    print("\n" + "=" * 60)
    print("TEST 1: Population Stability")
    print("=" * 60)

    us_rates = [0.12, 0.24, 0.35]
    initial_skills_per_rollout = []

    for rollout in range(3):
        sim.reset_state()
        skills = [a.skill for a in sim.state.agent_states]
        initial_skills_per_rollout.append(skills[:])

        # Set rates and run a few steps to exercise the environment
        sim.set_tax_rates(us_rates)
        for _ in range(3):
            await sim.step()

        print(f"  Rollout {rollout}: skills[:5] = {[f'{s:.2f}' for s in skills[:5]]}")

    # Verify initial skills match the injected values each time
    for i in range(3):
        assert initial_skills_per_rollout[i] == fixed_skills, \
            f"Rollout {i} initial skills differ from injected fixed_skills!"

    # Verify all rollouts are identical to each other
    for i in range(1, 3):
        assert initial_skills_per_rollout[0] == initial_skills_per_rollout[i], \
            f"Rollout {i} has different initial skills than rollout 0!"

    print("PASSED: All 3 rollouts had identical initial conditions matching injected skills")
    return True


# =============================================================================
# Test 2: SWF Sanity
# =============================================================================

async def test_swf_sanity(sim: AsyncLLMEconomist):
    """Test 2: SWF with US Federal rates is reasonable."""
    print("\n" + "=" * 60)
    print("TEST 2: SWF Sanity")
    print("=" * 60)

    sim.reset_state()
    us_rates = [0.12, 0.24, 0.35]
    sim.set_tax_rates(us_rates)

    # Run 64 steps (1 full tax year)
    num_steps = 64
    for step_i in range(num_steps):
        metrics = await sim.step()

    final_swf = sim.state.swf
    print(f"  Final SWF after {num_steps} steps: {final_swf:.4f}")
    print(f"  Tax rates: {[f'{r*100:.1f}%' for r in sim.state.tax_rates]}")
    print(f"  Mean income: ${metrics['mean_income']:.2f}")
    print(f"  Gini: {metrics['gini']:.3f}")
    print(f"  Mean utility: {metrics['mean_utility']:.2f}")

    # SWF should be finite and not NaN
    assert np.isfinite(final_swf), f"SWF is not finite: {final_swf}"

    # SWF should be positive (agents earn income, utility should be positive overall)
    assert final_swf > 0, f"SWF should be positive, got {final_swf}"

    # Check no individual agent has extreme utility
    extreme_agents = 0
    for agent in sim.state.agent_states:
        if abs(agent.utility) > 1e6:
            extreme_agents += 1
    assert extreme_agents == 0, \
        f"{extreme_agents} agents had extreme utility (|u| > 1e6)"

    print(f"PASSED: SWF = {final_swf:.4f} (finite, positive, no extreme utilities)")
    return True


# =============================================================================
# Test 3: SWF Sensitivity
# =============================================================================

async def test_swf_sensitivity(sim: AsyncLLMEconomist):
    """Test 3: Different tax rates produce different SWF."""
    print("\n" + "=" * 60)
    print("TEST 3: SWF Sensitivity")
    print("=" * 60)

    num_steps = 64

    # Run with near-zero taxes
    sim.reset_state()
    low_rates = [0.01, 0.01, 0.01]
    sim.set_tax_rates(low_rates)
    for _ in range(num_steps):
        await sim.step()
    swf_low = sim.state.swf
    print(f"  Near-zero tax rates {low_rates} -> SWF = {swf_low:.4f}")

    # Run with very high taxes
    sim.reset_state()
    high_rates = [0.90, 0.90, 0.90]
    sim.set_tax_rates(high_rates)
    for _ in range(num_steps):
        await sim.step()
    swf_high = sim.state.swf
    print(f"  Very high tax rates {high_rates} -> SWF = {swf_high:.4f}")

    # They should differ -- tax rates must affect the outcome
    assert swf_low != swf_high, \
        f"SWF did not change across tax regimes! low={swf_low:.4f}, high={swf_high:.4f}"

    print(f"  Difference: {abs(swf_low - swf_high):.4f}")
    print(f"PASSED: Tax rates affect SWF (low={swf_low:.4f}, high={swf_high:.4f})")
    return True


# =============================================================================
# Test 4: Worker Prompt Inspection
# =============================================================================

async def test_worker_prompts(sim: AsyncLLMEconomist):
    """Test 4: Worker prompts contain full context."""
    print("\n" + "=" * 60)
    print("TEST 4: Worker Prompt Inspection")
    print("=" * 60)

    sim.reset_state()
    us_rates = [0.12, 0.24, 0.35]
    sim.set_tax_rates(us_rates)

    # Run 1 step so agents have income/utility populated
    await sim.step()

    # Inspect prompts for first 3 agents
    all_checks_passed = True
    for idx in range(min(3, len(sim.state.agent_states))):
        agent = sim.state.agent_states[idx]
        system_prompt, user_prompt = sim._build_worker_prompt(agent, sim.state.timestep)

        print(f"\n  --- Agent {idx} ({agent.name}) ---")
        print(f"  System prompt (first 200 chars): {system_prompt[:200]}...")
        print(f"  User prompt (first 300 chars): {user_prompt[:300]}...")

        # Check system prompt contains persona text
        has_persona = len(agent.persona_prompt) > 0 and agent.persona_prompt[:20] in system_prompt
        if not has_persona:
            print(f"    WARN: persona text not found in system prompt")
            all_checks_passed = False

        # Check system prompt contains bracket info
        has_brackets = "tax bracket" in system_prompt.lower() or str(sim.state.tax_brackets[1]) in system_prompt
        if not has_brackets:
            print(f"    WARN: bracket info not found in system prompt")
            all_checks_passed = False

        # Check system prompt contains tax rates
        has_rates = "12.0%" in system_prompt or "0.12" in system_prompt
        if not has_rates:
            print(f"    WARN: tax rates not found in system prompt")
            all_checks_passed = False

        # Check user prompt contains skill level
        skill_str = f"${agent.skill:.2f}"
        has_skill = skill_str in user_prompt
        if not has_skill:
            print(f"    WARN: skill level ({skill_str}) not found in user prompt")
            all_checks_passed = False

        # Check user prompt contains income
        has_income = "income" in user_prompt.lower()
        if not has_income:
            print(f"    WARN: income info not found in user prompt")
            all_checks_passed = False

    if all_checks_passed:
        print(f"\nPASSED: Worker prompts contain persona, brackets, rates, skill, and income")
    else:
        print(f"\nFAILED: Some expected content missing from worker prompts")

    return all_checks_passed


# =============================================================================
# Test 5: Planner Action Test (Optional)
# =============================================================================

async def test_planner_actions():
    """Test 5: Planner policy generates valid actions (optional, requires model)."""
    print("\n" + "=" * 60)
    print("TEST 5: Planner Action Test (Optional)")
    print("=" * 60)

    try:
        import torch
        from experiments.run_reinforce_h3_optimized import PlannerPolicy, RLConfig
    except ImportError as e:
        print(f"  SKIP: Could not import PlannerPolicy or dependencies ({e})")
        return None

    planner_model_name = "google/gemma-3-4b-it"

    try:
        # Create a minimal RLConfig
        config = RLConfig(
            planner_model=planner_model_name,
            use_lora=True,
            lora_r=8,
            lora_alpha=8,
            lora_dropout=0.05,
        )

        policy = PlannerPolicy(
            model_name=planner_model_name,
            config=config,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        policy.load_model()
    except Exception as e:
        print(f"  SKIP: Model {planner_model_name} not available ({e})")
        return None

    # Sample 5 actions
    system_prompt = (
        "You are an AI tax policy planner. Respond with ONLY a JSON object.\n"
        "Required format: {\"tax_rates\": [r1, r2, r3]}\n"
        "The tax_rates array must have exactly 3 values between 0.0 and 0.99."
    )
    user_prompt = (
        "Set tax rates for 3 brackets: [0, 90000), [90000, 159100), [159100, +inf).\n"
        "Current mean income: $50,000. Gini: 0.42.\n"
        "Respond with ONLY the JSON object."
    )

    all_valid = True
    for trial in range(5):
        tax_rates, log_prob = policy.sample_action(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,
        )

        if tax_rates is None:
            print(f"  Trial {trial}: Failed to parse tax rates (log_prob={log_prob:.4f})")
            all_valid = False
            continue

        rates_in_range = all(0.0 <= r <= 1.0 for r in tax_rates)
        log_prob_finite = np.isfinite(log_prob)

        print(f"  Trial {trial}: rates={[f'{r:.3f}' for r in tax_rates]}, "
              f"log_prob={log_prob:.4f}, "
              f"in_range={rates_in_range}, finite_lp={log_prob_finite}")

        if not rates_in_range:
            print(f"    WARN: rates out of [0, 1] range")
            all_valid = False
        if not log_prob_finite:
            print(f"    WARN: log_prob is not finite")
            all_valid = False

    # Clean up GPU memory
    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if all_valid:
        print("PASSED: All 5 planner actions valid (rates in [0,1], finite log_probs)")
    else:
        print("FAILED: Some planner actions were invalid")

    return all_valid


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all validation tests."""
    print("=" * 60)
    print("REINFORCE++ v2 - Rollout Environment Validation")
    print("=" * 60)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}
    sim = None

    # -------------------------------------------------------------------------
    # Initialize shared simulator (tests 1-4 share it)
    # -------------------------------------------------------------------------
    try:
        print("\nInitializing shared simulator (this loads the vLLM engine)...")
        t0 = time.time()
        sim, fixed_skills = await create_shared_simulator(
            num_agents=32,
            max_timesteps=64,
            seed=42,
        )
        print(f"Simulator initialized in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"\nFATAL: Could not initialize simulator: {e}")
        traceback.print_exc()
        print("\nAll tests that require vLLM will be skipped.")
        print("Make sure the model 'Qwen/Qwen3-8B-AWQ' is downloaded and a GPU is available.")
        results = {
            'population_stability': False,
            'swf_sanity': False,
            'swf_sensitivity': False,
            'worker_prompts': False,
        }
        # Still try planner test (uses a separate model)
        try:
            results['planner_actions'] = await test_planner_actions()
        except Exception as e:
            print(f"FAILED: {e}")
            traceback.print_exc()
            results['planner_actions'] = False

        # Summary
        _print_summary(results)
        return False

    # -------------------------------------------------------------------------
    # Test 1: Population Stability
    # -------------------------------------------------------------------------
    try:
        t0 = time.time()
        results['population_stability'] = await test_population_stability(sim, fixed_skills)
        print(f"  (completed in {time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"FAILED: {e}")
        traceback.print_exc()
        results['population_stability'] = False

    # -------------------------------------------------------------------------
    # Test 2: SWF Sanity
    # -------------------------------------------------------------------------
    try:
        t0 = time.time()
        results['swf_sanity'] = await test_swf_sanity(sim)
        print(f"  (completed in {time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"FAILED: {e}")
        traceback.print_exc()
        results['swf_sanity'] = False

    # -------------------------------------------------------------------------
    # Test 3: SWF Sensitivity
    # -------------------------------------------------------------------------
    try:
        t0 = time.time()
        results['swf_sensitivity'] = await test_swf_sensitivity(sim)
        print(f"  (completed in {time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"FAILED: {e}")
        traceback.print_exc()
        results['swf_sensitivity'] = False

    # -------------------------------------------------------------------------
    # Test 4: Worker Prompt Inspection
    # -------------------------------------------------------------------------
    try:
        t0 = time.time()
        results['worker_prompts'] = await test_worker_prompts(sim)
        print(f"  (completed in {time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"FAILED: {e}")
        traceback.print_exc()
        results['worker_prompts'] = False

    # -------------------------------------------------------------------------
    # Test 5: Planner Action Test (optional, separate model)
    # -------------------------------------------------------------------------
    try:
        t0 = time.time()
        result = await test_planner_actions()
        if result is None:
            print("  (skipped - model not available)")
        else:
            print(f"  (completed in {time.time() - t0:.1f}s)")
        results['planner_actions'] = result
    except Exception as e:
        print(f"FAILED: {e}")
        traceback.print_exc()
        results['planner_actions'] = False

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    _print_summary(results)

    # Return True only for non-skipped tests
    mandatory = {k: v for k, v in results.items() if v is not None}
    return all(mandatory.values())


def _print_summary(results):
    """Print test result summary."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, result in results.items():
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "PASSED"
        else:
            status = "FAILED"
        print(f"  {name:30s} {status}")

    non_skipped = {k: v for k, v in results.items() if v is not None}
    passed = sum(1 for v in non_skipped.values() if v)
    total = len(non_skipped)
    skipped = sum(1 for v in results.values() if v is None)

    print(f"\n{passed}/{total} tests passed", end="")
    if skipped > 0:
        print(f" ({skipped} skipped)", end="")
    print()


if __name__ == '__main__':
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
