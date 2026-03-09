"""
Async main entry point for large-scale LLM Economist simulations.

Optimized for:
- 1000-100000+ agents
- Batch inference with vLLM
- Population-aligned personas
- RTX 5090 / H100 hardware
"""

import argparse
import asyncio
import logging
import os
import sys
import json
import time
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import random
import wandb

# Local imports
from .inference import ScalableInferenceEngine, InferenceConfig, BatchRequest
from .inference.config import (
    get_model_config, calculate_pareto_configs, estimate_throughput,
    SUPPORTED_MODELS, MODEL_ALIASES, QuantizationType
)
from .agents.persona_generator import (
    PopulationAlignedPersonaGenerator, generate_aligned_personas, Persona
)
from .agents.worker import ROLE_MESSAGES, PERSONAS, PERSONA_PERCENTS, distribute_fixed_personas
from .utils.common import rGB2

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Lightweight agent state for async simulation."""
    id: int
    name: str
    skill: float
    labor: float
    income: float  # pre-tax
    post_tax_income: float
    utility: float
    tax_paid: float
    persona_prompt: str
    role: str  # Persona role name (e.g. 'entrepreneur', 'teacher')
    utility_type: str  # egotistical, altruistic, adversarial
    history: List[Dict]
    satisfaction: float = 1.0  # Tax policy satisfaction (1.0=YES, 0.5=NO)
    adjusted_utility: float = 0.0  # utility * satisfaction


@dataclass
class SimulationState:
    """Global simulation state."""
    timestep: int
    tax_rates: List[float]
    tax_brackets: List[float]
    total_tax_collected: float
    rebate_per_agent: float
    swf: float  # Social welfare function
    agent_states: List[AgentState]


class AsyncLLMEconomist:
    """
    Async implementation of LLM Economist for massive scale.

    Key optimizations:
    - Batch all agent decisions together
    - Use vLLM continuous batching
    - Lightweight agent state (no full Agent objects)
    - Population-aligned personas
    """

    def __init__(
        self,
        num_agents: int,
        max_timesteps: int,
        model_name: str = "qwen3-30b-a3b",
        tensor_parallel_size: int = 2,
        tax_year_length: int = 128,
        scenario: str = "bounded",
        quantization: str = "awq",
        batch_size: int = 100,
        seed: int = 42,
        use_wandb: bool = False,
        debug: bool = False,
        disable_exploration: bool = False,
        disable_exploitation: bool = False,
        swf_weighting: str = "rawlsian",
        history_len: int = 50,
        bracket_setting: str = "three",
        external_planner: bool = False,
        fixed_skills: Optional[List[float]] = None,
        fixed_personas: Optional[Dict[str, str]] = None,
        gpu_memory_utilization: Optional[float] = None,
        max_model_len: int = 4096,
    ):
        self.max_model_len = max_model_len
        self.num_agents = num_agents
        self.max_timesteps = max_timesteps
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.tax_year_length = tax_year_length
        self.scenario = scenario
        self.batch_size = batch_size
        self.seed = seed
        self.use_wandb = use_wandb
        self.debug = debug
        self.disable_exploration = disable_exploration
        self.disable_exploitation = disable_exploitation
        self.swf_weighting = swf_weighting
        self.history_len = history_len
        self.bracket_setting = bracket_setting
        self.external_planner = external_planner
        self._gpu_memory_utilization = gpu_memory_utilization

        # Unique instance ID to prevent request ID collisions when sharing a vLLM engine
        self._instance_id = uuid.uuid4().hex[:8]

        # Fixed population (injected from outside for RL training)
        self._fixed_skills = fixed_skills
        self._fixed_personas = fixed_personas

        # Set seeds
        np.random.seed(seed)
        random.seed(seed)

        # Parse quantization
        self.quantization = QuantizationType(quantization) if quantization else QuantizationType.AWQ

        # Initialize inference engine (lazy)
        self.engine: Optional[ScalableInferenceEngine] = None

        # Initialize state
        self.state: Optional[SimulationState] = None
        self.personas: Dict[str, str] = {}

        # Tax configuration from bracket setting
        from llm_economist.utils.bracket import get_brackets, get_num_brackets
        self.tax_brackets = get_brackets(bracket_setting)
        num_b = get_num_brackets(bracket_setting)
        # Initial rates: uniform 15% across all brackets
        self.tax_rates = [0.15] * num_b

        # Logging
        self.metrics_history: List[Dict] = []

        # ICRL planner history for in-context reinforcement learning
        self.planner_history: List[Dict] = []

    async def initialize(self):
        """Initialize the simulation."""
        logger.info(f"Initializing AsyncLLMEconomist with {self.num_agents} agents")

        # Get model config
        model_config = get_model_config(self.model_name)
        logger.info(f"Using model: {model_config.name} ({model_config.hf_name})")

        # Initialize inference engine
        # Get quantization - use model's recommended if not overridden
        quant_value = None
        if self.quantization != QuantizationType.NONE:
            quant_value = self.quantization.value

        # Get dtype from model config (for Gemma 3 BF16)
        dtype = None
        if hasattr(model_config, 'recommended_quantization') and model_config.recommended_quantization == QuantizationType.NONE:
            dtype = "bfloat16"  # Use BF16 for models that don't need quantization

        engine_kwargs = dict(
            model_name=model_config.hf_name,
            tensor_parallel_size=self.tensor_parallel_size,
            quantization=quant_value,
            max_model_len=self.max_model_len,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            kv_cache_dtype="auto",  # Auto for better Blackwell compatibility
            max_num_seqs=min(512, self.batch_size * 2),
            text_only_mode=getattr(model_config, 'text_only_mode', False),
            dtype=dtype,
        )
        if self._gpu_memory_utilization is not None:
            engine_kwargs['gpu_memory_utilization'] = self._gpu_memory_utilization
        self.engine = ScalableInferenceEngine(**engine_kwargs)
        await self.engine.initialize()

        # Generate or use fixed personas
        if self._fixed_personas is not None:
            logger.info("Using fixed (injected) personas for RL training")
            self.personas = self._fixed_personas
            self._persona_roles = None  # RL training uses custom personas
        else:
            # Use original hand-crafted ROLE_MESSAGES personas from worker.py
            # These have strong, explicit tax opinions that drive realistic elasticity
            logger.info("Assigning original rich personas (ROLE_MESSAGES)...")
            np.random.seed(self.seed)
            role_list = distribute_fixed_personas(self.num_agents)
            self._persona_roles = role_list  # Store for reset
            # Build persona dict mapping index to full description
            self.personas = {}
            for i, role in enumerate(role_list):
                self.personas[i] = ROLE_MESSAGES[role]

        # Initialize agent states with fixed or sampled skills
        logger.info("Initializing agent states...")
        if self._fixed_skills is not None:
            skills = self._fixed_skills
        else:
            skills = self._sample_skills()
        self._initial_skills = skills  # Store for reset_state()

        agent_states = []
        for i in range(self.num_agents):
            if self._persona_roles is not None:
                role = self._persona_roles[i]
                persona_prompt = ROLE_MESSAGES[role]
            else:
                # RL training path: use injected personas
                persona_ids = list(self.personas.keys())
                persona_id = persona_ids[i % len(persona_ids)]
                role = 'default'
                persona_prompt = self.personas[persona_id]
            state = AgentState(
                id=i,
                name=f"worker_{i}",
                skill=skills[i],
                labor=40.0,  # Default 40 hours
                income=skills[i] * 40.0,  # Initial income
                post_tax_income=0.0,
                utility=0.0,
                tax_paid=0.0,
                persona_prompt=persona_prompt,
                role=role,
                utility_type='egotistical',  # Default
                history=[],
            )
            agent_states.append(state)

        # Initialize simulation state
        self.state = SimulationState(
            timestep=0,
            tax_rates=self.tax_rates.copy(),
            tax_brackets=self.tax_brackets.copy(),
            total_tax_collected=0.0,
            rebate_per_agent=0.0,
            swf=0.0,
            agent_states=agent_states,
        )

        logger.info(f"Initialization complete. Ready to simulate {self.num_agents} agents for {self.max_timesteps} steps.")

    def set_tax_rates(self, rates: List[float]):
        """Set tax rates externally (for RL training). Bypasses planner LLM call."""
        if len(rates) != len(self.state.tax_rates):
            raise ValueError(f"Expected {len(self.state.tax_rates)} rates, got {len(rates)}")
        self.state.tax_rates = [max(0.0, min(0.99, float(r))) for r in rates]

    def reset_state(self):
        """Reset agent states to initial conditions without reinitializing vLLM or personas.

        Enables reuse across RL rollouts: same population, fresh economic state.
        """
        from llm_economist.utils.bracket import get_num_brackets
        num_b = get_num_brackets(self.bracket_setting)

        for agent in self.state.agent_states:
            agent.labor = 40.0
            agent.income = agent.skill * 40.0
            agent.post_tax_income = 0.0
            agent.utility = 0.0
            agent.tax_paid = 0.0
            agent.satisfaction = 1.0
            agent.adjusted_utility = 0.0
            agent.history = []

        self.state.timestep = 0
        self.state.tax_rates = [0.15] * num_b
        self.state.total_tax_collected = 0.0
        self.state.rebate_per_agent = 0.0
        self.state.swf = 0.0
        self.metrics_history = []
        self.planner_history = []

    def _sample_skills(self) -> List[float]:
        """Sample skills from GB2 distribution (US income calibrated)."""
        # Use GB2 distribution fitted to US ACS data
        incomes = rGB2(self.num_agents)
        # Convert to hourly skill (income at 40 hrs/week)
        skills = [float(inc / 40.0) for inc in incomes]
        return skills

    def _build_worker_prompt(self, agent: AgentState, timestep: int) -> tuple:
        """Build system and user prompts for a worker agent.

        Mirrors the rich prompt from worker.py including bracket explanation,
        effective tax rate, historical context, running averages, labor
        direction nudge, and exploration/exploitation instructions.
        """
        from llm_economist.utils.bracket import get_bracket_prompt

        # --- Bracket explanation ---
        bracket_prompt, _ = get_bracket_prompt(self.bracket_setting)

        # --- Effective tax rate at current income ---
        if agent.income > 0:
            effective_rate = agent.tax_paid / agent.income * 100
        else:
            effective_rate = 0.0

        # --- Bracket-by-bracket breakdown ---
        bracket_detail_lines = []
        prev = 0
        for bracket_upper, rate in zip(self.state.tax_brackets[1:], self.state.tax_rates):
            if bracket_upper >= 10_000_000:
                bracket_detail_lines.append(f"  ${prev:,.0f}+  taxed at {rate*100:.1f}%")
            else:
                bracket_detail_lines.append(f"  ${prev:,.0f}–${bracket_upper:,.0f}  taxed at {rate*100:.1f}%")
            prev = bracket_upper
        bracket_detail = "\n".join(bracket_detail_lines)

        system_prompt = (
            f"You are {agent.name}, a citizen of Princetonia. "
            f"Your skill level is {agent.skill:.2f} with an expected income of "
            f"{agent.skill * 40:.2f} at 40 hours of labor each week.\n"
            f"{agent.persona_prompt}\n"
            "Each year you will have the option to choose the number of hours of "
            "labor to perform each week, from 0 to 100 hours. You can work overtime (>40 hours), "
            "undertime (<40 hours), or choose not to work at all (0 hours) if taxes make "
            "working not worth it. You will receive income z proportional "
            "to the number of hours worked and your skill level.\n"
            "Your goal is to maximize your adjusted utility.\n"
            "Isoelastic utility u~ = z~ - 0.0005 * labor^3.5\n"
            "where z~ = income - tax + rebate (post-tax income), and income = skill * labor.\n"
            "Your satisfaction r with tax policy (YES=1.0, NO=0.5) adjusts utility: u = r * u~.\n"
            "Higher tax rates mean less take-home pay per hour worked, so you may want to work fewer hours.\n"
            "Lower tax rates mean more take-home pay per hour, making additional work more rewarding.\n"
            "Make sure to sufficiently explore different amounts of LABOR before exploiting "
            "the best one for maximum utility u.\n"
            "Choose LABOR from [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] hours.\n"
            "Use the JSON format: {\"labor_hours\": X} and replace X with your answer.\n"
        )

        # --- Historical context ---
        history_text = ""
        if agent.history:
            K = min(self.history_len, len(agent.history))
            recent = agent.history[-K:]
            history_text = "Historical data:\n"
            for entry in recent:
                rates_str = [f"{r*100:.1f}%" for r in entry['tax_rates']]
                satisfaction = entry.get('satisfaction', 1.0)
                adj_u = entry.get('adjusted_utility', entry['utility'])
                satisfaction_str = "YES" if satisfaction >= 1.0 else "NO"
                history_text += (
                    f"Timestep {entry['timestep']}: "
                    f"TAX={rates_str}, "
                    f"skill s={agent.skill:.2f}, "
                    f"LABOR l={entry['labor']:.0f}, "
                    f"pre-tax income z={entry['income']:.2f}, "
                    f"tax_i={entry['tax_paid']:.2f}, "
                    f"rebate={entry.get('rebate', 0):.2f}, "
                    f"post-tax income z~={entry['post_tax_income']:.2f}, "
                    f"isoelastic u~={entry['utility']:.2f}, "
                    f"satisfaction r={satisfaction_str}, "
                    f"adjusted utility u={adj_u:.2f}\n"
                )

            # Best N timesteps by adjusted utility (helps workers identify optimal labor)
            sorted_history = sorted(
                agent.history,
                key=lambda x: x.get('adjusted_utility', x['utility']),
                reverse=True,
            )
            best_n = min(5, len(sorted_history))
            history_text += f"Best {best_n} timesteps:\n"
            for entry in sorted_history[:best_n]:
                adj_u = entry.get('adjusted_utility', entry['utility'])
                history_text += (
                    f"Timestep {entry['timestep']}: "
                    f"LABOR l={entry['labor']:.0f}, "
                    f"adjusted utility u={adj_u:.2f}\n"
                )

        # --- Running averages (using adjusted utility) ---
        avg_text = ""
        if agent.history:
            K = min(self.history_len, len(agent.history))
            recent = agent.history[-K:]
            avg_labor = round(np.mean([e['labor'] for e in recent]), -1)
            avg_utility = np.mean([e.get('adjusted_utility', e['utility']) for e in recent])
            avg_text = (
                f"The running average LABOR choice historically was average "
                f"LABOR={avg_labor:.0f} hours corresponding to average adjusted "
                f"utility u={avg_utility:.2f}. "
            )

        # --- Labor direction nudge (uses adjusted_utility to reflect tax satisfaction) ---
        nudge_text = ""
        if len(agent.history) >= 2:
            prev_entry = agent.history[-2]
            curr_entry = agent.history[-1]
            if curr_entry['labor'] != prev_entry['labor']:
                delta_l = curr_entry['labor'] - prev_entry['labor']
                delta_l_msg = 'Increasing' if delta_l > 0 else 'Decreasing'
                # Use adjusted_utility (utility * satisfaction) for the nudge
                # This way, if workers are unsatisfied with taxes, the utility
                # signal reflects that dissatisfaction and guides labor adjustments
                curr_u = curr_entry.get('adjusted_utility', curr_entry['utility'])
                prev_u = prev_entry.get('adjusted_utility', prev_entry['utility'])
                delta_u = curr_u - prev_u
                delta_u_msg = 'increased' if delta_u > 0 else 'decreased'
                if (delta_l > 0 and delta_u < 0) or (delta_l < 0 and delta_u > 0):
                    labor_action = f"too high and needs to be decreased below labor l={curr_entry['labor']:.0f}"
                elif (delta_l > 0 and delta_u > 0) or (delta_l < 0 and delta_u < 0):
                    labor_action = f"too low and needs to be increased above labor l={curr_entry['labor']:.0f}"
                else:
                    labor_action = "at a reasonable level"
                nudge_text = (
                    f"{delta_l_msg} labor {delta_u_msg} utility. "
                    f"This implies labor l is {labor_action}.\n"
                )

        # --- Exploration vs exploitation ---
        # Switch to exploit at tax year boundaries or near end of simulation
        # (matches worker.py's two-timescale coupling)
        at_tax_year_end = (timestep + 1) % self.tax_year_length == 0
        near_end = timestep > 0.9 * self.max_timesteps
        if at_tax_year_end or near_end:
            explore_text = "Choose your best amount of LABOR to perform."
        else:
            explore_text = (
                "Use the historical data to influence your answer in order to "
                "maximize utility u, while balancing exploration and exploitation "
                "by choosing varying amounts of LABOR. "
            )

        # --- Marginal tax analysis ---
        # Show workers how taxes affect the return on additional labor
        current_labor = max(agent.labor, 40)  # Use last labor or default
        current_income = agent.skill * current_labor
        marginal_rate = self._get_marginal_rate(current_income)
        net_per_hour = agent.skill * (1 - marginal_rate)

        marginal_text = (
            f"At your current income of ${current_income:,.0f}, your marginal tax rate is {marginal_rate*100:.0f}%.\n"
            f"Each additional hour of work earns ${agent.skill:.0f} pre-tax but only ${net_per_hour:.0f} after tax.\n"
            f"Labor cost for one more hour: {0.0005 * 3.5 * current_labor**2.5:.1f}\n"
        )

        # Show take-home comparison at different hours
        example_lines = ["Take-home analysis at current tax rates:"]
        for h in [20, 40, 60, 80]:
            gross = agent.skill * h
            tax = self._calculate_tax(gross)
            net = gross - tax + self.state.rebate_per_agent
            cost = 0.0005 * (h ** 3.5)
            u = net - cost
            # Marginal rate at this income level
            mr = self._get_marginal_rate(gross)
            take_home_per_hour = agent.skill * (1 - mr)
            example_lines.append(
                f"  {h}h: income=${gross:,.0f}, tax=${tax:,.0f} ({tax/max(gross,1)*100:.0f}% effective), "
                f"take-home/hr=${take_home_per_hour:.0f}, labor_cost={cost:.0f}, utility={u:.0f}"
            )
        examples = "\n".join(example_lines)

        user_prompt = (
            f"{history_text}"
            f"{nudge_text}"
            f"Timestep {timestep}:\n"
            f"{bracket_prompt}\n"
            f"Current tax schedule:\n{bracket_detail}\n"
            f"Your effective tax rate: {effective_rate:.1f}%\n"
            f"skill: s = {agent.skill:.2f}\n"
            f"{avg_text}\n"
            f"{marginal_text}\n"
            f"{examples}\n\n"
            f"Next year, you may perform LABOR: [0,10,20,30,40,50,60,70,80,90,100] hours. "
            f"{explore_text}\n"
            f"Respond with JSON: {{\"labor_hours\": <number>}}\n"
        )
        return system_prompt, user_prompt

    def _build_planner_prompt(self, timestep: int, worker_stats: List[Dict]) -> tuple:
        """Build system and user prompts for the tax planner."""
        if self.swf_weighting == 'utilitarian':
            swf_formula = "Social welfare = sum of utilities across all agents."
        else:
            swf_formula = "Social welfare = sum of (utility / pre_tax_income) across all agents."

        exploration_cue = ""
        if not self.disable_exploration:
            exploration_cue = "\n4. Explore diverse tax rates to find the best ones"

        exploitation_cue = ""
        if not self.disable_exploitation:
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
        incomes = [s['income'] for s in worker_stats]
        utilities = [s['utility'] for s in worker_stats]

        # Build historical context for ICRL
        history_text = ""
        if self.planner_history:
            # Last K timesteps of history
            K = min(self.history_len, len(self.planner_history))
            recent = self.planner_history[-K:]
            history_text += "Historical data:\n"
            for entry in recent:
                rates_str = [f"{r*100:.1f}%" for r in entry['tax_rates']]
                if self.swf_weighting == 'utilitarian':
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
            sorted_by_swf = sorted(self.planner_history, key=lambda x: x['swf'], reverse=True)
            top_n = min(5, len(sorted_by_swf))
            history_text += f"\nBest {top_n} timesteps:\n"
            for entry in sorted_by_swf[:top_n]:
                rates_str = [f"{r*100:.1f}%" for r in entry['tax_rates']]
                history_text += (
                    f"Timestep {entry['timestep']}: "
                    f"tax_rates={rates_str}, "
                    f"SWF={entry['swf']:.4f}\n"
                )
            history_text += "\n"

        # Exploration / exploitation cues (gated by flags)
        cue_text = ""
        if not self.disable_exploration and self.planner_history:
            # A: Anti-repetition — identify rates from the last 2 tax years to avoid
            recent_2 = self.planner_history[-2:]
            recent_rate_strs = [str([f"{r*100:.1f}%" for r in e['tax_rates']]) for e in recent_2]
            avoid_str = " and ".join(recent_rate_strs)

            # B: Check if bottom bracket has been frozen — add targeted nudge if so
            bottom_rates = [e['tax_rates'][0] for e in self.planner_history]
            bottom_frozen = len(set(round(r, 3) for r in bottom_rates)) == 1
            bottom_nudge = (
                f" Note: you have kept the bottom bracket fixed at {bottom_rates[0]*100:.1f}% for every tax year — "
                "consider testing lower rates (e.g. 5-10%) to incentivize low-income workers, "
                "or higher rates (e.g. 20-30%) to increase redistribution."
                if bottom_frozen else ""
            )

            cue_text += (
                "To find the true optimal policy, you must explore the full rate space. "
                f"(A) Do NOT choose rates identical to your last 2 tax years ({avoid_str}) — "
                "repeating known rates wastes a tax year and provides no new information. "
                "(B) Vary ALL brackets including the bottom bracket, not just the top ones."
                f"{bottom_nudge} "
                "(E) Unexplored rate combinations may yield much higher SWF than your current best — "
                "the cost of exploring is one tax year, but the reward of finding a better policy is permanent. "
                "Choose a combination you have never tried before. "
            )
        if not self.disable_exploitation and self.planner_history:
            # Best from all history
            K = min(self.history_len, len(self.planner_history))
            recent = self.planner_history[-K:]
            best_entry = max(recent, key=lambda x: x['swf'])
            best_rates = [f"{r*100:.1f}%" for r in best_entry['tax_rates']]
            # Best from the last 2 tax years specifically
            recent_2 = self.planner_history[-2:]
            best_recent = max(recent_2, key=lambda x: x['swf'])
            best_recent_rates = [f"{r*100:.1f}%" for r in best_recent['tax_rates']]
            cue_text += (
                f"The best marginal tax rates overall were TAX={best_rates} (SWF={best_entry['swf']:.4f}). "
                f"Among your last 2 tax years, the better choice was TAX={best_recent_rates} (SWF={best_recent['swf']:.4f}). "
                "Use these as a starting point, then adjust one or more brackets to improve further. "
            )

        user_prompt = f"""{history_text}Timestep {timestep}:
Current tax rates: {[f"{r*100:.1f}%" for r in self.state.tax_rates]}
Total tax collected: ${self.state.total_tax_collected:.2f}
Current SWF: {self.state.swf:.4f}

Worker statistics (N={len(worker_stats)}):
- Mean income: ${np.mean(incomes):.2f}
- Median income: ${np.median(incomes):.2f}
- Mean utility: {np.mean(utilities):.2f}
- Income Gini: {self._calculate_gini(incomes):.3f}

{cue_text}
Propose new tax rates.
Respond with JSON: {{"tax_rates": [rate1, rate2, ...], "reasoning": "<explanation>"}}
"""
        return system_prompt, user_prompt

    def _calculate_gini(self, values: List[float]) -> float:
        """Calculate Gini coefficient."""
        if not values or len(values) < 2:
            return 0.0
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        return (2 * np.sum((np.arange(1, n+1) * sorted_values)) / (n * cumsum[-1])) - (n + 1) / n

    async def step(self) -> Dict[str, Any]:
        """Execute one simulation step."""
        timestep = self.state.timestep
        is_tax_year_start = timestep % self.tax_year_length == 0

        start_time = time.time()

        # Step 1: Update tax rates if new tax year (skip if external planner controls rates)
        if is_tax_year_start and timestep > 0 and not self.external_planner:
            await self._update_tax_rates(timestep)

        # Step 2: Batch worker decisions
        worker_responses = await self._batch_worker_decisions(timestep)

        # Step 3: Update agent states based on responses
        parse_success = 0
        parse_fail = 0
        for i, (response, _) in enumerate(worker_responses):
            agent = self.state.agent_states[i]
            try:
                data = json.loads(response)
                new_labor = float(np.clip(data.get('labor_hours', 40), 0, 100))
                if self.debug and i < 3 and timestep < 3:
                    logger.debug(f"  Worker {i} response: labor_hours={data.get('labor_hours')}, parsed={new_labor:.1f}, prev={agent.labor:.1f}")
                agent.labor = new_labor
                parse_success += 1
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                parse_fail += 1
                if self.debug and i < 3:
                    logger.debug(f"  Worker {i} parse FAILED: {e}, raw={repr(response[:200])}")

            # Update income
            agent.income = agent.skill * agent.labor

        if self.debug and timestep < 5:
            logger.debug(f"  Step {timestep}: {parse_success} parsed OK, {parse_fail} failed")

        # Step 4: Apply taxes and calculate utilities
        self._apply_taxes()
        self._calculate_utilities()

        # Step 4a: Satisfaction assessment — workers evaluate tax policy
        # This creates the feedback loop that makes workers respond to tax changes:
        # satisfied=YES → r=1.0, satisfied=NO → r=0.5, adjusted_utility = utility * r
        await self._batch_satisfaction_assessments(timestep)

        # Step 4b: Record per-agent history for worker prompt context
        for agent in self.state.agent_states:
            agent.history.append({
                'timestep': timestep,
                'tax_rates': list(self.state.tax_rates),
                'labor': agent.labor,
                'income': agent.income,
                'tax_paid': agent.tax_paid,
                'rebate': self.state.rebate_per_agent,
                'post_tax_income': agent.post_tax_income,
                'utility': agent.utility,
                'satisfaction': agent.satisfaction,
                'adjusted_utility': agent.adjusted_utility,
            })

        # Step 5: Log metrics
        step_time = time.time() - start_time
        metrics = self._collect_metrics(timestep, step_time)
        self.metrics_history.append(metrics)

        # Step 6: Accumulate planner history for ICRL context
        self.planner_history.append({
            'timestep': timestep,
            'tax_rates': self.state.tax_rates.copy(),
            'swf': self.state.swf,
            'total_tax': self.state.total_tax_collected,
            'mean_income': metrics['mean_income'],
            'mean_utility': metrics['mean_utility'],
            'gini': metrics['gini'],
        })

        if self.debug or timestep % 100 == 0:
            logger.info(f"Step {timestep}: SWF={metrics['swf']:.4f}, "
                       f"mean_income=${metrics['mean_income']:.0f}, "
                       f"time={step_time:.2f}s")

        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'swf': metrics['swf'],
                'gini': metrics['gini'],
                'income/mean': metrics['mean_income'],
                'income/median': metrics['median_income'],
                'income/std': metrics['std_income'],
                'utility/mean': metrics['mean_utility'],
                'labor/mean': metrics['mean_labor'],
                'tax/total_collected': metrics['total_tax'],
                'tax/rebate_per_agent': metrics['rebate'],
                'tax/rates': metrics['tax_rates'],
                'time/step_seconds': step_time,
            }, step=timestep)

        # Advance timestep
        self.state.timestep += 1

        return metrics

    async def _batch_worker_decisions(self, timestep: int) -> List[tuple]:
        """Batch all worker decisions for efficient inference."""
        # Build prompts for all agents
        system_prompts = []
        user_prompts = []

        for agent in self.state.agent_states:
            sys, user = self._build_worker_prompt(agent, timestep)
            system_prompts.append(sys)
            user_prompts.append(user)

        # Process in batches
        all_responses = []

        for batch_start in range(0, self.num_agents, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.num_agents)

            batch = BatchRequest(
                request_ids=[f"{self._instance_id}_worker_{i}" for i in range(batch_start, batch_end)],
                prompts=user_prompts[batch_start:batch_end],
                system_prompts=system_prompts[batch_start:batch_end],
                temperatures=[0.7] * (batch_end - batch_start),
                max_tokens=256,
                json_format=True,
            )

            response = await self.engine.generate_batch(batch)

            for text, is_valid in zip(response.responses, response.is_json_valid):
                all_responses.append((text, is_valid))

        return all_responses

    async def _batch_satisfaction_assessments(self, timestep: int) -> None:
        """Batch satisfaction assessments for all workers.

        After utility is calculated, each worker evaluates whether they are
        satisfied with the current tax policy. This mirrors the satisfaction
        mechanism from worker.py:
          - YES → satisfaction = 1.0 (utility unchanged)
          - NO  → satisfaction = 0.5 (adjusted_utility halved)

        The adjusted_utility is used for the labor direction nudge, creating
        a feedback loop where workers respond to tax policy changes.
        """
        system_prompts = []
        user_prompts = []

        for agent in self.state.agent_states:
            sys_prompt = ""  # Match worker.py: system_prompt is empty for satisfaction

            # Build summary of this year's outcomes (matches worker.py message_history format)
            year_summary = (
                f"TAX: = {list(self.state.tax_rates)}\n"
                f"skill: s = {agent.skill:.2f}\n"
                f"LABOR: = l {agent.labor:.0f}\n"
                f"income: z = s * l = {agent.income:.2f}\n"
                f"tax_i = {agent.tax_paid:.2f}\n"
                f"rebate = {self.state.rebate_per_agent:.2f}\n"
                f"post-tax income: z~ = z - tax_i + rebate = {agent.post_tax_income:.2f}\n"
                f"isoelastic utility: u~ = z~ - c * l^d = {agent.utility:.2f}\n"
            )

            # Use the full ROLE_MESSAGES persona as prefix (matches worker.py line 425)
            # This is the key mechanism that creates persona-specific tax responses
            role_msg = agent.persona_prompt  # Full ROLE_MESSAGES text

            user_prompt = (
                f"{role_msg}\n"
                f"Based on your summary of this year:\n{year_summary}"
                " are you satisfied with the overall tax policy "
                "(including tax_i and rebate)?\n"
                'Let\'s think step by step. Your thought should be no more than '
                '4 sentences. Use the JSON format: '
                '{"thought": "<step-by-step-thinking>", "ANSWER": "X"} '
                'and replace "X" with "YES" or "NO".\n'
            )

            system_prompts.append(sys_prompt)
            user_prompts.append(user_prompt)

        # Batch inference
        all_responses = []
        for batch_start in range(0, self.num_agents, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.num_agents)
            batch = BatchRequest(
                request_ids=[f"{self._instance_id}_satisfaction_{i}" for i in range(batch_start, batch_end)],
                prompts=user_prompts[batch_start:batch_end],
                system_prompts=system_prompts[batch_start:batch_end],
                temperatures=[0.7] * (batch_end - batch_start),
                max_tokens=256,
                json_format=True,
            )
            response = await self.engine.generate_batch(batch)
            for text, is_valid in zip(response.responses, response.is_json_valid):
                all_responses.append((text, is_valid))

        # Parse satisfaction responses
        satisfied_count = 0
        unsatisfied_count = 0
        parse_fail = 0
        for i, (response, _) in enumerate(all_responses):
            agent = self.state.agent_states[i]
            try:
                data = json.loads(response)
                answer = str(data.get('ANSWER', data.get('answer', ''))).lower()
                if 'yes' in answer:
                    agent.satisfaction = 1.0
                    satisfied_count += 1
                elif 'no' in answer:
                    agent.satisfaction = 0.5
                    unsatisfied_count += 1
                else:
                    agent.satisfaction = 1.0  # Default to satisfied on parse ambiguity
                    parse_fail += 1
            except (json.JSONDecodeError, KeyError, TypeError):
                agent.satisfaction = 1.0  # Default to satisfied on parse failure
                parse_fail += 1

            agent.adjusted_utility = agent.utility * agent.satisfaction

        if self.debug or timestep % 50 == 0:
            logger.info(f"  Satisfaction step {timestep}: {satisfied_count} YES, "
                       f"{unsatisfied_count} NO, {parse_fail} parse failures")

    async def _update_tax_rates(self, timestep: int):
        """Update tax rates via planner LLM call."""
        worker_stats = [
            {'income': a.income, 'utility': a.utility}
            for a in self.state.agent_states
        ]

        sys_prompt, user_prompt = self._build_planner_prompt(timestep, worker_stats)

        response, _ = await self.engine.generate_single(
            sys_prompt, user_prompt,
            temperature=0.7,
            max_tokens=512,
            json_format=True,
        )

        try:
            data = json.loads(response)
            new_rates = data.get('tax_rates', self.state.tax_rates)
            # Validate rates (convert to float, handle strings like '10.0%' or '0.10')
            if len(new_rates) == len(self.state.tax_rates):
                parsed_rates = []
                for r in new_rates:
                    if isinstance(r, str):
                        r = r.strip().rstrip('%')
                        val = float(r)
                        # If value > 1, assume it's a percentage (e.g., 10.0 -> 0.10)
                        if val > 1:
                            val = val / 100.0
                    else:
                        val = float(r)
                    parsed_rates.append(max(0.0, min(0.99, val)))
                self.state.tax_rates = parsed_rates
                logger.info(f"Updated tax rates: {self.state.tax_rates}")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to parse planner response: {e}")

    def _apply_taxes(self):
        """Apply tax rates to all agents."""
        total_tax = 0.0

        for agent in self.state.agent_states:
            tax = self._calculate_tax(agent.income)
            agent.tax_paid = tax
            agent.post_tax_income = agent.income - tax
            total_tax += tax

        self.state.total_tax_collected = total_tax
        self.state.rebate_per_agent = total_tax / self.num_agents

        # Add rebate to post-tax income
        for agent in self.state.agent_states:
            agent.post_tax_income += self.state.rebate_per_agent

    def _get_marginal_rate(self, income: float) -> float:
        """Get the marginal tax rate for a given income level."""
        for i, bracket_upper in enumerate(self.state.tax_brackets[1:]):
            if income <= bracket_upper:
                return self.state.tax_rates[i]
        return self.state.tax_rates[-1]

    def _calculate_tax(self, income: float) -> float:
        """Calculate tax for a given income using bracket rates."""
        tax = 0.0
        prev_bracket = 0

        for bracket, rate in zip(self.tax_brackets[1:], self.state.tax_rates):
            if income <= prev_bracket:
                break
            taxable = min(income, bracket) - prev_bracket
            tax += taxable * rate
            prev_bracket = bracket

        # Top bracket
        if income > self.tax_brackets[-1]:
            tax += (income - self.tax_brackets[-1]) * self.state.tax_rates[-1]

        return tax

    def _calculate_utilities(self):
        """Calculate utilities for all agents."""
        # Isoelastic utility: u = z_tilde - c * l^delta
        # where z_tilde = post_tax_income, l = labor, c = cost coeff, delta = elasticity
        c = 0.0005
        delta = 3.5

        swf = 0.0
        for agent in self.state.agent_states:
            cost = c * (agent.labor ** delta)
            agent.utility = agent.post_tax_income - cost

            if self.swf_weighting == 'utilitarian':
                swf += agent.utility
            else:
                # rawlsian: weight by inverse pre-tax income (1/z)
                # Floor income at $1000 to prevent extreme ratios from GB2 left-tail draws
                swf += agent.utility / max(agent.income, 1000.0)

        self.state.swf = swf

    def _collect_metrics(self, timestep: int, step_time: float) -> Dict[str, Any]:
        """Collect metrics for this timestep."""
        incomes = [a.income for a in self.state.agent_states]
        utilities = [a.utility for a in self.state.agent_states]
        labors = [a.labor for a in self.state.agent_states]

        return {
            'timestep': timestep,
            'swf': self.state.swf,
            'mean_income': np.mean(incomes),
            'median_income': np.median(incomes),
            'std_income': np.std(incomes),
            'gini': self._calculate_gini(incomes),
            'mean_utility': np.mean(utilities),
            'mean_labor': np.mean(labors),
            'total_tax': self.state.total_tax_collected,
            'rebate': self.state.rebate_per_agent,
            'tax_rates': self.state.tax_rates.copy(),
            'step_time': step_time,
            'satisfaction_rate': np.mean([a.satisfaction for a in self.state.agent_states]),
            'mean_adjusted_utility': np.mean([a.adjusted_utility for a in self.state.agent_states]),
        }

    async def run(self) -> List[Dict]:
        """Run the full simulation."""
        logger.info(f"Starting simulation: {self.num_agents} agents, {self.max_timesteps} steps")

        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project='llm-economist',
                name=f'icrl_{self.model_name}_{self.scenario}_{self.num_agents}agents_seed{self.seed}',
                config={
                    'num_agents': self.num_agents,
                    'max_timesteps': self.max_timesteps,
                    'model_name': self.model_name,
                    'scenario': self.scenario,
                    'tax_year_length': self.tax_year_length,
                    'batch_size': self.batch_size,
                    'seed': self.seed,
                    'history_len': self.history_len,
                    'disable_exploration': self.disable_exploration,
                    'disable_exploitation': self.disable_exploitation,
                },
                tags=['icrl', 'in-context', self.scenario, f'agents_{self.num_agents}', f'seed_{self.seed}'],
                notes=f"""
ICRL (In-Context RL) Simulation
- Model: {self.model_name}
- Scenario: {self.scenario}
- Zero-shot LLM planner and workers
""",
            )
            logger.info("✓ Wandb initialized")

        start_time = time.time()

        for _ in range(self.max_timesteps):
            await self.step()

        total_time = time.time() - start_time
        logger.info(f"Simulation complete in {total_time:.1f}s ({total_time/60:.1f} min)")
        logger.info(f"Final SWF: {self.state.swf:.4f}")

        # Finish wandb
        if self.use_wandb:
            wandb.finish()

        return self.metrics_history

    def save_results(self, filepath: str):
        """Save simulation results to JSON."""
        results = {
            'config': {
                'num_agents': self.num_agents,
                'max_timesteps': self.max_timesteps,
                'model_name': self.model_name,
                'scenario': self.scenario,
                'seed': self.seed,
                'history_len': self.history_len,
                'disable_exploration': self.disable_exploration,
                'disable_exploitation': self.disable_exploitation,
                'bracket_setting': self.bracket_setting,
                'tax_year_length': self.tax_year_length,
            },
            'final_state': {
                'swf': self.state.swf,
                'tax_rates': self.state.tax_rates,
                'total_tax': self.state.total_tax_collected,
            },
            'metrics_history': self.metrics_history,
        }

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {filepath}")


async def main_async(args):
    """Async main entry point."""
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize simulator
    simulator = AsyncLLMEconomist(
        num_agents=args.num_agents,
        max_timesteps=args.max_timesteps,
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel,
        tax_year_length=args.tax_year_length,
        scenario=args.scenario,
        quantization=args.quantization,
        batch_size=args.batch_size,
        seed=args.seed,
        use_wandb=args.wandb,
        debug=args.debug,
        disable_exploration=args.disable_exploration,
        disable_exploitation=args.disable_exploitation,
        swf_weighting=args.swf_weighting,
        history_len=args.history_len,
        bracket_setting=args.bracket_setting,
        external_planner=args.external_planner,
        max_model_len=args.max_model_len,
    )

    await simulator.initialize()
    try:
        results = await simulator.run()
    except Exception as e:
        logger.error(f"Simulation failed at step {len(simulator.metrics_history)}: {e}")
        results = simulator.metrics_history
    finally:
        # Always save results (even partial) on crash
        if args.output and simulator.metrics_history:
            simulator.save_results(args.output)

    return results


def create_argument_parser():
    """Create argument parser for async simulator."""
    parser = argparse.ArgumentParser(description='Async LLM Economist Simulation')

    # Scale parameters
    parser.add_argument('--num-agents', type=int, default=1000,
                       help='Number of agents')
    parser.add_argument('--max-timesteps', type=int, default=500,
                       help='Maximum timesteps')

    # Model parameters
    parser.add_argument('--model', type=str, default='qwen3-30b-a3b',
                       choices=list(SUPPORTED_MODELS.keys()) + list(MODEL_ALIASES.keys()),
                       help='Model to use')
    parser.add_argument('--tensor-parallel', type=int, default=2,
                       help='Tensor parallel size (GPUs)')
    parser.add_argument('--quantization', type=str, default='awq',
                       choices=['awq', 'gptq', 'fp8', 'bitsandbytes', 'none'],
                       help='Quantization method (use bitsandbytes for unsloth models)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for inference')

    # Simulation parameters
    parser.add_argument('--scenario', type=str, default='bounded',
                       choices=['rational', 'bounded', 'democratic'],
                       help='Simulation scenario')
    parser.add_argument('--tax-year-length', type=int, default=128,
                       help='Steps per tax year')
    parser.add_argument('--bracket-setting', type=str, default='three',
                       choices=['flat', 'three', 'US_FED'],
                       help='Tax bracket structure (flat=1, three=3, US_FED=7 brackets)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Output
    parser.add_argument('--output', type=str, default='results.json',
                       help='Output file path')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable wandb logging')

    # ICRL ablation flags
    parser.add_argument('--history-len', type=int, default=50,
                       help='Number of past timesteps to include in planner ICRL context (default: 50)')
    parser.add_argument('--disable-exploration', action='store_true',
                       help='Disable exploration prompt cues in ICRL (ablation)')
    parser.add_argument('--disable-exploitation', action='store_true',
                       help='Disable exploitation prompt cues in ICRL (ablation)')
    parser.add_argument('--swf-weighting', default='rawlsian',
                       choices=['rawlsian', 'utilitarian'],
                       help='Social welfare function weighting scheme')

    # External planner (for RL training)
    parser.add_argument('--external-planner', action='store_true',
                       help='Skip LLM planner calls; tax rates set externally (for RL training)')
    parser.add_argument('--max-model-len', type=int, default=4096,
                       help='Max model sequence length for vLLM (increase for ICRL with long history)')

    # Utility
    parser.add_argument('--estimate-time', action='store_true',
                       help='Estimate runtime and exit')
    parser.add_argument('--pareto', action='store_true',
                       help='Show pareto-optimal configs and exit')

    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Handle utility commands
    if args.pareto:
        print("\n=== Pareto-Optimal Configurations for 6-hour Budget ===\n")
        configs = calculate_pareto_configs(
            time_budget_hours=6.0,
            model_name=args.model,
            num_gpus=args.tensor_parallel,
        )
        for i, cfg in enumerate(configs[:10]):
            print(f"{i+1}. {cfg['num_agents']:,} agents x {cfg['max_steps']:,} steps "
                  f"= {cfg['agent_step_product']:,} agent-steps "
                  f"(~{cfg['estimated_time_hours']:.1f} hours)")
        return

    if args.estimate_time:
        estimates = estimate_throughput(
            args.model,
            num_gpus=args.tensor_parallel,
            batch_size=args.batch_size,
        )
        total_steps = args.num_agents * args.max_timesteps
        estimated_hours = total_steps / estimates['estimated_agents_per_hour']
        print(f"\n=== Time Estimate ===")
        print(f"Model: {args.model} ({estimates['model_architecture']}, {estimates['model_active_params']}B active)")
        print(f"Throughput: {estimates['requests_per_second']:.1f} req/s")
        print(f"Total agent-steps: {total_steps:,}")
        print(f"Estimated time: {estimated_hours:.2f} hours ({estimated_hours*60:.0f} min)")
        return

    # Run simulation
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
