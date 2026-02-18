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
    utility_type: str  # egotistical, altruistic, adversarial
    history: List[Dict]


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
    ):
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

        # Tax configuration (US Federal 2024 brackets)
        self.tax_brackets = [0, 23000, 47000, 94000, 192000, 244000, 500000, 1000000]
        self.tax_rates = [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]  # Initial rates

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

        self.engine = ScalableInferenceEngine(
            model_name=model_config.hf_name,
            tensor_parallel_size=self.tensor_parallel_size,
            quantization=quant_value,
            max_model_len=32768,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            kv_cache_dtype="auto",  # Auto for better Blackwell compatibility
            max_num_seqs=min(512, self.batch_size * 2),
            text_only_mode=getattr(model_config, 'text_only_mode', False),
            dtype=dtype,
        )
        await self.engine.initialize()

        # Generate personas
        logger.info("Generating population-aligned personas...")
        self.personas = generate_aligned_personas(
            self.num_agents,
            use_llm_narratives=False,  # Faster without LLM narratives
            seed=self.seed,
        )

        # Initialize agent states
        logger.info("Initializing agent states...")
        skills = self._sample_skills()
        agent_states = []

        persona_ids = list(self.personas.keys())
        for i in range(self.num_agents):
            persona_id = persona_ids[i % len(persona_ids)]
            state = AgentState(
                id=i,
                name=f"worker_{i}",
                skill=skills[i],
                labor=40.0,  # Default 40 hours
                income=skills[i] * 40.0,  # Initial income
                post_tax_income=0.0,
                utility=0.0,
                tax_paid=0.0,
                persona_prompt=self.personas[persona_id],
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

    def _sample_skills(self) -> List[float]:
        """Sample skills from GB2 distribution (US income calibrated)."""
        # Use GB2 distribution fitted to US ACS data
        incomes = rGB2(self.num_agents)
        # Convert to hourly skill (income at 40 hrs/week)
        skills = [float(inc / 40.0) for inc in incomes]
        return skills

    def _build_worker_prompt(self, agent: AgentState, timestep: int) -> tuple:
        """Build system and user prompts for a worker agent."""
        system_prompt = f"""You are an economic agent in a tax simulation.
{agent.persona_prompt}

Your goal is to maximize your utility by choosing how many hours to work per week.
Utility = post_tax_income + rebate - cost_of_labor
where cost_of_labor increases with hours worked.

Current tax brackets: {self.state.tax_brackets}
Current tax rates: {[f"{r*100:.1f}%" for r in self.state.tax_rates]}
"""

        user_prompt = f"""Timestep {timestep}:
Your skill level (hourly wage): ${agent.skill:.2f}
Your current labor hours: {agent.labor:.0f}
Your current pre-tax income: ${agent.income:.2f}
Your last post-tax income + rebate: ${agent.post_tax_income:.2f}
Your last utility: {agent.utility:.2f}

Choose your labor hours for this period (0-100).
Consider the tax rates and your preferences.

Respond with JSON: {{"labor_hours": <number>, "reasoning": "<brief explanation>"}}
"""
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
            cue_text += "Use the historical data to influence your answer in order to maximize SWF, while balancing exploration and exploitation by choosing varying rates of TAX. "
            cue_text += "Try different rates of TAX before picking the one that corresponds to the highest SWF. "
        if not self.disable_exploitation and self.planner_history:
            # Compute best average tax rates from recent history
            K = min(self.history_len, len(self.planner_history))
            recent = self.planner_history[-K:]
            best_entry = max(recent, key=lambda x: x['swf'])
            best_rates = [f"{r*100:.1f}%" for r in best_entry['tax_rates']]
            cue_text += f"The best marginal tax rate historically was TAX={best_rates} corresponding to SWF={best_entry['swf']:.4f}. "

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

        # Step 1: Update tax rates if new tax year
        if is_tax_year_start and timestep > 0:
            await self._update_tax_rates(timestep)

        # Step 2: Batch worker decisions
        worker_responses = await self._batch_worker_decisions(timestep)

        # Step 3: Update agent states based on responses
        for i, (response, _) in enumerate(worker_responses):
            agent = self.state.agent_states[i]
            try:
                data = json.loads(response)
                agent.labor = float(np.clip(data.get('labor_hours', 40), 0, 100))
            except (json.JSONDecodeError, KeyError, TypeError):
                # Keep previous labor if parsing fails
                pass

            # Update income
            agent.income = agent.skill * agent.labor

        # Step 4: Apply taxes and calculate utilities
        self._apply_taxes()
        self._calculate_utilities()

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
                request_ids=[f"worker_{i}" for i in range(batch_start, batch_end)],
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
        c = 0.01
        delta = 2.0

        swf = 0.0
        for agent in self.state.agent_states:
            cost = c * (agent.labor ** delta)
            agent.utility = agent.post_tax_income - cost

            if self.swf_weighting == 'utilitarian':
                swf += agent.utility
            else:
                # rawlsian: weight by inverse pre-tax income (1/z)
                if agent.income > 0:
                    swf += agent.utility / agent.income

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
