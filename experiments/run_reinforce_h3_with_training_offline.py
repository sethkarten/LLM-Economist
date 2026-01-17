#!/usr/bin/env python3
"""
REINFORCE++ H3 Experiment: Baseline Comparison with PROPER TRAINING

This version includes actual policy gradient training (the original was missing this!)

H3: Single-decision baseline comparison
- Run US federal progressive tax as baseline (2024 rates)
- Each rollout: Planner makes ONE decision for new tax rates
- Roll out full tax year with those rates
- Reward = final_swf - baseline_swf (direct comparison)
- TRAIN the planner with policy gradients after each iteration

Usage:
    python experiments/run_reinforce_h3_with_training.py --seed 42 --num-iterations 50
"""

# Use vLLM legacy API to avoid V1 compilation issues
import os
os.environ['VLLM_USE_V1'] = '0'
# CRITICAL: Allow online mode for tokenizer initialization
# Models are cached locally, but transformers needs to check HF Hub for metadata
os.environ['HF_DATASETS_OFFLINE'] = '0'
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['TRANSFORMERS_OFFLINE'] = '0'

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# =============================================================================
# PROMPT CONFIGURATIONS
# =============================================================================

# H3: BASELINE COMPARISON PROMPT (beat US federal tax baseline)
H3_SYSTEM_PROMPT = """You are an AI tax policy planner optimizing social welfare in an economic simulation.

Your goal: Set tax rates that achieve HIGHER social welfare than the US federal progressive tax baseline.

Baseline Performance (US 2024 Federal Tax):
- Social Welfare: {baseline_swf:.1f}
- Gini: {baseline_gini:.3f}
- Labor: {baseline_labor:.1f} hours/week

Output tax policy as JSON to beat this baseline."""

H3_USER_TEMPLATE = """## Current Economic State

**Income Distribution:**
- Mean Income: ${mean_income:,.0f}
- Gini: {gini:.3f}

**US Federal Baseline to Beat:**
- SWF: {baseline_swf:.1f}
- Gini: {baseline_gini:.3f}
- Labor: {baseline_labor:.1f} hours/week

Set tax rates to maximize welfare above baseline.
Output: {{"tax_rates": [rate1, rate2, ...], "brackets": [threshold1, threshold2, ...]}}"""


@dataclass
class RLConfig:
    """Configuration for REINFORCE++ H3 training."""
    experiment: str = "h3"
    planner_model: str = "meta-llama/Llama-3.1-8B-Instruct"  # Trainable with LoRA
    worker_model: str = "meta-llama/Llama-3.2-1B"  # Smaller for vLLM to fit on same GPU

    # Environment
    num_agents: int = 100
    tax_year_length: int = 64  # 1 tax year per rollout

    # Training
    num_iterations: int = 50
    rollouts_per_iter: int = 16
    learning_rate: float = 1e-5
    kl_coef: float = 0.05
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0

    # Reward: Direct comparison to baseline (no complex weighting)
    # reward = final_swf - baseline_swf

    # LoRA
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # Checkpointing
    save_every: int = 10
    seed: int = 42

    def to_dict(self):
        return asdict(self)


def format_h3_system_prompt(baseline_metrics: Dict[str, float]) -> str:
    """Format H3 system prompt with baseline metrics."""
    return H3_SYSTEM_PROMPT.format(
        baseline_swf=baseline_metrics["swf"],
        baseline_gini=baseline_metrics["gini"],
        baseline_labor=baseline_metrics["mean_labor"],
    )


def format_h3_user_prompt(state: Dict[str, Any], baseline_metrics: Dict[str, float]) -> str:
    """Format H3 user prompt with current state and baseline."""
    return H3_USER_TEMPLATE.format(
        mean_income=state["mean_income"],
        gini=state["gini"],
        baseline_swf=baseline_metrics["swf"],
        baseline_gini=baseline_metrics["gini"],
        baseline_labor=baseline_metrics["mean_labor"],
    )


def compute_reward(final_swf: float, baseline_swf: float) -> float:
    """
    Compute reward for H3: Direct comparison to baseline.

    reward = final_swf - baseline_swf

    Positive reward means planner beat the US federal tax baseline.
    """
    return final_swf - baseline_swf


class PlannerPolicy:
    """
    Trainable planner policy with LoRA adapters.

    Uses torch-based generation to get log probabilities for training.
    """

    def __init__(self, model_name: str, config: RLConfig, device: str = "cuda"):
        self.model_name = model_name
        self.config = config
        self.device = device

        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load planner model with LoRA adapters for training."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        print(f"Loading trainable planner: {self.model_name}")

        # Use cached models (SLURM nodes have no internet)
        # For offline mode, use the snapshot directory directly instead of model name
        # This avoids transformers trying to contact HuggingFace API
        model_cache_path = "/scratch/gpfs/CHIJ/milkkarten/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

        print(f"Loading from snapshot: {model_cache_path}")

        # Load tokenizer from snapshot
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cache_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model in bfloat16 from snapshot
        self.model = AutoModelForCausalLM.from_pretrained(
            model_cache_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )

        # Add LoRA adapters
        if self.config.use_lora:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        self.model.to(self.device)

    def sample_action(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
    ) -> Tuple[Optional[List[float]], float]:
        """
        Sample tax policy action and compute log probability.

        Returns:
            Tuple of (tax_rates, log_prob)
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        full_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)

        # Generate with temperature sampling
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode response
        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute log probability
        log_prob = self._compute_log_prob(outputs.scores, generated_ids)

        # Parse tax rates
        tax_rates = self._parse_tax_rates(response)

        return tax_rates, log_prob

    def compute_log_prob_for_action(
        self,
        system_prompt: str,
        user_prompt: str,
        tax_rates: List[float],
    ) -> float:
        """
        Compute log probability of a specific action (for training).

        This is used during training to recompute log probs with gradients enabled.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        full_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Format expected response
        expected_response = json.dumps({"tax_rates": [round(r, 3) for r in tax_rates]})

        # Tokenize prompt and response
        prompt_ids = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        response_ids = self.tokenizer(expected_response, return_tensors="pt", add_special_tokens=False).to(self.device)

        # Concatenate
        input_ids = torch.cat([prompt_ids["input_ids"], response_ids["input_ids"]], dim=1)

        # Forward pass
        with torch.enable_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        # Compute log probs for response tokens
        response_start = prompt_ids["input_ids"].shape[1]
        response_logits = logits[0, response_start-1:-1, :]  # Shift by 1
        response_token_ids = input_ids[0, response_start:]

        log_probs = F.log_softmax(response_logits, dim=-1)
        selected_log_probs = log_probs[range(len(response_token_ids)), response_token_ids]

        return selected_log_probs.sum()

    def _compute_log_prob(self, scores: Tuple[torch.Tensor, ...], generated_ids: torch.Tensor) -> float:
        """Compute log probability of generated sequence."""
        total_log_prob = 0.0

        for i, (score, token_id) in enumerate(zip(scores, generated_ids)):
            if token_id == self.tokenizer.pad_token_id or token_id == self.tokenizer.eos_token_id:
                break
            log_probs = F.log_softmax(score[0], dim=-1)
            total_log_prob += log_probs[token_id].item()

        return total_log_prob

    def _parse_tax_rates(self, response: str) -> Optional[List[float]]:
        """Parse tax rates from model response."""
        import re

        try:
            # Extract JSON
            if "{" in response and "}" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                json_str = response[start:end]
                data = json.loads(json_str)

                if "tax_rates" in data:
                    rates = [float(r) for r in data["tax_rates"]]
                    # Clip to valid range
                    rates = [max(0.0, min(0.99, r)) for r in rates]
                    return rates
        except:
            pass

        return None

    def save_lora_weights(self, save_path: Path):
        """Save LoRA adapter weights."""
        if self.config.use_lora:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            print(f"Saved LoRA weights to {save_path}")

    def load_lora_weights(self, load_path: Path):
        """Load LoRA adapter weights."""
        if self.config.use_lora and load_path.exists():
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model,
                load_path,
                is_trainable=True
            )
            print(f"Loaded LoRA weights from {load_path}")


class REINFORCEExperiment:
    """
    Runs REINFORCE++ training for H3 experiment with baseline comparison.

    NOW WITH ACTUAL TRAINING!
    """

    def __init__(self, config: RLConfig, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.iteration = 0
        self.best_reward = float("-inf")
        self.metrics_history = []

        # Baseline metrics (computed once during setup)
        self.baseline_metrics = None

        # Models
        self.planner_policy = None  # Trainable torch model
        self.worker_engine = None   # vLLM for fast worker inference

        # Training
        self.optimizer = None
        self.scheduler = None

    async def setup(self, skip_baseline=False):
        """Initialize planner policy and worker engine, compute baseline."""
        from llm_economist.inference.async_engine import ScalableInferenceEngine
        from llm_economist.inference.config import get_model_config

        print(f"\n{'='*60}")
        print(f"Setting up H3 Experiment (WITH TRAINING)")
        print(f"{'='*60}")
        print(f"Planner: {self.config.planner_model}")
        print(f"Workers: {self.config.worker_model}")
        print(f"Agents: {self.config.num_agents}")
        print(f"{'='*60}\n")

        # Load trainable planner policy (torch-based)
        self.planner_policy = PlannerPolicy(
            model_name=self.config.planner_model,
            config=self.config,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.planner_policy.load_model()

        # Setup optimizer
        self.setup_optimizer()

        # Load vLLM engine for workers (fast parallel inference)
        model_config = get_model_config(self.config.worker_model)

        # Override quantization to None for offline mode (cached model is base FP16)
        quant_str = None

        print(f"\nLoading worker engine with {quant_str or 'no'} quantization...")

        self.worker_engine = ScalableInferenceEngine(
            model_name=model_config.hf_name,
            quantization=quant_str,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.45,  # Planner loads first (9GB), leave enough for worker+KV cache
            max_model_len=4096,
            text_only_mode=model_config.text_only_mode,
            enforce_eager=True,
            enable_prefix_caching=False,  # Incompatible with prefix caching
            enable_chunked_prefill=False,  # Disable for FlashInfer stability
        )

        print("Worker engine loaded.\n")

        # Compute baseline metrics
        if not skip_baseline:
            print("Computing baseline (US federal progressive tax 2024)...")
            self.baseline_metrics = await self._compute_baseline()
            print(f"Baseline SWF: {self.baseline_metrics['swf']:.2f}")
            print(f"Baseline Gini: {self.baseline_metrics['gini']:.3f}")
            print(f"Baseline Labor: {self.baseline_metrics['mean_labor']:.1f} hours/week\n")
        else:
            print("Skipping baseline computation (will load from checkpoint)\n")

    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        self.optimizer = AdamW(
            self.planner_policy.model.parameters(),
            lr=self.config.learning_rate,
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_iterations,
            eta_min=self.config.learning_rate * 0.1,
        )

        print(f"Optimizer: AdamW (lr={self.config.learning_rate})")
        print(f"Scheduler: CosineAnnealing\n")

    async def _compute_baseline(self) -> Dict[str, float]:
        """
        Compute baseline metrics using US federal progressive tax (2024).

        Returns baseline SWF, Gini, and labor to beat.
        """
        from llm_economist.agents.persona_generator import generate_aligned_personas
        from llm_economist.inference.async_engine import BatchRequest
        import re

        # Initialize with same seed for reproducibility
        np.random.seed(self.config.seed)

        skills = np.exp(np.random.randn(self.config.num_agents) * 0.5 + 3.5)
        labor = np.ones(self.config.num_agents) * 40

        # US federal tax 2024 brackets and rates
        us_brackets = [11000, 44725, 95375, 182100, 231250, 578125]
        us_rates = [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]

        # Generate personas
        personas = generate_aligned_personas(n=self.config.num_agents, seed=self.config.seed)
        persona_list = list(personas.values())

        # Run simulation with US federal tax
        for step in range(self.config.tax_year_length):
            incomes = skills * labor

            # Workers choose labor given US tax rates
            prompts = []
            for i, persona in enumerate(persona_list):
                prompt = f"""{persona}

Your skill level: {skills[i]:.1f}
Current income: ${incomes[i]:,.0f}
Tax rates: US federal progressive (10%-37%)

Hours to work this week (0-100)? Number only:"""
                prompts.append(prompt)

            batch = BatchRequest(
                request_ids=[f"baseline_w_{i}" for i in range(len(prompts))],
                prompts=prompts,
                system_prompts=["You are a worker deciding hours to work."] * len(prompts),
                temperatures=[0.7] * len(prompts),
                max_tokens=10,
            )

            response = await self.worker_engine.generate_batch(batch)

            # Parse labor choices
            for i, resp in enumerate(response.responses):
                try:
                    numbers = re.findall(r'\d+\.?\d*', resp)
                    if numbers:
                        labor[i] = min(100, max(0, float(numbers[0])))
                except:
                    pass

        # Final metrics
        final_incomes = skills * labor
        baseline_swf = self._compute_swf(final_incomes, us_rates, us_brackets)
        baseline_gini = self._compute_gini(final_incomes)
        baseline_labor = labor.mean()

        return {
            "swf": baseline_swf,
            "gini": baseline_gini,
            "mean_labor": baseline_labor,
        }

    async def collect_rollout(
        self,
        rollout_id: int,
    ) -> Tuple[Dict[str, Any], float, float]:
        """
        Collect a single rollout with H3 baseline comparison approach.

        1. Initialize economic state
        2. Planner makes ONE decision for new tax rates (with log prob)
        3. Roll out full tax year with those rates
        4. Reward = final_swf - baseline_swf

        Returns:
            Tuple of (rollout_data, reward, log_prob)
        """
        from llm_economist.agents.persona_generator import generate_aligned_personas
        from llm_economist.inference.async_engine import BatchRequest
        import re

        # Initialize economic state
        np.random.seed(self.config.seed + rollout_id + self.iteration * 1000)

        skills = np.exp(np.random.randn(self.config.num_agents) * 0.5 + 3.5)
        labor = np.ones(self.config.num_agents) * 40

        # US-style brackets (planner can modify rates)
        brackets = [11000, 44725, 95375, 182100, 231250, 578125]

        # Generate personas
        personas = generate_aligned_personas(n=self.config.num_agents, seed=self.config.seed + rollout_id)
        persona_list = list(personas.values())

        # Initial state for planner decision
        incomes = skills * labor
        initial_state = {
            "mean_income": incomes.mean(),
            "gini": self._compute_gini(incomes),
        }

        # Get planner action WITH LOG PROB (using trainable model)
        system_prompt = format_h3_system_prompt(self.baseline_metrics)
        user_prompt = format_h3_user_prompt(initial_state, self.baseline_metrics)

        tax_rates, log_prob = self.planner_policy.sample_action(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,
        )

        if not tax_rates:
            # Fallback to US federal rates if parsing fails
            tax_rates = [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]
            log_prob = -10.0  # Low log prob for failed parse

        # Run full tax year with planner's chosen rates
        for step in range(self.config.tax_year_length):
            incomes = skills * labor

            # Workers choose labor given planner's tax rates
            prompts = []
            for i, persona in enumerate(persona_list):
                prompt = f"""{persona}

Your skill level: {skills[i]:.1f}
Current income: ${incomes[i]:,.0f}
Tax rates: {tax_rates[0]*100:.0f}%-{tax_rates[-1]*100:.0f}%

Hours to work this week (0-100)? Number only:"""
                prompts.append(prompt)

            batch = BatchRequest(
                request_ids=[f"w_{i}" for i in range(len(prompts))],
                prompts=prompts,
                system_prompts=["You are a worker deciding hours to work."] * len(prompts),
                temperatures=[0.7] * len(prompts),
                max_tokens=10,
            )

            response = await self.worker_engine.generate_batch(batch)

            # Parse labor choices
            for i, resp in enumerate(response.responses):
                try:
                    numbers = re.findall(r'\d+\.?\d*', resp)
                    if numbers:
                        labor[i] = min(100, max(0, float(numbers[0])))
                except:
                    pass

        # Final state after full tax year
        final_incomes = skills * labor
        final_swf = self._compute_swf(final_incomes, tax_rates, brackets)

        # Compute reward: Direct comparison to baseline
        reward = compute_reward(final_swf, self.baseline_metrics["swf"])

        rollout_data = {
            "initial_state": initial_state,
            "action": {"tax_rates": tax_rates, "brackets": brackets},
            "final_swf": final_swf,
            "baseline_swf": self.baseline_metrics["swf"],
            "reward": reward,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

        return rollout_data, reward, log_prob

    def _compute_gini(self, values: np.ndarray) -> float:
        """Compute Gini coefficient."""
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        index = np.arange(1, n + 1)
        return float((2 * (index * sorted_vals).sum() / (n * sorted_vals.sum()) - (n + 1) / n))

    def _compute_swf(self, incomes: np.ndarray, tax_rates: List[float], brackets: List[float]) -> float:
        """Compute social welfare with numerical stability."""
        clipped_rates = [max(0.0, min(0.99, r)) for r in tax_rates]

        # Apply progressive taxes
        taxes = np.zeros_like(incomes)
        prev_bracket = 0.0
        all_brackets = brackets + [float('inf')]

        for i, bracket in enumerate(all_brackets):
            if i < len(clipped_rates):
                bracket_income = np.clip(incomes - prev_bracket, 0, bracket - prev_bracket)
                taxes += bracket_income * clipped_rates[i]
            prev_bracket = bracket

        post_tax = np.clip(incomes - taxes, 100.0, None)

        # Isoelastic utility
        eta = 1.5
        utilities = (post_tax ** (1 - eta) - 1) / (1 - eta)
        utilities = np.nan_to_num(utilities, nan=0.0, posinf=1e6, neginf=-1e6)
        utilities = np.clip(utilities, -100, 100)

        return float(utilities.sum())

    def train_step(self, rollouts: List[Tuple], iter_num: int) -> Dict[str, float]:
        """
        Perform policy gradient update on batch of rollouts.

        Args:
            rollouts: List of (rollout_data, reward, log_prob) tuples
            iter_num: Current iteration number

        Returns:
            Dictionary of training metrics
        """
        self.planner_policy.model.train()

        # Extract data
        rollout_data_list = [r[0] for r in rollouts]
        rewards = torch.tensor([r[1] for r in rollouts], dtype=torch.float32, device=self.planner_policy.device)
        old_log_probs = torch.tensor([r[2] for r in rollouts], dtype=torch.float32, device=self.planner_policy.device)

        # Compute advantages using group-relative baseline (GRPO-style)
        baseline = rewards.mean()
        advantages = rewards - baseline

        # Normalize advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Recompute log probs with gradients
        new_log_probs = []
        for rollout_data in rollout_data_list:
            log_prob = self.planner_policy.compute_log_prob_for_action(
                system_prompt=rollout_data["system_prompt"],
                user_prompt=rollout_data["user_prompt"],
                tax_rates=rollout_data["action"]["tax_rates"],
            )
            new_log_probs.append(log_prob)

        new_log_probs = torch.stack(new_log_probs)

        # Policy gradient loss: -E[log π(a|s) * A]
        pg_loss = -(new_log_probs * advantages).mean()

        # KL penalty (prevent drift from initial policy)
        kl_loss = torch.tensor(0.0, device=self.planner_policy.device)
        if self.config.kl_coef > 0:
            log_ratio = new_log_probs - old_log_probs
            kl_loss = (log_ratio ** 2).mean()  # Simplified KL

        # Entropy bonus (encourage exploration)
        entropy_loss = -new_log_probs.mean()

        # Total loss
        total_loss = (
            pg_loss
            + self.config.kl_coef * kl_loss
            - self.config.entropy_coef * entropy_loss
        )

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.planner_policy.model.parameters(),
                self.config.max_grad_norm
            )

        self.optimizer.step()
        self.scheduler.step()

        return {
            "loss/total": total_loss.item(),
            "loss/pg": pg_loss.item(),
            "loss/kl": kl_loss.item(),
            "loss/entropy": entropy_loss.item(),
            "advantage/mean": advantages.mean().item(),
            "advantage/std": advantages.std().item(),
            "lr": self.scheduler.get_last_lr()[0],
        }

    async def train(self):
        """Main training loop WITH GRADIENT UPDATES."""
        print(f"\n{'='*60}")
        print(f"Starting REINFORCE++ Training ({self.config.experiment.upper()})")
        print(f"{'='*60}")
        print(f"Iterations: {self.config.num_iterations}")
        print(f"Rollouts/iter: {self.config.rollouts_per_iter}")
        print(f"{'='*60}\n")

        start_iter = self.iteration  # Supports resume from checkpoint
        for iteration in range(start_iter, self.config.num_iterations):
            self.iteration = iteration
            iter_start = time.time()

            print(f"\nIteration {iteration+1}/{self.config.num_iterations}")

            # Collect rollouts
            rollouts = []
            for r in range(self.config.rollouts_per_iter):
                rollout_data, reward, log_prob = await self.collect_rollout(r)
                rollouts.append((rollout_data, reward, log_prob))

                if (r + 1) % 8 == 0:
                    print(f"  Rollouts: {r+1}/{self.config.rollouts_per_iter}")

            rewards = np.array([r[1] for r in rollouts])

            # TRAIN ON ROLLOUTS (THIS WAS MISSING!)
            train_metrics = self.train_step(rollouts, iteration)

            iter_time = time.time() - iter_start

            metrics = {
                "iteration": iteration,
                "reward_mean": float(rewards.mean()),
                "reward_std": float(rewards.std()),
                "reward_max": float(rewards.max()),
                "reward_min": float(rewards.min()),
                "time": iter_time,
                **train_metrics,  # Add training metrics
            }
            self.metrics_history.append(metrics)

            print(f"  Reward: {metrics['reward_mean']:.3f} ± {metrics['reward_std']:.3f}")
            print(f"  Loss: {metrics['loss/total']:.3f} (PG: {metrics['loss/pg']:.3f})")
            print(f"  LR: {metrics['lr']:.2e}")
            print(f"  Time: {iter_time:.1f}s")

            # Track best
            if metrics["reward_mean"] > self.best_reward:
                self.best_reward = metrics["reward_mean"]
                self.save_checkpoint("best")

            # Periodic save
            if (iteration + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"iter_{iteration+1}")

        # Final save
        self.save_checkpoint("final")
        print(f"\nTraining complete! Best reward: {self.best_reward:.3f}")

    def save_checkpoint(self, name: str):
        """Save training checkpoint including LoRA weights."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "config": self.config.to_dict(),
            "iteration": self.iteration,
            "best_reward": self.best_reward,
            "metrics_history": self.metrics_history,
            "baseline_metrics": self.baseline_metrics,  # Cache baseline
        }

        with open(checkpoint_dir / "state.json", "w") as f:
            json.dump(state, f, indent=2)

        # Save LoRA weights
        self.planner_policy.save_lora_weights(checkpoint_dir / "planner_lora")

        # Save optimizer state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }, checkpoint_dir / "optimizer.pt")

        print(f"  Saved: {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load training state from checkpoint (metadata only). Returns True if successful."""
        state_file = Path(checkpoint_path) / "state.json"
        if not state_file.exists():
            print(f"No checkpoint found at {checkpoint_path}")
            return False

        with open(state_file) as f:
            state = json.load(f)

        self.iteration = state.get("iteration", 0) + 1  # Resume from next iteration
        self.best_reward = state.get("best_reward", float("-inf"))
        self.metrics_history = state.get("metrics_history", [])
        self.baseline_metrics = state.get("baseline_metrics")  # Load cached baseline

        print(f"Resumed from checkpoint: {checkpoint_path}")
        print(f"  Starting at iteration {self.iteration}")
        print(f"  Best reward so far: {self.best_reward:.3f}")

        if self.baseline_metrics:
            print(f"  Loaded cached baseline: SWF={self.baseline_metrics['swf']:.2f}, Gini={self.baseline_metrics['gini']:.3f}")

        # Store checkpoint path for loading model weights after setup()
        self.checkpoint_path = checkpoint_path

        return True

    def load_model_weights(self):
        """Load LoRA weights and optimizer state (call after setup())."""
        if not hasattr(self, 'checkpoint_path') or not self.checkpoint_path:
            return

        # Load LoRA weights
        lora_path = Path(self.checkpoint_path) / "planner_lora"
        if lora_path.exists():
            self.planner_policy.load_lora_weights(lora_path)
            print(f"  Loaded LoRA weights from {lora_path}")

        # Load optimizer state
        optimizer_path = Path(self.checkpoint_path) / "optimizer.pt"
        if optimizer_path.exists():
            checkpoint = torch.load(optimizer_path)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            print(f"  Loaded optimizer state")

        return True

    async def shutdown(self):
        """Cleanup."""
        if self.worker_engine:
            await self.worker_engine.shutdown()


async def main():
    parser = argparse.ArgumentParser(
        description="REINFORCE++ H3 Experiment: Baseline Comparison WITH TRAINING",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-iterations", type=int, default=50,
                       help="Number of training iterations")
    parser.add_argument("--rollouts-per-iter", type=int, default=16,
                       help="Rollouts per iteration")
    parser.add_argument("--num-agents", type=int, default=100,
                       help="Number of worker agents")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output", type=str, default="models/reinforce_h3_trained",
                       help="Output directory")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")

    args = parser.parse_args()

    # Create config
    config = RLConfig(
        experiment="h3",
        num_agents=args.num_agents,
        num_iterations=args.num_iterations,
        rollouts_per_iter=args.rollouts_per_iter,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    output_dir = args.output

    # Create experiment
    experiment = REINFORCEExperiment(config, output_dir)

    try:
        # Load checkpoint metadata first if resuming
        skip_baseline = False
        if args.resume:
            success = experiment.load_checkpoint(args.resume)
            # Skip baseline computation if it was cached in checkpoint
            if success and experiment.baseline_metrics is not None:
                skip_baseline = True
                print("Using cached baseline from checkpoint (skipping expensive recomputation)\n")

        # Setup model and engines
        await experiment.setup(skip_baseline=skip_baseline)

        # Load model weights after setup (if resuming)
        if args.resume:
            experiment.load_model_weights()

        await experiment.train()
    finally:
        await experiment.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
