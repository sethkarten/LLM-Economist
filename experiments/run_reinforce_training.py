#!/usr/bin/env python3
"""
REINFORCE++ v2 Training: Full AsyncLLMEconomist Rollouts

Key difference from run_reinforce_h3_optimized.py:
- Rollouts use the ACTUAL simulation (AsyncLLMEconomist with external_planner=True)
  instead of a simplified approximation
- Same SWF formula, worker prompts, tax logic, and persona generation as production
- Multi-tax-year rollouts (planner sets rates at the start of each tax year)
- Supports dynamic bracket settings (flat/three/US_FED)

Usage:
    # Quick local test
    uv run python experiments/run_reinforce_training.py \
        --num-agents 16 --num-rollouts 4 --num-iterations 3 --no-wandb

    # Full training on 2x5090
    uv run python experiments/run_reinforce_training.py \
        --planner-model google/gemma-3-4b-it \
        --worker-model Qwen/Qwen3-8B-AWQ \
        --num-agents 32 --num-iterations 500

    # SLURM cluster (B200)
    uv run python experiments/run_reinforce_training.py \
        --planner-model google/gemma-3-4b-it \
        --worker-model Qwen/Qwen3-8B-AWQ \
        --num-agents 100 --num-iterations 1000 \
        --output-dir results/reinforce_v2/b200_seed42
"""

# Environment setup — must come before any torch/vLLM imports
import os
os.environ['VLLM_USE_V1'] = '0'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
# SLURM compute nodes have no internet — force offline mode
if 'SLURM_JOB_ID' in os.environ:
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HOME'] = '/scratch/gpfs/CHIJ/milkkarten/huggingface'
    os.environ['TRANSFORMERS_CACHE'] = '/scratch/gpfs/CHIJ/milkkarten/huggingface/hub'
    os.environ['WANDB_MODE'] = 'offline'
    os.environ['WANDB_DIR'] = '/scratch/gpfs/CHIJ/milkkarten/LLM-Economist/wandb'

import argparse
import asyncio
import json
import math
import random
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import wandb


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

PLANNER_SYSTEM_PROMPT = """You are an AI tax policy planner in an economic simulation with {num_agents} workers.

Social welfare function (rawlsian):
  SWF = sum( utility_i / max(income_i, $1000) ) for all workers
  utility_i = post_tax_income_i + rebate - 0.0005 * labor_hours_i^3.5

Your goal: Set marginal tax rates for {num_brackets} brackets to maximize SWF.

Tax brackets: {brackets}

Respond with ONLY a JSON object: {{"tax_rates": [r1, r2, ...]}}
Each rate must be between 0.0 and 0.99."""

PLANNER_USER_TEMPLATE = """## Current Economic State

- Number of workers: {num_agents}
- Mean pre-tax income: ${mean_income:,.0f}
- Median pre-tax income: ${median_income:,.0f}
- Income Gini: {gini:.3f}
- Current tax rates: {current_rates}
- Previous SWF: {prev_swf:.4f}
- US Federal baseline SWF: {baseline_swf:.4f}

Set {num_brackets} marginal tax rates (0.0-0.99) to maximize social welfare.
Respond with ONLY: {{"tax_rates": [r1, r2, ...]}}"""


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for REINFORCE++ v2 training with full simulation rollouts."""
    planner_model: str = "google/gemma-3-4b-it"
    worker_model: str = "Qwen/Qwen3-8B-AWQ"
    num_agents: int = 32
    bracket_setting: str = "three"  # flat (1), three (3), or US_FED (7)
    num_rollouts: int = 16          # batch size per iteration
    tax_year_length: int = 64       # simulation steps per tax year
    num_tax_years: int = 4          # rollout = num_tax_years * tax_year_length steps
    num_iterations: int = 500
    lr: float = 1e-6  # Conservative LR to prevent format collapse
    kl_coef: float = 0.05  # Anchor near base model to prevent format collapse
    entropy_coef: float = 0.01
    clip_grad: float = 1.0
    lora_r: int = 8
    lora_alpha: int = 8
    lora_targets: Tuple = ("gate_proj", "up_proj", "down_proj")
    seed: int = 42
    output_dir: str = "results/reinforce_v2"
    gpu_memory_util: float = 0.85   # for worker vLLM engine
    parallel_rollouts: int = 4      # concurrent rollouts (unused; kept for future)
    use_wandb: bool = True


# =============================================================================
# FIXED POPULATION
# =============================================================================

class FixedPopulation:
    """Fixed population for RL training. Generated ONCE, reused across all rollouts."""

    def __init__(self, num_agents: int, seed: int = 42):
        self.num_agents = num_agents
        self.seed = seed

        # Generate skills from GB2 distribution with fixed seed
        np.random.seed(seed)
        random.seed(seed)

        from llm_economist.utils.common import rGB2
        incomes = rGB2(num_agents)
        self.skills = [float(inc / 40.0) for inc in incomes]

        # Generate personas with fixed seed (no LLM narratives for speed)
        from llm_economist.agents.persona_generator import generate_aligned_personas
        self.personas = generate_aligned_personas(
            n=num_agents,
            use_llm_narratives=False,
            seed=seed,
        )


# =============================================================================
# PLANNER POLICY (LoRA-tuned causal LM)
# =============================================================================

class PlannerPolicy:
    """
    Trainable planner policy with LoRA adapters.

    Uses torch-based generation to get log probabilities for training.
    Adapted from run_reinforce_h3_optimized.py with dynamic bracket support.
    """

    def __init__(
        self,
        model_name: str,
        config: TrainingConfig,
        device: str = "cuda",
        force_single_gpu: bool = False,
    ):
        self.model_name = model_name
        self.config = config
        self.device = device
        self.force_single_gpu = force_single_gpu

        # Derive number of brackets from bracket_setting
        from llm_economist.utils.bracket import get_num_brackets
        self.num_brackets = get_num_brackets(config.bracket_setting)

        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load planner model with LoRA adapters for training."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        print(f"Loading trainable planner: {self.model_name}")

        model_name_or_path = self.model_name
        print(f"Loading from: {model_name_or_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model in BF16
        # In 2-GPU mode, force planner to cuda:0 only (cuda:1 is for vLLM)
        device_map = {"": 0} if self.force_single_gpu else "auto"
        print(f"Loading planner with device_map={device_map} (force_single_gpu={self.force_single_gpu})", flush=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing for memory efficiency")

        # Add LoRA adapters (MLP-only targets)
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=0.0,
            target_modules=list(self.config.lora_targets),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        self.model.to(self.device)

    # Forced JSON prefix to prevent format collapse during training.
    # The model only generates numeric values + closing brackets.
    JSON_PREFIX = '{"tax_rates": ['

    def sample_action(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_new_tokens: int = 64,
    ) -> Tuple[Optional[List[float]], float, str]:
        """
        Sample tax policy action using forced JSON prefix for robustness.

        Returns:
            Tuple of (tax_rates or None, log_prob, raw_suffix)
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

        # Force JSON prefix as part of the prompt
        prompt_with_prefix = full_prompt + self.JSON_PREFIX
        inputs = self.tokenizer(prompt_with_prefix, return_tensors="pt").to(self.device)

        # CRITICAL: switch to eval mode for generation.
        # Train mode + gradient checkpointing + LoRA dropout causes degeneration.
        was_training = self.model.training
        self.model.eval()

        # Generate only the suffix (numeric values + closing brackets)
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

        # Restore training mode
        if was_training:
            self.model.train()

        # Decode suffix
        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        raw_suffix = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Truncate at first '}' to prevent rambling
        if '}' in raw_suffix:
            raw_suffix = raw_suffix[:raw_suffix.index('}') + 1]

        # Compute log probability of suffix
        log_prob = self._compute_log_prob(outputs.scores, generated_ids)

        # Reconstruct full JSON and parse
        full_response = self.JSON_PREFIX + raw_suffix
        tax_rates = self._parse_tax_rates(full_response)

        if tax_rates is None:
            print(f"  [DEBUG] Failed to parse forced-prefix output: '{raw_suffix}'")

        return tax_rates, log_prob, raw_suffix

    def _compute_log_prob_batch_internal(
        self,
        prompts_and_suffixes: List[Tuple[str, str, str]],
        enable_grad: bool = True,
    ) -> torch.Tensor:
        """
        Compute log probabilities for a batch of actions using forced JSON prefix.

        Args:
            prompts_and_suffixes: List of (system_prompt, user_prompt, raw_suffix) tuples
                where raw_suffix is the generated text after JSON_PREFIX
            enable_grad: Whether to enable gradient computation

        Returns:
            Tensor of log probabilities, shape (batch_size,)
        """
        batch_input_ids = []
        batch_response_lengths = []

        for system_prompt, user_prompt, raw_suffix in prompts_and_suffixes:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            full_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Prompt includes forced JSON prefix
            prompt_with_prefix = full_prompt + self.JSON_PREFIX

            # Tokenize prompt (with prefix) and suffix separately
            prompt_ids = self.tokenizer(prompt_with_prefix, return_tensors="pt")["input_ids"]
            suffix_ids = self.tokenizer(
                raw_suffix, return_tensors="pt", add_special_tokens=False
            )["input_ids"]

            input_ids = torch.cat([prompt_ids, suffix_ids], dim=1).squeeze(0)

            batch_input_ids.append(input_ids)
            batch_response_lengths.append(suffix_ids.shape[1])

        # Pad to same length
        max_len = max(ids.shape[0] for ids in batch_input_ids)
        padded_input_ids = []
        attention_mask = []

        for ids in batch_input_ids:
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                ids = torch.cat([
                    torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=ids.dtype),
                    ids,
                ])
            padded_input_ids.append(ids)
            mask = torch.ones_like(ids)
            if pad_len > 0:
                mask[:pad_len] = 0
            attention_mask.append(mask)

        # Process in chunks. Larger chunks are faster but use more memory.
        # A6000 (48GB) can handle batch=8 easily; RTX 5090 (32GB) needs smaller chunks.
        chunk_size = min(4, len(padded_input_ids))
        all_log_probs = []

        grad_context = torch.enable_grad() if enable_grad else torch.no_grad()

        for chunk_start in range(0, len(padded_input_ids), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(padded_input_ids))
            chunk_input_ids = padded_input_ids[chunk_start:chunk_end]
            chunk_attention_mask = [attention_mask[i] for i in range(chunk_start, chunk_end)]

            # Stack chunk into batch
            input_ids_batch = torch.stack(chunk_input_ids).to(self.device)
            attention_mask_batch = torch.stack(chunk_attention_mask).to(self.device)

            # Forward pass for chunk
            with grad_context:
                outputs = self.model(input_ids_batch, attention_mask=attention_mask_batch)
                logits = outputs.logits

            # Compute log probs for each sample in chunk
            for i in range(len(chunk_input_ids)):
                global_i = chunk_start + i
                response_len = batch_response_lengths[global_i]

                # Find where response starts (accounting for left-padding)
                pad_len = max_len - batch_input_ids[global_i].shape[0]
                prompt_len = batch_input_ids[global_i].shape[0] - response_len
                response_start = pad_len + prompt_len

                # Extract response logits and token IDs
                response_logits = logits[i, response_start - 1:response_start + response_len - 1, :]
                response_token_ids = input_ids_batch[i, response_start:response_start + response_len]

                # Compute log probs
                log_probs = F.log_softmax(response_logits, dim=-1)
                selected_log_probs = log_probs[range(len(response_token_ids)), response_token_ids]

                all_log_probs.append(selected_log_probs.mean())

        return torch.stack(all_log_probs)

    def compute_log_prob_batch(
        self,
        prompts_and_suffixes: List[Tuple[str, str, str]],
    ) -> torch.Tensor:
        """Compute log probs with gradients for training.

        Args:
            prompts_and_suffixes: List of (system_prompt, user_prompt, raw_suffix) tuples
        """
        return self._compute_log_prob_batch_internal(prompts_and_suffixes, enable_grad=True)

    def compute_ref_log_prob_batch(
        self,
        prompts_and_suffixes: List[Tuple[str, str, str]],
    ) -> torch.Tensor:
        """Compute log probs from frozen base model (LoRA disabled) for KL penalty."""
        self.model.disable_adapter_layers()
        with torch.no_grad():
            ref_log_probs = self._compute_log_prob_batch_internal(
                prompts_and_suffixes, enable_grad=False
            )
        self.model.enable_adapter_layers()
        return ref_log_probs

    def _compute_log_prob(
        self,
        scores: Tuple[torch.Tensor, ...],
        generated_ids: torch.Tensor,
    ) -> float:
        """Compute mean log probability of generated sequence from generation scores."""
        total_log_prob = 0.0
        count = 0

        for score, token_id in zip(scores, generated_ids):
            if token_id == self.tokenizer.pad_token_id or token_id == self.tokenizer.eos_token_id:
                break
            log_probs = F.log_softmax(score[0], dim=-1)
            total_log_prob += log_probs[token_id].item()
            count += 1

        return total_log_prob / max(count, 1)

    def _parse_tax_rates(self, response: str) -> Optional[List[float]]:
        """Parse tax rates from model response. Accepts any length array."""
        try:
            if "{" in response and "}" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                json_str = response[start:end]
                data = json.loads(json_str)

                if "tax_rates" in data:
                    rates = [float(r) for r in data["tax_rates"]]
                    # Clip to valid range
                    rates = [max(0.0, min(0.99, r)) for r in rates]

                    # Pad or truncate to match expected bracket count
                    if len(rates) < self.num_brackets:
                        rates = rates + [0.15] * (self.num_brackets - len(rates))
                    elif len(rates) > self.num_brackets:
                        rates = rates[:self.num_brackets]

                    return rates
        except (ValueError, json.JSONDecodeError, TypeError, KeyError):
            pass

        return None

    def save_lora_weights(self, save_path):
        """Save LoRA adapter weights."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Saved LoRA weights to {save_path}")

    def load_lora_weights(self, load_path):
        """Load LoRA adapter weights."""
        load_path = Path(load_path)
        if load_path.exists():
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model,
                load_path,
                is_trainable=True,
            )
            print(f"Loaded LoRA weights from {load_path}")


# =============================================================================
# ROLLOUT ENVIRONMENT
# =============================================================================

class RolloutEnvironment:
    """Wraps AsyncLLMEconomist with external_planner=True for RL rollouts."""

    def __init__(
        self,
        config: TrainingConfig,
        population: FixedPopulation,
        worker_gpu: Optional[str] = None,
    ):
        self.config = config
        self.population = population
        self.worker_gpu = worker_gpu  # Physical GPU ID for vLLM (e.g. "1")
        self.sim = None

    async def initialize(self):
        """Create and initialize AsyncLLMEconomist instance with external_planner=True."""
        from llm_economist.main_async import AsyncLLMEconomist

        total_steps = self.config.tax_year_length * self.config.num_tax_years

        # In 2-GPU mode, restrict vLLM to the worker GPU so it doesn't collide
        # with the planner on GPU 0. vLLM always uses cuda:0, so we make the
        # worker GPU the only visible device for the spawned engine process.
        original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if self.worker_gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.worker_gpu
            print(f"[RolloutEnv] Set CUDA_VISIBLE_DEVICES={self.worker_gpu} for vLLM worker engine")

        # In single-GPU mode (no worker_gpu), reduce memory for vLLM so planner fits too
        gpu_mem_util = self.config.gpu_memory_util if self.worker_gpu is not None else 0.50

        self.sim = AsyncLLMEconomist(
            num_agents=self.config.num_agents,
            max_timesteps=total_steps,
            model_name=self.config.worker_model,
            tensor_parallel_size=1,
            tax_year_length=self.config.tax_year_length,
            scenario="bounded",
            quantization="awq",
            batch_size=self.config.num_agents,
            seed=self.config.seed,
            external_planner=True,
            fixed_skills=self.population.skills,
            fixed_personas=self.population.personas,
            bracket_setting=self.config.bracket_setting,
            gpu_memory_utilization=gpu_mem_util,
            history_len=5,
            max_model_len=8192,
        )
        await self.sim.initialize()

        # Restore original CUDA_VISIBLE_DEVICES so planner can still use GPU 0
        if self.worker_gpu is not None:
            if original_cuda_visible is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
            else:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            print(f"[RolloutEnv] Restored CUDA_VISIBLE_DEVICES={original_cuda_visible}")

    async def run_rollout(
        self,
        tax_rates_per_year: List[List[float]],
    ) -> Tuple[List[float], float]:
        """
        Run a complete rollout with externally-specified tax rates.

        Args:
            tax_rates_per_year: List of rate arrays, one per tax year.

        Returns:
            (swf_per_year, final_swf) -- SWF at end of each tax year, and final SWF.
        """
        self.sim.reset_state()
        swf_per_year = []

        for year_idx, rates in enumerate(tax_rates_per_year):
            self.sim.set_tax_rates(rates)
            # Run tax_year_length steps
            for step_idx in range(self.config.tax_year_length):
                await self.sim.step()
            swf_per_year.append(self.sim.state.swf)

        return swf_per_year, self.sim.state.swf

    def get_observation(self) -> Dict[str, Any]:
        """Get current economic state for planner observation."""
        agents = self.sim.state.agent_states
        incomes = [a.income for a in agents]
        return {
            'num_agents': len(agents),
            'mean_income': float(np.mean(incomes)),
            'median_income': float(np.median(incomes)),
            'gini': self.sim._calculate_gini(incomes),
            'current_rates': list(self.sim.state.tax_rates),
            'prev_swf': self.sim.state.swf,
            'tax_brackets': list(self.sim.state.tax_brackets),
        }


# =============================================================================
# REINFORCE++ TRAINER
# =============================================================================

class REINFORCETrainer:
    """REINFORCE++ trainer using full AsyncLLMEconomist rollouts."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.population = FixedPopulation(config.num_agents, config.seed)
        self.policy: Optional[PlannerPolicy] = None
        self.env: Optional[RolloutEnvironment] = None
        self.optimizer: Optional[AdamW] = None
        self.scheduler: Optional[LambdaLR] = None
        self.baseline_swf: float = 0.0
        self.best_swf: float = float('-inf')
        self.metrics_history: List[Dict[str, Any]] = []

    async def setup(self):
        """Initialize policy, environment, and compute baseline."""
        from llm_economist.utils.bracket import get_num_brackets

        # Detect GPU setup
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        visible_gpus = [
            int(g.strip()) for g in cuda_visible.split(",") if g.strip().isdigit()
        ] if cuda_visible else []
        use_2gpu = len(visible_gpus) >= 2
        print(f"[SETUP] CUDA_VISIBLE_DEVICES='{cuda_visible}', 2-GPU mode={use_2gpu}", flush=True)

        # Load planner policy
        planner_device = "cuda:0" if use_2gpu else ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PlannerPolicy(
            model_name=self.config.planner_model,
            config=self.config,
            device=planner_device,
            force_single_gpu=use_2gpu,
        )
        self.policy.load_model()

        # Setup optimizer with linear warmup + cosine decay
        self.optimizer = AdamW(self.policy.model.parameters(), lr=self.config.lr)
        warmup_steps = max(1, int(0.1 * self.config.num_iterations))

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, self.config.num_iterations - warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        print(f"Optimizer: AdamW (lr={self.config.lr})")
        print(f"Scheduler: Linear warmup ({warmup_steps} steps) + Cosine decay")

        # Create rollout environment (single env reused across rollouts)
        # In 2-GPU mode, assign vLLM to the second visible GPU
        worker_gpu = str(visible_gpus[1]) if use_2gpu else None
        self.env = RolloutEnvironment(self.config, self.population, worker_gpu=worker_gpu)
        await self.env.initialize()

        # Compute US Federal baseline SWF
        self.baseline_swf = await self._compute_baseline()
        print(f"US Federal baseline SWF: {self.baseline_swf:.4f}")

    async def _compute_baseline(self) -> float:
        """Run simulation with representative US tax rates to get baseline SWF."""
        from llm_economist.utils.bracket import get_num_brackets
        num_b = get_num_brackets(self.config.bracket_setting)

        if num_b == 3:
            us_rates = [0.12, 0.24, 0.35]
        elif num_b == 7:
            us_rates = [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]
        else:
            us_rates = [0.22]  # flat

        # Run one full rollout with US rates repeated for each tax year
        rates_per_year = [us_rates] * self.config.num_tax_years
        _, baseline_swf = await self.env.run_rollout(rates_per_year)
        return baseline_swf

    def _format_observation(
        self,
        obs: Dict[str, Any],
        baseline_swf: float,
    ) -> Tuple[str, str]:
        """Format observation into system + user prompts for planner."""
        from llm_economist.utils.bracket import get_num_brackets
        num_b = get_num_brackets(self.config.bracket_setting)

        system_prompt = PLANNER_SYSTEM_PROMPT.format(
            num_agents=obs['num_agents'],
            num_brackets=num_b,
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
            num_brackets=num_b,
        )
        return system_prompt, user_prompt

    async def collect_rollouts(self) -> List[Dict[str, Any]]:
        """Collect a batch of rollouts sequentially."""
        from llm_economist.utils.bracket import get_num_brackets

        rollouts = []
        num_b = get_num_brackets(self.config.bracket_setting)

        for r_idx in range(self.config.num_rollouts):
            # Reset state and get initial observation
            self.env.sim.reset_state()
            obs = self.env.get_observation()
            sys_prompt, user_prompt = self._format_observation(obs, self.baseline_swf)

            # Sample planner action (with forced JSON prefix)
            tax_rates, log_prob, raw_suffix = self.policy.sample_action(sys_prompt, user_prompt)
            format_success = tax_rates is not None

            if not tax_rates:
                # Fallback to uniform moderate rates
                tax_rates = [0.15] * num_b
                raw_suffix = ", ".join(str(r) for r in tax_rates) + "]}"
                log_prob = -10.0
                print(f"  [WARNING] Rollout {r_idx}: Failed to parse, using fallback rates")

            # Run rollout: same rates each tax year
            rates_per_year = [tax_rates] * self.config.num_tax_years
            swf_trajectory, final_swf = await self.env.run_rollout(rates_per_year)

            rollouts.append({
                'system_prompt': sys_prompt,
                'user_prompt': user_prompt,
                'tax_rates': tax_rates,
                'raw_suffix': raw_suffix,
                'log_prob': log_prob,
                'final_swf': final_swf,
                'swf_trajectory': swf_trajectory,
                'format_success': format_success,
                'reward': final_swf - self.baseline_swf,
            })

        return rollouts

    def train_step(self, rollouts: List[Dict[str, Any]], iter_num: int) -> Dict[str, float]:
        """REINFORCE policy gradient step with positive-only advantages.

        Key stability features:
        - Forced JSON prefix (handled in PlannerPolicy) prevents format collapse
        - Positive-only advantages: only reinforce good actions, never push model
          away from valid outputs (which can corrupt formatting)
        - KL penalty anchors near base model
        - Per-sample gradient accumulation for memory efficiency
        """
        self.policy.model.train()

        # Filter to format-successful rollouts only
        valid = [r for r in rollouts if r['format_success']]
        if len(valid) < 2:
            print("  Skipping training step: too few valid rollouts")
            return {
                'skipped': True,
                'loss/total': 0.0,
                'loss/pg': 0.0,
                'loss/kl': 0.0,
                'loss/entropy': 0.0,
                'advantage/mean': 0.0,
                'advantage/std': 0.0,
                'grad_norm': 0.0,
                'lr': self.scheduler.get_last_lr()[0],
            }

        rewards = torch.tensor(
            [r['reward'] for r in valid], dtype=torch.float32, device=self.policy.device
        )

        # GRPO advantages: normalize within batch
        advantages = rewards - rewards.mean()
        if rewards.std() > 1e-8:
            advantages = advantages / (rewards.std() + 1e-8)

        # Positive-only advantages: only reinforce above-average actions.
        # This prevents pushing the model AWAY from specific token sequences,
        # which is what causes format collapse in standard REINFORCE.
        advantages = torch.clamp(advantages, min=0.0)

        # Use raw_suffix for exact token-level consistency with generation
        prompts_and_suffixes = [
            (r['system_prompt'], r['user_prompt'], r['raw_suffix'])
            for r in valid
        ]

        t0 = time.time()
        self.optimizer.zero_grad()

        total_pg = 0.0
        total_kl = 0.0
        total_ent = 0.0
        n = len(valid)

        # Process each sample individually: forward + backward, accumulate grads
        for i in range(n):
            pa = prompts_and_suffixes[i]
            adv = advantages[i]

            # Forward with gradients
            new_lp = self.policy.compute_log_prob_batch([pa])
            pg_i = -(new_lp[0] * adv)
            ent_i = -new_lp[0]

            kl_i = torch.tensor(0.0, device=self.policy.device)
            if self.config.kl_coef > 0:
                ref_lp = self.policy.compute_ref_log_prob_batch([pa])
                kl_i = new_lp[0] - ref_lp[0]

            loss_i = pg_i + self.config.kl_coef * kl_i - self.config.entropy_coef * ent_i
            (loss_i / n).backward()

            total_pg += pg_i.item()
            total_kl += kl_i.item()
            total_ent += ent_i.item()

        t1 = time.time()

        # Compute gradient norm before clipping
        grad_norm = 0.0
        for p in self.policy.model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        if self.config.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy.model.parameters(), self.config.clip_grad
            )

        self.optimizer.step()
        self.scheduler.step()

        print(
            f"  [TIMING] train_step={t1-t0:.1f}s ({n} samples, {(t1-t0)/n:.1f}s/sample) "
            f"grad_norm={grad_norm:.4f}",
            flush=True,
        )

        return {
            'loss/total': total_pg / n + self.config.kl_coef * total_kl / n - self.config.entropy_coef * total_ent / n,
            'loss/pg': total_pg / n,
            'loss/kl': total_kl / n,
            'loss/entropy': total_ent / n,
            'advantage/mean': advantages.mean().item(),
            'advantage/std': advantages.std().item(),
            'grad_norm': grad_norm,
            'lr': self.scheduler.get_last_lr()[0],
        }

    async def train(self):
        """Main training loop."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_dict = asdict(self.config)
        # Convert tuple to list for JSON serialization
        config_dict['lora_targets'] = list(config_dict['lora_targets'])
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Wandb init
        if self.config.use_wandb:
            wandb.init(
                project='llm-economist',
                name=f'reinforce_v2_{self.config.bracket_setting}_seed{self.config.seed}',
                config=config_dict,
                tags=['reinforce_v2', f'seed_{self.config.seed}', self.config.bracket_setting],
            )

        for iteration in range(self.config.num_iterations):
            iter_start = time.time()

            # Collect rollouts
            rollouts = await self.collect_rollouts()

            rewards = [r['reward'] for r in rollouts]
            format_rate = sum(1 for r in rollouts if r['format_success']) / len(rollouts)

            # Train step
            train_metrics = self.train_step(rollouts, iteration)

            iter_time = time.time() - iter_start

            mean_swf = float(np.mean([r['final_swf'] for r in rollouts]))
            max_swf = float(max(r['final_swf'] for r in rollouts))
            mean_reward = float(np.mean(rewards))

            # Build metrics
            metrics = {
                'iteration': iteration,
                'reward/mean': mean_reward,
                'reward/std': float(np.std(rewards)),
                'swf/mean': mean_swf,
                'swf/best_in_batch': max_swf,
                'swf/baseline': self.baseline_swf,
                'format_success_rate': format_rate,
                'time': iter_time,
                **train_metrics,
            }
            self.metrics_history.append(metrics)

            print(
                f"Iter {iteration + 1}/{self.config.num_iterations} | "
                f"Reward: {mean_reward:.3f} | SWF: {mean_swf:.4f} (baseline: {self.baseline_swf:.4f}) | "
                f"Loss: {train_metrics.get('loss/total', 0):.3f} | "
                f"Format: {format_rate * 100:.0f}% | Time: {iter_time:.1f}s"
            )

            if self.config.use_wandb:
                wandb.log(metrics, step=iteration)

            # Track best
            if mean_swf > self.best_swf:
                self.best_swf = mean_swf
                self.policy.save_lora_weights(output_dir / 'best')
                print(f"  New best SWF: {mean_swf:.4f}")

            # Checkpoint every 10 iterations
            if iteration % 10 == 0:
                self.policy.save_lora_weights(output_dir / 'checkpoints' / f'iter_{iteration}')
                # Save optimizer state
                torch.save({
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'iteration': iteration,
                    'best_swf': self.best_swf,
                    'baseline_swf': self.baseline_swf,
                }, output_dir / 'checkpoints' / f'iter_{iteration}' / 'optimizer.pt')
                with open(output_dir / 'metrics_history.json', 'w') as f:
                    json.dump(self.metrics_history, f, indent=2)

        # Final save
        self.policy.save_lora_weights(output_dir / 'final')
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'iteration': self.config.num_iterations - 1,
            'best_swf': self.best_swf,
            'baseline_swf': self.baseline_swf,
        }, output_dir / 'final' / 'optimizer.pt')
        with open(output_dir / 'metrics_history.json', 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

        if self.config.use_wandb:
            wandb.finish()

        print(f"\nTraining complete. Best SWF: {self.best_swf:.4f} (baseline: {self.baseline_swf:.4f})")

    async def shutdown(self):
        """Cleanup: shut down the vLLM engine."""
        if self.env and self.env.sim and self.env.sim.engine:
            await self.env.sim.engine.shutdown()


# =============================================================================
# MAIN / ARGPARSE
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='REINFORCE++ v2 Training with Full Simulation Rollouts',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--planner-model', type=str, default='google/gemma-3-4b-it',
                        help='HuggingFace model for trainable planner')
    parser.add_argument('--worker-model', type=str, default='Qwen/Qwen3-8B-AWQ',
                        help='HuggingFace model for frozen workers (vLLM)')
    parser.add_argument('--num-agents', type=int, default=32,
                        help='Number of worker agents')
    parser.add_argument('--bracket-setting', type=str, default='three',
                        choices=['flat', 'three', 'US_FED'],
                        help='Tax bracket configuration')
    parser.add_argument('--num-rollouts', type=int, default=16,
                        help='Rollouts (batch size) per training iteration')
    parser.add_argument('--tax-year-length', type=int, default=64,
                        help='Simulation steps per tax year')
    parser.add_argument('--num-tax-years', type=int, default=4,
                        help='Tax years per rollout')
    parser.add_argument('--num-iterations', type=int, default=500,
                        help='Total training iterations')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--kl-coef', type=float, default=0.05,
                        help='KL divergence penalty coefficient')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='Entropy bonus coefficient')
    parser.add_argument('--clip-grad', type=float, default=1.0,
                        help='Gradient clipping norm (0 to disable)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str, default='results/reinforce_v2',
                        help='Output directory for checkpoints and metrics')
    parser.add_argument('--gpu-memory-util', type=float, default=0.85,
                        help='GPU memory utilization for vLLM worker engine')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable wandb logging')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint directory')
    return parser.parse_args()


async def main():
    args = parse_args()

    config = TrainingConfig(
        planner_model=args.planner_model,
        worker_model=args.worker_model,
        num_agents=args.num_agents,
        bracket_setting=args.bracket_setting,
        num_rollouts=args.num_rollouts,
        tax_year_length=args.tax_year_length,
        num_tax_years=args.num_tax_years,
        num_iterations=args.num_iterations,
        lr=args.lr,
        kl_coef=args.kl_coef,
        entropy_coef=args.entropy_coef,
        clip_grad=args.clip_grad,
        seed=args.seed,
        output_dir=args.output_dir,
        gpu_memory_util=args.gpu_memory_util,
        use_wandb=not args.no_wandb,
    )

    print(f"\n{'=' * 60}")
    print(f"REINFORCE++ v2 Training Configuration")
    print(f"{'=' * 60}")
    print(f"Planner:         {config.planner_model}")
    print(f"Workers:         {config.worker_model}")
    print(f"Agents:          {config.num_agents}")
    print(f"Brackets:        {config.bracket_setting}")
    print(f"Rollouts/iter:   {config.num_rollouts}")
    print(f"Tax year length: {config.tax_year_length}")
    print(f"Tax years:       {config.num_tax_years}")
    print(f"Steps/rollout:   {config.tax_year_length * config.num_tax_years}")
    print(f"Iterations:      {config.num_iterations}")
    print(f"LR:              {config.lr}")
    print(f"KL coef:         {config.kl_coef}")
    print(f"Entropy coef:    {config.entropy_coef}")
    print(f"Seed:            {config.seed}")
    print(f"Output:          {config.output_dir}")
    print(f"{'=' * 60}\n")

    trainer = REINFORCETrainer(config)

    try:
        # Handle checkpoint resume
        if args.resume:
            resume_dir = Path(args.resume)
            state_file = resume_dir / 'optimizer.pt'
            if state_file.exists():
                print(f"Will resume from checkpoint: {args.resume}")

        await trainer.setup()

        # Load checkpoint weights if resuming
        if args.resume:
            resume_dir = Path(args.resume)
            # Load LoRA weights
            trainer.policy.load_lora_weights(resume_dir)
            # Load optimizer state
            opt_file = resume_dir / 'optimizer.pt'
            if opt_file.exists():
                checkpoint = torch.load(opt_file, weights_only=False)
                trainer.optimizer.load_state_dict(checkpoint['optimizer'])
                trainer.scheduler.load_state_dict(checkpoint['scheduler'])
                trainer.best_swf = checkpoint.get('best_swf', float('-inf'))
                start_iter = checkpoint.get('iteration', -1) + 1
                print(f"Resumed from iteration {start_iter}, best SWF: {trainer.best_swf:.4f}")
                # Load previous metrics
                metrics_file = Path(config.output_dir) / 'metrics_history.json'
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        trainer.metrics_history = json.load(f)

        await trainer.train()
    finally:
        await trainer.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
