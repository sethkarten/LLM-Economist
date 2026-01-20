#!/usr/bin/env python3
"""
REINFORCE++ Training with 2×GPU Setup (Optimized for B200/H200)

GPU 1: vLLM Rollout Worker (fast inference for workers)
GPU 2: Unsloth LoRA Training (fast finetuning for planner)

Architecture:
- Planner: Small LLM with LoRA (trainable) - uses unsloth for 2-4x speedup
- Workers: Same or different LLM (frozen) - uses vLLM for fast batch inference
- Budget: 48 hours max per experiment
- Goal: Maximize steps per second (SPS)

Usage:
    # Single GPU test (uses both for vLLM)
    python run_reinforce_2gpu.py --model gemma3-4b --num-agents 100 --seed 42

    # Production 2×B200 (GPU 0: vLLM, GPU 1: Training)
    CUDA_VISIBLE_DEVICES=0,1 python run_reinforce_2gpu.py \\
        --model gemma3-4b --num-agents 100 --seed 42 \\
        --num-iterations 1000 --rollouts-per-iter 32
"""

import os
# CRITICAL: Set offline mode before any HF imports
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HOME'] = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))

# Wandb offline for SLURM
os.environ['WANDB_MODE'] = os.environ.get('WANDB_MODE', 'offline')
os.environ['WANDB_DIR'] = os.environ.get('WANDB_DIR', './wandb')

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

# Unsloth for fast LoRA training
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    print("WARNING: unsloth not available, falling back to standard PEFT")
    UNSLOTH_AVAILABLE = False
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model


@dataclass
class ReinforceConfig:
    """Configuration for REINFORCE++ training."""
    # Models
    planner_model: str = "google/gemma-2-2b-it"  # Trainable planner
    worker_model: str = "google/gemma-3-4b-it"   # Fixed workers

    # Scale
    num_agents: int = 100
    tax_year_length: int = 64

    # Training
    num_iterations: int = 1000
    rollouts_per_iter: int = 16
    learning_rate: float = 1e-5
    kl_coef: float = 0.05
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0

    # LoRA (unsloth optimized)
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_4bit: bool = True  # 4-bit quantization for training

    # Checkpointing
    save_every: int = 10

    # Other
    seed: int = 42


class UnslothPlannerPolicy:
    """
    Planner policy using Unsloth for fast LoRA training.

    Unsloth optimizations:
    - 2-4x faster training than standard PEFT
    - Lower memory usage with 4-bit quantization
    - Optimized kernels for LoRA
    """

    def __init__(self, model_name: str, config: ReinforceConfig, device: str = "cuda:1"):
        self.model_name = model_name
        self.config = config
        self.device = device

        # Model will be loaded in load_model()
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model with Unsloth or fallback to standard PEFT."""
        print(f"\nLoading plannerunsloth {self.model_name}...")

        if UNSLOTH_AVAILABLE:
            # Unsloth: 2-4x faster than standard PEFT
            max_seq_length = 2048
            dtype = None  # Auto-detect
            load_in_4bit = self.config.lora_4bit

            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=f"unsloth/{self.model_name.split('/')[-1]}",  # Use unsloth versions
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
                device_map={"": torch.device(self.device)},
            )

            # Add LoRA adapters
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
                use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
                random_state=self.config.seed,
            )

            print(f"✓ Loaded with Unsloth (4-bit LoRA)")

        else:
            # Fallback to standard PEFT
            print("Using standard PEFT (slower)")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.config.lora_4bit else torch.float32,
                device_map={"": torch.device(self.device)},
            )

            # Add LoRA
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )

            self.model = get_peft_model(self.model, lora_config)
            print(f"✓ Loaded with standard PEFT")

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable params: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")

    def generate_tax_policy(self, system_prompt: str, user_prompt: str,
                           temperature: float = 0.7) -> Tuple[List[float], float]:
        """
        Generate tax policy and compute log probability.

        Returns:
            (tax_rates, log_prob)
        """
        # Format prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate with sampling
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=temperature,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response
        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute log probability
        log_prob = self._compute_log_prob(outputs.scores, generated_ids)

        # Parse tax rates from response
        tax_rates = self._parse_tax_rates(response)

        return tax_rates, log_prob

    def _compute_log_prob(self, scores: List[torch.Tensor], token_ids: torch.Tensor) -> float:
        """Compute log probability of generated sequence."""
        log_probs = []
        for score, token_id in zip(scores, token_ids):
            log_prob = F.log_softmax(score[0], dim=-1)[token_id]
            log_probs.append(log_prob.item())
        return sum(log_probs)

    def _parse_tax_rates(self, response: str) -> List[float]:
        """Parse tax rates from LLM response."""
        try:
            # Try JSON parsing first
            data = json.loads(response)
            rates = data.get('tax_rates', [0.1, 0.15, 0.22, 0.24, 0.32, 0.35, 0.37])
        except:
            # Fallback: extract numbers
            import re
            numbers = re.findall(r'0?\.\d+|\d+%', response)
            rates = []
            for num in numbers[:7]:  # Take first 7 numbers
                if '%' in num:
                    rates.append(float(num.replace('%', '')) / 100)
                else:
                    rates.append(float(num))

            if len(rates) < 7:
                # Default US federal rates
                rates = [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]

        # Clip to valid range [0, 1]
        return [max(0.0, min(1.0, r)) for r in rates[:7]]

    def compute_action_log_prob(self, system_prompt: str, user_prompt: str,
                                tax_rates: List[float]) -> float:
        """
        Recompute log probability for a specific action (for training).
        This enables gradients.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Target response
        target = json.dumps({"tax_rates": [round(r, 3) for r in tax_rates]})

        # Tokenize
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        target_ids = self.tokenizer(target, return_tensors="pt", add_special_tokens=False).to(self.device)

        # Concatenate
        input_ids = torch.cat([prompt_ids["input_ids"], target_ids["input_ids"]], dim=1)

        # Forward pass with gradients
        with torch.enable_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        # Compute log probs for target tokens
        target_start = prompt_ids["input_ids"].shape[1]
        target_logits = logits[0, target_start-1:-1, :]
        target_token_ids = input_ids[0, target_start:]

        log_probs = F.log_softmax(target_logits, dim=-1)
        selected_log_probs = log_probs[range(len(target_token_ids)), target_token_ids]

        return selected_log_probs.sum()


# TODO: Continue implementation with:
# 1. vLLM worker engine setup (async_engine.py integration)
# 2. REINFORCE++ trainer class
# 3. Training loop with policy gradients
# 4. Wandb logging
# 5. Checkpoint saving/loading
# 6. Main entry point with argument parsing

print("✓ REINFORCE++ 2×GPU script template created")
print("This is a starting point - needs completion with:")
print("1. vLLM worker engine")
print("2. REINFORCE++ training logic")
print("3. Full integration")
