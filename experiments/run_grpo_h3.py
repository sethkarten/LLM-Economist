#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) H3 Experiment

Replaces REINFORCE++ for planner training. Key improvements:
- Samples G completions per prompt, ranks within group (cancels prompt-level noise)
- PPO-style clipping bounds policy change per step (prevents JSON format collapse)
- KL penalty against frozen reference model (reverse KL)
- Fused worker batching: all G*num_agents worker prompts in a single vLLM call per timestep

Why GRPO works where REINFORCE++ failed:
- REINFORCE++ takes a single PG step that can destroy JSON formatting
- GRPO's clipping (epsilon=0.2) ensures the policy ratio stays in [0.8, 1.2]
- Group-relative advantages cancel prompt-level noise, giving cleaner gradients
- Format gating: format failures get reward=-1.0, dominating the advantage signal

Usage:
    # Test on single A6000
    python experiments/run_grpo_h3.py --gpu a6000 --seed 42 --num-iterations 3

    # Full experiment on 2x A6000
    python experiments/run_grpo_h3.py --gpu a6000_2gpu --seed 42

    # Production on B200
    python experiments/run_grpo_h3.py --gpu b200 --seed 42 --num-iterations 100
"""

# Use vLLM legacy API to avoid V1 compilation issues
import os
os.environ['VLLM_USE_V1'] = '0'
# CRITICAL: Disable PyTorch compilation to avoid 10+ min hang
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
# Only enable offline mode on SLURM (compute nodes have no internet)
# SSH resources like Cynthia/Pikachu have internet access and can download models
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
import time
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import wandb

# =============================================================================
# PROMPT CONFIGURATIONS (same as REINFORCE++ H3)
# =============================================================================

H3_SYSTEM_PROMPT = """You are an AI tax policy planner optimizing social welfare in an economic simulation.

Your goal: Set tax rates that achieve HIGHER social welfare than the US federal progressive tax baseline.

Baseline Performance (US 2024 Federal Tax):
- Social Welfare: {baseline_swf:.1f}
- Gini: {baseline_gini:.3f}
- Labor: {baseline_labor:.1f} hours/week

You MUST respond with ONLY a JSON object. No explanation, no strategy, no other text.
Required format: {{"tax_rates": [r1, r2, r3, r4, r5, r6, r7]}}

The tax_rates array must have exactly 7 values (0.0-0.99) for the 7 US tax brackets.
Example: {{"tax_rates": [0.08, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]}}"""

H3_USER_TEMPLATE = """## Current Economic State

**Income Distribution:**
- Mean Income: ${mean_income:,.0f}
- Gini: {gini:.3f}

**US Federal Baseline to Beat:**
- SWF: {baseline_swf:.1f}
- Gini: {baseline_gini:.3f}
- Labor: {baseline_labor:.1f} hours/week

Respond with ONLY the JSON object.
Required format: {{"tax_rates": [r1, r2, r3, r4, r5, r6, r7]}}"""


# =============================================================================
# GRPO CONFIG (extends RLConfig with group-relative parameters)
# =============================================================================

@dataclass
class GRPOConfig:
    """Configuration for GRPO H3 training."""
    experiment: str = "h3_grpo"
    planner_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    worker_model: str = "gemma3-4b"

    # Environment
    num_agents: int = 100
    tax_year_length: int = 16  # Sufficient for single-decision H3

    # GRPO-specific
    group_size: int = 8           # G: completions per prompt
    epsilon: float = 0.2         # PPO clip range
    num_groups_per_iter: int = 4  # Number of prompt groups per iteration
    grpo_beta: float = 0.05     # KL penalty coefficient (reverse KL)

    # Training
    num_iterations: int = 100
    learning_rate: float = 5e-7   # More conservative than REINFORCE++ (was 1e-6)
    max_grad_norm: float = 0.5
    temperature: float = 0.8     # Sampling temperature for diverse completions

    # LoRA
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.05

    # GPU optimization
    gpu_memory_utilization: float = 0.20  # Share GPU with planner
    enable_prefix_caching: bool = True

    # Checkpointing
    save_every: int = 10
    seed: int = 42

    # Format safety
    format_lr_halve_threshold: float = 0.5  # Halve LR if format_success_rate < this

    def to_dict(self):
        return asdict(self)

    @classmethod
    def for_gpu(cls, gpu_type: str = "a6000", **kwargs):
        """Create config optimized for specific GPU type."""
        if gpu_type.lower() == "b200":
            return cls(
                num_groups_per_iter=8,
                gpu_memory_utilization=0.75,
                **kwargs
            )
        elif gpu_type.lower() == "b200_2gpu":
            defaults_2gpu = {
                "num_groups_per_iter": 16,
                "gpu_memory_utilization": 0.90,
                "num_agents": 1000,
                "num_iterations": 100,
            }
            merged = {**defaults_2gpu, **kwargs}
            return cls(**merged)
        elif gpu_type.lower() == "a6000_2gpu":
            defaults_2gpu = {
                "num_groups_per_iter": 8,
                "gpu_memory_utilization": 0.85,
                "num_agents": 1000,
                "num_iterations": 100,
            }
            merged = {**defaults_2gpu, **kwargs}
            return cls(**merged)
        else:  # A6000 1-GPU or default
            return cls(
                num_groups_per_iter=4,
                gpu_memory_utilization=0.30,
                **kwargs
            )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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


def compute_reward(final_swf: float, baseline_swf: float, format_success: bool = True) -> float:
    """
    Compute reward for H3 with format gating.

    Format gating: format failure = -1.0 (hard gate, dominates advantage signal).
    Otherwise: scaled SWF improvement in roughly [0, 1] range.
    """
    if not format_success:
        return -1.0  # Hard format gate

    raw_improvement = final_swf - baseline_swf
    expected_range = 40.0  # SWF improvement we're aiming for
    scaled_reward = raw_improvement / expected_range
    # Small format bonus for valid JSON
    return scaled_reward + 0.1


def parse_tax_rates(response: str) -> Optional[List[float]]:
    """Parse tax rates from model response.

    Handles Qwen3 thinking mode by stripping <think>...</think> tags
    and extracting JSON from anywhere in the response.
    """
    try:
        # Strip Qwen3 thinking tags if present
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        # Also handle case where thinking tag is not closed (model ran out of tokens)
        cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL).strip()

        # Try to find JSON in cleaned response
        text_to_parse = cleaned if cleaned else response
        if "{" in text_to_parse and "}" in text_to_parse:
            start = text_to_parse.index("{")
            end = text_to_parse.rindex("}") + 1
            json_str = text_to_parse[start:end]
            data = json.loads(json_str)

            if "tax_rates" in data:
                rates = [float(r) for r in data["tax_rates"]]
                rates = [max(0.0, min(0.99, r)) for r in rates]
                return rates

        # Fallback: try to extract 7 floats from the response
        numbers = re.findall(r'0\.\d+', text_to_parse)
        if len(numbers) >= 7:
            rates = [max(0.0, min(0.99, float(n))) for n in numbers[:7]]
            return rates
    except Exception:
        pass
    return None


# =============================================================================
# PLANNER POLICY (same as REINFORCE++ but with sample_action_group)
# =============================================================================

class PlannerPolicy:
    """
    Trainable planner policy with LoRA adapters.

    Uses torch-based generation to get log probabilities for training.
    Extended with sample_action_group() for GRPO.
    """

    def __init__(self, model_name: str, config: GRPOConfig, device: str = "cuda", force_single_gpu: bool = False):
        self.model_name = model_name
        self.config = config
        self.device = device
        self.force_single_gpu = force_single_gpu

        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load planner model with LoRA adapters for training."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        print(f"Loading trainable planner: {self.model_name}")

        model_name_or_path = self.model_name
        print(f"Loading from: {model_name_or_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device_map = {"": 0} if self.force_single_gpu else "auto"
        print(f"Loading planner with device_map={device_map} (force_single_gpu={self.force_single_gpu})", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )

        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing for memory efficiency")

        if self.config.use_lora:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["gate_proj", "up_proj", "down_proj"],
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
        temperature: float = 0.8,
        max_new_tokens: int = 512,
    ) -> Tuple[Optional[List[float]], float, str]:
        """
        Sample a single tax policy action and compute log probability.

        Returns:
            Tuple of (tax_rates, log_prob, raw_response)
        """
        # For Qwen3 models: append /no_think to disable thinking mode
        # This prevents the model from spending all tokens on <think> tags
        effective_user_prompt = user_prompt
        if "qwen3" in self.model_name.lower() or "Qwen3" in self.model_name:
            effective_user_prompt = user_prompt + " /no_think"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": effective_user_prompt},
        ]

        full_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)

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

        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        log_prob = self._compute_log_prob(outputs.scores, generated_ids)
        tax_rates = parse_tax_rates(response)

        # Debug: log first few responses to diagnose format issues
        if not hasattr(self, '_sample_count'):
            self._sample_count = 0
        self._sample_count += 1
        if self._sample_count <= 3 or (tax_rates is None and self._sample_count <= 10):
            print(f"  [DEBUG sample {self._sample_count}] raw response: {response[:200]}", flush=True)
            print(f"  [DEBUG sample {self._sample_count}] parsed: {tax_rates is not None}", flush=True)

        return tax_rates, log_prob, response

    def sample_action_group(
        self,
        system_prompt: str,
        user_prompt: str,
        group_size: int = 8,
        temperature: float = 0.8,
    ) -> List[Tuple[Optional[List[float]], float, str]]:
        """
        Sample G diverse completions for the same prompt.

        Returns:
            List of (tax_rates, log_prob, raw_response) tuples, length G.
        """
        results = []
        for g in range(group_size):
            tax_rates, log_prob, response = self.sample_action(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
            )
            results.append((tax_rates, log_prob, response))
        return results

    def _compute_log_prob_batch_internal(
        self,
        prompts_and_actions: List[Tuple[str, str, List[float]]],
        enable_grad: bool = True,
    ) -> torch.Tensor:
        """
        Compute log probabilities for a batch of (system_prompt, user_prompt, tax_rates).

        Processes one sample at a time to avoid OOM on shared GPU.
        """
        all_log_probs = []
        grad_context = torch.enable_grad() if enable_grad else torch.no_grad()

        for system_prompt, user_prompt, tax_rates in prompts_and_actions:
            # Apply /no_think for Qwen3 models (consistent with sample_action)
            effective_user_prompt = user_prompt
            if "qwen3" in self.model_name.lower() or "Qwen3" in self.model_name:
                effective_user_prompt = user_prompt + " /no_think"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": effective_user_prompt},
            ]

            full_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            expected_response = json.dumps({"tax_rates": [round(r, 3) for r in tax_rates]})

            prompt_ids = self.tokenizer(full_prompt, return_tensors="pt")["input_ids"]
            response_ids = self.tokenizer(expected_response, return_tensors="pt", add_special_tokens=False)["input_ids"]

            input_ids = torch.cat([prompt_ids, response_ids], dim=1).to(self.device)

            with grad_context:
                outputs = self.model(input_ids)
                logits = outputs.logits

            response_start = prompt_ids.shape[1]
            response_logits = logits[0, response_start - 1:-1, :]
            response_token_ids = input_ids[0, response_start:]

            log_probs = F.log_softmax(response_logits, dim=-1)
            selected_log_probs = log_probs[range(len(response_token_ids)), response_token_ids]

            all_log_probs.append(selected_log_probs.mean())

        return torch.stack(all_log_probs)

    def compute_log_prob_batch(
        self,
        prompts_and_actions: List[Tuple[str, str, List[float]]],
    ) -> torch.Tensor:
        """Compute log probabilities with gradients enabled (for training)."""
        return self._compute_log_prob_batch_internal(prompts_and_actions, enable_grad=True)

    def compute_ref_log_prob_batch(
        self,
        prompts_and_actions: List[Tuple[str, str, List[float]]],
    ) -> torch.Tensor:
        """Compute log probs from frozen base model (LoRA disabled) for KL penalty."""
        self.model.disable_adapter_layers()
        with torch.no_grad():
            ref_log_probs = self._compute_log_prob_batch_internal(prompts_and_actions, enable_grad=False)
        self.model.enable_adapter_layers()
        return ref_log_probs

    def _compute_log_prob(self, scores: Tuple[torch.Tensor, ...], generated_ids: torch.Tensor) -> float:
        """Compute mean log probability of generated sequence."""
        total_log_prob = 0.0
        count = 0

        for i, (score, token_id) in enumerate(zip(scores, generated_ids)):
            if token_id == self.tokenizer.pad_token_id or token_id == self.tokenizer.eos_token_id:
                break
            log_probs = F.log_softmax(score[0], dim=-1)
            total_log_prob += log_probs[token_id].item()
            count += 1

        return total_log_prob / max(count, 1)

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


# =============================================================================
# GRPO EXPERIMENT
# =============================================================================

class GRPOExperiment:
    """
    GRPO training for H3 experiment.

    Core loop:
    1. Fix an economic state (skills, personas, seed)
    2. Planner samples G tax policies (group)
    3. Run G parallel worker simulations with FUSED batches
    4. Compute group-relative advantages
    5. PPO-clip policy gradient update with KL penalty
    """

    def __init__(self, config: GRPOConfig, output_dir: str):
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
        self.planner_policy = None
        self.worker_engine = None

        # Worker engine config (for restart in 1-GPU mode)
        self._worker_engine_config = None
        self._use_2gpu_mode = False

        # Training
        self.optimizer = None
        self.scheduler = None

        # Format-aware LR tracking
        self._lr_halved = False

    async def setup(self, skip_baseline=False):
        """Initialize planner policy and worker engine, compute baseline."""
        from llm_economist.inference.async_engine import ScalableInferenceEngine
        from llm_economist.inference.config import get_model_config

        print(f"\n{'='*60}")
        print(f"Setting up GRPO H3 Experiment")
        print(f"{'='*60}")
        print(f"Planner: {self.config.planner_model}")
        print(f"Workers: {self.config.worker_model}")
        print(f"Agents: {self.config.num_agents}")
        print(f"Group size (G): {self.config.group_size}")
        print(f"Groups/iter: {self.config.num_groups_per_iter}")
        print(f"Epsilon (clip): {self.config.epsilon}")
        print(f"Beta (KL): {self.config.grpo_beta}")
        print(f"{'='*60}\n")

        # Detect 2-GPU mode BEFORE loading planner
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        print(f"[DEBUG] CUDA_VISIBLE_DEVICES = '{cuda_visible}'", flush=True)
        visible_gpus = [int(g.strip()) for g in cuda_visible.split(",") if g.strip().isdigit()] if cuda_visible else []
        self._use_2gpu_mode = len(visible_gpus) >= 2
        print(f"[DEBUG] visible_gpus = {visible_gpus}, 2-GPU mode = {self._use_2gpu_mode}", flush=True)

        # Load trainable planner policy
        self.planner_policy = PlannerPolicy(
            model_name=self.config.planner_model,
            config=self.config,
            device="cuda:0" if self._use_2gpu_mode else ("cuda" if torch.cuda.is_available() else "cpu"),
            force_single_gpu=self._use_2gpu_mode,
        )
        self.planner_policy.load_model()

        # Setup optimizer
        self.setup_optimizer()

        # Load vLLM engine for workers
        model_config = get_model_config(self.config.worker_model)
        quant_str = None

        # In offline mode, resolve cached model path
        worker_model_path = model_config.hf_name
        if os.environ.get('HF_HUB_OFFLINE') == '1':
            hf_cache = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
            model_dir_name = f"models--{model_config.hf_name.replace('/', '--')}"

            possible_paths = [
                os.path.join(hf_cache, model_dir_name),
                os.path.join(hf_cache, 'hub', model_dir_name),
                os.path.join('/data1/milkkarten/.cache/huggingface', model_dir_name),
            ]

            for model_cache_dir in possible_paths:
                if os.path.exists(model_cache_dir):
                    snapshots_dir = os.path.join(model_cache_dir, 'snapshots')
                    if os.path.exists(snapshots_dir):
                        snapshots = os.listdir(snapshots_dir)
                        if snapshots:
                            worker_model_path = os.path.join(snapshots_dir, snapshots[0])
                            print(f"Offline mode: using cached model at {worker_model_path}")
                            break
            else:
                print(f"WARNING: Could not find cached model for {model_config.hf_name}")

        print(f"\nLoading worker engine with {quant_str or 'no'} quantization...", flush=True)

        if self._use_2gpu_mode:
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            visible_gpus = [int(g.strip()) for g in cuda_visible.split(",") if g.strip().isdigit()]
            worker_gpu_id = visible_gpus[1]
            print(f"2-GPU mode: planner on GPU {visible_gpus[0]}, worker server on GPU {worker_gpu_id}", flush=True)

            from llm_economist.inference.vllm_server_engine import VLLMServerEngine
            self.worker_engine = VLLMServerEngine(
                model_name=worker_model_path,
                gpu_id=worker_gpu_id,
                port=8100,
                gpu_memory_utilization=0.85,
                max_model_len=4096,
                quantization=quant_str,
            )
        else:
            worker_gpu_mem = self.config.gpu_memory_utilization
            print(f"1-GPU mode: worker and planner sharing cuda:0")
            print(f"  Worker GPU memory: {worker_gpu_mem} (shared with planner)")

            self._worker_engine_config = {
                'model_name': worker_model_path,
                'quantization': quant_str,
                'tensor_parallel_size': 1,
                'gpu_memory_utilization': worker_gpu_mem,
                'max_model_len': 4096,
                'text_only_mode': model_config.text_only_mode,
                'enforce_eager': True,
                'enable_prefix_caching': self.config.enable_prefix_caching,
                'enable_chunked_prefill': False,
            }

            self.worker_engine = ScalableInferenceEngine(**self._worker_engine_config)

        # Start vLLM server if using 2-GPU mode
        if self._use_2gpu_mode:
            print("Starting vLLM server (this may take 1-2 minutes)...", flush=True)
            await self.worker_engine.start(timeout=180)

        print("Worker engine loaded.\n", flush=True)

        # Compute baseline metrics
        if not skip_baseline:
            print("Computing baseline (US federal progressive tax 2024)...", flush=True)
            self.baseline_metrics = await self._compute_baseline()
            print(f"Baseline SWF: {self.baseline_metrics['swf']:.2f}", flush=True)
            print(f"Baseline Gini: {self.baseline_metrics['gini']:.3f}")
            print(f"Baseline Labor: {self.baseline_metrics['mean_labor']:.1f} hours/week\n")
        else:
            print("Skipping baseline computation (will load from checkpoint)\n")

    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler with 10% linear warmup."""
        self.optimizer = AdamW(
            self.planner_policy.model.parameters(),
            lr=self.config.learning_rate,
        )

        warmup_steps = max(1, int(0.1 * self.config.num_iterations))

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, self.config.num_iterations - warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        print(f"Optimizer: AdamW (lr={self.config.learning_rate})")
        print(f"Scheduler: Linear warmup ({warmup_steps} steps) + Cosine decay\n")

    def setup_wandb(self, offline: bool = False):
        """Initialize wandb logging."""
        if offline:
            os.environ['WANDB_MODE'] = 'offline'
            print("[WANDB] Running in offline mode")

        wandb.init(
            project='llm-economist',
            name=f'h3_grpo_{self.config.planner_model.split("/")[-1]}_seed{self.config.seed}',
            config=asdict(self.config),
            tags=['h3', 'grpo', f'seed_{self.config.seed}', f'agents_{self.config.num_agents}'],
            notes=f"""
GRPO H3 Training
- Planner: {self.config.planner_model} (LoRA finetuned)
- Workers: {self.config.worker_model} (frozen)
- Baseline SWF: {self.baseline_metrics['swf']:.2f}
- Group size: {self.config.group_size}
- Epsilon (clip): {self.config.epsilon}
- Beta (KL): {self.config.grpo_beta}
- Goal: Beat US Federal Tax baseline
""",
        )

        wandb.config.update({
            'baseline_swf': self.baseline_metrics['swf'],
            'baseline_gini': self.baseline_metrics['gini'],
            'baseline_labor': self.baseline_metrics['mean_labor'],
        })

        mode_str = "offline" if offline else "online"
        print(f"Wandb initialized ({mode_str} mode)\n")

    async def _stop_worker_engine(self):
        """Prepare for training in 1-GPU mode."""
        if not self._use_2gpu_mode:
            print("[1-GPU] Freeing GPU memory for training...", flush=True)
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print("[1-GPU] Ready for training", flush=True)

    async def _start_worker_engine(self):
        """Prepare for rollouts in 1-GPU mode."""
        if not self._use_2gpu_mode:
            print("[1-GPU] Ready for next rollout collection", flush=True)

    # =========================================================================
    # BASELINE COMPUTATION
    # =========================================================================

    async def _compute_baseline(self) -> Dict[str, float]:
        """Compute baseline metrics using US federal progressive tax (2024)."""
        from llm_economist.agents.persona_generator import generate_aligned_personas
        from llm_economist.inference.async_engine import BatchRequest

        np.random.seed(self.config.seed)
        skills = np.exp(np.random.randn(self.config.num_agents) * 0.5 + 3.5)
        labor = np.ones(self.config.num_agents) * 40

        us_brackets = [11000, 44725, 95375, 182100, 231250, 578125]
        us_rates = [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]

        personas = generate_aligned_personas(n=self.config.num_agents, seed=self.config.seed)
        persona_list = list(personas.values())

        for step in range(self.config.tax_year_length):
            print(f"[BASELINE] Step {step}/{self.config.tax_year_length}")
            incomes = skills * labor

            prompts = []
            for i, persona in enumerate(persona_list):
                prompt = f"""{persona}

Your skill level: {skills[i]:.1f}
Current income: ${incomes[i]:,.0f}
Tax rates: US federal progressive (10%-37%)

Hours to work this week (0-100)? Number only:"""
                prompts.append(prompt)

            batch = BatchRequest(
                request_ids=[f"baseline_step{step}_w_{i}" for i in range(len(prompts))],
                prompts=prompts,
                system_prompts=["You are a worker deciding hours to work."] * len(prompts),
                temperatures=[0.7] * len(prompts),
                max_tokens=10,
            )

            response = await self.worker_engine.generate_batch(batch)
            print(f"[BASELINE] Step {step} - received {len(response.responses)} responses")

            for i, resp in enumerate(response.responses):
                try:
                    numbers = re.findall(r'\d+\.?\d*', resp)
                    if numbers:
                        labor[i] = min(100, max(0, float(numbers[0])))
                except Exception:
                    pass
            print(f"[BASELINE] Step {step} - parsed labor choices, mean={labor.mean():.1f}")

        final_incomes = skills * labor
        baseline_swf = self._compute_swf(final_incomes, us_rates, us_brackets)
        baseline_gini = self._compute_gini(final_incomes)
        baseline_labor = labor.mean()

        return {
            "swf": baseline_swf,
            "gini": baseline_gini,
            "mean_labor": baseline_labor,
        }

    # =========================================================================
    # GRPO CORE: COLLECT GROUP ROLLOUTS WITH FUSED BATCHING
    # =========================================================================

    async def collect_group_rollouts_optimized(
        self,
        group_id: int,
    ) -> List[Dict[str, Any]]:
        """
        Collect G rollouts for ONE fixed economic state with FUSED worker batching.

        Algorithm:
        1. Fix one economic state (skills, personas, seed)
        2. Sample G tax policies from planner
        3. For each timestep, fuse ALL G*num_agents worker prompts into a single vLLM call
        4. Each simulation uses SAME workers, only tax policy differs
        5. Compute G rewards with format gating

        Returns:
            List of G rollout dicts, each containing:
            - tax_rates, log_prob, reward, format_success,
            - system_prompt, user_prompt, final_swf
        """
        from llm_economist.agents.persona_generator import generate_aligned_personas
        from llm_economist.inference.async_engine import BatchRequest

        G = self.config.group_size
        N = self.config.num_agents

        # Fix ONE economic state for the entire group
        group_seed = self.config.seed + group_id + self.iteration * 10000
        np.random.seed(group_seed)

        skills = np.exp(np.random.randn(N) * 0.5 + 3.5)
        brackets = [11000, 44725, 95375, 182100, 231250, 578125]

        # Generate personas (same for all G simulations)
        personas = generate_aligned_personas(n=N, seed=group_seed)
        persona_list = list(personas.values())

        # Initial state for planner
        initial_labor = np.ones(N) * 40
        initial_incomes = skills * initial_labor
        state = {
            "mean_income": float(initial_incomes.mean()),
            "gini": float(self._compute_gini(initial_incomes)),
        }

        system_prompt = format_h3_system_prompt(self.baseline_metrics)
        user_prompt = format_h3_user_prompt(state, self.baseline_metrics)

        # Step 2: Sample G diverse tax policies from planner
        print(f"  [Group {group_id}] Sampling {G} planner completions...", flush=True)
        group_samples = self.planner_policy.sample_action_group(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            group_size=G,
            temperature=self.config.temperature,
        )

        # Parse results: list of (tax_rates, log_prob, raw_response)
        group_tax_rates = []
        group_log_probs = []
        group_format_success = []
        group_raw_responses = []

        for tax_rates, log_prob, raw_resp in group_samples:
            format_ok = tax_rates is not None
            group_format_success.append(format_ok)
            group_raw_responses.append(raw_resp)

            if not format_ok:
                # Use US federal rates as fallback (but mark format failure)
                tax_rates = [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]
                log_prob = -10.0
                print(f"  [Group {group_id}] WARNING: Format failure in sample, using fallback", flush=True)

            group_tax_rates.append(tax_rates)
            group_log_probs.append(log_prob)

        # Step 3: Run G parallel simulations with FUSED worker batches
        # Each simulation: same skills & personas, different tax_rates
        # We maintain G separate labor arrays
        group_labor = [np.ones(N) * 40 for _ in range(G)]

        for step in range(self.config.tax_year_length):
            # Build FUSED batch: G * N prompts in a single vLLM call
            all_prompts = []
            all_request_ids = []

            for g in range(G):
                incomes_g = skills * group_labor[g]
                tax_rates_g = group_tax_rates[g]

                for i, persona in enumerate(persona_list):
                    prompt = f"""{persona}

Your skill level: {skills[i]:.1f}
Current income: ${incomes_g[i]:,.0f}
Tax rates: {tax_rates_g[0]*100:.0f}%-{tax_rates_g[-1]*100:.0f}%

Hours to work this week (0-100)? Number only:"""
                    all_prompts.append(prompt)
                    all_request_ids.append(f"grp{group_id}_g{g}_s{step}_w{i}")

            # Single fused vLLM call for all G*N prompts
            fused_batch = BatchRequest(
                request_ids=all_request_ids,
                prompts=all_prompts,
                system_prompts=["You are a worker deciding hours to work."] * len(all_prompts),
                temperatures=[0.7] * len(all_prompts),
                max_tokens=10,
            )

            response = await self.worker_engine.generate_batch(fused_batch)

            # Parse responses back into G separate labor arrays
            resp_idx = 0
            for g in range(G):
                for i in range(N):
                    try:
                        numbers = re.findall(r'\d+\.?\d*', response.responses[resp_idx])
                        if numbers:
                            group_labor[g][i] = min(100, max(0, float(numbers[0])))
                    except Exception:
                        pass
                    resp_idx += 1

            if step % 4 == 0:
                mean_labors = [group_labor[g].mean() for g in range(G)]
                print(f"  [Group {group_id}] Step {step}/{self.config.tax_year_length} "
                      f"- mean labor across G: {np.mean(mean_labors):.1f}", flush=True)

        # Step 4: Compute G rewards
        rollouts = []
        for g in range(G):
            final_incomes_g = skills * group_labor[g]
            final_swf_g = self._compute_swf(final_incomes_g, group_tax_rates[g], brackets)
            reward_g = compute_reward(
                final_swf_g,
                self.baseline_metrics["swf"],
                format_success=group_format_success[g],
            )

            rollouts.append({
                "tax_rates": group_tax_rates[g],
                "log_prob": group_log_probs[g],
                "reward": reward_g,
                "format_success": group_format_success[g],
                "final_swf": final_swf_g,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "raw_response": group_raw_responses[g],
            })

        rewards = [r["reward"] for r in rollouts]
        format_rate = sum(group_format_success) / G
        print(f"  [Group {group_id}] Rewards: mean={np.mean(rewards):.3f}, "
              f"std={np.std(rewards):.3f}, format={format_rate*100:.0f}%", flush=True)

        return rollouts

    # =========================================================================
    # GRPO TRAINING STEP
    # =========================================================================

    def grpo_train_step(
        self,
        all_group_rollouts: List[List[Dict[str, Any]]],
        iter_num: int,
    ) -> Dict[str, float]:
        """
        GRPO policy gradient update.

        For each group:
        1. Compute group-relative advantages: A_i = (r_i - mean(r)) / (std(r) + 1e-4)
        2. Recompute new log probs (with gradients)
        3. Compute ratio = exp(new_lp - old_lp)
        4. Clipped surrogate loss: -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
        5. Reverse KL penalty if beta > 0

        Args:
            all_group_rollouts: List of groups, each group is a list of G rollout dicts
            iter_num: Current iteration

        Returns:
            Dictionary of training metrics
        """
        self.planner_policy.model.train()
        device = self.planner_policy.device
        epsilon = self.config.epsilon
        beta = self.config.grpo_beta

        # Flatten all valid rollouts and compute per-group advantages
        flat_prompts_actions = []   # (system_prompt, user_prompt, tax_rates)
        flat_old_log_probs = []     # old log probs (detached)
        flat_advantages = []         # group-relative advantages

        total_format_success = 0
        total_completions = 0

        for group_rollouts in all_group_rollouts:
            total_completions += len(group_rollouts)

            # Compute group-relative advantages
            group_rewards = torch.tensor(
                [r["reward"] for r in group_rollouts],
                dtype=torch.float32,
                device=device,
            )
            reward_mean = group_rewards.mean()
            reward_std = group_rewards.std()
            group_advantages = (group_rewards - reward_mean) / (reward_std + 1e-4)

            for i, rollout in enumerate(group_rollouts):
                if not rollout["format_success"]:
                    # Skip format failures from training (action is fabricated)
                    total_format_success += 0
                    continue

                total_format_success += 1
                flat_prompts_actions.append(
                    (rollout["system_prompt"], rollout["user_prompt"], rollout["tax_rates"])
                )
                flat_old_log_probs.append(rollout["log_prob"])
                flat_advantages.append(group_advantages[i].item())

        if len(flat_prompts_actions) < 2:
            print("  Skipping training step: too few valid completions across groups", flush=True)
            return {
                "skipped": True,
                "loss/total": 0.0,
                "loss/pg_clipped": 0.0,
                "loss/kl": 0.0,
                "advantage/mean": 0.0,
                "advantage/std": 0.0,
                "ratio/mean": 0.0,
                "ratio/clipped_frac": 0.0,
                "group_reward_var": 0.0,
                "lr": self.scheduler.get_last_lr()[0],
            }

        format_success_rate = total_format_success / max(total_completions, 1)

        # Format-aware safety: halve LR if format success is collapsing
        if format_success_rate < self.config.format_lr_halve_threshold and not self._lr_halved:
            print(f"  FORMAT SAFETY: format_success_rate={format_success_rate:.2f} < {self.config.format_lr_halve_threshold}")
            print(f"  Halving learning rate from {self.config.learning_rate:.2e} to {self.config.learning_rate/2:.2e}")
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 2
            self._lr_halved = True

        old_log_probs_tensor = torch.tensor(flat_old_log_probs, dtype=torch.float32, device=device)
        advantages_tensor = torch.tensor(flat_advantages, dtype=torch.float32, device=device)

        # Recompute log probs with gradients
        new_log_probs = self.planner_policy.compute_log_prob_batch(flat_prompts_actions)

        # PPO-style clipped surrogate loss
        log_ratio = new_log_probs - old_log_probs_tensor
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)

        surr1 = ratio * advantages_tensor
        surr2 = clipped_ratio * advantages_tensor
        pg_loss = -torch.min(surr1, surr2).mean()

        # Compute clipping statistics
        clipped_frac = ((ratio - 1.0).abs() > epsilon).float().mean().item()

        # Reverse KL penalty against frozen base model
        kl_loss = torch.tensor(0.0, device=device)
        if beta > 0:
            ref_log_probs = self.planner_policy.compute_ref_log_prob_batch(flat_prompts_actions)
            # Reverse KL: KL(ref || policy) = E_ref[log(ref/policy)]
            # = E_policy[exp(ref_lp - new_lp) - (ref_lp - new_lp) - 1]
            kl_divergence = torch.exp(ref_log_probs - new_log_probs) - (ref_log_probs - new_log_probs) - 1
            kl_loss = kl_divergence.mean()

        total_loss = pg_loss + beta * kl_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            grad_norm_before = torch.nn.utils.clip_grad_norm_(
                self.planner_policy.model.parameters(), float('inf')
            )
            print(f"  Grad norm before clipping: {grad_norm_before:.4f}")
            torch.nn.utils.clip_grad_norm_(
                self.planner_policy.model.parameters(),
                self.config.max_grad_norm,
            )

        self.optimizer.step()
        self.scheduler.step()

        # Compute group reward variance (diversity collapse detection)
        all_reward_vars = []
        for group_rollouts in all_group_rollouts:
            group_rewards = [r["reward"] for r in group_rollouts]
            all_reward_vars.append(np.var(group_rewards))
        mean_group_reward_var = float(np.mean(all_reward_vars))

        return {
            "loss/total": total_loss.item(),
            "loss/pg_clipped": pg_loss.item(),
            "loss/kl": kl_loss.item(),
            "advantage/mean": advantages_tensor.mean().item(),
            "advantage/std": advantages_tensor.std().item(),
            "ratio/mean": ratio.mean().item(),
            "ratio/clipped_frac": clipped_frac,
            "group_reward_var": mean_group_reward_var,
            "lr": self.scheduler.get_last_lr()[0],
        }

    # =========================================================================
    # MAIN TRAINING LOOP
    # =========================================================================

    async def train(self):
        """Main GRPO training loop."""
        G = self.config.group_size
        num_groups = self.config.num_groups_per_iter

        print(f"\n{'='*60}")
        print(f"Starting GRPO Training (H3)")
        print(f"{'='*60}")
        print(f"Iterations: {self.config.num_iterations}")
        print(f"Groups/iter: {num_groups}")
        print(f"Group size (G): {G}")
        print(f"Total completions/iter: {num_groups * G}")
        print(f"{'='*60}\n")

        start_iter = self.iteration
        for iteration in range(start_iter, self.config.num_iterations):
            self.iteration = iteration
            iter_start = time.time()

            print(f"\nIteration {iteration+1}/{self.config.num_iterations}")

            # Collect group rollouts
            # Each group: fix one state, sample G policies, run G sims with fused batches
            all_group_rollouts = []
            for grp_idx in range(num_groups):
                group_rollouts = await self.collect_group_rollouts_optimized(
                    group_id=grp_idx,
                )
                all_group_rollouts.append(group_rollouts)

            # Aggregate metrics across all groups
            all_rewards = []
            all_format_success = []
            for group_rollouts in all_group_rollouts:
                for r in group_rollouts:
                    all_rewards.append(r["reward"])
                    all_format_success.append(r["format_success"])

            rewards_arr = np.array(all_rewards)
            format_success_rate = sum(all_format_success) / len(all_format_success)

            # Free worker engine memory before training (1-GPU mode)
            await self._stop_worker_engine()

            # GRPO training step
            train_metrics = self.grpo_train_step(all_group_rollouts, iteration)

            iter_time = time.time() - iter_start

            metrics = {
                "iteration": iteration,
                "reward_mean": float(rewards_arr.mean()),
                "reward_std": float(rewards_arr.std()),
                "reward_max": float(rewards_arr.max()),
                "reward_min": float(rewards_arr.min()),
                "format_success_rate": format_success_rate,
                "time": iter_time,
                **train_metrics,
            }
            self.metrics_history.append(metrics)

            print(f"  Reward: {metrics['reward_mean']:.3f} +/- {metrics['reward_std']:.3f}")
            print(f"  Format Success: {format_success_rate*100:.1f}%")
            print(f"  Loss: {metrics['loss/total']:.4f} (PG-clip: {metrics['loss/pg_clipped']:.4f}, KL: {metrics['loss/kl']:.4f})")
            print(f"  Ratio: mean={metrics['ratio/mean']:.3f}, clipped={metrics['ratio/clipped_frac']*100:.1f}%")
            print(f"  Group reward var: {metrics['group_reward_var']:.4f}")
            print(f"  LR: {metrics['lr']:.2e}")
            print(f"  Time: {iter_time:.1f}s")

            # Log to wandb
            actual_swf = self.baseline_metrics['swf'] + metrics['reward_mean'] * 40.0  # Undo scaling
            swf_improvement_pct = (metrics['reward_mean'] * 40.0 / max(abs(self.baseline_metrics['swf']), 1e-8)) * 100

            wandb.log({
                # Social Welfare
                'swf/actual': actual_swf,
                'swf/baseline': self.baseline_metrics['swf'],
                'swf/improvement_pct': swf_improvement_pct,
                'swf/best': self.baseline_metrics['swf'] + self.best_reward * 40.0,

                # Rewards
                'reward/mean': metrics['reward_mean'],
                'reward/std': metrics['reward_std'],
                'reward/max': metrics['reward_max'],
                'reward/min': metrics['reward_min'],
                'reward/best': self.best_reward,

                # GRPO-specific losses
                'loss/total': metrics['loss/total'],
                'loss/pg_clipped': metrics['loss/pg_clipped'],
                'loss/kl_reverse': metrics['loss/kl'],

                # PPO-clip diagnostics
                'ratio/mean': metrics['ratio/mean'],
                'ratio/clipped_frac': metrics['ratio/clipped_frac'],

                # Advantages
                'advantage/mean': metrics['advantage/mean'],
                'advantage/std': metrics['advantage/std'],

                # Diversity collapse detection
                'grpo/group_reward_variance': metrics['group_reward_var'],

                # Optimizer
                'optimizer/learning_rate': metrics['lr'],

                # Performance
                'time/iteration_seconds': iter_time,

                # Format success
                'format/success_rate': format_success_rate,
            }, step=iteration)

            # Track best
            if metrics["reward_mean"] > self.best_reward:
                self.best_reward = metrics["reward_mean"]
                self.save_checkpoint("best")

            # Periodic save
            if (iteration + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"iter_{iteration+1}")

            # Restart worker engine for next iteration (if not last)
            if iteration < self.config.num_iterations - 1:
                await self._start_worker_engine()

        # Final save
        self.save_checkpoint("final")
        print(f"\nGRPO Training complete! Best reward: {self.best_reward:.3f}")

        wandb.finish()

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _compute_gini(self, values: np.ndarray) -> float:
        """Compute Gini coefficient."""
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        index = np.arange(1, n + 1)
        return float((2 * (index * sorted_vals).sum() / (n * sorted_vals.sum()) - (n + 1) / n))

    def _compute_swf(self, incomes: np.ndarray, tax_rates: List[float], brackets: List[float]) -> float:
        """Compute social welfare with numerical stability."""
        clipped_rates = [max(0.0, min(0.99, r)) for r in tax_rates]

        taxes = np.zeros_like(incomes)
        prev_bracket = 0.0
        all_brackets = brackets + [float('inf')]

        for i, bracket in enumerate(all_brackets):
            if i < len(clipped_rates):
                bracket_income = np.clip(incomes - prev_bracket, 0, bracket - prev_bracket)
                taxes += bracket_income * clipped_rates[i]
            prev_bracket = bracket

        post_tax = np.clip(incomes - taxes, 100.0, None)

        eta = 1.5
        utilities = (post_tax ** (1 - eta) - 1) / (1 - eta)
        utilities = np.nan_to_num(utilities, nan=0.0, posinf=1e6, neginf=-1e6)
        utilities = np.clip(utilities, -100, 100)

        return float(utilities.sum())

    # =========================================================================
    # CHECKPOINTING
    # =========================================================================

    def save_checkpoint(self, name: str):
        """Save training checkpoint including LoRA weights."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "config": self.config.to_dict(),
            "iteration": self.iteration,
            "best_reward": self.best_reward,
            "metrics_history": self.metrics_history,
            "baseline_metrics": self.baseline_metrics,
            "lr_halved": self._lr_halved,
        }

        with open(checkpoint_dir / "state.json", "w") as f:
            json.dump(state, f, indent=2)

        self.planner_policy.save_lora_weights(checkpoint_dir / "planner_lora")

        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }, checkpoint_dir / "optimizer.pt")

        print(f"  Saved: {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load training state from checkpoint. Returns True if successful."""
        state_file = Path(checkpoint_path) / "state.json"
        if not state_file.exists():
            print(f"No checkpoint found at {checkpoint_path}")
            return False

        with open(state_file) as f:
            state = json.load(f)

        self.iteration = state.get("iteration", 0) + 1
        self.best_reward = state.get("best_reward", float("-inf"))
        self.metrics_history = state.get("metrics_history", [])
        self.baseline_metrics = state.get("baseline_metrics")
        self._lr_halved = state.get("lr_halved", False)

        print(f"Resumed from checkpoint: {checkpoint_path}")
        print(f"  Starting at iteration {self.iteration}")
        print(f"  Best reward so far: {self.best_reward:.3f}")

        if self.baseline_metrics:
            print(f"  Loaded cached baseline: SWF={self.baseline_metrics['swf']:.2f}, Gini={self.baseline_metrics['gini']:.3f}")

        self.checkpoint_path = checkpoint_path
        return True

    def load_model_weights(self):
        """Load LoRA weights and optimizer state (call after setup())."""
        if not hasattr(self, 'checkpoint_path') or not self.checkpoint_path:
            return

        lora_path = Path(self.checkpoint_path) / "planner_lora"
        if lora_path.exists():
            self.planner_policy.load_lora_weights(lora_path)
            print(f"  Loaded LoRA weights from {lora_path}")

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


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="GRPO H3 Experiment: Group Relative Policy Optimization for Tax Planner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gpu", type=str, default="a6000",
                        choices=["a6000", "a6000_2gpu", "b200", "b200_2gpu"],
                        help="GPU type")
    parser.add_argument("--num-iterations", type=int, default=None,
                        help="Number of training iterations")
    parser.add_argument("--num-groups-per-iter", type=int, default=None,
                        help="Number of prompt groups per iteration")
    parser.add_argument("--group-size", type=int, default=None,
                        help="Number of completions per group (G)")
    parser.add_argument("--num-agents", type=int, default=None,
                        help="Number of worker agents")
    parser.add_argument("--learning-rate", type=float, default=5e-7,
                        help="Learning rate")
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="PPO clip range")
    parser.add_argument("--grpo-beta", type=float, default=0.05,
                        help="KL penalty coefficient (reverse KL)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature for planner completions")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, default="models/grpo_h3_trained",
                        help="Output directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--wandb-offline", action="store_true",
                        help="Run wandb in offline mode (for SLURM clusters)")

    args = parser.parse_args()

    config_kwargs = {
        "experiment": "h3_grpo",
        "learning_rate": args.learning_rate,
        "epsilon": args.epsilon,
        "grpo_beta": args.grpo_beta,
        "temperature": args.temperature,
        "seed": args.seed,
    }

    if args.num_agents is not None:
        config_kwargs["num_agents"] = args.num_agents
    if args.num_iterations is not None:
        config_kwargs["num_iterations"] = args.num_iterations
    if args.num_groups_per_iter is not None:
        config_kwargs["num_groups_per_iter"] = args.num_groups_per_iter
    if args.group_size is not None:
        config_kwargs["group_size"] = args.group_size

    config = GRPOConfig.for_gpu(args.gpu, **config_kwargs)

    print(f"\n{'='*60}")
    print(f"GRPO Configuration: {args.gpu.upper()}-optimized")
    print(f"{'='*60}")
    print(f"group_size (G): {config.group_size}")
    print(f"num_groups_per_iter: {config.num_groups_per_iter}")
    print(f"total completions/iter: {config.group_size * config.num_groups_per_iter}")
    print(f"epsilon (clip): {config.epsilon}")
    print(f"beta (KL): {config.grpo_beta}")
    print(f"temperature: {config.temperature}")
    print(f"learning_rate: {config.learning_rate}")
    print(f"lora_r: {config.lora_r}, lora_alpha: {config.lora_alpha}")
    print(f"tax_year_length: {config.tax_year_length}")
    print(f"gpu_memory_utilization: {config.gpu_memory_utilization}")
    print(f"{'='*60}\n")

    output_dir = args.output

    experiment = GRPOExperiment(config, output_dir)

    try:
        skip_baseline = False
        if args.resume:
            success = experiment.load_checkpoint(args.resume)
            if success and experiment.baseline_metrics is not None:
                skip_baseline = True
                print("Using cached baseline from checkpoint (skipping expensive recomputation)\n")

        await experiment.setup(skip_baseline=skip_baseline)

        if args.resume:
            experiment.load_model_weights()

        experiment.setup_wandb(offline=args.wandb_offline)
        await experiment.train()
    finally:
        await experiment.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
