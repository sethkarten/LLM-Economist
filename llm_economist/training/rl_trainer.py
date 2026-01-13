#!/usr/bin/env python3
"""
REINFORCE++ trainer for planner policy optimization.

Implements policy gradient training with:
- Group-relative baseline (GRPO-style)
- KL penalty to prevent drift from base model
- Entropy bonus for exploration
- Distributed rollout collection support

Usage:
    # Local training (single GPU)
    python -m llm_economist.training.rl_trainer \
        --planner-model qwen3-4b \
        --worker-model qwen3-30b-a3b \
        --num-agents 100 \
        --output models/planner-rl

    # Distributed training (coordinator mode)
    python -m llm_economist.training.rl_trainer \
        --mode coordinator \
        --rollout-dir /shared/rollouts \
        --output models/planner-rl
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import math

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np


@dataclass
class RLConfig:
    """Configuration for REINFORCE++ training."""
    # Model
    planner_model: str = "Qwen/Qwen3-4B-Instruct"
    worker_model: str = "Qwen/Qwen3-30B-A3B-Instruct"
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05

    # Environment
    num_agents: int = 500
    steps_per_rollout: int = 256  # ~2 tax years
    tax_year_length: int = 128

    # Training
    learning_rate: float = 1e-5
    batch_size: int = 64  # rollouts per gradient update
    num_iterations: int = 100
    max_grad_norm: float = 1.0

    # REINFORCE++ specifics
    kl_coef: float = 0.1  # KL penalty coefficient
    entropy_coef: float = 0.01  # Entropy bonus
    gamma: float = 0.99  # Discount factor
    use_group_baseline: bool = True  # GRPO-style baseline

    # Reward shaping
    swf_weight: float = 1.0
    gini_weight: float = 0.1  # Bonus for reducing inequality
    labor_weight: float = 0.1  # Penalty for reducing labor

    # Checkpointing
    save_every: int = 10
    eval_every: int = 5

    # Hardware
    device: str = "cuda"
    use_4bit: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RLConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Rollout:
    """A single planner rollout (one tax year of interaction)."""
    # Observation (state before action)
    observation: Dict[str, Any]

    # Action taken
    action: Dict[str, Any]  # {"tax_rates": [...], "brackets": [...]}
    action_log_prob: float  # log π(a|s)

    # Outcome
    reward: float
    next_observation: Dict[str, Any]

    # Metadata
    rollout_id: str
    worker_id: str
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation": self.observation,
            "action": self.action,
            "action_log_prob": self.action_log_prob,
            "reward": self.reward,
            "next_observation": self.next_observation,
            "rollout_id": self.rollout_id,
            "worker_id": self.worker_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Rollout":
        return cls(**d)


@dataclass
class RolloutBatch:
    """Batch of rollouts for training."""
    observations: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    log_probs: torch.Tensor  # [batch_size]
    rewards: torch.Tensor  # [batch_size]
    advantages: torch.Tensor  # [batch_size] - computed from rewards

    @classmethod
    def from_rollouts(
        cls,
        rollouts: List[Rollout],
        use_group_baseline: bool = True,
        device: str = "cuda"
    ) -> "RolloutBatch":
        """Create batch from list of rollouts with advantage computation."""
        observations = [r.observation for r in rollouts]
        actions = [r.action for r in rollouts]
        log_probs = torch.tensor(
            [r.action_log_prob for r in rollouts],
            dtype=torch.float32,
            device=device
        )
        rewards = torch.tensor(
            [r.reward for r in rollouts],
            dtype=torch.float32,
            device=device
        )

        # Compute advantages using group-relative baseline (GRPO-style)
        if use_group_baseline:
            baseline = rewards.mean()
            advantages = rewards - baseline
            # Normalize advantages
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = rewards

        return cls(
            observations=observations,
            actions=actions,
            log_probs=log_probs,
            rewards=rewards,
            advantages=advantages,
        )


class PlannerPolicy:
    """
    Wrapper around the planner LLM for policy gradient training.

    Handles:
    - Action sampling (tax policy generation)
    - Log probability computation
    - KL divergence from reference model
    """

    def __init__(
        self,
        model_name: str,
        config: RLConfig,
        reference_model: Optional[Any] = None,
    ):
        self.model_name = model_name
        self.config = config
        self.reference_model = reference_model

        self.model = None
        self.tokenizer = None
        self.device = config.device

    def load_model(self):
        """Load the policy model with LoRA adapters."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        print(f"Loading planner policy: {self.model_name}")

        # Quantization config
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if not self.config.use_4bit else None,
        )

        # Prepare for training
        if self.config.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

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

    def format_observation(self, obs: Dict[str, Any]) -> str:
        """Format observation into prompt string."""
        from .dataset import PLANNER_SYSTEM_PROMPT, PLANNER_USER_TEMPLATE

        # Format current tax policy
        tax_rates = obs.get("current_tax_rates", [0.1, 0.2, 0.3])
        brackets = obs.get("current_brackets", [50000, 100000])

        tax_policy_lines = []
        for i, rate in enumerate(tax_rates):
            if i == 0:
                upper = brackets[0] if brackets else "∞"
                tax_policy_lines.append(f"  - $0 - ${upper:,.0f}: {rate*100:.1f}%")
            elif i < len(brackets):
                lower = brackets[i-1]
                upper = brackets[i]
                tax_policy_lines.append(f"  - ${lower:,.0f} - ${upper:,.0f}: {rate*100:.1f}%")
            else:
                lower = brackets[-1] if brackets else 0
                tax_policy_lines.append(f"  - ${lower:,.0f}+: {rate*100:.1f}%")

        current_tax_policy = "\n".join(tax_policy_lines)

        # Format historical trends
        def format_trend(values: List[float]) -> str:
            if not values:
                return "N/A"
            if len(values) == 1:
                return f"{values[0]:.3f}"
            trend = "↑" if values[-1] > values[0] else "↓" if values[-1] < values[0] else "→"
            return f"{values[-1]:.3f} ({trend} from {values[0]:.3f})"

        swf_trend = format_trend(obs.get("historical_swf", []))
        gini_trend = format_trend(obs.get("historical_gini", []))
        labor_trend = format_trend(obs.get("historical_labor", []))

        prompt = f"""## Economic State (Tax Year {obs.get('tax_year', 0)})

**Aggregate Metrics:**
- Total Income: ${obs.get('total_income', 0):,.0f}
- Average Income: ${obs.get('avg_income', 0):,.0f}
- Median Income: ${obs.get('median_income', 0):,.0f}
- Income Gini: {obs.get('income_gini', 0):.3f}
- Total Labor Supply: {obs.get('total_labor', 0):,.0f} hours
- Number of Agents: {obs.get('num_agents', 0)}

**Current Tax Policy:**
{current_tax_policy}

**Historical Trends:**
- SWF: {swf_trend}
- Gini: {gini_trend}
- Labor: {labor_trend}

Based on this economic state, determine the optimal tax policy.
Respond with JSON: {{"tax_rates": [...], "brackets": [...]}}"""

        return prompt

    def sample_action(
        self,
        observation: Dict[str, Any],
        temperature: float = 0.7,
        max_new_tokens: int = 256,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Sample a tax policy action from the policy.

        Returns:
            Tuple of (action_dict, log_probability)
        """
        from .dataset import PLANNER_SYSTEM_PROMPT

        prompt = self.format_observation(observation)

        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        full_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

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

        # Compute log probability of generated sequence
        log_prob = self._compute_sequence_log_prob(outputs.scores, generated_ids)

        # Parse action from response
        action = self._parse_action(response)

        return action, log_prob

    def _compute_sequence_log_prob(
        self,
        scores: Tuple[torch.Tensor, ...],
        generated_ids: torch.Tensor
    ) -> float:
        """Compute log probability of generated sequence."""
        total_log_prob = 0.0

        for i, (score, token_id) in enumerate(zip(scores, generated_ids)):
            if token_id == self.tokenizer.pad_token_id:
                break
            log_probs = F.log_softmax(score[0], dim=-1)
            total_log_prob += log_probs[token_id].item()

        return total_log_prob

    def _parse_action(self, response: str) -> Dict[str, Any]:
        """Parse tax policy from model response."""
        import json

        default_action = {
            "tax_rates": [0.1, 0.2, 0.3, 0.35],
            "brackets": [30000, 70000, 150000],
        }

        try:
            # Extract JSON from response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            elif "{" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                json_str = response[start:end]
            else:
                return default_action

            action = json.loads(json_str)

            # Validate
            if "tax_rates" not in action or "brackets" not in action:
                return default_action

            # Clamp tax rates to valid range
            action["tax_rates"] = [max(0.0, min(1.0, r)) for r in action["tax_rates"]]

            return action

        except (json.JSONDecodeError, ValueError, IndexError):
            return default_action

    def compute_log_prob(
        self,
        observation: Dict[str, Any],
        action: Dict[str, Any],
    ) -> float:
        """Compute log probability of a specific action given observation."""
        from .dataset import PLANNER_SYSTEM_PROMPT

        prompt = self.format_observation(observation)
        action_str = json.dumps(action)

        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": action_str},
        ]

        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Compute log prob of action tokens
        # Find where action starts in the sequence
        action_tokens = self.tokenizer(action_str, return_tensors="pt")["input_ids"][0]

        # This is simplified - in practice we'd find exact position
        log_probs = F.log_softmax(logits[0, -len(action_tokens)-1:-1], dim=-1)
        action_log_prob = sum(
            log_probs[i, token_id].item()
            for i, token_id in enumerate(action_tokens)
            if i < len(log_probs)
        )

        return action_log_prob

    def compute_entropy(self, observation: Dict[str, Any]) -> float:
        """Compute entropy of policy at given observation (for exploration bonus)."""
        # Simplified: sample multiple actions and estimate entropy
        log_probs = []
        for _ in range(5):
            _, log_prob = self.sample_action(observation, temperature=1.0)
            log_probs.append(log_prob)

        # Estimate entropy from samples
        log_probs = torch.tensor(log_probs)
        entropy = -log_probs.mean().item()
        return entropy

    def save(self, output_dir: str):
        """Save model and tokenizer."""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save config
        with open(os.path.join(output_dir, "rl_config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    @classmethod
    def load(cls, checkpoint_dir: str, config: Optional[RLConfig] = None) -> "PlannerPolicy":
        """Load model from checkpoint."""
        if config is None:
            config_path = os.path.join(checkpoint_dir, "rl_config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = RLConfig.from_dict(json.load(f))
            else:
                config = RLConfig()

        policy = cls(checkpoint_dir, config)
        policy.load_model()
        return policy


class REINFORCETrainer:
    """
    REINFORCE++ trainer for planner policy.

    Implements policy gradient with:
    - Group-relative baseline (reduces variance)
    - KL penalty (prevents drift from base model)
    - Entropy bonus (encourages exploration)
    """

    def __init__(
        self,
        policy: PlannerPolicy,
        config: RLConfig,
        output_dir: str,
    ):
        self.policy = policy
        self.config = config
        self.output_dir = output_dir

        # Training state
        self.iteration = 0
        self.total_rollouts = 0
        self.best_reward = float("-inf")

        # Metrics history
        self.metrics_history: List[Dict[str, float]] = []

        # Setup optimizer
        self.optimizer = None
        self.scheduler = None

    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        trainable_params = [p for p in self.policy.model.parameters() if p.requires_grad]

        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_iterations,
            eta_min=self.config.learning_rate * 0.1,
        )

    def train_step(self, batch: RolloutBatch) -> Dict[str, float]:
        """
        Perform one training step on a batch of rollouts.

        Returns:
            Dictionary of metrics
        """
        self.policy.model.train()

        # Compute new log probs for actions in batch
        new_log_probs = []
        for obs, action in zip(batch.observations, batch.actions):
            log_prob = self.policy.compute_log_prob(obs, action)
            new_log_probs.append(log_prob)

        new_log_probs = torch.tensor(
            new_log_probs,
            dtype=torch.float32,
            device=self.config.device,
            requires_grad=True
        )

        # Policy gradient loss: -E[log π(a|s) * A]
        pg_loss = -(new_log_probs * batch.advantages).mean()

        # KL penalty (optional, to prevent drift)
        kl_loss = torch.tensor(0.0, device=self.config.device)
        if self.config.kl_coef > 0 and self.policy.reference_model is not None:
            # Compute KL divergence from reference model
            # Simplified: use ratio of log probs
            log_ratio = new_log_probs - batch.log_probs
            kl_loss = log_ratio.mean()

        # Entropy bonus (optional, for exploration)
        entropy_loss = torch.tensor(0.0, device=self.config.device)
        if self.config.entropy_coef > 0:
            # Approximate entropy from log probs
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
                self.policy.model.parameters(),
                self.config.max_grad_norm
            )

        self.optimizer.step()
        self.scheduler.step()

        metrics = {
            "loss/total": total_loss.item(),
            "loss/pg": pg_loss.item(),
            "loss/kl": kl_loss.item(),
            "loss/entropy": entropy_loss.item(),
            "reward/mean": batch.rewards.mean().item(),
            "reward/std": batch.rewards.std().item(),
            "reward/max": batch.rewards.max().item(),
            "reward/min": batch.rewards.min().item(),
            "advantage/mean": batch.advantages.mean().item(),
            "advantage/std": batch.advantages.std().item(),
            "lr": self.scheduler.get_last_lr()[0],
        }

        return metrics

    def train(
        self,
        rollout_collector,
        num_iterations: Optional[int] = None,
    ):
        """
        Main training loop.

        Args:
            rollout_collector: Object that can collect rollouts (env + workers)
            num_iterations: Number of training iterations (overrides config)
        """
        num_iterations = num_iterations or self.config.num_iterations

        print(f"\n{'='*60}")
        print("Starting REINFORCE++ Training")
        print(f"{'='*60}")
        print(f"Policy: {self.policy.model_name}")
        print(f"Iterations: {num_iterations}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"{'='*60}\n")

        self.setup_optimizer()

        for iteration in range(num_iterations):
            self.iteration = iteration

            # Collect rollouts
            print(f"\nIteration {iteration+1}/{num_iterations}")
            print("Collecting rollouts...")

            rollouts = rollout_collector.collect(
                policy=self.policy,
                num_rollouts=self.config.batch_size,
            )

            self.total_rollouts += len(rollouts)

            # Create batch with advantages
            batch = RolloutBatch.from_rollouts(
                rollouts,
                use_group_baseline=self.config.use_group_baseline,
                device=self.config.device,
            )

            # Training step
            print("Training...")
            metrics = self.train_step(batch)
            self.metrics_history.append(metrics)

            # Logging
            print(f"  Loss: {metrics['loss/total']:.4f}")
            print(f"  Reward: {metrics['reward/mean']:.4f} ± {metrics['reward/std']:.4f}")
            print(f"  Advantage: {metrics['advantage/mean']:.4f}")
            print(f"  LR: {metrics['lr']:.2e}")

            # Checkpointing
            if (iteration + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_{iteration+1}")

            # Track best model
            if metrics["reward/mean"] > self.best_reward:
                self.best_reward = metrics["reward/mean"]
                self.save_checkpoint("best")

        # Final save
        self.save_checkpoint("final")
        print(f"\nTraining complete! Best reward: {self.best_reward:.4f}")

    def save_checkpoint(self, name: str):
        """Save training checkpoint."""
        checkpoint_dir = os.path.join(self.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model
        self.policy.save(checkpoint_dir)

        # Save training state
        state = {
            "iteration": self.iteration,
            "total_rollouts": self.total_rollouts,
            "best_reward": self.best_reward,
            "metrics_history": self.metrics_history,
            "config": self.config.to_dict(),
        }

        with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
            json.dump(state, f, indent=2)

        print(f"  Saved checkpoint: {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir: str):
        """Load training checkpoint."""
        state_path = os.path.join(checkpoint_dir, "training_state.json")

        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)

            self.iteration = state["iteration"]
            self.total_rollouts = state["total_rollouts"]
            self.best_reward = state["best_reward"]
            self.metrics_history = state["metrics_history"]

        self.policy = PlannerPolicy.load(checkpoint_dir, self.config)


class RolloutCollector:
    """
    Collects rollouts by running the planner in the economic simulation.

    Each rollout:
    1. Planner observes economic state
    2. Planner samples tax policy
    3. Workers respond for one tax year
    4. Compute reward from SWF change
    """

    def __init__(
        self,
        worker_model: str,
        config: RLConfig,
        seed: int = 42,
    ):
        self.worker_model = worker_model
        self.config = config
        self.seed = seed

        self.simulation = None

    def setup_simulation(self):
        """Initialize the economic simulation with workers."""
        # This would integrate with the main async simulation
        # For now, we'll create a simplified version
        pass

    def collect(
        self,
        policy: PlannerPolicy,
        num_rollouts: int,
    ) -> List[Rollout]:
        """
        Collect rollouts by running the policy in the environment.

        Args:
            policy: Planner policy to collect rollouts for
            num_rollouts: Number of rollouts to collect

        Returns:
            List of Rollout objects
        """
        rollouts = []

        for i in range(num_rollouts):
            rollout = self._collect_single_rollout(policy, rollout_id=f"rollout_{i}")
            rollouts.append(rollout)

        return rollouts

    def _collect_single_rollout(
        self,
        policy: PlannerPolicy,
        rollout_id: str,
    ) -> Rollout:
        """Collect a single rollout (one tax year)."""
        # Get current observation from simulation
        observation = self._get_observation()

        # Sample action from policy
        action, log_prob = policy.sample_action(observation)

        # Step environment (workers respond to new tax policy)
        next_observation, reward = self._step_environment(action)

        return Rollout(
            observation=observation,
            action=action,
            action_log_prob=log_prob,
            reward=reward,
            next_observation=next_observation,
            rollout_id=rollout_id,
            worker_id="local",
            timestamp=time.time(),
        )

    def _get_observation(self) -> Dict[str, Any]:
        """Get current economic state observation."""
        # Placeholder - would come from actual simulation
        return {
            "tax_year": 0,
            "total_income": 5000000,
            "avg_income": 50000,
            "median_income": 45000,
            "income_gini": 0.35,
            "total_labor": 400000,
            "num_agents": self.config.num_agents,
            "current_tax_rates": [0.1, 0.2, 0.3, 0.35],
            "current_brackets": [30000, 70000, 150000],
            "historical_swf": [0.5, 0.52, 0.51],
            "historical_gini": [0.36, 0.35, 0.35],
            "historical_labor": [390000, 395000, 400000],
        }

    def _step_environment(
        self,
        action: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float]:
        """
        Apply tax policy and run workers for one tax year.

        Returns:
            Tuple of (next_observation, reward)
        """
        # Placeholder - would run actual simulation
        # For now, compute a synthetic reward based on the action

        tax_rates = action.get("tax_rates", [0.2, 0.3, 0.35, 0.4])

        # Simple reward: penalize very high or very low rates
        avg_rate = np.mean(tax_rates)
        progressivity = tax_rates[-1] - tax_rates[0] if len(tax_rates) > 1 else 0

        # Reward heuristic (placeholder)
        reward = (
            -abs(avg_rate - 0.25) * 2  # Optimal average ~25%
            + progressivity * 0.5       # Reward progressivity
            + np.random.normal(0, 0.1)  # Noise
        )

        next_observation = self._get_observation()
        next_observation["tax_year"] += 1

        return next_observation, reward


def main():
    parser = argparse.ArgumentParser(description="REINFORCE++ Planner Training")

    # Model arguments
    parser.add_argument("--planner-model", "-p", type=str, default="Qwen/Qwen3-4B-Instruct",
                       help="Planner model to finetune")
    parser.add_argument("--worker-model", "-w", type=str, default="Qwen/Qwen3-30B-A3B-Instruct",
                       help="Worker model (fixed)")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output directory for checkpoints")

    # Training arguments
    parser.add_argument("--num-iterations", type=int, default=100,
                       help="Number of training iterations")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Rollouts per training step")
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--kl-coef", type=float, default=0.1,
                       help="KL penalty coefficient")

    # Environment arguments
    parser.add_argument("--num-agents", type=int, default=100,
                       help="Number of worker agents")
    parser.add_argument("--steps-per-rollout", type=int, default=256,
                       help="Steps per rollout (tax year length)")

    # Other
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create config
    config = RLConfig(
        planner_model=args.planner_model,
        worker_model=args.worker_model,
        num_agents=args.num_agents,
        steps_per_rollout=args.steps_per_rollout,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        kl_coef=args.kl_coef,
    )

    # Create policy
    policy = PlannerPolicy(args.planner_model, config)
    policy.load_model()

    # Create rollout collector
    collector = RolloutCollector(args.worker_model, config, seed=args.seed)

    # Create trainer
    trainer = REINFORCETrainer(policy, config, args.output)

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(collector, num_iterations=args.num_iterations)


if __name__ == "__main__":
    main()
