#!/usr/bin/env python3
"""
G1: Pure RL AI Economist Baseline.

Trains both planner AND workers using standard PPO (no LLMs).
This establishes an upper bound for what pure RL can achieve.

Usage:
    python -m llm_economist.training.rl_baseline \
        --num-agents 1000 \
        --training-steps 500000 \
        --output models/rl_baseline/
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal, Categorical
import numpy as np


@dataclass
class RLBaselineConfig:
    """Configuration for RL baseline training."""
    # Environment
    num_agents: int = 1000
    num_brackets: int = 7
    max_timesteps: int = 2000
    tax_year_length: int = 128

    # Training
    training_steps: int = 500000
    batch_size: int = 2048
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Network architecture
    planner_hidden_dim: int = 256
    worker_hidden_dim: int = 128

    # Checkpointing
    save_every: int = 10000
    eval_every: int = 5000

    # Hardware
    device: str = "cuda"
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WorkerNetwork(nn.Module):
    """
    Worker policy network (2-layer MLP).
    Input: skill, current income, tax rates
    Output: labor hours (continuous)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_std_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))

        mean = torch.sigmoid(self.mean_head(h)) * 100  # Labor hours 0-100
        log_std = torch.clamp(self.log_std_head(h), -2, 2)
        value = self.value_head(h)

        return mean, log_std, value

    def get_action(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self.forward(x)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.clamp(0, 100), log_prob, value


class PlannerNetwork(nn.Module):
    """
    Planner policy network (3-layer MLP).
    Input: aggregate economic stats
    Output: tax rates for each bracket
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.log_std_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))

        mean = torch.sigmoid(self.mean_head(h))  # Tax rates 0-1
        log_std = torch.clamp(self.log_std_head(h), -2, 0)  # Smaller variance
        value = self.value_head(h)

        return mean, log_std, value

    def get_action(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self.forward(x)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action.clamp(0, 0.99), log_prob, value


class EconomicEnvironment:
    """
    Simplified economic environment for RL training.
    """

    def __init__(self, config: RLBaselineConfig):
        self.config = config
        self.num_agents = config.num_agents
        self.num_brackets = config.num_brackets
        self.device = config.device

        # Tax brackets (US-style)
        self.brackets = torch.tensor(
            [10000, 40000, 85000, 160000, 200000, 500000],
            dtype=torch.float32, device=self.device
        )

        # Agent skills (log-normal distribution)
        self.reset()

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset environment to initial state."""
        # Initialize agent skills (log-normal)
        self.skills = torch.exp(
            torch.randn(self.num_agents, device=self.device) * 0.5 + 3.5
        )  # Mean skill ~33

        # Initial tax rates (flat 20%)
        self.tax_rates = torch.ones(self.num_brackets, device=self.device) * 0.2

        # Initialize labor at 40 hours
        self.labor = torch.ones(self.num_agents, device=self.device) * 40

        # Compute initial state
        self.incomes = self.skills * self.labor
        self.timestep = 0

        return self._get_state()

    def _get_state(self) -> Dict[str, torch.Tensor]:
        """Get current state representation."""
        # Worker state: skill, income, tax rates (broadcasted)
        worker_state = torch.stack([
            self.skills,
            self.incomes,
            self._get_marginal_rate(self.incomes),
        ], dim=-1)

        # Planner state: aggregate statistics
        # Normalize values to prevent overflow
        income_mean = self.incomes.mean() / 1000  # Scale down
        income_std = self.incomes.std() / 1000
        income_median = self.incomes.median() / 1000
        gini = self._compute_gini(self.incomes)
        labor_mean = self.labor.mean()
        labor_std = self.labor.std()
        swf = self._compute_swf() / self.num_agents  # Normalize by agents

        planner_state = torch.tensor([
            income_mean,
            income_std,
            income_median,
            gini,
            labor_mean,
            labor_std,
            swf,
        ], device=self.device)

        # Replace any NaNs with zeros
        planner_state = torch.nan_to_num(planner_state, nan=0.0, posinf=1e6, neginf=-1e6)

        return {
            "worker_state": worker_state,
            "planner_state": planner_state,
            "tax_rates": self.tax_rates.clone(),
        }

    def _get_marginal_rate(self, incomes: torch.Tensor) -> torch.Tensor:
        """Get marginal tax rate for each income level."""
        rates = torch.zeros_like(incomes)
        for i, (bracket, rate) in enumerate(zip(
            [0] + self.brackets.tolist() + [float('inf')],
            self.tax_rates
        )):
            if i < len(self.tax_rates):
                mask = incomes > bracket
                rates[mask] = self.tax_rates[min(i, len(self.tax_rates)-1)]
        return rates

    def _apply_taxes(self, incomes: torch.Tensor) -> torch.Tensor:
        """Apply progressive tax schedule to incomes."""
        taxes = torch.zeros_like(incomes)
        prev_bracket = 0.0

        for i, bracket in enumerate(self.brackets.tolist() + [float('inf')]):
            bracket_income = torch.clamp(incomes - prev_bracket, 0, bracket - prev_bracket)
            taxes += bracket_income * self.tax_rates[min(i, len(self.tax_rates)-1)]
            prev_bracket = bracket

        return incomes - taxes

    def _compute_gini(self, values: torch.Tensor) -> torch.Tensor:
        """Compute Gini coefficient."""
        sorted_vals = torch.sort(values)[0]
        n = len(sorted_vals)
        index = torch.arange(1, n + 1, device=self.device, dtype=torch.float32)
        total = sorted_vals.sum()
        if total < 1e-8:
            return torch.tensor(0.0, device=self.device)
        return (2 * (index * sorted_vals).sum() / (n * total) - (n + 1) / n)

    def _compute_swf(self) -> torch.Tensor:
        """Compute social welfare function (sum of utilities)."""
        post_tax = self._apply_taxes(self.incomes)
        # Isoelastic utility: (z^(1-eta) - 1) / (1-eta) with eta=1.5
        # Add small epsilon to avoid pow of zero
        eta = 1.5
        post_tax = torch.clamp(post_tax, min=1e-6)
        utilities = (post_tax.pow(1-eta) - 1) / (1-eta)
        return utilities.sum()

    def step(
        self,
        worker_actions: torch.Tensor,
        planner_action: Optional[torch.Tensor] = None,
        update_planner: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, bool]:
        """
        Step the environment.

        Args:
            worker_actions: Labor hours for each worker [num_agents]
            planner_action: New tax rates [num_brackets] (optional)
            update_planner: Whether to update planner policy this step

        Returns:
            next_state, worker_rewards, planner_reward, done
        """
        # Update tax rates if planner acts
        if update_planner and planner_action is not None:
            self.tax_rates = planner_action.squeeze()

        # Workers choose labor
        self.labor = worker_actions.squeeze().clamp(0, 100)

        # Compute incomes
        self.incomes = self.skills * self.labor

        # Compute post-tax income
        post_tax = self._apply_taxes(self.incomes)

        # Worker utility: post_tax income - labor cost
        labor_cost = 0.5 * (self.labor / 50).pow(2)  # Quadratic labor cost
        worker_utilities = post_tax / 1000 - labor_cost  # Scale down income

        # Worker rewards: utility change
        worker_rewards = worker_utilities

        # Planner reward: SWF
        swf = self._compute_swf()
        gini = self._compute_gini(self.incomes)
        planner_reward = swf / self.num_agents - 10 * gini  # Penalize inequality

        self.timestep += 1
        done = self.timestep >= self.config.max_timesteps

        return self._get_state(), worker_rewards, planner_reward, done


class PPOBuffer:
    """Buffer for storing trajectories for PPO training."""

    def __init__(self, size: int, obs_dim: int, action_dim: int = 1, device: str = "cuda"):
        self.size = size
        self.device = device

        self.observations = torch.zeros(size, obs_dim, device=device)
        self.actions = torch.zeros(size, action_dim, device=device)
        self.log_probs = torch.zeros(size, 1, device=device)
        self.rewards = torch.zeros(size, 1, device=device)
        self.values = torch.zeros(size, 1, device=device)
        self.dones = torch.zeros(size, 1, device=device)
        self.advantages = torch.zeros(size, 1, device=device)
        self.returns = torch.zeros(size, 1, device=device)

        self.ptr = 0
        self.path_start_idx = 0

    def store(self, obs, action, log_prob, reward, value, done):
        """Store a single transition."""
        self.observations[self.ptr] = obs.squeeze()
        self.actions[self.ptr] = action.squeeze()
        self.log_probs[self.ptr] = log_prob.squeeze()
        self.rewards[self.ptr] = reward.squeeze()
        self.values[self.ptr] = value.squeeze()
        self.dones[self.ptr] = done.squeeze()
        self.ptr += 1

    def finish_path(self, last_value: float, gamma: float, lam: float):
        """Compute GAE advantages when trajectory ends."""
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = self.rewards[path_slice]
        values = self.values[path_slice]

        # Compute GAE
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = last_gae = delta + gamma * lam * last_gae

        self.advantages[path_slice] = advantages
        self.returns[path_slice] = advantages + values
        self.path_start_idx = self.ptr

    def get(self):
        """Get all data from buffer."""
        return (
            self.observations[:self.ptr],
            self.actions[:self.ptr],
            self.log_probs[:self.ptr],
            self.returns[:self.ptr],
            self.advantages[:self.ptr],
        )

    def clear(self):
        """Reset buffer."""
        self.ptr = 0
        self.path_start_idx = 0


class RLBaselineTrainer:
    """
    PPO trainer for both planner and workers.
    """

    def __init__(self, config: RLBaselineConfig, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = config.device

        # Initialize environment
        self.env = EconomicEnvironment(config)

        # Initialize networks
        worker_input_dim = 3  # skill, income, marginal_rate
        planner_input_dim = 7  # aggregate stats

        self.worker_net = WorkerNetwork(
            worker_input_dim, config.worker_hidden_dim
        ).to(self.device)

        self.planner_net = PlannerNetwork(
            planner_input_dim, config.num_brackets, config.planner_hidden_dim
        ).to(self.device)

        # Optimizers
        self.worker_optimizer = Adam(
            self.worker_net.parameters(), lr=config.learning_rate
        )
        self.planner_optimizer = Adam(
            self.planner_net.parameters(), lr=config.learning_rate
        )

        # Buffers
        self.worker_buffer = PPOBuffer(
            config.batch_size, worker_input_dim, action_dim=1, device=self.device
        )
        self.planner_buffer = PPOBuffer(
            config.batch_size // config.tax_year_length, planner_input_dim,
            action_dim=config.num_brackets, device=self.device
        )

        # Metrics
        self.metrics_history = []
        self.total_steps = 0
        self.best_swf = float("-inf")

    def collect_rollouts(self) -> Dict[str, float]:
        """Collect rollouts from environment."""
        state = self.env.reset()
        episode_rewards = []
        episode_swfs = []

        for step in range(self.config.batch_size):
            # Get worker actions
            worker_state = state["worker_state"]
            with torch.no_grad():
                worker_actions, worker_log_probs, worker_values = \
                    self.worker_net.get_action(worker_state)

            # Get planner action (every tax_year_length steps)
            update_planner = (step % self.config.tax_year_length == 0)
            planner_action = None

            if update_planner:
                planner_state = state["planner_state"]
                with torch.no_grad():
                    planner_action, planner_log_prob, planner_value = \
                        self.planner_net.get_action(planner_state.unsqueeze(0))

                # Store planner transition
                self.planner_buffer.store(
                    planner_state, planner_action, planner_log_prob,
                    torch.tensor([0.0], device=self.device),  # Reward filled later
                    planner_value, torch.tensor([0.0], device=self.device)
                )

            # Step environment
            next_state, worker_rewards, planner_reward, done = self.env.step(
                worker_actions, planner_action, update_planner
            )

            # Store worker transitions (sample a subset for efficiency)
            sample_idx = torch.randint(0, self.config.num_agents, (1,)).item()
            self.worker_buffer.store(
                worker_state[sample_idx],
                worker_actions[sample_idx],
                worker_log_probs[sample_idx],
                worker_rewards[sample_idx].unsqueeze(0),
                worker_values[sample_idx],
                torch.tensor([float(done)], device=self.device)
            )

            episode_rewards.append(worker_rewards.mean().item())
            episode_swfs.append(self.env._compute_swf().item())

            state = next_state

            if done:
                # Finish paths
                self.worker_buffer.finish_path(
                    0, self.config.gamma, self.config.gae_lambda
                )
                self.planner_buffer.finish_path(
                    0, self.config.gamma, self.config.gae_lambda
                )
                state = self.env.reset()

        return {
            "mean_reward": np.mean(episode_rewards),
            "mean_swf": np.mean(episode_swfs),
            "final_swf": episode_swfs[-1] if episode_swfs else 0,
        }

    def update_policy(self, network, optimizer, buffer, epochs: int = 4):
        """Update policy using PPO."""
        obs, actions, old_log_probs, returns, advantages = buffer.get()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        for _ in range(epochs):
            # Get new log probs and values
            if network == self.worker_net:
                mean, log_std, values = network(obs)
                std = log_std.exp()
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
            else:
                mean, log_std, values = network(obs)
                std = log_std.exp()
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().sum(dim=-1).mean()

            # Policy loss (PPO clip)
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values, returns)

            # Total loss
            loss = (
                policy_loss
                + self.config.value_coef * value_loss
                - self.config.entropy_coef * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), self.config.max_grad_norm)
            optimizer.step()

            total_loss += loss.item()

        buffer.clear()
        return total_loss / epochs

    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print("G1: RL AI Economist Baseline Training (PPO)")
        print(f"{'='*60}")
        print(f"Agents: {self.config.num_agents}")
        print(f"Training steps: {self.config.training_steps}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"{'='*60}\n")

        start_time = time.time()

        while self.total_steps < self.config.training_steps:
            # Collect rollouts
            rollout_metrics = self.collect_rollouts()
            self.total_steps += self.config.batch_size

            # Update policies
            worker_loss = self.update_policy(
                self.worker_net, self.worker_optimizer, self.worker_buffer
            )
            planner_loss = self.update_policy(
                self.planner_net, self.planner_optimizer, self.planner_buffer
            )

            # Log metrics
            metrics = {
                "step": self.total_steps,
                "worker_loss": worker_loss,
                "planner_loss": planner_loss,
                **rollout_metrics,
            }
            self.metrics_history.append(metrics)

            # Print progress
            elapsed = time.time() - start_time
            steps_per_sec = self.total_steps / elapsed
            eta = (self.config.training_steps - self.total_steps) / steps_per_sec

            print(f"Step {self.total_steps:,}/{self.config.training_steps:,} "
                  f"| SWF: {rollout_metrics['mean_swf']:.1f} "
                  f"| Worker Loss: {worker_loss:.4f} "
                  f"| Planner Loss: {planner_loss:.4f} "
                  f"| ETA: {eta/60:.1f}min")

            # Save checkpoint
            if self.total_steps % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_{self.total_steps}")

            # Track best
            if rollout_metrics['mean_swf'] > self.best_swf:
                self.best_swf = rollout_metrics['mean_swf']
                self.save_checkpoint("best")

        # Final save
        self.save_checkpoint("final")
        print(f"\nTraining complete! Best SWF: {self.best_swf:.1f}")
        print(f"Total time: {(time.time() - start_time)/3600:.2f} hours")

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(exist_ok=True)

        torch.save({
            "worker_net": self.worker_net.state_dict(),
            "planner_net": self.planner_net.state_dict(),
            "worker_optimizer": self.worker_optimizer.state_dict(),
            "planner_optimizer": self.planner_optimizer.state_dict(),
            "total_steps": self.total_steps,
            "best_swf": self.best_swf,
            "config": self.config.to_dict(),
        }, checkpoint_dir / "checkpoint.pt")

        with open(checkpoint_dir / "metrics.json", "w") as f:
            json.dump(self.metrics_history[-100:], f, indent=2)  # Last 100 entries

        print(f"  Saved checkpoint: {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.worker_net.load_state_dict(checkpoint["worker_net"])
        self.planner_net.load_state_dict(checkpoint["planner_net"])
        self.worker_optimizer.load_state_dict(checkpoint["worker_optimizer"])
        self.planner_optimizer.load_state_dict(checkpoint["planner_optimizer"])
        self.total_steps = checkpoint["total_steps"]
        self.best_swf = checkpoint["best_swf"]

        print(f"Loaded checkpoint from step {self.total_steps}")


def main():
    parser = argparse.ArgumentParser(description="G1: RL AI Economist Baseline")

    parser.add_argument("--num-agents", type=int, default=1000)
    parser.add_argument("--training-steps", type=int, default=500000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create config
    config = RLBaselineConfig(
        num_agents=args.num_agents,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        seed=args.seed,
    )

    # Create trainer
    trainer = RLBaselineTrainer(config, args.output)

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
