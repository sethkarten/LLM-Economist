#!/usr/bin/env python3
"""
Local test for REINFORCE++ trainer components.

Tests:
1. Rollout data structure
2. Batch creation with advantage computation
3. Policy forward pass (mocked)
4. Training step gradient flow

Run:
    python experiments/local_tests/test_rl_trainer.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np


def test_rollout_dataclass():
    """Test Rollout dataclass creation and conversion."""
    from llm_economist.training.rl_trainer import Rollout

    rollout = Rollout(
        observation={
            "tax_year": 5,
            "total_income": 5000000,
            "avg_income": 50000,
            "income_gini": 0.35,
            "num_agents": 100,
        },
        action={
            "tax_rates": [0.1, 0.2, 0.3],
            "brackets": [30000, 70000],
        },
        action_log_prob=-2.5,
        reward=0.05,
        next_observation={"tax_year": 6},
        rollout_id="test_001",
        worker_id="local",
        timestamp=0.0,
    )

    # Test conversion
    d = rollout.to_dict()
    assert d["observation"]["tax_year"] == 5
    assert d["reward"] == 0.05

    # Test reconstruction
    rollout2 = Rollout.from_dict(d)
    assert rollout2.action["tax_rates"] == [0.1, 0.2, 0.3]

    print("✓ Rollout dataclass test passed")


def test_rollout_batch():
    """Test RolloutBatch with advantage computation."""
    from llm_economist.training.rl_trainer import Rollout, RolloutBatch

    # Create test rollouts with varying rewards
    rollouts = []
    for i in range(10):
        rollout = Rollout(
            observation={"tax_year": i, "num_agents": 100},
            action={"tax_rates": [0.1 + i * 0.01]},
            action_log_prob=-2.0 - i * 0.1,
            reward=0.1 * (i - 5),  # Range: -0.5 to 0.4
            next_observation={"tax_year": i + 1},
            rollout_id=f"test_{i}",
            worker_id="local",
            timestamp=float(i),
        )
        rollouts.append(rollout)

    # Create batch
    batch = RolloutBatch.from_rollouts(rollouts, use_group_baseline=True, device="cpu")

    # Check shapes
    assert len(batch.observations) == 10
    assert batch.log_probs.shape == (10,)
    assert batch.rewards.shape == (10,)
    assert batch.advantages.shape == (10,)

    # Check advantage normalization
    assert abs(batch.advantages.mean().item()) < 1e-5, "Advantages should be mean-centered"
    assert abs(batch.advantages.std().item() - 1.0) < 0.1, "Advantages should be normalized"

    print("✓ RolloutBatch test passed")


def test_rl_config():
    """Test RLConfig serialization."""
    from llm_economist.training.rl_trainer import RLConfig

    config = RLConfig(
        planner_model="test-model",
        num_agents=50,
        learning_rate=1e-4,
        batch_size=32,
    )

    # Test to_dict
    d = config.to_dict()
    assert d["num_agents"] == 50
    assert d["learning_rate"] == 1e-4

    # Test from_dict
    config2 = RLConfig.from_dict(d)
    assert config2.planner_model == "test-model"
    assert config2.batch_size == 32

    print("✓ RLConfig test passed")


def test_reward_computation():
    """Test RL reward computation from trajectories."""
    from llm_economist.training.data_collector import (
        PlannerObservation, PlannerOutcome, compute_rl_reward
    )

    # Create test observation
    observation = PlannerObservation(
        timestep=128,
        tax_year=1,
        total_labor_supply=40000,
        total_income=5000000,
        avg_income=50000,
        median_income=45000,
        income_gini=0.35,
        income_by_bracket=[30, 40, 20, 10],
        bracket_thresholds=[30000, 70000, 150000],
        current_tax_rates=[0.1, 0.2, 0.3, 0.35],
        current_brackets=[30000, 70000, 150000],
        historical_swf=[0.5, 0.52],
        historical_gini=[0.36, 0.35],
        historical_labor=[39000, 40000],
        avg_labor_hours=40.0,
        labor_variance=100.0,
        num_agents=100,
    )

    # Test positive outcome
    good_outcome = PlannerOutcome(
        next_swf=0.55,
        swf_change=0.03,  # Improvement
        next_gini=0.33,
        gini_change=-0.02,  # Reduction (good)
        next_total_labor=41000,
        labor_change=1000,  # Increase
        next_total_income=5200000,
        income_change=200000,
        income_by_bracket_next=[25, 42, 23, 10],
        bracket_mobility=0.1,
    )

    reward = compute_rl_reward(good_outcome, observation)
    assert reward > 0, f"Good outcome should have positive reward, got {reward}"

    # Test negative outcome
    bad_outcome = PlannerOutcome(
        next_swf=0.45,
        swf_change=-0.07,  # Decline
        next_gini=0.40,
        gini_change=0.05,  # Increase (bad)
        next_total_labor=35000,
        labor_change=-5000,  # Decrease
        next_total_income=4500000,
        income_change=-500000,
        income_by_bracket_next=[35, 38, 18, 9],
        bracket_mobility=0.2,
    )

    bad_reward = compute_rl_reward(bad_outcome, observation)
    assert bad_reward < reward, f"Bad outcome should have lower reward"

    print("✓ Reward computation test passed")


def test_trajectory_to_rollout_conversion():
    """Test conversion from trajectory to RL rollout format."""
    from llm_economist.training.data_collector import (
        PlannerTrajectory, PlannerObservation, PlannerAction, PlannerOutcome,
        trajectory_to_rl_rollout
    )

    obs = PlannerObservation(
        timestep=128,
        tax_year=1,
        total_labor_supply=40000,
        total_income=5000000,
        avg_income=50000,
        median_income=45000,
        income_gini=0.35,
        income_by_bracket=[30, 40, 20, 10],
        bracket_thresholds=[30000, 70000, 150000],
        current_tax_rates=[0.1, 0.2, 0.3, 0.35],
        current_brackets=[30000, 70000, 150000],
        historical_swf=[0.5, 0.52],
        historical_gini=[0.36, 0.35],
        historical_labor=[39000, 40000],
        avg_labor_hours=40.0,
        labor_variance=100.0,
        num_agents=100,
    )

    action = PlannerAction(
        new_tax_rates=[0.12, 0.22, 0.32, 0.37],
        new_brackets=[32000, 72000, 155000],
        reasoning="Slight increase to improve SWF",
    )

    outcome = PlannerOutcome(
        next_swf=0.55,
        swf_change=0.03,
        next_gini=0.33,
        gini_change=-0.02,
        next_total_labor=41000,
        labor_change=1000,
        next_total_income=5200000,
        income_change=200000,
        income_by_bracket_next=[25, 42, 23, 10],
        bracket_mobility=0.1,
    )

    trajectory = PlannerTrajectory(
        observation=obs,
        action=action,
        outcome=outcome,
        simulation_id="test_sim",
        model_name="test_model",
        scenario="bounded",
        seed=42,
        quality_score=0.8,
    )

    # Convert to rollout format
    rollout = trajectory_to_rl_rollout(trajectory)

    assert rollout["observation"]["tax_year"] == 1
    assert rollout["action"]["tax_rates"] == [0.12, 0.22, 0.32, 0.37]
    assert "reward" in rollout
    assert rollout["next_observation"]["tax_year"] == 2

    print("✓ Trajectory to rollout conversion test passed")


def test_mock_training_step():
    """Test training step with mock gradients."""
    from llm_economist.training.rl_trainer import RLConfig, RolloutBatch, Rollout

    # Create mock rollouts
    rollouts = []
    for i in range(8):
        rollouts.append(Rollout(
            observation={"tax_year": i},
            action={"tax_rates": [0.2]},
            action_log_prob=-2.0,
            reward=np.random.randn() * 0.1,
            next_observation={"tax_year": i + 1},
            rollout_id=f"mock_{i}",
            worker_id="test",
            timestamp=0.0,
        ))

    batch = RolloutBatch.from_rollouts(rollouts, use_group_baseline=True, device="cpu")

    # Mock policy gradient computation
    log_probs = torch.randn(8, requires_grad=True)
    pg_loss = -(log_probs * batch.advantages).mean()

    # Check gradient flows
    pg_loss.backward()
    assert log_probs.grad is not None, "Gradients should flow through"

    print("✓ Mock training step test passed")


def run_all_tests():
    """Run all local tests."""
    print("\n" + "="*50)
    print("Running Local RL Trainer Tests")
    print("="*50 + "\n")

    test_rollout_dataclass()
    test_rollout_batch()
    test_rl_config()
    test_reward_computation()
    test_trajectory_to_rollout_conversion()
    test_mock_training_step()

    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50 + "\n")


if __name__ == "__main__":
    run_all_tests()
