"""
Data collector for planner trajectories from LLM Economist simulations.

Collects (state, action, reward) tuples from the planner's tax policy decisions
to enable:
- Supervised finetuning (SFT)
- Direct Preference Optimization (DPO)
- Reinforcement learning (REINFORCE++, GRPO)

The collector integrates with both the simulation loop and the RL training pipeline.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np


@dataclass
class PlannerObservation:
    """Observation seen by the planner at decision time."""
    timestep: int
    tax_year: int

    # Aggregate economic state
    total_labor_supply: float
    total_income: float
    avg_income: float
    median_income: float
    income_gini: float

    # Income distribution by bracket
    income_by_bracket: List[float]  # [bracket_0_count, bracket_1_count, ...]
    bracket_thresholds: List[float]

    # Current tax policy
    current_tax_rates: List[float]
    current_brackets: List[float]

    # Historical metrics (last N tax years)
    historical_swf: List[float]
    historical_gini: List[float]
    historical_labor: List[float]

    # Agent behavior summary
    avg_labor_hours: float
    labor_variance: float
    num_agents: int


@dataclass
class PlannerAction:
    """Tax policy action taken by the planner."""
    new_tax_rates: List[float]
    new_brackets: List[float]
    reasoning: Optional[str] = None


@dataclass
class PlannerOutcome:
    """Outcome after planner's action (observed next tax year)."""
    next_swf: float
    swf_change: float
    next_gini: float
    gini_change: float
    next_total_labor: float
    labor_change: float
    next_total_income: float
    income_change: float

    # Distributional outcomes
    income_by_bracket_next: List[float]
    bracket_mobility: float  # Fraction of agents who changed brackets


@dataclass
class PlannerTrajectory:
    """A single (observation, action, outcome) tuple for training."""
    observation: PlannerObservation
    action: PlannerAction
    outcome: PlannerOutcome

    # Metadata
    simulation_id: str
    model_name: str
    scenario: str
    seed: int

    # Quality score for filtering
    quality_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "observation": asdict(self.observation),
            "action": asdict(self.action),
            "outcome": asdict(self.outcome),
            "simulation_id": self.simulation_id,
            "model_name": self.model_name,
            "scenario": self.scenario,
            "seed": self.seed,
            "quality_score": self.quality_score,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlannerTrajectory":
        """Load from dictionary."""
        return cls(
            observation=PlannerObservation(**d["observation"]),
            action=PlannerAction(**d["action"]),
            outcome=PlannerOutcome(**d["outcome"]),
            simulation_id=d["simulation_id"],
            model_name=d["model_name"],
            scenario=d["scenario"],
            seed=d["seed"],
            quality_score=d.get("quality_score", 0.0),
        )


class SimulationDataCollector:
    """
    Collects planner trajectories during simulation runs.

    Usage:
        collector = SimulationDataCollector(
            simulation_id="exp_001",
            model_name="qwen3-30b-a3b",
            scenario="bounded",
            seed=42
        )

        # During simulation, at each tax year boundary:
        collector.record_observation(state)
        collector.record_action(tax_policy, reasoning)

        # After observing next tax year:
        collector.record_outcome(next_state)

        # Save all trajectories
        collector.save("data/trajectories/")
    """

    def __init__(
        self,
        simulation_id: str,
        model_name: str,
        scenario: str = "bounded",
        seed: int = 42,
        tax_year_length: int = 128,
        history_length: int = 5,
    ):
        self.simulation_id = simulation_id
        self.model_name = model_name
        self.scenario = scenario
        self.seed = seed
        self.tax_year_length = tax_year_length
        self.history_length = history_length

        self.trajectories: List[PlannerTrajectory] = []
        self._current_observation: Optional[PlannerObservation] = None
        self._current_action: Optional[PlannerAction] = None

        # Historical metrics buffer
        self._swf_history: List[float] = []
        self._gini_history: List[float] = []
        self._labor_history: List[float] = []

    def record_observation(
        self,
        timestep: int,
        agent_states: List[Dict[str, Any]],
        current_tax_rates: List[float],
        current_brackets: List[float],
        swf: float,
    ) -> PlannerObservation:
        """
        Record the planner's observation at a tax year boundary.

        Args:
            timestep: Current simulation timestep
            agent_states: List of agent state dicts with 'income', 'labor_hours', etc.
            current_tax_rates: Current marginal tax rates
            current_brackets: Current income bracket thresholds
            swf: Current social welfare function value

        Returns:
            PlannerObservation object
        """
        tax_year = timestep // self.tax_year_length

        # Extract income and labor data
        incomes = np.array([s.get("income", 0) for s in agent_states])
        labor_hours = np.array([s.get("labor_hours", 0) for s in agent_states])

        # Calculate aggregate statistics
        total_labor = float(np.sum(labor_hours))
        total_income = float(np.sum(incomes))
        avg_income = float(np.mean(incomes)) if len(incomes) > 0 else 0.0
        median_income = float(np.median(incomes)) if len(incomes) > 0 else 0.0

        # Calculate Gini coefficient
        income_gini = self._calculate_gini(incomes)

        # Income by bracket
        income_by_bracket = self._count_by_bracket(incomes, current_brackets)

        # Update history
        self._swf_history.append(swf)
        self._gini_history.append(income_gini)
        self._labor_history.append(total_labor)

        # Trim history
        self._swf_history = self._swf_history[-self.history_length:]
        self._gini_history = self._gini_history[-self.history_length:]
        self._labor_history = self._labor_history[-self.history_length:]

        observation = PlannerObservation(
            timestep=timestep,
            tax_year=tax_year,
            total_labor_supply=total_labor,
            total_income=total_income,
            avg_income=avg_income,
            median_income=median_income,
            income_gini=income_gini,
            income_by_bracket=income_by_bracket,
            bracket_thresholds=list(current_brackets),
            current_tax_rates=list(current_tax_rates),
            current_brackets=list(current_brackets),
            historical_swf=list(self._swf_history),
            historical_gini=list(self._gini_history),
            historical_labor=list(self._labor_history),
            avg_labor_hours=float(np.mean(labor_hours)) if len(labor_hours) > 0 else 0.0,
            labor_variance=float(np.var(labor_hours)) if len(labor_hours) > 0 else 0.0,
            num_agents=len(agent_states),
        )

        self._current_observation = observation
        return observation

    def record_action(
        self,
        new_tax_rates: List[float],
        new_brackets: List[float],
        reasoning: Optional[str] = None,
    ) -> PlannerAction:
        """
        Record the planner's tax policy action.

        Args:
            new_tax_rates: New marginal tax rates chosen by planner
            new_brackets: New income bracket thresholds
            reasoning: Optional reasoning from the LLM

        Returns:
            PlannerAction object
        """
        action = PlannerAction(
            new_tax_rates=list(new_tax_rates),
            new_brackets=list(new_brackets),
            reasoning=reasoning,
        )

        self._current_action = action
        return action

    def record_outcome(
        self,
        agent_states: List[Dict[str, Any]],
        next_swf: float,
        prev_swf: float,
    ) -> Optional[PlannerTrajectory]:
        """
        Record the outcome after the planner's action takes effect.

        Call this at the next tax year boundary after record_action().

        Args:
            agent_states: Agent states after one tax year under new policy
            next_swf: SWF value after policy change
            prev_swf: SWF value before policy change

        Returns:
            Complete PlannerTrajectory if observation and action were recorded
        """
        if self._current_observation is None or self._current_action is None:
            return None

        # Extract metrics
        incomes = np.array([s.get("income", 0) for s in agent_states])
        labor_hours = np.array([s.get("labor_hours", 0) for s in agent_states])

        next_gini = self._calculate_gini(incomes)
        next_total_labor = float(np.sum(labor_hours))
        next_total_income = float(np.sum(incomes))

        # Calculate changes
        swf_change = next_swf - prev_swf
        gini_change = next_gini - self._current_observation.income_gini
        labor_change = next_total_labor - self._current_observation.total_labor_supply
        income_change = next_total_income - self._current_observation.total_income

        # Income by bracket (using new brackets)
        income_by_bracket_next = self._count_by_bracket(
            incomes, self._current_action.new_brackets
        )

        # Calculate bracket mobility (simplified)
        bracket_mobility = self._calculate_bracket_mobility(
            self._current_observation.income_by_bracket,
            income_by_bracket_next
        )

        outcome = PlannerOutcome(
            next_swf=next_swf,
            swf_change=swf_change,
            next_gini=next_gini,
            gini_change=gini_change,
            next_total_labor=next_total_labor,
            labor_change=labor_change,
            next_total_income=next_total_income,
            income_change=income_change,
            income_by_bracket_next=income_by_bracket_next,
            bracket_mobility=bracket_mobility,
        )

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            self._current_observation, self._current_action, outcome
        )

        trajectory = PlannerTrajectory(
            observation=self._current_observation,
            action=self._current_action,
            outcome=outcome,
            simulation_id=self.simulation_id,
            model_name=self.model_name,
            scenario=self.scenario,
            seed=self.seed,
            quality_score=quality_score,
        )

        self.trajectories.append(trajectory)

        # Reset current observation/action
        self._current_observation = None
        self._current_action = None

        return trajectory

    def _calculate_gini(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient."""
        if len(values) == 0 or np.sum(values) == 0:
            return 0.0

        sorted_values = np.sort(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        return (2 * np.sum((np.arange(1, n + 1) * sorted_values)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])

    def _count_by_bracket(
        self, incomes: np.ndarray, brackets: List[float]
    ) -> List[float]:
        """Count agents in each income bracket."""
        brackets = sorted(brackets) + [float('inf')]
        counts = []
        for i in range(len(brackets)):
            lower = brackets[i - 1] if i > 0 else 0
            upper = brackets[i]
            count = np.sum((incomes >= lower) & (incomes < upper))
            counts.append(float(count))
        return counts

    def _calculate_bracket_mobility(
        self, prev_counts: List[float], next_counts: List[float]
    ) -> float:
        """Estimate bracket mobility (simplified - total movement)."""
        if len(prev_counts) != len(next_counts):
            return 0.0
        total = sum(prev_counts)
        if total == 0:
            return 0.0
        movement = sum(abs(p - n) for p, n in zip(prev_counts, next_counts))
        return movement / (2 * total)  # Normalized to [0, 1]

    def _calculate_quality_score(
        self,
        observation: PlannerObservation,
        action: PlannerAction,
        outcome: PlannerOutcome,
    ) -> float:
        """
        Calculate quality score for trajectory filtering.

        Higher scores indicate:
        - Positive SWF improvement
        - Gini reduction (more equality)
        - Maintained or increased labor supply
        """
        score = 0.0

        # SWF improvement (main objective)
        if outcome.swf_change > 0:
            score += min(outcome.swf_change / 0.1, 1.0) * 0.5
        else:
            score += max(outcome.swf_change / 0.1, -1.0) * 0.3

        # Gini reduction (equality)
        if outcome.gini_change < 0:
            score += min(-outcome.gini_change / 0.05, 0.5) * 0.3

        # Labor supply maintenance
        if outcome.labor_change >= 0:
            score += 0.2
        else:
            labor_pct_change = outcome.labor_change / max(observation.total_labor_supply, 1)
            score += max(labor_pct_change * 2, -0.2)

        return max(0.0, min(1.0, score))

    def get_trajectories(
        self, min_quality: float = 0.0
    ) -> List[PlannerTrajectory]:
        """Get trajectories filtered by minimum quality score."""
        return [t for t in self.trajectories if t.quality_score >= min_quality]

    def save(self, output_dir: str, filename: Optional[str] = None):
        """Save collected trajectories to JSON file."""
        os.makedirs(output_dir, exist_ok=True)

        if filename is None:
            filename = f"trajectories_{self.simulation_id}_{self.seed}.json"

        filepath = os.path.join(output_dir, filename)

        data = {
            "simulation_id": self.simulation_id,
            "model_name": self.model_name,
            "scenario": self.scenario,
            "seed": self.seed,
            "num_trajectories": len(self.trajectories),
            "trajectories": [t.to_dict() for t in self.trajectories],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return filepath

    @classmethod
    def load(cls, filepath: str) -> "SimulationDataCollector":
        """Load trajectories from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        collector = cls(
            simulation_id=data["simulation_id"],
            model_name=data["model_name"],
            scenario=data.get("scenario", "bounded"),
            seed=data.get("seed", 42),
        )

        collector.trajectories = [
            PlannerTrajectory.from_dict(t) for t in data["trajectories"]
        ]

        return collector


def merge_trajectory_files(
    input_files: List[str],
    output_file: str,
    min_quality: float = 0.0,
) -> int:
    """
    Merge multiple trajectory files into one.

    Args:
        input_files: List of trajectory JSON files
        output_file: Output merged file path
        min_quality: Minimum quality score to include

    Returns:
        Number of trajectories in merged file
    """
    all_trajectories = []

    for filepath in input_files:
        collector = SimulationDataCollector.load(filepath)
        all_trajectories.extend(collector.get_trajectories(min_quality))

    # Create merged output
    merged_data = {
        "merged_from": input_files,
        "num_trajectories": len(all_trajectories),
        "min_quality_filter": min_quality,
        "trajectories": [t.to_dict() for t in all_trajectories],
    }

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(merged_data, f, indent=2)

    return len(all_trajectories)


def compute_rl_reward(
    outcome: PlannerOutcome,
    observation: PlannerObservation,
    swf_weight: float = 1.0,
    gini_weight: float = 0.1,
    labor_weight: float = 0.1,
) -> float:
    """
    Compute RL reward from trajectory outcome.

    The reward is designed to encourage:
    - SWF improvement (primary objective)
    - Gini reduction (equality bonus)
    - Labor supply maintenance (economic stability)

    Args:
        outcome: PlannerOutcome from simulation
        observation: PlannerObservation (for normalization)
        swf_weight: Weight for SWF component
        gini_weight: Weight for Gini component
        labor_weight: Weight for labor component

    Returns:
        Scalar reward value
    """
    # Normalized SWF change (main objective)
    if observation.historical_swf and observation.historical_swf[-1] != 0:
        prev_swf = observation.historical_swf[-1]
        swf_reward = outcome.swf_change / abs(prev_swf)
    else:
        swf_reward = outcome.swf_change

    # Gini reduction bonus (negative change = good)
    gini_reward = -outcome.gini_change

    # Labor stability (penalize large drops)
    if observation.total_labor_supply > 0:
        labor_pct_change = outcome.labor_change / observation.total_labor_supply
        labor_reward = min(labor_pct_change, 0.1)  # Cap positive reward
    else:
        labor_reward = 0.0

    # Combine rewards
    reward = (
        swf_weight * swf_reward
        + gini_weight * gini_reward
        + labor_weight * labor_reward
    )

    return float(reward)


def trajectory_to_rl_rollout(
    trajectory: PlannerTrajectory,
    swf_weight: float = 1.0,
    gini_weight: float = 0.1,
    labor_weight: float = 0.1,
) -> Dict[str, Any]:
    """
    Convert a PlannerTrajectory to RL Rollout format.

    This format is compatible with the REINFORCE++ trainer in rl_trainer.py.

    Args:
        trajectory: PlannerTrajectory object
        swf_weight: Weight for SWF in reward
        gini_weight: Weight for Gini in reward
        labor_weight: Weight for labor in reward

    Returns:
        Dict in Rollout format: {observation, action, reward, ...}
    """
    obs = trajectory.observation

    # Convert observation to dict format expected by RL trainer
    observation_dict = {
        "tax_year": obs.tax_year,
        "total_income": obs.total_income,
        "avg_income": obs.avg_income,
        "median_income": obs.median_income,
        "income_gini": obs.income_gini,
        "total_labor": obs.total_labor_supply,
        "num_agents": obs.num_agents,
        "current_tax_rates": obs.current_tax_rates,
        "current_brackets": obs.current_brackets,
        "historical_swf": obs.historical_swf,
        "historical_gini": obs.historical_gini,
        "historical_labor": obs.historical_labor,
    }

    # Convert action to dict format
    action_dict = {
        "tax_rates": trajectory.action.new_tax_rates,
        "brackets": trajectory.action.new_brackets,
    }
    if trajectory.action.reasoning:
        action_dict["reasoning"] = trajectory.action.reasoning

    # Compute reward
    reward = compute_rl_reward(
        trajectory.outcome,
        trajectory.observation,
        swf_weight=swf_weight,
        gini_weight=gini_weight,
        labor_weight=labor_weight,
    )

    # Next observation (simplified - just increment tax year)
    next_observation_dict = observation_dict.copy()
    next_observation_dict["tax_year"] = obs.tax_year + 1
    next_observation_dict["income_gini"] = trajectory.outcome.next_gini
    next_observation_dict["total_labor"] = trajectory.outcome.next_total_labor
    next_observation_dict["total_income"] = trajectory.outcome.next_total_income

    return {
        "observation": observation_dict,
        "action": action_dict,
        "action_log_prob": 0.0,  # Not available from offline data
        "reward": reward,
        "next_observation": next_observation_dict,
        "rollout_id": f"{trajectory.simulation_id}_{obs.tax_year}",
        "worker_id": "offline",
        "timestamp": 0.0,
    }


def convert_trajectories_to_rl_dataset(
    trajectory_files: List[str],
    output_file: str,
    min_quality: float = 0.0,
    swf_weight: float = 1.0,
    gini_weight: float = 0.1,
    labor_weight: float = 0.1,
) -> int:
    """
    Convert trajectory files to RL dataset format.

    Args:
        trajectory_files: List of trajectory JSON files
        output_file: Output file path
        min_quality: Minimum quality score to include
        swf_weight: Weight for SWF in reward
        gini_weight: Weight for Gini in reward
        labor_weight: Weight for labor in reward

    Returns:
        Number of rollouts in output file
    """
    all_rollouts = []

    for filepath in trajectory_files:
        collector = SimulationDataCollector.load(filepath)

        for traj in collector.get_trajectories(min_quality):
            rollout = trajectory_to_rl_rollout(
                traj,
                swf_weight=swf_weight,
                gini_weight=gini_weight,
                labor_weight=labor_weight,
            )
            all_rollouts.append(rollout)

    # Save as JSONL (one rollout per line)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        for rollout in all_rollouts:
            f.write(json.dumps(rollout) + "\n")

    return len(all_rollouts)
