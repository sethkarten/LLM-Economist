"""
Dataset formatting for finetuning small planner models.

Converts collected planner trajectories into formats suitable for:
- Supervised Fine-Tuning (SFT)
- Direct Preference Optimization (DPO)

Target models: Qwen3-4B, Phi-4, Gemma3-4B, or similar ~3-4B parameter models.
"""

import json
import os
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .data_collector import PlannerTrajectory, PlannerObservation, PlannerAction


# Prompt template for planner
PLANNER_SYSTEM_PROMPT = """You are an AI tax policy planner in an economic simulation.
Your goal is to set tax rates that maximize social welfare while maintaining economic productivity.

You observe:
- Current economic state (incomes, labor, inequality)
- Historical trends in social welfare and Gini coefficient
- Current tax policy

You must choose:
- New marginal tax rates for each income bracket
- Adjustments to bracket thresholds (optional)

Respond with a JSON object containing your tax policy decision."""

PLANNER_USER_TEMPLATE = """## Economic State (Tax Year {tax_year})

**Aggregate Metrics:**
- Total Income: ${total_income:,.0f}
- Average Income: ${avg_income:,.0f}
- Median Income: ${median_income:,.0f}
- Income Gini: {income_gini:.3f}
- Total Labor Supply: {total_labor:,.0f} hours
- Number of Agents: {num_agents}

**Current Tax Policy:**
{current_tax_policy}

**Historical Trends (last {history_len} tax years):**
- SWF: {swf_trend}
- Gini: {gini_trend}
- Labor: {labor_trend}

**Income Distribution:**
{income_distribution}

Based on this economic state, determine the optimal tax policy for the next tax year.
Respond with JSON: {{"tax_rates": [...], "brackets": [...], "reasoning": "..."}}"""


@dataclass
class SFTExample:
    """Single example for supervised fine-tuning."""
    system: str
    user: str
    assistant: str
    quality_score: float = 0.0

    def to_messages(self) -> List[Dict[str, str]]:
        """Convert to chat message format."""
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
            {"role": "assistant", "content": self.assistant},
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "messages": self.to_messages(),
            "quality_score": self.quality_score,
        }


@dataclass
class DPOExample:
    """Single example for Direct Preference Optimization."""
    system: str
    user: str
    chosen: str  # Better response
    rejected: str  # Worse response
    chosen_score: float = 0.0
    rejected_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.user},
            ],
            "chosen": {"role": "assistant", "content": self.chosen},
            "rejected": {"role": "assistant", "content": self.rejected},
            "chosen_score": self.chosen_score,
            "rejected_score": self.rejected_score,
        }


class PlannerDataset:
    """
    Dataset class for planner finetuning.

    Converts PlannerTrajectory objects into SFT or DPO training examples.
    """

    def __init__(
        self,
        trajectories: List[PlannerTrajectory],
        system_prompt: str = PLANNER_SYSTEM_PROMPT,
        user_template: str = PLANNER_USER_TEMPLATE,
    ):
        self.trajectories = trajectories
        self.system_prompt = system_prompt
        self.user_template = user_template

    @classmethod
    def from_files(cls, filepaths: List[str]) -> "PlannerDataset":
        """Load trajectories from multiple JSON files."""
        from .data_collector import SimulationDataCollector

        all_trajectories = []
        for filepath in filepaths:
            collector = SimulationDataCollector.load(filepath)
            all_trajectories.extend(collector.trajectories)

        return cls(all_trajectories)

    @classmethod
    def from_directory(cls, directory: str, pattern: str = "*.json") -> "PlannerDataset":
        """Load all trajectory files from a directory."""
        import glob
        filepaths = glob.glob(os.path.join(directory, pattern))
        return cls.from_files(filepaths)

    def _format_observation(self, obs: PlannerObservation) -> str:
        """Format observation into user prompt."""
        # Format current tax policy
        tax_policy_lines = []
        for i, (rate, bracket) in enumerate(zip(obs.current_tax_rates, obs.current_brackets)):
            if i == 0:
                tax_policy_lines.append(f"  - $0 - ${bracket:,.0f}: {rate*100:.1f}%")
            elif i < len(obs.current_brackets):
                prev_bracket = obs.current_brackets[i-1]
                tax_policy_lines.append(f"  - ${prev_bracket:,.0f} - ${bracket:,.0f}: {rate*100:.1f}%")
        if obs.current_tax_rates:
            last_bracket = obs.current_brackets[-1] if obs.current_brackets else 0
            last_rate = obs.current_tax_rates[-1]
            tax_policy_lines.append(f"  - ${last_bracket:,.0f}+: {last_rate*100:.1f}%")

        current_tax_policy = "\n".join(tax_policy_lines) if tax_policy_lines else "No policy set"

        # Format trends
        def format_trend(values: List[float]) -> str:
            if not values:
                return "N/A"
            if len(values) == 1:
                return f"{values[0]:.3f}"
            trend = "↑" if values[-1] > values[0] else "↓" if values[-1] < values[0] else "→"
            return f"{values[-1]:.3f} ({trend} from {values[0]:.3f})"

        swf_trend = format_trend(obs.historical_swf)
        gini_trend = format_trend(obs.historical_gini)
        labor_trend = format_trend(obs.historical_labor)

        # Format income distribution
        dist_lines = []
        brackets = obs.bracket_thresholds + [float('inf')]
        for i, count in enumerate(obs.income_by_bracket):
            lower = brackets[i-1] if i > 0 else 0
            upper = brackets[i] if i < len(brackets) else "∞"
            if upper == float('inf'):
                upper = "∞"
            pct = (count / obs.num_agents * 100) if obs.num_agents > 0 else 0
            dist_lines.append(f"  - ${lower:,.0f}-{upper}: {int(count)} agents ({pct:.1f}%)")

        income_distribution = "\n".join(dist_lines) if dist_lines else "N/A"

        return self.user_template.format(
            tax_year=obs.tax_year,
            total_income=obs.total_income,
            avg_income=obs.avg_income,
            median_income=obs.median_income,
            income_gini=obs.income_gini,
            total_labor=obs.total_labor_supply,
            num_agents=obs.num_agents,
            current_tax_policy=current_tax_policy,
            history_len=len(obs.historical_swf),
            swf_trend=swf_trend,
            gini_trend=gini_trend,
            labor_trend=labor_trend,
            income_distribution=income_distribution,
        )

    def _format_action(self, action: PlannerAction) -> str:
        """Format action into assistant response."""
        response = {
            "tax_rates": action.new_tax_rates,
            "brackets": action.new_brackets,
        }
        if action.reasoning:
            response["reasoning"] = action.reasoning

        return json.dumps(response, indent=2)

    def create_sft_examples(
        self,
        min_quality: float = 0.3,
        max_examples: Optional[int] = None,
    ) -> List[SFTExample]:
        """
        Create SFT examples from trajectories.

        Args:
            min_quality: Minimum quality score to include
            max_examples: Maximum number of examples (None for all)

        Returns:
            List of SFTExample objects
        """
        # Filter by quality
        filtered = [t for t in self.trajectories if t.quality_score >= min_quality]

        # Sort by quality (best first)
        filtered.sort(key=lambda t: t.quality_score, reverse=True)

        if max_examples:
            filtered = filtered[:max_examples]

        examples = []
        for traj in filtered:
            user_prompt = self._format_observation(traj.observation)
            assistant_response = self._format_action(traj.action)

            examples.append(SFTExample(
                system=self.system_prompt,
                user=user_prompt,
                assistant=assistant_response,
                quality_score=traj.quality_score,
            ))

        return examples

    def create_dpo_pairs(
        self,
        min_score_diff: float = 0.1,
        max_pairs: Optional[int] = None,
    ) -> List[DPOExample]:
        """
        Create DPO preference pairs from trajectories.

        Pairs trajectories with similar observations but different outcomes
        to create preference learning examples.

        Args:
            min_score_diff: Minimum quality score difference for a valid pair
            max_pairs: Maximum number of pairs (None for all)

        Returns:
            List of DPOExample objects
        """
        # Group trajectories by similar observations
        # Use tax_year and approximate income level as grouping key
        groups: Dict[Tuple[int, int], List[PlannerTrajectory]] = {}

        for traj in self.trajectories:
            # Create grouping key: (tax_year, income_bucket)
            income_bucket = int(traj.observation.avg_income / 10000)  # $10k buckets
            key = (traj.observation.tax_year, income_bucket)

            if key not in groups:
                groups[key] = []
            groups[key].append(traj)

        # Create pairs within each group
        pairs = []
        for key, group_trajs in groups.items():
            if len(group_trajs) < 2:
                continue

            # Sort by quality score
            group_trajs.sort(key=lambda t: t.quality_score, reverse=True)

            # Create pairs: each high-quality with each low-quality
            for i, better in enumerate(group_trajs):
                for worse in group_trajs[i+1:]:
                    score_diff = better.quality_score - worse.quality_score
                    if score_diff >= min_score_diff:
                        user_prompt = self._format_observation(better.observation)
                        chosen = self._format_action(better.action)
                        rejected = self._format_action(worse.action)

                        pairs.append(DPOExample(
                            system=self.system_prompt,
                            user=user_prompt,
                            chosen=chosen,
                            rejected=rejected,
                            chosen_score=better.quality_score,
                            rejected_score=worse.quality_score,
                        ))

        # Sort by score difference (most informative first)
        pairs.sort(key=lambda p: p.chosen_score - p.rejected_score, reverse=True)

        if max_pairs:
            pairs = pairs[:max_pairs]

        return pairs

    def save_sft_dataset(
        self,
        output_path: str,
        format: str = "jsonl",
        min_quality: float = 0.3,
        train_ratio: float = 0.9,
    ) -> Tuple[str, str]:
        """
        Save SFT dataset to files.

        Args:
            output_path: Output directory or file path
            format: Output format ('jsonl' or 'json')
            min_quality: Minimum quality score
            train_ratio: Fraction for training set

        Returns:
            Tuple of (train_path, eval_path)
        """
        examples = self.create_sft_examples(min_quality=min_quality)

        # Shuffle and split
        random.shuffle(examples)
        split_idx = int(len(examples) * train_ratio)
        train_examples = examples[:split_idx]
        eval_examples = examples[split_idx:]

        # Ensure output directory exists
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_path = output_dir / f"train.{format}"
        eval_path = output_dir / f"eval.{format}"

        # Write files
        if format == "jsonl":
            with open(train_path, "w") as f:
                for ex in train_examples:
                    f.write(json.dumps(ex.to_dict()) + "\n")

            with open(eval_path, "w") as f:
                for ex in eval_examples:
                    f.write(json.dumps(ex.to_dict()) + "\n")
        else:
            with open(train_path, "w") as f:
                json.dump([ex.to_dict() for ex in train_examples], f, indent=2)

            with open(eval_path, "w") as f:
                json.dump([ex.to_dict() for ex in eval_examples], f, indent=2)

        return str(train_path), str(eval_path)

    def save_dpo_dataset(
        self,
        output_path: str,
        format: str = "jsonl",
        min_score_diff: float = 0.1,
        train_ratio: float = 0.9,
    ) -> Tuple[str, str]:
        """
        Save DPO dataset to files.

        Args:
            output_path: Output directory
            format: Output format ('jsonl' or 'json')
            min_score_diff: Minimum score difference for pairs
            train_ratio: Fraction for training set

        Returns:
            Tuple of (train_path, eval_path)
        """
        pairs = self.create_dpo_pairs(min_score_diff=min_score_diff)

        # Shuffle and split
        random.shuffle(pairs)
        split_idx = int(len(pairs) * train_ratio)
        train_pairs = pairs[:split_idx]
        eval_pairs = pairs[split_idx:]

        # Ensure output directory exists
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_path = output_dir / f"dpo_train.{format}"
        eval_path = output_dir / f"dpo_eval.{format}"

        # Write files
        if format == "jsonl":
            with open(train_path, "w") as f:
                for pair in train_pairs:
                    f.write(json.dumps(pair.to_dict()) + "\n")

            with open(eval_path, "w") as f:
                for pair in eval_pairs:
                    f.write(json.dumps(pair.to_dict()) + "\n")
        else:
            with open(train_path, "w") as f:
                json.dump([pair.to_dict() for pair in train_pairs], f, indent=2)

            with open(eval_path, "w") as f:
                json.dump([pair.to_dict() for pair in eval_pairs], f, indent=2)

        return str(train_path), str(eval_path)


def create_sft_dataset(
    trajectory_dir: str,
    output_dir: str,
    min_quality: float = 0.3,
    train_ratio: float = 0.9,
) -> Tuple[str, str]:
    """
    Convenience function to create SFT dataset from trajectory files.

    Args:
        trajectory_dir: Directory containing trajectory JSON files
        output_dir: Output directory for dataset files
        min_quality: Minimum quality score to include
        train_ratio: Train/eval split ratio

    Returns:
        Tuple of (train_path, eval_path)
    """
    dataset = PlannerDataset.from_directory(trajectory_dir)
    return dataset.save_sft_dataset(
        output_dir,
        min_quality=min_quality,
        train_ratio=train_ratio,
    )


def create_dpo_dataset(
    trajectory_dir: str,
    output_dir: str,
    min_score_diff: float = 0.1,
    train_ratio: float = 0.9,
) -> Tuple[str, str]:
    """
    Convenience function to create DPO dataset from trajectory files.

    Args:
        trajectory_dir: Directory containing trajectory JSON files
        output_dir: Output directory for dataset files
        min_score_diff: Minimum quality score difference for pairs
        train_ratio: Train/eval split ratio

    Returns:
        Tuple of (train_path, eval_path)
    """
    dataset = PlannerDataset.from_directory(trajectory_dir)
    return dataset.save_dpo_dataset(
        output_dir,
        min_score_diff=min_score_diff,
        train_ratio=train_ratio,
    )
