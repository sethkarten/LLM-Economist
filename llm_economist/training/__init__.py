"""
Training module for finetuning small planners on LLM Economist data.

Key components:
- Data collection from simulation runs
- Dataset formatting for SFT/DPO
- LoRA finetuning scripts
- Evaluation against base models

Usage:
    # Collect trajectories during simulation
    from llm_economist.training import SimulationDataCollector
    collector = SimulationDataCollector("exp_001", "qwen3-30b-a3b")

    # At each tax year:
    collector.record_observation(timestep, agent_states, tax_rates, brackets, swf)
    collector.record_action(new_rates, new_brackets, reasoning)
    collector.record_outcome(next_agent_states, next_swf, prev_swf)

    # Save trajectories
    collector.save("data/trajectories/")

    # Create training dataset
    from llm_economist.training import create_sft_dataset
    train_path, eval_path = create_sft_dataset("data/trajectories/", "data/sft/")

    # Finetune (from command line)
    # python -m llm_economist.training.finetune_planner \\
    #     --model qwen3-4b --data data/sft/ --output models/planner
"""

from .data_collector import (
    SimulationDataCollector,
    PlannerTrajectory,
    PlannerObservation,
    PlannerAction,
    PlannerOutcome,
    merge_trajectory_files,
)
from .dataset import (
    PlannerDataset,
    SFTExample,
    DPOExample,
    create_sft_dataset,
    create_dpo_dataset,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_TEMPLATE,
)
from .data_collector import (
    compute_rl_reward,
    trajectory_to_rl_rollout,
    convert_trajectories_to_rl_dataset,
)

__all__ = [
    # Data collection
    'SimulationDataCollector',
    'PlannerTrajectory',
    'PlannerObservation',
    'PlannerAction',
    'PlannerOutcome',
    'merge_trajectory_files',
    # Dataset creation
    'PlannerDataset',
    'SFTExample',
    'DPOExample',
    'create_sft_dataset',
    'create_dpo_dataset',
    # RL reward and conversion
    'compute_rl_reward',
    'trajectory_to_rl_rollout',
    'convert_trajectories_to_rl_dataset',
    # Prompts
    'PLANNER_SYSTEM_PROMPT',
    'PLANNER_USER_TEMPLATE',
]
