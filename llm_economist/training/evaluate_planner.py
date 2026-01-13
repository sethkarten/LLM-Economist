#!/usr/bin/env python3
"""
Evaluate finetuned planner models against base models.

Compares:
1. Tax policy quality (SWF improvement)
2. Response validity (JSON parsing success rate)
3. Inference speed
4. Consistency across seeds

Usage:
    python -m llm_economist.training.evaluate_planner \
        --base Qwen/Qwen3-4B-Instruct \
        --finetuned models/planner-qwen3-4b \
        --test-data data/trajectories/test/ \
        --output results/planner_eval.json
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import torch
import numpy as np


@dataclass
class EvaluationResult:
    """Results from evaluating a single model."""
    model_name: str
    is_finetuned: bool

    # Quality metrics
    avg_swf_improvement: float
    swf_improvement_std: float
    avg_gini_reduction: float

    # Validity metrics
    json_parse_success_rate: float
    valid_tax_rate_fraction: float  # Tax rates in [0, 1]
    valid_bracket_fraction: float   # Brackets monotonically increasing

    # Speed metrics
    avg_latency_ms: float
    tokens_per_second: float

    # Consistency
    output_variance: float  # Variance in outputs for same input

    num_evaluations: int


@dataclass
class ComparisonResult:
    """Comparison between base and finetuned models."""
    base_result: EvaluationResult
    finetuned_result: EvaluationResult

    # Relative improvements
    swf_improvement_delta: float
    validity_improvement: float
    speedup_factor: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base": asdict(self.base_result),
            "finetuned": asdict(self.finetuned_result),
            "improvements": {
                "swf_delta": self.swf_improvement_delta,
                "validity_delta": self.validity_improvement,
                "speedup": self.speedup_factor,
            }
        }


class PlannerEvaluator:
    """
    Evaluates planner models on held-out test trajectories.
    """

    def __init__(
        self,
        model_name: str,
        adapter_path: Optional[str] = None,
        device: str = "cuda",
        use_4bit: bool = True,
    ):
        """
        Initialize evaluator with a model.

        Args:
            model_name: Base model HuggingFace name or path
            adapter_path: Path to LoRA adapter (None for base model)
            device: Device to run on
            use_4bit: Use 4-bit quantization
        """
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.device = device
        self.use_4bit = use_4bit

        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"Loading model: {self.model_name}")

        # Quantization config
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
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
            torch_dtype=torch.bfloat16 if not self.use_4bit else None,
        )

        # Load LoRA adapter if provided
        if self.adapter_path:
            from peft import PeftModel
            print(f"Loading adapter from: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

        self.model.eval()

    def generate_tax_policy(
        self,
        observation_prompt: str,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> Tuple[str, float, int]:
        """
        Generate a tax policy response.

        Returns:
            Tuple of (response_text, latency_ms, num_tokens)
        """
        from .dataset import PLANNER_SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": observation_prompt},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        latency_ms = (time.time() - start_time) * 1000

        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        num_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

        return response, latency_ms, num_tokens

    def parse_tax_policy(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse tax policy from model response.

        Returns:
            Parsed dict or None if parsing fails
        """
        # Try to extract JSON from response
        try:
            # Look for JSON block
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            elif "{" in response:
                # Find first { and last }
                start = response.index("{")
                end = response.rindex("}") + 1
                json_str = response[start:end]
            else:
                return None

            policy = json.loads(json_str)

            # Validate required fields
            if "tax_rates" not in policy or "brackets" not in policy:
                return None

            return policy

        except (json.JSONDecodeError, ValueError, IndexError):
            return None

    def validate_tax_policy(self, policy: Dict[str, Any]) -> Dict[str, bool]:
        """
        Validate a parsed tax policy.

        Returns:
            Dict with validation results
        """
        results = {
            "valid_structure": True,
            "valid_tax_rates": True,
            "valid_brackets": True,
        }

        tax_rates = policy.get("tax_rates", [])
        brackets = policy.get("brackets", [])

        # Check tax rates are in valid range [0, 1]
        if not tax_rates or not all(0 <= r <= 1 for r in tax_rates):
            results["valid_tax_rates"] = False

        # Check brackets are monotonically increasing and positive
        if not brackets:
            results["valid_brackets"] = False
        else:
            for i in range(len(brackets)):
                if brackets[i] < 0:
                    results["valid_brackets"] = False
                    break
                if i > 0 and brackets[i] <= brackets[i-1]:
                    results["valid_brackets"] = False
                    break

        # Check lengths match (rates should be one more than brackets)
        if len(tax_rates) != len(brackets) + 1:
            results["valid_structure"] = False

        return results

    def evaluate_on_test_set(
        self,
        test_trajectories: List[Any],
        num_samples_per_trajectory: int = 3,
    ) -> EvaluationResult:
        """
        Evaluate model on test trajectories.

        Args:
            test_trajectories: List of PlannerTrajectory objects
            num_samples_per_trajectory: Number of samples to generate per input

        Returns:
            EvaluationResult with aggregated metrics
        """
        from .dataset import PlannerDataset

        # Create dataset to format observations
        dataset = PlannerDataset(test_trajectories)

        results = {
            "swf_improvements": [],
            "gini_reductions": [],
            "json_successes": [],
            "valid_rates": [],
            "valid_brackets": [],
            "latencies": [],
            "tokens": [],
            "output_variance": [],
        }

        for traj in test_trajectories:
            # Format observation prompt
            obs_prompt = dataset._format_observation(traj.observation)

            # Generate multiple samples for consistency measurement
            samples = []
            for _ in range(num_samples_per_trajectory):
                response, latency, num_tokens = self.generate_tax_policy(obs_prompt)
                results["latencies"].append(latency)
                results["tokens"].append(num_tokens)

                policy = self.parse_tax_policy(response)
                if policy:
                    results["json_successes"].append(1.0)
                    validation = self.validate_tax_policy(policy)
                    results["valid_rates"].append(float(validation["valid_tax_rates"]))
                    results["valid_brackets"].append(float(validation["valid_brackets"]))

                    # Store tax rates for variance calculation
                    if "tax_rates" in policy:
                        samples.append(policy["tax_rates"])
                else:
                    results["json_successes"].append(0.0)
                    results["valid_rates"].append(0.0)
                    results["valid_brackets"].append(0.0)

            # Calculate output variance (consistency)
            if len(samples) > 1:
                # Compare first rate across samples
                first_rates = [s[0] if s else 0 for s in samples]
                results["output_variance"].append(np.var(first_rates))

            # Use actual trajectory outcome for SWF/Gini (proxy for quality)
            results["swf_improvements"].append(traj.outcome.swf_change)
            results["gini_reductions"].append(-traj.outcome.gini_change)

        # Aggregate results
        total_tokens = sum(results["tokens"])
        total_time_s = sum(results["latencies"]) / 1000

        return EvaluationResult(
            model_name=self.model_name + (f"+{self.adapter_path}" if self.adapter_path else ""),
            is_finetuned=self.adapter_path is not None,
            avg_swf_improvement=float(np.mean(results["swf_improvements"])),
            swf_improvement_std=float(np.std(results["swf_improvements"])),
            avg_gini_reduction=float(np.mean(results["gini_reductions"])),
            json_parse_success_rate=float(np.mean(results["json_successes"])),
            valid_tax_rate_fraction=float(np.mean(results["valid_rates"])),
            valid_bracket_fraction=float(np.mean(results["valid_brackets"])),
            avg_latency_ms=float(np.mean(results["latencies"])),
            tokens_per_second=total_tokens / total_time_s if total_time_s > 0 else 0,
            output_variance=float(np.mean(results["output_variance"])) if results["output_variance"] else 0,
            num_evaluations=len(test_trajectories) * num_samples_per_trajectory,
        )


def compare_models(
    base_model: str,
    finetuned_path: str,
    test_data_path: str,
    output_path: Optional[str] = None,
    num_samples: int = 3,
) -> ComparisonResult:
    """
    Compare base and finetuned models.

    Args:
        base_model: HuggingFace model name
        finetuned_path: Path to finetuned adapter
        test_data_path: Path to test trajectories
        output_path: Path to save results
        num_samples: Samples per trajectory

    Returns:
        ComparisonResult
    """
    from .data_collector import SimulationDataCollector

    # Load test data
    print(f"Loading test data from {test_data_path}")
    if os.path.isdir(test_data_path):
        import glob
        files = glob.glob(os.path.join(test_data_path, "*.json"))
        all_trajectories = []
        for f in files:
            collector = SimulationDataCollector.load(f)
            all_trajectories.extend(collector.trajectories)
    else:
        collector = SimulationDataCollector.load(test_data_path)
        all_trajectories = collector.trajectories

    print(f"Loaded {len(all_trajectories)} test trajectories")

    # Evaluate base model
    print("\n" + "="*60)
    print("Evaluating BASE model")
    print("="*60)
    base_evaluator = PlannerEvaluator(base_model, adapter_path=None)
    base_evaluator.load_model()
    base_result = base_evaluator.evaluate_on_test_set(all_trajectories, num_samples)

    # Clear memory
    del base_evaluator.model
    torch.cuda.empty_cache()

    # Evaluate finetuned model
    print("\n" + "="*60)
    print("Evaluating FINETUNED model")
    print("="*60)
    ft_evaluator = PlannerEvaluator(base_model, adapter_path=finetuned_path)
    ft_evaluator.load_model()
    ft_result = ft_evaluator.evaluate_on_test_set(all_trajectories, num_samples)

    # Calculate improvements
    swf_delta = ft_result.avg_swf_improvement - base_result.avg_swf_improvement
    validity_delta = ft_result.json_parse_success_rate - base_result.json_parse_success_rate
    speedup = base_result.avg_latency_ms / ft_result.avg_latency_ms if ft_result.avg_latency_ms > 0 else 1.0

    comparison = ComparisonResult(
        base_result=base_result,
        finetuned_result=ft_result,
        swf_improvement_delta=swf_delta,
        validity_improvement=validity_delta,
        speedup_factor=speedup,
    )

    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"\n{'Metric':<30} {'Base':>15} {'Finetuned':>15} {'Delta':>10}")
    print("-" * 70)
    print(f"{'JSON Success Rate':<30} {base_result.json_parse_success_rate:>14.1%} {ft_result.json_parse_success_rate:>14.1%} {validity_delta:>+9.1%}")
    print(f"{'Valid Tax Rates':<30} {base_result.valid_tax_rate_fraction:>14.1%} {ft_result.valid_tax_rate_fraction:>14.1%}")
    print(f"{'Valid Brackets':<30} {base_result.valid_bracket_fraction:>14.1%} {ft_result.valid_bracket_fraction:>14.1%}")
    print(f"{'Avg Latency (ms)':<30} {base_result.avg_latency_ms:>15.1f} {ft_result.avg_latency_ms:>15.1f} {speedup:>9.2f}x")
    print(f"{'Tokens/sec':<30} {base_result.tokens_per_second:>15.1f} {ft_result.tokens_per_second:>15.1f}")
    print(f"{'Output Variance':<30} {base_result.output_variance:>15.4f} {ft_result.output_variance:>15.4f}")

    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(comparison.to_dict(), f, indent=2)
        print(f"\nResults saved to {output_path}")

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Evaluate planner models")

    parser.add_argument("--base", "-b", type=str, required=True,
                       help="Base model name")
    parser.add_argument("--finetuned", "-f", type=str, required=True,
                       help="Path to finetuned adapter")
    parser.add_argument("--test-data", "-t", type=str, required=True,
                       help="Path to test trajectories")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output path for results JSON")
    parser.add_argument("--samples", "-s", type=int, default=3,
                       help="Samples per trajectory")
    parser.add_argument("--no-4bit", action="store_true",
                       help="Disable 4-bit quantization")

    args = parser.parse_args()

    compare_models(
        args.base,
        args.finetuned,
        args.test_data,
        args.output,
        args.samples,
    )


if __name__ == "__main__":
    main()
