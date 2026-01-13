#!/usr/bin/env python3
"""
LoRA finetuning script for small planner models.

Finetunes a ~3-4B parameter model (Qwen3-4B, Phi-4, Gemma3-4B) on collected
planner trajectories to create an efficient specialist planner.

Usage:
    # SFT training
    python -m llm_economist.training.finetune_planner \
        --model Qwen/Qwen3-4B-Instruct \
        --data data/trajectories/ \
        --output models/planner-qwen3-4b \
        --method sft

    # DPO training (requires SFT checkpoint first)
    python -m llm_economist.training.finetune_planner \
        --model models/planner-qwen3-4b \
        --data data/trajectories/ \
        --output models/planner-qwen3-4b-dpo \
        --method dpo

Requirements:
    pip install transformers peft trl bitsandbytes accelerate datasets
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from datasets import Dataset, load_dataset


# Default training hyperparameters
DEFAULT_SFT_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_seq_length": 2048,
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 100,
    "fp16": True,
    "bf16": False,  # Set to True if supported
}

DEFAULT_DPO_CONFIG = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 5e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_length": 2048,
    "max_prompt_length": 1024,
    "beta": 0.1,  # DPO beta parameter
    "logging_steps": 10,
    "save_steps": 50,
    "eval_steps": 50,
    "fp16": True,
}

# LoRA configuration
DEFAULT_LORA_CONFIG = {
    "r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# Supported base models
SUPPORTED_MODELS = {
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B-Instruct",
    "phi-4": "microsoft/phi-4",
    "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    "gemma3-4b": "google/gemma-3-4b-it",
    "gemma3-1b": "google/gemma-3-1b-it",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "olmo-1b": "allenai/OLMo-1B-0724-Instruct",
}


def get_model_name(model_arg: str) -> str:
    """Resolve model name from shorthand or full name."""
    if model_arg.lower() in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_arg.lower()]
    return model_arg  # Assume it's a full HF name or path


def load_and_prepare_model(
    model_name: str,
    lora_config: Dict[str, Any],
    use_4bit: bool = True,
    device_map: str = "auto",
):
    """
    Load base model with LoRA adapters for training.

    Args:
        model_name: HuggingFace model name or local path
        lora_config: LoRA configuration dictionary
        use_4bit: Whether to use 4-bit quantization
        device_map: Device mapping strategy

    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print(f"Loading model: {model_name}")

    # Quantization config for efficient training
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not use_4bit else None,
    )

    # Prepare for training
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Add LoRA adapters
    peft_config = LoraConfig(**lora_config)
    model = get_peft_model(model, peft_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model, tokenizer


def prepare_sft_dataset(
    data_path: str,
    tokenizer,
    max_seq_length: int = 2048,
) -> tuple:
    """
    Prepare SFT dataset from trajectory files.

    Args:
        data_path: Path to trajectory directory or JSONL file
        tokenizer: HuggingFace tokenizer
        max_seq_length: Maximum sequence length

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    from .dataset import PlannerDataset

    # Check if it's a directory or file
    if os.path.isdir(data_path):
        dataset = PlannerDataset.from_directory(data_path)
    else:
        dataset = PlannerDataset.from_files([data_path])

    # Create SFT examples
    sft_examples = dataset.create_sft_examples(min_quality=0.3)

    if len(sft_examples) == 0:
        raise ValueError(f"No training examples found in {data_path}")

    print(f"Created {len(sft_examples)} SFT examples")

    # Convert to HF dataset format
    def format_example(example):
        messages = example.to_messages()
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text, "quality_score": example.quality_score}

    formatted = [format_example(ex) for ex in sft_examples]

    # Split into train/eval
    split_idx = int(len(formatted) * 0.9)
    train_data = formatted[:split_idx]
    eval_data = formatted[split_idx:]

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    return train_dataset, eval_dataset


def prepare_dpo_dataset(
    data_path: str,
    tokenizer,
    max_length: int = 2048,
) -> tuple:
    """
    Prepare DPO dataset from trajectory files.

    Args:
        data_path: Path to trajectory directory or JSONL file
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    from .dataset import PlannerDataset

    # Load trajectories
    if os.path.isdir(data_path):
        dataset = PlannerDataset.from_directory(data_path)
    else:
        dataset = PlannerDataset.from_files([data_path])

    # Create DPO pairs
    dpo_pairs = dataset.create_dpo_pairs(min_score_diff=0.1)

    if len(dpo_pairs) == 0:
        raise ValueError(f"No DPO pairs found in {data_path}")

    print(f"Created {len(dpo_pairs)} DPO pairs")

    # Convert to HF dataset format
    def format_pair(pair):
        prompt = tokenizer.apply_chat_template(
            pair.to_dict()["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            "prompt": prompt,
            "chosen": pair.chosen,
            "rejected": pair.rejected,
        }

    formatted = [format_pair(pair) for pair in dpo_pairs]

    # Split into train/eval
    split_idx = int(len(formatted) * 0.9)
    train_data = formatted[:split_idx]
    eval_data = formatted[split_idx:]

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    return train_dataset, eval_dataset


def run_sft_training(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
    config: Dict[str, Any],
):
    """Run SFT training loop."""
    from transformers import TrainingArguments
    from trl import SFTTrainer

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        fp16=config["fp16"],
        bf16=config.get("bf16", False),
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        max_seq_length=config["max_seq_length"],
        dataset_text_field="text",
        packing=True,  # Efficient packing of sequences
    )

    print("Starting SFT training...")
    trainer.train()

    # Save final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model saved to {output_dir}")
    return trainer


def run_dpo_training(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
    config: Dict[str, Any],
    ref_model=None,
):
    """Run DPO training loop."""
    from transformers import TrainingArguments
    from trl import DPOTrainer, DPOConfig

    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        max_length=config["max_length"],
        max_prompt_length=config["max_prompt_length"],
        beta=config["beta"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        fp16=config["fp16"],
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="tensorboard",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()

    # Save final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model saved to {output_dir}")
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Finetune small planner model")

    # Model arguments
    parser.add_argument("--model", "-m", type=str, required=True,
                       help=f"Base model (shorthand or HF name). Options: {list(SUPPORTED_MODELS.keys())}")
    parser.add_argument("--data", "-d", type=str, required=True,
                       help="Path to trajectory data (directory or file)")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output directory for finetuned model")

    # Training method
    parser.add_argument("--method", type=str, choices=["sft", "dpo"], default="sft",
                       help="Training method: sft or dpo")

    # Training config overrides
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=64,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128,
                       help="LoRA alpha")

    # Hardware
    parser.add_argument("--no-4bit", action="store_true",
                       help="Disable 4-bit quantization")
    parser.add_argument("--bf16", action="store_true",
                       help="Use bfloat16 instead of fp16")

    # Other
    parser.add_argument("--min-quality", type=float, default=0.3,
                       help="Minimum quality score for training examples")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")

    args = parser.parse_args()

    # Resolve model name
    model_name = get_model_name(args.model)
    print(f"Using model: {model_name}")

    # Setup configs
    if args.method == "sft":
        config = DEFAULT_SFT_CONFIG.copy()
    else:
        config = DEFAULT_DPO_CONFIG.copy()

    # Apply overrides
    if args.epochs:
        config["num_train_epochs"] = args.epochs
    if args.batch_size:
        config["per_device_train_batch_size"] = args.batch_size
        config["per_device_eval_batch_size"] = args.batch_size
    if args.lr:
        config["learning_rate"] = args.lr
    if args.bf16:
        config["bf16"] = True
        config["fp16"] = False

    lora_config = DEFAULT_LORA_CONFIG.copy()
    lora_config["r"] = args.lora_r
    lora_config["lora_alpha"] = args.lora_alpha

    # Load model
    model, tokenizer = load_and_prepare_model(
        model_name,
        lora_config,
        use_4bit=not args.no_4bit,
    )

    # Prepare dataset
    if args.method == "sft":
        train_dataset, eval_dataset = prepare_sft_dataset(
            args.data,
            tokenizer,
            max_seq_length=config["max_seq_length"],
        )

        run_sft_training(
            model,
            tokenizer,
            train_dataset,
            eval_dataset,
            args.output,
            config,
        )
    else:
        train_dataset, eval_dataset = prepare_dpo_dataset(
            args.data,
            tokenizer,
            max_length=config["max_length"],
        )

        run_dpo_training(
            model,
            tokenizer,
            train_dataset,
            eval_dataset,
            args.output,
            config,
        )

    print("\nTraining complete!")
    print(f"Model saved to: {args.output}")
    print("\nTo use the finetuned model:")
    print(f"  from peft import PeftModel")
    print(f"  model = PeftModel.from_pretrained(base_model, '{args.output}')")


if __name__ == "__main__":
    main()
