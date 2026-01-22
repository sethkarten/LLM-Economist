#!/usr/bin/env python3
"""Test what the trained planner actually outputs."""

import os
import sys
import json
from pathlib import Path

# Set up paths
os.environ['HF_HOME'] = '/scratch/gpfs/CHIJ/milkkarten/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/gpfs/CHIJ/milkkarten/huggingface/hub'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    # Load the trained planner
    base_model = "Qwen/Qwen3-4B-Instruct-2507"
    lora_path = Path("/scratch/gpfs/CHIJ/milkkarten/LLM-Economist/models/reinforce_h3_trained/best/planner_lora")

    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    if lora_path.exists():
        print(f"Loading LoRA weights from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    else:
        print("No LoRA weights found, using base model")

    # Create the prompt (same as in training)
    system_prompt = """You are an AI tax policy planner optimizing social welfare in an economic simulation.

Your goal: Set tax rates that achieve HIGHER social welfare than the US federal progressive tax baseline.

Baseline Performance (US 2024 Federal Tax):
- Social Welfare: 195.0
- Gini: 0.305
- Labor: 56.5 hours/week

Set tax rates to maximize welfare above baseline.
Output: {"tax_rates": [rate1, rate2, ...], "brackets": [threshold1, threshold2, ...]}"""

    user_prompt = """Current economic state:
- Mean income: $50,000
- Income std: $25,000
- Mean labor: 40 hours

Set optimal tax rates to beat baseline SWF of 195.0."""

    # Generate response
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print("\n=== Generating 5 sample tax policies ===\n")

    for i in range(5):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"Sample {i+1}:")
        print(f"  Raw output: {response[:300]}...")

        # Try to parse
        try:
            if "{" in response and "}" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                json_str = response[start:end]
                data = json.loads(json_str)
                if "tax_rates" in data:
                    rates = data["tax_rates"]
                    print(f"  Parsed rates: {rates}")

                    # Compare to US federal
                    us_rates = [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]
                    if len(rates) == len(us_rates):
                        diff = sum(abs(r - u) for r, u in zip(rates, us_rates)) / len(rates)
                        print(f"  Avg diff from US federal: {diff:.4f}")
                else:
                    print("  No tax_rates in JSON")
            else:
                print("  No JSON found in response")
        except Exception as e:
            print(f"  Parse error: {e}")
        print()

if __name__ == "__main__":
    main()
