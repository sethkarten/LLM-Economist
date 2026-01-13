#!/usr/bin/env python3
"""Benchmark vLLM + AWQ models for max throughput on RTX 5090."""
import os
os.environ['HF_HOME'] = '/mnt/storage/models'
os.environ['VLLM_ATTENTION_BACKEND'] = 'TRITON_ATTN'

import time
import gc
import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

# Base models with FP8 dynamic quantization
MODELS = [
    ("Qwen3-8B", "Qwen/Qwen3-8B"),
    ("Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"),
    ("Mistral-7B-v0.3", "mistralai/Mistral-7B-Instruct-v0.3"),
    ("Gemma-3-12B", "google/gemma-3-12b-it"),
    ("OLMo-3-7B", "allenai/OLMo-3-7B-Instruct"),
]

# Batch of test prompts (simulate agent decisions)
TEST_PROMPTS = [
    f"You are worker {i}. Skill={0.3+i*0.07:.2f}, tax=30%. Hours to work (0-10)? Just the number."
    for i in range(50)  # 50 concurrent requests for throughput test
]

def benchmark_model(name, model_id, use_awq=True):
    """Benchmark a single model."""
    print(f"\n{'='*60}")
    print(f"Testing: {name} ({model_id})")
    print(f"{'='*60}")

    try:
        start_load = time.time()

        # Configure based on model type
        kwargs = {
            'model': model_id,
            'dtype': 'float16',
            'enforce_eager': True,  # Blackwell compatibility
            'max_model_len': 4096,
            'gpu_memory_utilization': 0.85,
            'trust_remote_code': True,
            'download_dir': '/mnt/storage/models',
        }

        # Use FP8 dynamic quantization for all models
        kwargs['quantization'] = 'fp8'

        llm = LLM(**kwargs)
        load_time = time.time() - start_load
        print(f"Load time: {load_time:.1f}s")

        params = SamplingParams(max_tokens=50, temperature=0.7)

        # Warmup
        _ = llm.generate(["Hello"], params)

        # Benchmark
        start_gen = time.time()
        outputs = llm.generate(TEST_PROMPTS, params)
        gen_time = time.time() - start_gen

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        throughput = len(TEST_PROMPTS) / gen_time
        tokens_per_sec = total_tokens / gen_time

        print(f"Throughput: {throughput:.1f} req/s")
        print(f"Tokens/sec: {tokens_per_sec:.1f}")
        print(f"Sample: {outputs[0].outputs[0].text[:80]}")

        result = {
            'model': name,
            'model_id': model_id,
            'load_time': load_time,
            'throughput_rps': throughput,
            'tokens_per_sec': tokens_per_sec,
            'status': 'OK'
        }

        # Cleanup
        del llm
        destroy_model_parallel()
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)

        return result

    except Exception as e:
        print(f"ERROR: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        return {'model': name, 'model_id': model_id, 'status': f'FAILED: {str(e)[:60]}'}


def main():
    results = []

    for name, model_id in MODELS:
        result = benchmark_model(name, model_id)
        results.append(result)

    # Summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY - vLLM + AWQ (RTX 5090)")
    print("="*80)
    print(f"{'Model':<25} {'req/s':<10} {'tok/s':<12} {'Load(s)':<10} {'Status'}")
    print("-"*80)
    for r in results:
        if r['status'] == 'OK':
            print(f"{r['model']:<25} {r['throughput_rps']:<10.1f} {r['tokens_per_sec']:<12.1f} {r['load_time']:<10.1f} {r['status']}")
        else:
            print(f"{r['model']:<25} {'N/A':<10} {'N/A':<12} {'N/A':<10} {r['status'][:30]}")

    return results


if __name__ == '__main__':
    main()
