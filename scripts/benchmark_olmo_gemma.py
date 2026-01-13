#!/usr/bin/env python3
"""Benchmark OLMo-3-7B and Gemma-3-12B on RTX 5090."""
import os
os.environ['HF_HOME'] = '/mnt/storage/models'
os.environ['VLLM_ATTENTION_BACKEND'] = 'TRITON_ATTN'

import time
import gc
import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

# Models to test
MODELS = [
    # OLMo-3-7B - no AWQ available, test FP8 only
    ("OLMo-3-7B", "allenai/OLMo-3-7B-Instruct", "fp8"),
    # Gemma-3-12B - try AWQ
    ("Gemma-3-12B-AWQ", "pytorch/gemma-3-12b-it-AWQ-INT4", "awq"),
    # Gemma-3-12B - try bfloat16 (no quant)
    ("Gemma-3-12B-BF16", "google/gemma-3-12b-it", "none"),
]

# Test prompts
TEST_PROMPTS = [
    f"You are worker {i}. Skill={0.3+i*0.07:.2f}, tax=30%. Hours to work (0-10)? Just the number."
    for i in range(50)
]

def benchmark_model(name, model_id, quant):
    """Benchmark a single model."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Model: {model_id}, Quant: {quant}")
    print(f"{'='*60}")

    try:
        start_load = time.time()

        kwargs = {
            'model': model_id,
            'enforce_eager': True,
            'max_model_len': 4096,
            'gpu_memory_utilization': 0.85,
            'trust_remote_code': True,
            'download_dir': '/mnt/storage/models',
        }

        if quant == 'fp8':
            kwargs['quantization'] = 'fp8'
        elif quant == 'awq':
            kwargs['quantization'] = 'awq'
        elif quant == 'none':
            kwargs['dtype'] = 'bfloat16'

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
            'quant': quant,
            'throughput_rps': throughput,
            'tokens_per_sec': tokens_per_sec,
            'load_time': load_time,
            'status': 'OK'
        }

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
        return {'model': name, 'quant': quant, 'status': f'FAILED: {str(e)[:60]}'}


def main():
    results = []

    for name, model_id, quant in MODELS:
        result = benchmark_model(name, model_id, quant)
        results.append(result)

    # Summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY - OLMo & Gemma (RTX 5090 Blackwell)")
    print("="*80)
    print(f"{'Model':<20} {'Quant':<10} {'req/s':<10} {'tok/s':<12} {'Status'}")
    print("-"*80)

    for r in results:
        if r.get('status') == 'OK':
            print(f"{r['model']:<20} {r['quant']:<10} {r['throughput_rps']:<10.1f} {r['tokens_per_sec']:<12.1f} OK")
        else:
            print(f"{r['model']:<20} {r.get('quant', 'N/A'):<10} {'N/A':<10} {'N/A':<12} {r['status'][:30]}")

    return results


if __name__ == '__main__':
    main()
