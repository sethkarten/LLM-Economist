#!/usr/bin/env python3
"""Compare AWQ vs FP8 performance on RTX 5090 Blackwell."""
import os
os.environ['HF_HOME'] = '/mnt/storage/models'
os.environ['VLLM_ATTENTION_BACKEND'] = 'TRITON_ATTN'

import time
import gc
import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

# Models to compare: (name, base_model_id, awq_model_id)
MODELS = [
    ("Mistral-7B-v0.3", "mistralai/Mistral-7B-Instruct-v0.3", "solidrust/Mistral-7B-Instruct-v0.3-AWQ"),
    ("Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"),
    ("Qwen3-8B", "Qwen/Qwen3-8B", "Qwen/Qwen3-8B-AWQ"),
]

# Test prompts
TEST_PROMPTS = [
    f"You are worker {i}. Skill={0.3+i*0.07:.2f}, tax=30%. Hours to work (0-10)? Just the number."
    for i in range(50)
]

def benchmark_model(name, model_id, quant_method):
    """Benchmark a single model with given quantization."""
    print(f"\n{'='*60}")
    print(f"Testing: {name} ({quant_method})")
    print(f"Model: {model_id}")
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

        if quant_method == 'fp8':
            kwargs['dtype'] = 'float16'
            kwargs['quantization'] = 'fp8'
        elif quant_method == 'awq':
            kwargs['quantization'] = 'awq'

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
            'quant': quant_method,
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
        return {'model': name, 'quant': quant_method, 'status': f'FAILED: {str(e)[:60]}'}


def main():
    results = []

    for name, base_id, awq_id in MODELS:
        # Test FP8 first
        result_fp8 = benchmark_model(name, base_id, 'fp8')
        results.append(result_fp8)

        # Then test AWQ
        result_awq = benchmark_model(name, awq_id, 'awq')
        results.append(result_awq)

    # Summary
    print("\n" + "="*90)
    print("BENCHMARK SUMMARY - AWQ vs FP8 (RTX 5090 Blackwell)")
    print("="*90)
    print(f"{'Model':<20} {'Quant':<8} {'req/s':<10} {'tok/s':<12} {'Load(s)':<10} {'Status'}")
    print("-"*90)

    for r in results:
        if r.get('status') == 'OK':
            print(f"{r['model']:<20} {r['quant']:<8} {r['throughput_rps']:<10.1f} {r['tokens_per_sec']:<12.1f} {r['load_time']:<10.1f} {r['status']}")
        else:
            print(f"{r['model']:<20} {r.get('quant', 'N/A'):<8} {'N/A':<10} {'N/A':<12} {'N/A':<10} {r['status'][:30]}")

    # Find winners
    print("\n" + "="*90)
    print("WINNERS BY MODEL")
    print("="*90)

    model_results = {}
    for r in results:
        if r.get('status') == 'OK':
            name = r['model']
            if name not in model_results:
                model_results[name] = {}
            model_results[name][r['quant']] = r['throughput_rps']

    for name, quants in model_results.items():
        if 'fp8' in quants and 'awq' in quants:
            winner = 'AWQ' if quants['awq'] > quants['fp8'] else 'FP8'
            diff = abs(quants['awq'] - quants['fp8']) / max(quants.values()) * 100
            print(f"{name}: {winner} wins by {diff:.1f}% ({quants['awq']:.1f} AWQ vs {quants['fp8']:.1f} FP8)")

    return results


if __name__ == '__main__':
    main()
