"""
Configuration for supported models and inference settings.

Optimized for RTX 5090 (32GB VRAM) with focus on ~30B frontier models.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum


class QuantizationType(Enum):
    """Supported quantization methods."""
    NONE = "none"
    AWQ = "awq"
    GPTQ = "gptq"
    FP8 = "fp8"
    INT8 = "int8"
    FP4 = "fp4"  # NVFP4 on Blackwell
    BNB = "bitsandbytes"  # Unsloth-style quantization


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    hf_name: str  # HuggingFace model name
    total_params: float  # Billions
    active_params: float  # Billions (for MoE models)
    architecture: str  # dense, moe, mamba-moe
    context_length: int
    recommended_quantization: QuantizationType
    vram_fp16: float  # GB needed for FP16
    vram_int4: float  # GB needed for INT4
    supports_thinking: bool = False
    chat_template: str = "default"
    text_only_mode: bool = False  # For multimodal models, use limit_mm_per_prompt={'image': 0}
    notes: str = ""


# Supported frontier models (~30B class, optimized for RTX 5090)
SUPPORTED_MODELS: Dict[str, ModelConfig] = {
    # Qwen 3 Models
    "qwen3-30b-a3b": ModelConfig(
        name="Qwen3-30B-A3B",
        hf_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        total_params=30.0,
        active_params=3.0,
        architecture="moe",
        context_length=131072,
        recommended_quantization=QuantizationType.AWQ,
        vram_fp16=60.0,
        vram_int4=15.0,
        supports_thinking=True,
        chat_template="qwen",
        notes="Fastest MoE model, only 3B active params"
    ),
    "qwen3-32b": ModelConfig(
        name="Qwen3-32B",
        hf_name="Qwen/Qwen3-32B-Instruct",
        total_params=32.0,
        active_params=32.0,
        architecture="dense",
        context_length=131072,
        recommended_quantization=QuantizationType.AWQ,
        vram_fp16=64.0,
        vram_int4=16.0,
        supports_thinking=True,
        chat_template="qwen",
        notes="Dense model, higher quality but slower"
    ),

    # OLMo 3 Models
    "olmo3-32b-instruct": ModelConfig(
        name="OLMo-2-0325-32B-Instruct",
        hf_name="allenai/OLMo-2-0325-32B-Instruct",
        total_params=32.0,
        active_params=32.0,
        architecture="dense",
        context_length=32768,
        recommended_quantization=QuantizationType.FP8,
        vram_fp16=64.0,
        vram_int4=16.0,
        supports_thinking=False,
        chat_template="olmo",
        notes="Fully open model, Apache 2.0. No AWQ available, use FP8."
    ),
    "olmo3-32b-think": ModelConfig(
        name="OLMo-3-32B-Think",
        hf_name="allenai/OLMo-3-1B-32B-1125-Think",
        total_params=32.0,
        active_params=32.0,
        architecture="dense",
        context_length=32768,
        recommended_quantization=QuantizationType.AWQ,
        vram_fp16=64.0,
        vram_int4=16.0,
        supports_thinking=True,
        chat_template="olmo",
        notes="Reasoning model with explicit thinking"
    ),

    # NVIDIA Nemotron Models
    "nemotron-30b-a3b": ModelConfig(
        name="Nemotron-3-Nano-30B-A3B",
        hf_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        total_params=31.6,
        active_params=3.2,
        architecture="mamba-moe",
        context_length=1000000,  # 1M context!
        recommended_quantization=QuantizationType.AWQ,
        vram_fp16=63.0,
        vram_int4=16.0,
        supports_thinking=True,
        chat_template="nemotron",
        notes="Hybrid Mamba-MoE, 1M context, fully open training data"
    ),

    # Google Gemma 3 Models
    "gemma3-27b": ModelConfig(
        name="Gemma-3-27B-IT",
        hf_name="google/gemma-3-27b-it",
        total_params=27.0,
        active_params=27.0,
        architecture="dense",
        context_length=128000,
        recommended_quantization=QuantizationType.AWQ,
        vram_fp16=54.0,
        vram_int4=14.0,
        supports_thinking=False,
        chat_template="gemma",
        notes="Multimodal capable, 128K context"
    ),

    # === LOCAL TESTING MODELS (7-8B, AWQ on RTX 5090 Blackwell) ===
    # Benchmarked on RTX 5090 with vLLM 0.13.0 + AWQ quantization
    # AWQ is ~8-9% faster than FP8 on Blackwell
    # Use quantization="awq" and enforce_eager=True for Blackwell GPUs

    # Mistral-7B-v0.3 AWQ - FASTEST (67.1 req/s, 3328 tok/s)
    "mistral-7b-v0.3": ModelConfig(
        name="Mistral-7B-Instruct-v0.3-AWQ",
        hf_name="solidrust/Mistral-7B-Instruct-v0.3-AWQ",
        total_params=7.0,
        active_params=7.0,
        architecture="dense",
        context_length=32768,
        recommended_quantization=QuantizationType.AWQ,
        vram_fp16=14.0,
        vram_int4=4.0,
        supports_thinking=False,
        chat_template="mistral",
        notes="FASTEST on RTX 5090: 67.1 req/s with AWQ"
    ),

    # Llama-3.1-8B AWQ (65.9 req/s, 3295 tok/s)
    "llama-3.1-8b": ModelConfig(
        name="Llama-3.1-8B-Instruct-AWQ",
        hf_name="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        total_params=8.0,
        active_params=8.0,
        architecture="dense",
        context_length=131072,
        recommended_quantization=QuantizationType.AWQ,
        vram_fp16=16.0,
        vram_int4=5.0,
        supports_thinking=False,
        chat_template="llama",
        notes="RTX 5090: 65.9 req/s with AWQ, 128K context"
    ),

    # Qwen3-8B AWQ (60.5 req/s, 3027 tok/s) - has thinking mode
    "qwen3-8b": ModelConfig(
        name="Qwen3-8B-AWQ",
        hf_name="Qwen/Qwen3-8B-AWQ",
        total_params=8.0,
        active_params=8.0,
        architecture="dense",
        context_length=131072,
        recommended_quantization=QuantizationType.AWQ,
        vram_fp16=16.0,
        vram_int4=5.0,
        supports_thinking=True,
        chat_template="qwen",
        notes="RTX 5090: 60.5 req/s with AWQ, supports thinking mode"
    ),

    # OLMo-3-7B-Instruct FP8 (31.5 req/s - slower, no AWQ available)
    "olmo3-7b": ModelConfig(
        name="OLMo-3-7B-Instruct",
        hf_name="allenai/OLMo-3-7B-Instruct",
        total_params=7.0,
        active_params=7.0,
        architecture="dense",
        context_length=32768,
        recommended_quantization=QuantizationType.FP8,
        vram_fp16=14.0,
        vram_int4=4.5,
        supports_thinking=False,
        chat_template="olmo",
        notes="RTX 5090: 31.5 req/s with FP8, fully open Apache 2.0 (2x slower)"
    ),

    # Gemma 3 Models (multimodal, use text_only_mode=True for text-only inference)
    # Benchmarked on RTX 5090 with vLLM 0.13.0 + BF16 + text-only mode
    # IMPORTANT: Requires limit_mm_per_prompt={'image': 0} for Blackwell compatibility

    # Gemma-3-4B BF16 text-only - FASTEST OVERALL (92.0 req/s)
    "gemma3-4b": ModelConfig(
        name="Gemma-3-4B-IT",
        hf_name="google/gemma-3-4b-it",
        total_params=4.0,
        active_params=4.0,
        architecture="dense",
        context_length=128000,
        recommended_quantization=QuantizationType.NONE,  # BF16 is faster than FP8!
        vram_fp16=8.6,
        vram_int4=3.0,
        supports_thinking=False,
        chat_template="gemma",
        text_only_mode=True,  # Multimodal model, disable vision encoder for text-only
        notes="RTX 5090: 92.0 req/s with BF16 text-only - FASTEST MODEL"
    ),

    # Gemma-3-4B-PT (base, no instruction tuning) - may be less anchored to defaults
    "gemma3-4b-pt": ModelConfig(
        name="Gemma-3-4B-PT",
        hf_name="google/gemma-3-4b-pt",
        total_params=4.0,
        active_params=4.0,
        architecture="dense",
        context_length=128000,
        recommended_quantization=QuantizationType.NONE,
        vram_fp16=8.6,
        vram_int4=3.0,
        supports_thinking=False,
        chat_template="gemma",
        text_only_mode=True,  # Multimodal model, disable vision encoder for text-only
        notes="Base model (no IT), may explore more diverse outputs"
    ),

    # Gemma-3-12B BF16 text-only (65.8 req/s)
    "gemma3-12b": ModelConfig(
        name="Gemma-3-12B-IT",
        hf_name="google/gemma-3-12b-it",
        total_params=12.0,
        active_params=12.0,
        architecture="dense",
        context_length=128000,
        recommended_quantization=QuantizationType.NONE,  # BF16 only, FP8 OOMs
        vram_fp16=23.3,
        vram_int4=7.0,
        supports_thinking=False,
        chat_template="gemma",
        text_only_mode=True,  # Multimodal model, disable vision encoder for text-only
        notes="RTX 5090: 65.8 req/s with BF16 text-only, higher quality than 4B"
    ),

    # Unsloth optimized models (pre-quantized with bitsandbytes)
    "unsloth-qwen3-8b": ModelConfig(
        name="Qwen3-8B-Unsloth-4bit",
        hf_name="unsloth/Qwen3-8B-bnb-4bit",
        total_params=8.0,
        active_params=8.0,
        architecture="dense",
        context_length=131072,
        recommended_quantization=QuantizationType.BNB,  # BitsAndBytes quantized
        vram_fp16=16.0,
        vram_int4=5.0,
        supports_thinking=True,
        chat_template="qwen",
        notes="Unsloth optimized 4-bit, fast local inference"
    ),
    "unsloth-qwen3-14b": ModelConfig(
        name="Qwen3-14B-Unsloth-4bit",
        hf_name="unsloth/Qwen3-14B-bnb-4bit",
        total_params=14.0,
        active_params=14.0,
        architecture="dense",
        context_length=131072,
        recommended_quantization=QuantizationType.BNB,  # BitsAndBytes quantized
        vram_fp16=28.0,
        vram_int4=8.0,
        supports_thinking=True,
        chat_template="qwen",
        notes="Unsloth optimized 4-bit, best quality/speed for local"
    ),
    "unsloth-llama31-8b": ModelConfig(
        name="Llama-3.1-8B-Unsloth-4bit",
        hf_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        total_params=8.0,
        active_params=8.0,
        architecture="dense",
        context_length=128000,
        recommended_quantization=QuantizationType.BNB,  # BitsAndBytes quantized
        vram_fp16=16.0,
        vram_int4=5.0,
        supports_thinking=False,
        chat_template="llama",
        notes="Unsloth optimized 4-bit Llama"
    ),

    # Small models for local testing / finetuning
    "qwen3-4b": ModelConfig(
        name="Qwen3-4B-Instruct",
        hf_name="Qwen/Qwen3-4B",
        total_params=4.0,
        active_params=4.0,
        architecture="dense",
        context_length=131072,
        recommended_quantization=QuantizationType.NONE,
        vram_fp16=8.0,
        vram_int4=3.0,
        supports_thinking=True,
        chat_template="qwen",
        notes="Small model for local testing and planner finetuning"
    ),
    "qwen3-1.7b": ModelConfig(
        name="Qwen3-1.7B",
        hf_name="Qwen/Qwen3-1.7B",
        total_params=1.7,
        active_params=1.7,
        architecture="dense",
        context_length=131072,
        recommended_quantization=QuantizationType.NONE,
        vram_fp16=4.0,
        vram_int4=1.5,
        supports_thinking=True,
        chat_template="qwen",
        notes="Tiny model for quick tests"
    ),
    # Legacy models for comparison
    "llama31-8b": ModelConfig(
        name="Llama-3.1-8B-Instruct",
        hf_name="meta-llama/Llama-3.1-8B-Instruct",
        total_params=8.0,
        active_params=8.0,
        architecture="dense",
        context_length=128000,
        recommended_quantization=QuantizationType.AWQ,
        vram_fp16=16.0,
        vram_int4=5.0,
        supports_thinking=False,
        chat_template="llama",
        notes="Current baseline model"
    ),
    "llama32-1b": ModelConfig(
        name="Llama-3.2-1B",
        hf_name="meta-llama/Llama-3.2-1B",
        total_params=1.0,
        active_params=1.0,
        architecture="dense",
        context_length=128000,
        recommended_quantization=QuantizationType.NONE,
        vram_fp16=3.0,
        vram_int4=1.5,
        supports_thinking=False,
        chat_template="llama",
        notes="Tiny model for worker inference"
    ),
}


# Model name aliases for convenience
MODEL_ALIASES = {
    "qwen3": "qwen3-30b-a3b",
    "qwen": "qwen3-30b-a3b",
    "olmo3": "olmo3-32b-instruct",
    "olmo": "olmo3-32b-instruct",
    "olmo-think": "olmo3-32b-think",
    "nemotron": "nemotron-30b-a3b",
    "gemma3": "gemma3-27b",
    "gemma": "gemma3-27b",
    "llama": "llama31-8b",
    # Unsloth aliases (local inference)
    "unsloth": "unsloth-qwen3-8b",
    "unsloth-8b": "unsloth-qwen3-8b",
    "unsloth-14b": "unsloth-qwen3-14b",
    "local": "unsloth-qwen3-8b",  # Default local model
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get model configuration by name or alias."""
    # Check aliases first
    resolved_name = MODEL_ALIASES.get(model_name.lower(), model_name.lower())

    if resolved_name in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[resolved_name]

    # Try to find by HuggingFace name
    for config in SUPPORTED_MODELS.values():
        if config.hf_name.lower() == model_name.lower():
            return config

    raise ValueError(f"Unknown model: {model_name}. Supported: {list(SUPPORTED_MODELS.keys())}")


@dataclass
class InferenceConfig:
    """Configuration for inference engine."""
    # Model settings
    model_name: str = "qwen3-30b-a3b"
    quantization: QuantizationType = QuantizationType.AWQ

    # Hardware settings
    tensor_parallel_size: int = 1  # Number of GPUs
    gpu_memory_utilization: float = 0.95

    # Performance settings
    max_model_len: int = 8192  # Context window to use
    max_num_seqs: int = 256  # Max concurrent sequences
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    kv_cache_dtype: str = "fp8"  # fp8 or auto

    # Batching settings
    batch_size: int = 100  # Agents per batch
    max_tokens: int = 256  # Max tokens per response
    temperature: float = 0.7

    # Memory management
    swap_space: int = 4  # GB of CPU swap

    def __post_init__(self):
        """Validate and adjust config based on model."""
        model_config = get_model_config(self.model_name)

        # Auto-adjust settings based on model
        if model_config.architecture == "moe":
            # MoE models can handle larger batches
            self.max_num_seqs = min(512, self.max_num_seqs * 2)

        if model_config.context_length > 100000:
            # For long-context models, may need to reduce batch size
            self.max_num_seqs = min(128, self.max_num_seqs)

    def to_engine_args(self) -> dict:
        """Convert to vLLM engine arguments."""
        model_config = get_model_config(self.model_name)

        return {
            'model': model_config.hf_name,
            'tensor_parallel_size': self.tensor_parallel_size,
            'gpu_memory_utilization': self.gpu_memory_utilization,
            'max_model_len': self.max_model_len,
            'quantization': self.quantization.value if self.quantization != QuantizationType.NONE else None,
            'kv_cache_dtype': self.kv_cache_dtype,
            'enable_prefix_caching': self.enable_prefix_caching,
            'enable_chunked_prefill': self.enable_chunked_prefill,
            'max_num_seqs': self.max_num_seqs,
            'swap_space': self.swap_space,
            'enforce_eager': False,
            'trust_remote_code': True,
        }


# Presets for different hardware configurations
HARDWARE_PRESETS = {
    "rtx5090_single": InferenceConfig(
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_model_len=8192,
        max_num_seqs=256,
        batch_size=100,
    ),
    "rtx5090_dual": InferenceConfig(
        tensor_parallel_size=2,
        gpu_memory_utilization=0.95,
        max_model_len=16384,
        max_num_seqs=512,
        batch_size=200,
    ),
    "h100_single": InferenceConfig(
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_model_len=32768,
        max_num_seqs=512,
        batch_size=200,
    ),
}


def estimate_throughput(
    model_name: str,
    num_gpus: int = 1,
    batch_size: int = 100,
    avg_input_tokens: int = 500,
    avg_output_tokens: int = 100,
    quantization: QuantizationType = QuantizationType.AWQ
) -> Dict[str, float]:
    """
    Estimate throughput for a given configuration.

    Returns dict with:
    - requests_per_second
    - tokens_per_second
    - time_per_batch_seconds
    - estimated_agents_per_hour
    """
    model_config = get_model_config(model_name)

    # Base throughput estimates (tokens/second) based on benchmarks
    # RTX 5090 with AWQ int4
    if model_config.architecture == "moe" and model_config.active_params <= 5:
        # MoE with small active params (like Qwen3-30B-A3B with 3B active)
        base_throughput = 15000  # tok/s with batching
    elif model_config.architecture == "mamba-moe":
        # Mamba-MoE hybrid
        base_throughput = 12000
    elif model_config.total_params <= 10:
        # Small models (~8B)
        base_throughput = 25000
    elif model_config.total_params <= 30:
        # Medium models (~27-32B)
        base_throughput = 8000
    else:
        # Large models
        base_throughput = 4000

    # Adjust for quantization
    quant_multipliers = {
        QuantizationType.NONE: 0.5,  # FP16 is slower
        QuantizationType.FP8: 0.8,
        QuantizationType.AWQ: 1.0,
        QuantizationType.GPTQ: 0.9,
        QuantizationType.INT8: 0.7,
        QuantizationType.FP4: 1.2,  # NVFP4 is fastest
    }
    throughput = base_throughput * quant_multipliers.get(quantization, 1.0)

    # Scale with number of GPUs (sublinear)
    throughput *= num_gpus ** 0.8

    # Calculate estimates
    tokens_per_request = avg_input_tokens + avg_output_tokens
    requests_per_second = throughput / tokens_per_request
    time_per_batch = batch_size / requests_per_second
    agents_per_hour = requests_per_second * 3600

    return {
        'requests_per_second': requests_per_second,
        'tokens_per_second': throughput,
        'time_per_batch_seconds': time_per_batch,
        'estimated_agents_per_hour': agents_per_hour,
        'model_active_params': model_config.active_params,
        'model_architecture': model_config.architecture,
    }


def calculate_pareto_configs(
    time_budget_hours: float = 6.0,
    model_name: str = "qwen3-30b-a3b",
    num_gpus: int = 2,
    min_steps: int = 50,
    min_agents: int = 100,
) -> List[Dict]:
    """
    Calculate pareto-optimal experiment configurations for a time budget.

    Returns list of configs sorted by (agents * steps) product.
    """
    estimates = estimate_throughput(model_name, num_gpus)
    time_budget_seconds = time_budget_hours * 3600

    # Time per agent-step (seconds)
    time_per_agent_step = 1.0 / estimates['requests_per_second']

    configs = []

    # Try different agent counts
    for num_agents in [100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]:
        if num_agents < min_agents:
            continue

        # Calculate max steps for this agent count
        max_steps = int(time_budget_seconds / (num_agents * time_per_agent_step))

        if max_steps < min_steps:
            continue

        # Calculate actual time
        actual_time = num_agents * max_steps * time_per_agent_step

        configs.append({
            'num_agents': num_agents,
            'max_steps': max_steps,
            'estimated_time_hours': actual_time / 3600,
            'agent_step_product': num_agents * max_steps,
            'agents_per_step_time': num_agents * time_per_agent_step,
        })

    # Sort by agent*step product (higher is better)
    configs.sort(key=lambda x: x['agent_step_product'], reverse=True)

    return configs
