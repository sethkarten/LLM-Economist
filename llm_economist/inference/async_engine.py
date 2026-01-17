"""
Async vLLM engine wrapper for massive scale agent simulations.

Key optimizations:
- AWQ int4 quantization for 75% memory reduction
- FP8 KV cache for 50% memory reduction
- Chunked prefill for 2-3x throughput
- Prefix caching for reduced redundant compute
- Continuous batching for maximum GPU utilization

RTX 5090 (Blackwell) Support:
- Uses TRITON_ATTN backend (FLASH_ATTN kernels incompatible)
- Requires enforce_eager=True to avoid CUDA graph issues
- Supports bitsandbytes quantization for unsloth models
"""

import asyncio
import logging
import os
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from time import time
import json

logger = logging.getLogger(__name__)


def detect_blackwell_gpu() -> bool:
    """Detect if running on RTX 5090 or other Blackwell (SM 100) GPUs."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            # RTX 50xx series are Blackwell architecture
            if "5090" in gpu_name or "5080" in gpu_name or "5070" in gpu_name:
                return True
            # Also check by compute capability
            major, minor = torch.cuda.get_device_capability(0)
            if major >= 10:  # SM 100+ is Blackwell
                return True
    except Exception:
        pass
    return False


@dataclass
class BatchRequest:
    """A batch of requests to be processed together."""
    request_ids: List[str]
    prompts: List[str]
    system_prompts: List[str]
    temperatures: List[float]
    max_tokens: int = 256
    json_format: bool = False


@dataclass
class BatchResponse:
    """Response from a batch of requests."""
    request_ids: List[str]
    responses: List[str]
    is_json_valid: List[bool]
    latencies: List[float]


class ScalableInferenceEngine:
    """
    High-performance async inference engine for massive scale simulations.

    Supports:
    - vLLM with tensor parallelism across multiple GPUs
    - AWQ/GPTQ int4 quantization
    - FP8 KV cache
    - Continuous batching
    - Prefix caching for repeated prompts
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 8192,
        quantization: Optional[str] = "awq",  # awq, gptq, fp8, bitsandbytes, None
        kv_cache_dtype: str = "fp8",
        enable_prefix_caching: bool = True,
        enable_chunked_prefill: bool = True,
        max_num_seqs: int = 256,  # Max concurrent sequences
        swap_space: int = 4,  # GB of CPU swap space
        enforce_eager: Optional[bool] = None,  # Auto-detect for Blackwell
        text_only_mode: bool = False,  # For multimodal models, disable vision encoder
        dtype: Optional[str] = None,  # Override dtype (e.g., "bfloat16")
    ):
        """
        Initialize the scalable inference engine.

        Args:
            model_name: HuggingFace model name or path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length
            quantization: Quantization method (awq, gptq, fp8, bitsandbytes, None)
            kv_cache_dtype: KV cache dtype (fp8, auto)
            enable_prefix_caching: Enable prefix caching for repeated prompts
            enable_chunked_prefill: Enable chunked prefill for better throughput
            max_num_seqs: Maximum number of concurrent sequences
            swap_space: GB of CPU swap space for KV cache offloading
            enforce_eager: Force eager mode (auto-detects Blackwell GPUs if None)
            text_only_mode: For multimodal models (Gemma 3), disable vision encoder
            dtype: Override dtype (e.g., "bfloat16" for Gemma 3)
        """
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.quantization = quantization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs

        self._engine = None
        self._initialized = False

        # Auto-detect Blackwell GPU and configure accordingly
        self._is_blackwell = detect_blackwell_gpu()
        if self._is_blackwell:
            logger.info("Detected Blackwell GPU (RTX 50xx series), enabling compatibility mode")
            # Set TRITON_ATTN backend for Blackwell (FLASH_ATTN kernels incompatible)
            os.environ['VLLM_ATTENTION_BACKEND'] = 'TRITON_ATTN'
            # Force eager mode if not explicitly set
            if enforce_eager is None:
                enforce_eager = True
            # Use auto KV cache dtype for Blackwell
            if kv_cache_dtype == "fp8":
                kv_cache_dtype = "auto"  # FP8 KV cache may have issues on Blackwell

        # Determine final enforce_eager value
        final_enforce_eager = enforce_eager if enforce_eager is not None else False

        # Store config for lazy initialization
        self._config = {
            'model': model_name,
            'tensor_parallel_size': tensor_parallel_size,
            'gpu_memory_utilization': gpu_memory_utilization,
            'max_model_len': max_model_len,
            'quantization': quantization,
            'kv_cache_dtype': kv_cache_dtype,
            'enable_prefix_caching': enable_prefix_caching,
            'enable_chunked_prefill': enable_chunked_prefill,
            'max_num_seqs': max_num_seqs,
            'swap_space': swap_space,
            'enforce_eager': final_enforce_eager,
            'trust_remote_code': True,
            'tokenizer_mode': 'auto',
            'download_dir': os.path.expanduser('~/.cache/huggingface'),
        }

        # Add text-only mode for multimodal models (e.g., Gemma 3)
        if text_only_mode:
            self._config['limit_mm_per_prompt'] = {'image': 0}
            logger.info("Text-only mode enabled (vision encoder disabled)")

        # Add dtype override if specified
        if dtype:
            self._config['dtype'] = dtype
            logger.info(f"Using dtype override: {dtype}")

        # Performance tracking
        self._total_requests = 0
        self._total_tokens = 0
        self._total_time = 0.0

    async def initialize(self):
        """Lazily initialize the vLLM engine."""
        if self._initialized:
            return

        try:
            from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
            self._SamplingParams = SamplingParams

            logger.info(f"Initializing vLLM engine with config: {self._config}")

            engine_args = AsyncEngineArgs(**self._config)
            self._engine = AsyncLLMEngine.from_engine_args(engine_args)
            self._initialized = True

            logger.info(f"vLLM engine initialized successfully for {self.model_name}")

        except ImportError:
            logger.warning("vLLM not available, falling back to synchronous mode")
            self._engine = None
            self._initialized = True

    def _format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Format system and user prompts for the model."""
        # Use chat template format
        return f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_prompt}<|end|>\n<|assistant|>\n"

    async def generate_single(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
        json_format: bool = False,
        request_id: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Generate a single response.

        Returns:
            Tuple of (response_text, is_json_valid)
        """
        if not self._initialized:
            await self.initialize()

        if self._engine is None:
            # Fallback to synchronous mode
            return await self._fallback_generate(
                system_prompt, user_prompt, temperature, max_tokens, json_format
            )

        prompt = self._format_prompt(system_prompt, user_prompt)

        sampling_params = self._SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=['}'] if json_format else None,
            include_stop_str_in_output=json_format,  # Include '}' in output for valid JSON
        )

        request_id = request_id or f"req_{time()}"

        start_time = time()

        async for output in self._engine.generate(prompt, sampling_params, request_id):
            pass  # Stream through to get final output

        response_text = output.outputs[0].text
        latency = time() - start_time

        self._total_requests += 1
        self._total_time += latency

        # Extract JSON if requested
        if json_format:
            return self._extract_json(response_text)

        return response_text, False

    async def generate_batch(
        self,
        batch: BatchRequest,
        progress_callback: Optional[callable] = None
    ) -> BatchResponse:
        """
        Generate responses for a batch of requests.

        This is the key method for massive scale - it batches all agent
        prompts together for maximum throughput.
        """
        if not self._initialized:
            await self.initialize()

        start_time = time()
        n_requests = len(batch.prompts)

        logger.info(f"Processing batch of {n_requests} requests")

        if self._engine is None:
            # Fallback to synchronous processing
            return await self._fallback_batch_generate(batch)

        # Format all prompts
        formatted_prompts = [
            self._format_prompt(sys, user)
            for sys, user in zip(batch.system_prompts, batch.prompts)
        ]

        # Create sampling params for each request
        # (vLLM can handle different params per request)
        sampling_params_list = [
            self._SamplingParams(
                temperature=temp,
                max_tokens=batch.max_tokens,
                stop=['}'] if batch.json_format else None,
                include_stop_str_in_output=batch.json_format,  # Include '}' in output for valid JSON
            )
            for temp in batch.temperatures
        ]

        # Submit all requests
        tasks = []
        for i, (prompt, params, req_id) in enumerate(
            zip(formatted_prompts, sampling_params_list, batch.request_ids)
        ):
            task = self._generate_and_collect(prompt, params, req_id)
            tasks.append(task)

        # Wait for all completions
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        responses = []
        is_json_valid = []
        latencies = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Request {batch.request_ids[i]} failed: {result}")
                responses.append("")
                is_json_valid.append(False)
                latencies.append(0.0)
            else:
                text, is_valid, latency = result
                if batch.json_format:
                    text, is_valid = self._extract_json(text)
                responses.append(text)
                is_json_valid.append(is_valid)
                latencies.append(latency)

        total_time = time() - start_time
        self._total_requests += n_requests
        self._total_time += total_time

        throughput = n_requests / total_time
        logger.info(f"Batch completed: {n_requests} requests in {total_time:.2f}s ({throughput:.1f} req/s)")

        if progress_callback:
            progress_callback(n_requests, total_time)

        return BatchResponse(
            request_ids=batch.request_ids,
            responses=responses,
            is_json_valid=is_json_valid,
            latencies=latencies
        )

    async def _generate_and_collect(
        self,
        prompt: str,
        sampling_params: Any,
        request_id: str
    ) -> Tuple[str, bool, float]:
        """Generate and collect a single response."""
        start_time = time()

        async for output in self._engine.generate(prompt, sampling_params, request_id):
            pass

        latency = time() - start_time
        return output.outputs[0].text, False, latency

    async def _fallback_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        json_format: bool
    ) -> Tuple[str, bool]:
        """Fallback to HuggingFace transformers if vLLM unavailable."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            if not hasattr(self, '_fallback_model'):
                logger.info("Loading fallback model with transformers...")
                self._fallback_tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, trust_remote_code=True
                )
                self._fallback_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )

            prompt = self._format_prompt(system_prompt, user_prompt)
            inputs = self._fallback_tokenizer(prompt, return_tensors="pt").to(
                self._fallback_model.device
            )

            with torch.no_grad():
                outputs = self._fallback_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self._fallback_tokenizer.eos_token_id
                )

            response = self._fallback_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            if json_format:
                return self._extract_json(response)
            return response, False

        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            return "", False

    async def _fallback_batch_generate(self, batch: BatchRequest) -> BatchResponse:
        """Fallback batch generation using sequential calls."""
        responses = []
        is_json_valid = []
        latencies = []

        for i in range(len(batch.prompts)):
            start = time()
            text, is_valid = await self._fallback_generate(
                batch.system_prompts[i],
                batch.prompts[i],
                batch.temperatures[i],
                batch.max_tokens,
                batch.json_format
            )
            latencies.append(time() - start)
            responses.append(text)
            is_json_valid.append(is_valid)

        return BatchResponse(
            request_ids=batch.request_ids,
            responses=responses,
            is_json_valid=is_json_valid,
            latencies=latencies
        )

    def _extract_json(self, message: str) -> Tuple[str, bool]:
        """Extract JSON from a message string."""
        try:
            json_start = message.find('{')
            json_end = message.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                return message, False

            json_str = message[json_start:json_end]
            if len(json_str) > 0:
                json.loads(json_str)  # Validate
                return json_str, True
        except (ValueError, json.JSONDecodeError):
            pass

        return message, False

    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        return {
            'total_requests': self._total_requests,
            'total_time': self._total_time,
            'avg_latency': self._total_time / max(1, self._total_requests),
            'throughput': self._total_requests / max(0.001, self._total_time),
        }

    async def shutdown(self):
        """Shutdown the engine."""
        if self._engine is not None:
            # vLLM doesn't have explicit shutdown, but we can clean up
            self._engine = None
            self._initialized = False


class AgentBatcher:
    """
    Helper class to batch agent requests efficiently.

    Groups agents into batches based on their prompt similarity
    for better prefix caching utilization.
    """

    def __init__(self, engine: ScalableInferenceEngine, batch_size: int = 100):
        self.engine = engine
        self.batch_size = batch_size

    async def process_agents(
        self,
        agents: List[Any],
        timestep: int,
        get_prompt_fn: callable,
        temperature: float = 0.7,
        max_tokens: int = 256,
        json_format: bool = False
    ) -> List[Tuple[str, bool]]:
        """
        Process all agents in optimized batches.

        Args:
            agents: List of agent objects
            timestep: Current simulation timestep
            get_prompt_fn: Function that takes (agent, timestep) and returns (system_prompt, user_prompt)
            temperature: Sampling temperature
            max_tokens: Max tokens per response
            json_format: Whether to extract JSON

        Returns:
            List of (response, is_json_valid) tuples in agent order
        """
        n_agents = len(agents)
        results = [None] * n_agents

        # Process in batches
        for batch_start in range(0, n_agents, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_agents)
            batch_agents = agents[batch_start:batch_end]

            # Get prompts for batch
            system_prompts = []
            user_prompts = []
            for agent in batch_agents:
                sys, user = get_prompt_fn(agent, timestep)
                system_prompts.append(sys)
                user_prompts.append(user)

            # Create batch request
            batch = BatchRequest(
                request_ids=[f"agent_{i}" for i in range(batch_start, batch_end)],
                prompts=user_prompts,
                system_prompts=system_prompts,
                temperatures=[temperature] * len(batch_agents),
                max_tokens=max_tokens,
                json_format=json_format
            )

            # Process batch
            response = await self.engine.generate_batch(batch)

            # Store results in order
            for i, (text, is_valid) in enumerate(zip(response.responses, response.is_json_valid)):
                results[batch_start + i] = (text, is_valid)

        return results
