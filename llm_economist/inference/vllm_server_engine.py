"""
vLLM Server Engine - Runs vLLM in a separate process for true multi-GPU support.

This solves the problem where torch.cuda.set_device() doesn't work for vLLM.
By running vLLM as an OpenAI-compatible server in a subprocess with its own
CUDA_VISIBLE_DEVICES, we can dedicate a specific GPU to vLLM inference.
"""

import os
import subprocess
import time
import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple
import aiohttp

# Import BatchRequest/BatchResponse from async_engine for interface compatibility
from llm_economist.inference.async_engine import BatchRequest, BatchResponse

logger = logging.getLogger(__name__)


class VLLMServerEngine:
    """
    vLLM inference engine that runs as a separate server process.

    This enables true multi-GPU support by running vLLM in a subprocess
    with its own CUDA_VISIBLE_DEVICES environment variable.
    """

    def __init__(
        self,
        model_name: str,
        gpu_id: int = 0,
        port: int = 8100,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        quantization: Optional[str] = None,
    ):
        """
        Initialize vLLM server engine.

        Args:
            model_name: HuggingFace model name or path
            gpu_id: Physical GPU ID to use (not CUDA device index)
            port: Port for the vLLM server
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length
            quantization: Quantization method (awq, gptq, None)
        """
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.quantization = quantization

        self._server_process = None
        self._base_url = f"http://localhost:{port}"
        self._initialized = False

    async def start(self, timeout: int = 120):
        """Start the vLLM server in a subprocess."""
        if self._initialized:
            return

        # Build the vLLM server command
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_name,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-model-len", str(self.max_model_len),
            "--trust-remote-code",
            "--enforce-eager",  # For stability
        ]

        if self.quantization:
            cmd.extend(["--quantization", self.quantization])

        # Set environment for the subprocess - ONLY the target GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        # Also disable torch compile in subprocess
        env["TORCH_COMPILE_DISABLE"] = "1"
        env["TORCHDYNAMO_DISABLE"] = "1"
        env["VLLM_USE_V1"] = "0"

        print(f"[VLLMServer] Starting vLLM server on physical GPU {self.gpu_id}, port {self.port}", flush=True)
        print(f"[VLLMServer] Model: {self.model_name}", flush=True)
        print(f"[VLLMServer] Command: {' '.join(cmd)}", flush=True)
        logger.info(f"Starting vLLM server on GPU {self.gpu_id}, port {self.port}")

        # Start the server process
        self._server_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for the server to be ready
        start_time = time.time()
        check_count = 0
        while time.time() - start_time < timeout:
            check_count += 1
            if check_count % 10 == 0:
                elapsed = time.time() - start_time
                print(f"[VLLMServer] Waiting for server... ({elapsed:.0f}s elapsed)", flush=True)

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self._base_url}/health", timeout=5) as resp:
                        if resp.status == 200:
                            # Do a test completion to ensure model is loaded
                            test_payload = {
                                "model": self.model_name,
                                "prompt": "Hello",
                                "max_tokens": 1,
                                "temperature": 0.0,
                            }
                            async with session.post(
                                f"{self._base_url}/v1/completions",
                                json=test_payload,
                                timeout=aiohttp.ClientTimeout(total=60)
                            ) as test_resp:
                                if test_resp.status == 200:
                                    elapsed = time.time() - start_time
                                    print(f"[VLLMServer] ✓ Server ready on port {self.port} (took {elapsed:.1f}s)")
                                    logger.info(f"vLLM server ready on port {self.port}")
                                    self._initialized = True
                                    return
                                else:
                                    print(f"[VLLMServer] Server healthy but model not ready yet...")
            except Exception as e:
                if check_count % 10 == 0:
                    print(f"[VLLMServer] Waiting... (error: {type(e).__name__})")
            await asyncio.sleep(2)

        # Server didn't start in time - try to get error output
        if self._server_process:
            stderr_output = ""
            try:
                # Non-blocking read of stderr
                self._server_process.stderr.flush()
                stderr_output = self._server_process.stderr.read(4096).decode() if self._server_process.stderr else ""
            except:
                pass
            print(f"[VLLMServer] ✗ Server failed to start within {timeout}s")
            if stderr_output:
                print(f"[VLLMServer] Stderr: {stderr_output[:1000]}")

        self.stop()
        raise RuntimeError(f"vLLM server failed to start within {timeout}s")

    def stop(self):
        """Stop the vLLM server."""
        if self._server_process:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
            self._server_process = None
        self._initialized = False

    async def shutdown(self):
        """Async shutdown method for interface compatibility with ScalableInferenceEngine."""
        self.stop()

    async def generate_batch(self, batch: BatchRequest) -> BatchResponse:
        """
        Generate responses for a batch of prompts.

        Args:
            batch: BatchRequest with prompts, system_prompts, temperatures, and parameters

        Returns:
            BatchResponse with generated texts
        """
        if not self._initialized:
            await self.start()

        responses = []
        latencies = []
        is_json_valid = []

        # Limit concurrent requests to avoid overwhelming the server
        max_concurrent = 20  # Process in batches of 20
        connector = aiohttp.TCPConnector(limit=max_concurrent)

        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for i, prompt in enumerate(batch.prompts):
                # Combine system prompt with user prompt for completions API
                system_prompt = batch.system_prompts[i] if i < len(batch.system_prompts) else ""
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

                # Get per-prompt temperature
                temperature = batch.temperatures[i] if i < len(batch.temperatures) else 0.7

                task = self._generate_single(
                    session, full_prompt, batch.max_tokens,
                    temperature, 0.9  # top_p
                )
                tasks.append(task)

            # Process in chunks to avoid overwhelming the server
            results = []
            for chunk_start in range(0, len(tasks), max_concurrent):
                chunk_end = min(chunk_start + max_concurrent, len(tasks))
                chunk_tasks = tasks[chunk_start:chunk_end]
                chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
                results.extend(chunk_results)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"Request {i} failed: {type(result).__name__}: {result}", flush=True)
                    logger.error(f"Request {i} failed: {type(result).__name__}: {result}")
                    responses.append("")
                    latencies.append(0.0)
                    is_json_valid.append(False)
                else:
                    text, latency = result
                    responses.append(text)
                    latencies.append(latency)
                    # Check if JSON format was requested and response is valid
                    if batch.json_format:
                        try:
                            import json
                            json.loads(text)
                            is_json_valid.append(True)
                        except:
                            is_json_valid.append(False)
                    else:
                        is_json_valid.append(True)

        return BatchResponse(
            request_ids=batch.request_ids,
            responses=responses,
            is_json_valid=is_json_valid,
            latencies=latencies,
        )

    async def _generate_single(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Tuple[str, float]:
        """Generate a single response."""
        start_time = time.time()

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        async with session.post(
            f"{self._base_url}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),  # Increased timeout
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"vLLM server error: {text}")

            data = await resp.json()
            generated_text = data["choices"][0]["text"]
            latency = time.time() - start_time

            return generated_text, latency

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
