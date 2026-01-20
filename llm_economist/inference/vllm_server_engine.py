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

        # Kill any existing vLLM processes on this port (cleanup from previous runs)
        try:
            kill_cmd = f"pkill -9 -f 'vllm.*--port {self.port}' 2>/dev/null || true"
            subprocess.run(kill_cmd, shell=True, timeout=5)
            await asyncio.sleep(1)  # Give time for port to be released
            print(f"[VLLMServer] Cleaned up any stale processes on port {self.port}", flush=True)
        except Exception as e:
            logger.warning(f"Cleanup failed (non-fatal): {e}")

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

        # Pass HF_TOKEN if available, otherwise enable offline mode
        if os.environ.get("HF_TOKEN"):
            env["HF_TOKEN"] = os.environ["HF_TOKEN"]
            print(f"[VLLMServer] Passing HF_TOKEN to subprocess", flush=True)
        else:
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            print(f"[VLLMServer] No HF_TOKEN - enabling offline mode for subprocess", flush=True)

        print(f"[VLLMServer] Starting vLLM server on physical GPU {self.gpu_id}, port {self.port}", flush=True)
        print(f"[VLLMServer] Model: {self.model_name}", flush=True)
        print(f"[VLLMServer] Command: {' '.join(cmd)}", flush=True)
        logger.info(f"Starting vLLM server on GPU {self.gpu_id}, port {self.port}")

        # Start the server process - redirect stderr to a file for debugging
        import tempfile
        self._stderr_file = tempfile.NamedTemporaryFile(mode='w+', prefix='vllm_stderr_', suffix='.log', delete=False)
        self._stdout_file = tempfile.NamedTemporaryFile(mode='w+', prefix='vllm_stdout_', suffix='.log', delete=False)
        print(f"[VLLMServer] Stderr log: {self._stderr_file.name}", flush=True)

        self._server_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._stdout_file,
            stderr=self._stderr_file,
        )
        print(f"[VLLMServer] Process started with PID {self._server_process.pid}", flush=True)

        # Wait for the server to be ready
        start_time = time.time()
        check_count = 0
        while time.time() - start_time < timeout:
            check_count += 1

            # Check if process died
            poll_result = self._server_process.poll()
            if poll_result is not None:
                print(f"[VLLMServer] ✗ Process died with exit code {poll_result}", flush=True)
                # Read stderr
                self._stderr_file.flush()
                self._stderr_file.seek(0)
                stderr_content = self._stderr_file.read()
                if stderr_content:
                    print(f"[VLLMServer] Stderr output:\n{stderr_content[-2000:]}", flush=True)
                self.stop()
                raise RuntimeError(f"vLLM server process died with exit code {poll_result}")

            if check_count % 10 == 0:
                elapsed = time.time() - start_time
                print(f"[VLLMServer] Waiting for server... ({elapsed:.0f}s elapsed)", flush=True)
                # Print any stderr output so far
                try:
                    self._stderr_file.flush()
                    current_pos = self._stderr_file.tell()
                    self._stderr_file.seek(0)
                    stderr_so_far = self._stderr_file.read()
                    self._stderr_file.seek(current_pos)
                    if stderr_so_far and len(stderr_so_far) > 10:
                        print(f"[VLLMServer] Recent stderr: ...{stderr_so_far[-500:]}", flush=True)
                except Exception as e:
                    pass

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

        # Server didn't start in time - get error output
        print(f"[VLLMServer] ✗ Server failed to start within {timeout}s", flush=True)
        if self._server_process:
            try:
                self._stderr_file.flush()
                self._stderr_file.seek(0)
                stderr_content = self._stderr_file.read()
                if stderr_content:
                    print(f"[VLLMServer] Full stderr:\n{stderr_content[-3000:]}", flush=True)
            except:
                pass

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
        # Clean up temp files
        for attr in ['_stderr_file', '_stdout_file']:
            if hasattr(self, attr) and getattr(self, attr):
                try:
                    getattr(self, attr).close()
                except:
                    pass

    async def shutdown(self):
        """Async shutdown method for interface compatibility with ScalableInferenceEngine."""
        self.stop()

    async def generate_batch(self, batch: BatchRequest) -> BatchResponse:
        """
        Generate responses for a batch of prompts using concurrent requests.

        Args:
            batch: BatchRequest with prompts, system_prompts, temperatures, and parameters

        Returns:
            BatchResponse with generated texts
        """
        if not self._initialized:
            await self.start()

        responses = [""] * len(batch.prompts)
        latencies = [0.0] * len(batch.prompts)
        is_json_valid = [False] * len(batch.prompts)

        # Use concurrent requests with semaphore to limit parallelism
        # This is much faster than sequential while avoiding overwhelming the server
        max_concurrent = 32  # Limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_single(session: aiohttp.ClientSession, idx: int) -> None:
            """Process a single request with semaphore limiting."""
            async with semaphore:
                system_prompt = batch.system_prompts[idx] if idx < len(batch.system_prompts) else ""
                prompt = batch.prompts[idx]
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                temperature = batch.temperatures[idx] if idx < len(batch.temperatures) else 0.7

                try:
                    text, latency = await self._generate_single(
                        session, full_prompt, batch.max_tokens,
                        temperature, 0.9  # top_p
                    )
                    responses[idx] = text
                    latencies[idx] = latency
                    # Check JSON validity
                    if batch.json_format:
                        try:
                            import json
                            json.loads(text)
                            is_json_valid[idx] = True
                        except:
                            is_json_valid[idx] = False
                    else:
                        is_json_valid[idx] = True
                except Exception as e:
                    logger.warning(f"Request {idx} failed: {type(e).__name__}: {e}")
                    # Keep defaults (empty string, 0 latency, False validity)

        # Process all requests concurrently with semaphore limiting
        start_time = time.time()
        connector = aiohttp.TCPConnector(limit=max_concurrent * 2)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [process_single(session, i) for i in range(len(batch.prompts))]
            await asyncio.gather(*tasks)

        elapsed = time.time() - start_time
        success_count = sum(1 for r in responses if r)
        print(f"[VLLMServer] Batch: {success_count}/{len(batch.prompts)} in {elapsed:.1f}s ({len(batch.prompts)/elapsed:.1f} req/s)", flush=True)

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
        max_retries: int = 3,
    ) -> Tuple[str, float]:
        """Generate a single response with retry logic."""
        start_time = time.time()

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        last_error = None
        for attempt in range(max_retries):
            try:
                async with session.post(
                    f"{self._base_url}/v1/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"vLLM server error: {text}")

                    data = await resp.json()
                    generated_text = data["choices"][0]["text"]
                    latency = time.time() - start_time

                    return generated_text, latency
            except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                    logger.warning(f"Request failed (attempt {attempt+1}/{max_retries}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    raise

        raise last_error if last_error else RuntimeError("All retries failed")

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
