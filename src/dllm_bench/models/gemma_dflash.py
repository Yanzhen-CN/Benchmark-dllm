"""Gemma 4 served with the official DFlash speculative-decoding draft.

Gemma 4 DFlash is currently implemented by vLLM rather than Transformers.
The adapter owns one local vLLM server for the lifetime of the matrix process,
uses the OpenAI-compatible streaming endpoint for per-sample timing, and reads
vLLM's Prometheus counters to persist speculative acceptance statistics.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from ..data_paths import data_root
from ..interfaces import GenerationRequest, GenerationResult, RunStatus, TimingResult
from ..resource.energy import measure_energy_joules
from ..resource.vram import measure_peak_device_memory_gb
from .base import _looks_like_oom
from .prompting import tokenize_instruction_prompt


DEFAULT_TARGET_CHECKPOINT = "google/gemma-4-26B-A4B-it"
DEFAULT_DRAFT_CHECKPOINT = "z-lab/gemma-4-26B-A4B-it-DFlash"
_METRIC_NAMES = (
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
)


def _server_log_tail(path: Path, lines: int = 80) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


class _ManagedVLLMServer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._signature: tuple[Any, ...] | None = None
        self._log_handle = None
        self._log_path: Path | None = None

    def ensure(
        self,
        *,
        target: str,
        draft: str,
        host: str,
        port: int,
        startup_timeout_seconds: float,
        num_speculative_tokens: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        gpu_memory_utilization: float,
    ) -> str:
        import requests

        base_url = f"http://{host}:{port}"
        signature = (
            target,
            draft,
            host,
            port,
            num_speculative_tokens,
            max_model_len,
            max_num_batched_tokens,
            gpu_memory_utilization,
        )
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                if self._signature != signature:
                    raise RuntimeError("a differently configured managed DFlash server is already running")
                return base_url
            self.close()

            executable = Path(sys.executable).with_name(
                "vllm.exe" if os.name == "nt" else "vllm"
            )
            if not executable.is_file():
                raise RuntimeError(
                    f"vLLM executable not found in {executable.parent}; "
                    "run `python setup_venv.py --matrix configs/experiments/dg_comparison.yaml "
                    "-m gemma_dflash` first"
                )
            log_dir = data_root() / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = log_dir / "gemma_dflash_vllm.log"
            self._log_handle = self._log_path.open("w", encoding="utf-8")
            print(f"Gemma DFlash vLLM startup log: {self._log_path}", flush=True)
            speculative = json.dumps(
                {
                    "method": "dflash",
                    "model": draft,
                    "num_speculative_tokens": num_speculative_tokens,
                    "attention_backend": "flash_attn",
                },
                separators=(",", ":"),
            )
            command = [
                str(executable),
                "serve",
                target,
                "--host",
                host,
                "--port",
                str(port),
                "--served-model-name",
                target,
                "--speculative-config",
                speculative,
                "--attention-backend",
                "triton_attn",
                "--max-model-len",
                str(max_model_len),
                "--max-num-batched-tokens",
                str(max_num_batched_tokens),
                "--gpu-memory-utilization",
                str(gpu_memory_utilization),
                "--trust-remote-code",
            ]
            environment = os.environ.copy()
            environment.setdefault("PYTHONUNBUFFERED", "1")
            environment["NO_PROXY"] = ",".join(
                filter(None, [environment.get("NO_PROXY", ""), "127.0.0.1", "localhost"])
            )
            self._process = subprocess.Popen(
                command,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=(os.name != "nt"),
            )
            self._signature = signature

        deadline = time.monotonic() + startup_timeout_seconds
        health_url = f"{base_url}/health"
        next_status_report = time.monotonic() + 15.0
        last_status_line = ""
        while time.monotonic() < deadline:
            process = self._process
            if process is None or process.poll() is not None:
                tail = _server_log_tail(self._log_path) if self._log_path else ""
                self.close()
                raise RuntimeError(f"Gemma DFlash vLLM server exited during startup:\n{tail}")
            try:
                if requests.get(health_url, timeout=2).ok:
                    return base_url
            except requests.RequestException:
                pass
            now = time.monotonic()
            if now >= next_status_report and self._log_path is not None:
                tail = _server_log_tail(self._log_path, lines=8)
                status_lines = [line.strip() for line in tail.splitlines() if line.strip()]
                status_line = status_lines[-1] if status_lines else ""
                if status_line and status_line != last_status_line:
                    print(f"\n[vLLM startup] {status_line}", flush=True)
                    last_status_line = status_line
                next_status_report = now + 15.0
            time.sleep(2)
        tail = _server_log_tail(self._log_path) if self._log_path else ""
        self.close()
        raise RuntimeError(
            f"Gemma DFlash vLLM server did not become ready within "
            f"{startup_timeout_seconds:.0f}s:\n{tail}"
        )

    def close(self) -> None:
        process, self._process = self._process, None
        self._signature = None
        if process is not None and process.poll() is None:
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


_SERVER = _ManagedVLLMServer()
atexit.register(_SERVER.close)


def _parse_prometheus_metrics(text: str) -> dict[str, float]:
    values = {name: 0.0 for name in _METRIC_NAMES}
    found: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or " " not in line:
            continue
        left, raw_value = line.rsplit(None, 1)
        metric = left.split("{", 1)[0]
        if metric.endswith("_total"):
            metric = metric[:-6]
        if metric not in values:
            continue
        try:
            values[metric] += float(raw_value)
            found.add(metric)
        except ValueError:
            continue
    return {name: values[name] for name in found}


class GemmaDFlashAdapter:
    """Local vLLM deployment adapter; intentionally excluded from trace analysis."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_TARGET_CHECKPOINT,
        draft_model_name_or_path: str = DEFAULT_DRAFT_CHECKPOINT,
        config_name: str = "dflash",
        num_speculative_tokens: int = 15,
        max_model_len: int = 16384,
        max_num_batched_tokens: int = 32768,
        gpu_memory_utilization: float = 0.90,
        server_host: str = "127.0.0.1",
        server_port: int = 8000,
        startup_timeout_seconds: float = 1800.0,
        request_timeout_seconds: float = 7200.0,
        require_spec_metrics: bool = True,
        enable_thinking: bool = False,
    ) -> None:
        # Keep the persisted row name as ``gemma_dflash`` (model + variant),
        # matching ``gemma_ar-baseline`` without a duplicated suffix.
        self.name = "gemma"
        self.config_name = config_name
        self.supports_trace = False
        self.natively_measures_resources = False
        self.measurement_protocol = "local-vllm-streaming-v1-metrics-outside-window"
        self._model_name = model_name_or_path
        self._draft_model_name = draft_model_name_or_path
        self._num_speculative_tokens = int(num_speculative_tokens)
        self._max_model_len = int(max_model_len)
        self._max_num_batched_tokens = int(max_num_batched_tokens)
        self._gpu_memory_utilization = float(gpu_memory_utilization)
        self._server_host = server_host
        self._server_port = int(os.environ.get("DLLM_DFLASH_PORT", server_port))
        self._startup_timeout_seconds = float(startup_timeout_seconds)
        self._request_timeout_seconds = float(request_timeout_seconds)
        self._require_spec_metrics = bool(require_spec_metrics)
        self._enable_thinking = bool(enable_thinking)
        self._base_url: str | None = None
        self._tokenizer = None
        self._inference_dtype = "bfloat16"
        self.execution_path = "vllm-dflash-speculative"
        self.sampling_profile = "greedy-non-thinking"
        self.inference_optimizations = (
            "kv-cache",
            "dflash-block-diffusion-draft",
            f"speculative-block-{self._num_speculative_tokens + 1}",
        )

    def _ensure_loaded(self) -> None:
        import requests
        from transformers import AutoTokenizer
        from dllm_bench.models.device_transfer import run_gpu_loading_operation

        external = os.environ.get("DLLM_DFLASH_SERVER_URL")
        if external:
            self._base_url = external.rstrip("/")
            response = requests.get(f"{self._base_url}/health", timeout=10)
            response.raise_for_status()
        else:
            self._base_url = run_gpu_loading_operation(
                lambda: _SERVER.ensure(
                    target=self._model_name,
                    draft=self._draft_model_name,
                    host=self._server_host,
                    port=self._server_port,
                    startup_timeout_seconds=self._startup_timeout_seconds,
                    num_speculative_tokens=self._num_speculative_tokens,
                    max_model_len=self._max_model_len,
                    max_num_batched_tokens=self._max_num_batched_tokens,
                    gpu_memory_utilization=self._gpu_memory_utilization,
                ),
                label="Gemma DFlash target + draft via vLLM",
            )
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

    def warm(self) -> None:
        self._ensure_loaded()

    def _encode_prompt(self, request: GenerationRequest) -> list[int]:
        encoded = tokenize_instruction_prompt(
            self._tokenizer,
            request.prompt,
            device="cpu",
            chat_template_kwargs={"enable_thinking": self._enable_thinking},
            target_input_tokens=request.config.get("target_input_tokens"),
        )
        input_ids = encoded["input_ids"]
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return [int(token_id) for token_id in input_ids]

    def _read_spec_metrics(self) -> dict[str, float]:
        import requests

        response = requests.get(f"{self._base_url}/metrics", timeout=10)
        response.raise_for_status()
        return _parse_prometheus_metrics(response.text)

    def _stream_completion(self, input_ids: list[int], request: GenerationRequest) -> tuple[str, int, float | None]:
        import requests

        payload = {
            "model": self._model_name,
            "prompt": input_ids,
            "max_tokens": request.max_new_tokens,
            "temperature": float(request.config.get("temperature", 0.0)),
            "seed": request.seed,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        started = time.perf_counter()
        first_token_seconds: float | None = None
        chunks: list[str] = []
        completion_tokens: int | None = None
        with requests.post(
            f"{self._base_url}/v1/completions",
            json=payload,
            stream=True,
            timeout=self._request_timeout_seconds,
        ) as response:
            if not response.ok:
                raise RuntimeError(
                    f"vLLM completion failed ({response.status_code}): {response.text}"
                )
            for raw_line in response.iter_lines(decode_unicode=True):
                if isinstance(raw_line, bytes):
                    raw_line = raw_line.decode("utf-8", errors="replace")
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data = raw_line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                usage = event.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    completion_tokens = int(usage["completion_tokens"])
                choices = event.get("choices") or []
                if choices:
                    text = choices[0].get("text", "")
                    if text:
                        if first_token_seconds is None:
                            first_token_seconds = time.perf_counter() - started
                        chunks.append(text)
        output_text = "".join(chunks)
        if completion_tokens is None:
            completion_tokens = len(
                self._tokenizer.encode(output_text, add_special_tokens=False)
            )
        return output_text, completion_tokens, first_token_seconds

    def warmup_generation(self, request: GenerationRequest) -> None:
        self._ensure_loaded()
        self._stream_completion(self._encode_prompt(request), request)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self._ensure_loaded()
        input_ids = self._encode_prompt(request)
        try:
            before = self._read_spec_metrics()
            if self._require_spec_metrics and not all(name in before for name in _METRIC_NAMES):
                raise RuntimeError("vLLM did not expose the required speculative-decoding counters")
            with measure_peak_device_memory_gb() as vram, measure_energy_joules(
                synchronize=False
            ) as energy:
                started = time.perf_counter()
                output_text, output_tokens, ttft = self._stream_completion(input_ids, request)
                elapsed = time.perf_counter() - started
            after = self._read_spec_metrics()
            deltas = {
                name: max(0.0, after.get(name, before.get(name, 0.0)) - before.get(name, 0.0))
                for name in _METRIC_NAMES
            }
            drafts = int(deltas["vllm:spec_decode_num_drafts"])
            drafted_tokens = int(deltas["vllm:spec_decode_num_draft_tokens"])
            accepted_tokens = int(deltas["vllm:spec_decode_num_accepted_tokens"])
            acceptance_rate = (
                accepted_tokens / drafted_tokens if drafted_tokens else None
            )
            mean_acceptance_length = (
                1.0 + accepted_tokens / drafts if drafts else None
            )
            tpot = (
                (elapsed - ttft) / (output_tokens - 1)
                if ttft is not None and output_tokens > 1
                else None
            )
            return GenerationResult(
                request=request,
                output_text=output_text,
                status=RunStatus.SUCCESS,
                num_forward_passes=0,
                final_valid_length=output_tokens,
                timing=TimingResult(elapsed, source="measured"),
                energy_joules=energy.joules if energy.available else None,
                peak_vram_gb=vram.peak_gb if vram.available else None,
                extra={
                    "input_tokens": len(input_ids),
                    "generation_backend": "vllm",
                    "target_checkpoint": self._model_name,
                    "draft_checkpoint": self._draft_model_name,
                    "num_speculative_tokens": self._num_speculative_tokens,
                    "target_verification_passes": drafts,
                    "drafted_tokens": drafted_tokens,
                    "accepted_draft_tokens": accepted_tokens,
                    "draft_acceptance_rate": acceptance_rate,
                    "mean_acceptance_length": mean_acceptance_length,
                    "time_to_first_token_seconds": ttft,
                    "time_per_output_token_seconds": tpot,
                    "peak_vram_backend": "nvml-device-used",
                    "trace_unavailable_reason": "vllm_public_api_exposes_acceptance_counters_not_per-token_verification_history",
                },
            )
        except Exception as exc:  # persist failures/OOM through the common stage
            log_tail = _server_log_tail(_SERVER._log_path) if _SERVER._log_path else ""
            message = f"{exc}\n{log_tail}" if log_tail else str(exc)
            return GenerationResult(
                request=request,
                output_text="",
                status=RunStatus.OOM if _looks_like_oom(RuntimeError(message)) else RunStatus.FAILED,
                error_message=message,
            )
