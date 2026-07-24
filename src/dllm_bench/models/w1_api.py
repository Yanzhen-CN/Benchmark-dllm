"""W1 (Appendix D.3): closed-source, API-only — standard/jump/gidd configs.

Does not use :class:`~dllm_bench.models.base.BaseModelAdapter`: W1 is not a
local HF model, so there is nothing for us to wrap with our own GPU-synced
timing/energy protocol. Timing comes from whatever the API itself reports
and is tagged ``source="self_reported"`` on :class:`TimingResult` — per
section 5, that must be surfaced separately in reports, never blended into
the same column as our own measured numbers. Energy per Sample is left
``None`` (no local energy-counter access to a third-party API).

Whether the API's trace payload is granular enough for Part 4 (per-position
mask state, per-position entropy/confidence at each step) is explicitly
**unverified** as of the design doc (Appendix D.3) — ``trace_available``
defaults to ``False`` until that is confirmed against the real API response,
and ``_parse_trace`` is a best-effort placeholder for whatever shape it
turns out to have.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from ..interfaces import (
    GenerationRequest,
    GenerationResult,
    PositionState,
    RunStatus,
    TimingResult,
    TraceStep,
)

W1Config = Literal["standard", "jump", "gidd"]


class W1ApiAdapter:
    def __init__(
        self,
        config_name: W1Config,
        api_base_url: str,
        api_key: str | None = None,
        checkpoint: str = "w1",
        trace_available: bool = False,
        timeout_s: float = 120.0,
    ) -> None:
        self.name = "w1"
        self.config_name = config_name
        self.supports_trace = trace_available
        self.natively_measures_resources = True
        self._api_base_url = api_base_url.rstrip("/")
        self._api_key = api_key or os.environ.get("W1_API_KEY")
        self._checkpoint = checkpoint
        self._timeout_s = timeout_s

    def generate(self, request: GenerationRequest) -> GenerationResult:
        import requests

        payload = {
            "model": self._checkpoint,
            "prompt": request.prompt,
            "max_tokens": request.max_new_tokens,
            "mode": self.config_name,
            "temperature": request.config.get("temperature", 0.0),
            "seed": request.seed,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

        try:
            response = requests.post(
                f"{self._api_base_url}/generate",
                json=payload,
                headers=headers,
                timeout=self._timeout_s,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001 - any transport/API failure -> FAILED status, not a crash
            return GenerationResult(
                request=request,
                output_text="",
                status=RunStatus.FAILED,
                error_message=str(exc),
            )

        text = data.get("text", "")
        timing = _extract_self_reported_timing(data)
        trace = _parse_trace(data["trace"]) if self.supports_trace and "trace" in data else []

        return GenerationResult(
            request=request,
            output_text=text,
            status=RunStatus.SUCCESS,
            trace=trace,
            num_forward_passes=len(trace),
            final_valid_length=data.get("output_length", len(text.split())),
            timing=timing,
            energy_joules=None,
            extra={
                "raw_response_meta": {k: v for k, v in data.items() if k not in ("text", "trace")}
            },
        )


def _extract_self_reported_timing(data: dict[str, Any]) -> TimingResult | None:
    seconds = data.get("latency_seconds")
    if seconds is None:
        tokens_per_second = data.get("tokens_per_second")
        total_tokens = data.get("total_tokens")
        if tokens_per_second and total_tokens:
            seconds = total_tokens / tokens_per_second
    if seconds is None:
        return None
    return TimingResult(wall_clock_seconds=seconds, source="self_reported")


def _parse_trace(raw_trace: list[dict[str, Any]]) -> list[TraceStep]:
    """Best-effort conversion of the API's trace payload. Schema unverified
    (Appendix D.3) — adjust the field names below once confirmed."""
    trace: list[TraceStep] = []
    for step_index, raw_step in enumerate(raw_trace):
        try:
            canvas = raw_step.get("canvas", raw_step.get("token_ids", []))
            committed = raw_step.get("committed_positions", raw_step.get("accepted_positions", []))
            entropy_by_position = raw_step.get("entropy_by_position")
            position_states = [
                PositionState.ACCEPTED if i in committed else PositionState.MASKED
                for i in range(len(canvas))
            ]
            trace.append(
                TraceStep(
                    forward_index=step_index,
                    token_ids=list(canvas),
                    position_states=position_states,
                    committed_positions=list(committed),
                    decoded_text=raw_step.get("decoded_text", ""),
                    entropy_by_position=entropy_by_position,
                )
            )
        except Exception:  # noqa: BLE001 - unknown/changed schema shouldn't break the whole run
            continue
    return trace
