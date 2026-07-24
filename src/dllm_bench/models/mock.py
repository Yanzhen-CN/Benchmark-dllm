"""Pure-Python fake diffusion backend.

Exists purely to exercise the rest of the framework (orchestrator, metrics,
report generation) without a GPU, real weights, or network access. It
simulates a canvas of word-level positions that get unmasked over a handful
of forward passes, populating every :class:`~dllm_bench.interfaces.TraceStep`
field a real diffusion adapter would (mask state, committed positions,
per-position entropy/top-1 confidence) so Part 4 analyses can be unit-tested
against something deterministic.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from ..interfaces import (
    GenerationRequest,
    GenerationResult,
    PositionState,
    RunStatus,
    TraceStep,
)
from .base import BaseModelAdapter

MASK_PLACEHOLDER = "[MASK]"


def _default_response(request: GenerationRequest) -> str:
    words = request.prompt.strip().split()
    seed_word = words[-1] if words else "answer"
    return f"The result is {seed_word} done"


class MockDiffusionAdapter(BaseModelAdapter):
    """Deterministic (given ``request.seed``) fake diffusion generator."""

    def __init__(
        self,
        name: str = "mock",
        config_name: str = "default",
        response_fn: Callable[[GenerationRequest], str] | None = None,
        steps: int = 8,
    ) -> None:
        self.name = name
        self.config_name = config_name
        self.supports_trace = True
        self.natively_measures_resources = False
        self._response_fn = response_fn or _default_response
        self._steps = steps

    def _generate_core(self, request: GenerationRequest) -> GenerationResult:
        rng = random.Random(request.seed)
        target_text = self._response_fn(request)
        tokens = target_text.split(" ")
        n = min(len(tokens), max(request.max_new_tokens, 1))
        n = max(n, 1)
        tokens = tokens[:n]
        n = len(tokens)

        steps = max(1, min(self._steps, n))
        order = list(range(n))
        rng.shuffle(order)
        chunks: list[list[int]] = [[] for _ in range(steps)]
        for i, pos in enumerate(order):
            chunks[i % steps].append(pos)

        committed: set[int] = set()
        trace: list[TraceStep] = []
        for step_index, chunk in enumerate(chunks):
            committed.update(chunk)
            position_states = [
                PositionState.ACCEPTED if p in committed else PositionState.MASKED
                for p in range(n)
            ]
            token_ids = [
                hash(tokens[p]) % 50_000 if p in committed else -1 for p in range(n)
            ]
            token_texts = [
                tokens[p] if p in committed else MASK_PLACEHOLDER for p in range(n)
            ]
            remaining = [p for p in range(n) if p not in committed]
            progress = (step_index + 1) / steps
            entropy_by_position = {
                p: max(0.0, (1.0 - progress) * rng.random()) for p in remaining
            }
            top1_by_position = {
                p: min(1.0, progress * (0.5 + 0.5 * rng.random())) for p in remaining
            }
            decoded = " ".join(
                tokens[p] if p in committed else MASK_PLACEHOLDER for p in range(n)
            )
            trace.append(
                TraceStep(
                    forward_index=step_index,
                    token_ids=token_ids,
                    position_states=position_states,
                    committed_positions=sorted(chunk),
                    decoded_text=decoded,
                    entropy_by_position=entropy_by_position or None,
                    top1_confidence_by_position=top1_by_position or None,
                    token_texts=token_texts,
                )
            )

        return GenerationResult(
            request=request,
            output_text=" ".join(tokens),
            status=RunStatus.SUCCESS,
            trace=trace,
            num_forward_passes=steps,
            final_valid_length=n,
        )
