"""Unified model interface shared by every backend (AR, diffusion, API).

Every adapter — HF autoregressive, HF-style diffusion, W1 API, or the test
mock — implements :class:`ModelAdapter` and returns a :class:`GenerationResult`.
The runner modules (``runner/*.py``), the metrics modules
(``metrics/*.py``) and the visualization layer (``visual/public/*.py``) only ever depend on
these types, never on a concrete backend. This is what lets the design
document's "only swap the model, not the pipeline" requirement hold in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class RunStatus(str, Enum):
    """Appendix B / 3.1 `Run Status` category."""

    SUCCESS = "success"
    OOM = "oom"
    TRUNCATED = "truncated"
    FAILED = "failed"


class PositionState(str, Enum):
    """Per-position state recorded in a :class:`TraceStep` (design doc 4.1)."""

    MASKED = "masked"
    VISIBLE = "visible"
    ACCEPTED = "accepted"


@dataclass
class GenerationRequest:
    """Input to a single sample generation.

    ``config`` carries model-specific knobs (e.g. ``steps``, ``block_length``,
    ``steps_per_block`` for diffusion models, or ``temperature``/``seed`` for
    every model) so the interface stays identical across backends while each
    adapter reads whatever subset of ``config`` it understands.
    """

    prompt: str
    max_new_tokens: int
    config: dict[str, Any] = field(default_factory=dict)
    sample_id: str | None = None
    seed: int = 42


@dataclass
class TraceStep:
    """One forward pass, as required by design doc 4.1.

    ``token_ids``/``position_states`` describe the *entire* canvas at this
    step. ``committed_positions`` is the subset written or modified during
    this step (used by 4.2's final-stable-step and 4.4's commit-order).
    ``entropy_by_position``/``top1_confidence_by_position`` are optional —
    only backends that expose per-position output distributions (needed for
    4.5 Remaining-token Certainty) populate them; W1's availability depends on
    what its API actually returns (see Appendix D.3, unresolved as of writing).
    ``token_texts`` is the human-readable string per position (parallel to
    ``token_ids``), for backends that can cheaply provide it — the trace
    visualizer (``visual/public/token_grid_viz.py``) shows it in each grid cell,
    falling back to ``str(token_id)`` when a backend doesn't populate it.
    """

    forward_index: int
    token_ids: list[int]
    position_states: list[PositionState]
    committed_positions: list[int]
    decoded_text: str
    entropy_by_position: dict[int, float] | None = None
    top1_confidence_by_position: dict[int, float] | None = None
    token_texts: list[str] | None = None


@dataclass
class ForwardProfile:
    """One actual top-level model forward in a profiling run."""

    forward_index: int
    phase: str
    wall_clock_seconds: float | None = None
    compute_tflops: float | None = None
    accepted_tokens: int | None = None
    active_tokens: int | None = None
    eligible_tokens: int | None = None
    input_tokens: int | None = None
    kv_cache_tokens: int | None = None
    attention_tokens: int | None = None
    uses_kv_cache: bool | None = None
    stores_kv: bool | None = None


@dataclass
class TimingResult:
    """Appendix B timing protocol: GPU-synced wall-clock around one sample."""

    wall_clock_seconds: float
    source: str = "measured"
    """"measured" (our own before/after+sync protocol) or "self_reported"
    (e.g. W1 API Tps) — must be surfaced separately in reports, never mixed
    into the same column as "measured" numbers (design doc section 5)."""


@dataclass
class GenerationResult:
    request: GenerationRequest
    output_text: str
    status: RunStatus
    trace: list[TraceStep] = field(default_factory=list)
    forward_profiles: list[ForwardProfile] = field(default_factory=list)
    num_forward_passes: int = 0
    final_valid_length: int = 0
    timing: TimingResult | None = None
    energy_joules: float | None = None
    compute_tflops: float | None = None
    peak_vram_gb: float | None = None
    error_message: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def has_trace(self) -> bool:
        return len(self.trace) > 0


@runtime_checkable
class ModelAdapter(Protocol):
    """Contract every model backend must satisfy.

    ``name``/``config_name`` identify the model+config pair for reports (e.g.
    "illada" / "p1"). ``supports_trace`` and ``natively_measures_resources``
    let the runner and visualization layers know which Part 3/4 analyses are
    even possible for this backend (see design doc section 5's per-model
    table and Appendix D.3 for the W1 caveats).
    """

    name: str
    config_name: str
    supports_trace: bool
    natively_measures_resources: bool

    def generate(self, request: GenerationRequest) -> GenerationResult:
        ...
