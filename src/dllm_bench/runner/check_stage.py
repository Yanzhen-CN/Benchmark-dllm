"""Read-only validation of generated, scored, and visualized matrix artifacts.

Generation JSON can contain multi-gigabyte traces.  The checker deliberately
parses only the top-level prefix that precedes ``trace`` and records whether
the trace array is empty, so validating a transferred run does not duplicate
the trace in memory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

from ..registry import load_yaml
from .matrix import MatrixJob
from .output_layout import (
    resolve_model_output_dir,
    resolve_score_output_dir,
    visualization_output_dir,
)


_MAX_PREFIX_CHARS = 16 * 1024 * 1024
_VALID_GENERATION_STATUSES = {"success", "truncated"}


@dataclass
class ArtifactCheck:
    model: str
    variant: str
    dataset: str
    optional: bool
    generation_dir: str
    expected_samples: int | None = None
    actual_samples: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    trace_samples: int = 0
    score_complete: bool | None = None
    visualization_present: bool | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


def _read_generation_prefix(path: Path) -> dict[str, Any]:
    """Load the GenerationResult fields before its potentially huge trace."""
    prefix: list[str] = []
    prefix_chars = 0
    trace_nonempty = False
    found_trace = False
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith('"trace"'):
                found_trace = True
                value = stripped.split(":", 1)[1].strip().rstrip(",")
                trace_nonempty = value != "[]"
                prefix.extend(['  "trace": []\n', "}\n"])
                break
            prefix.append(line)
            prefix_chars += len(line)
            if prefix_chars > _MAX_PREFIX_CHARS:
                raise ValueError(
                    f"generation prefix exceeds {_MAX_PREFIX_CHARS} characters"
                )
    if not found_trace:
        raise ValueError("top-level trace field is missing or not line-delimited")
    payload = json.loads("".join(prefix))
    payload["_trace_nonempty"] = trace_nonempty
    return payload


def _normalise_length(value: str) -> int:
    normalised = str(value).strip().lower()
    if normalised.endswith("k"):
        return int(normalised[:-1]) * 1000
    return int(normalised)


def expected_sample_count(
    job: MatrixJob,
    dataset_config: dict[str, Any],
    n_samples_override: int | None = None,
) -> int | None:
    if n_samples_override is not None:
        return n_samples_override
    if job.n_samples is not None:
        return job.n_samples
    if dataset_config.get("sample_size") is not None:
        return int(dataset_config["sample_size"])

    dataset_name = str(dataset_config.get("dataset", job.dataset_config.stem))
    if dataset_name == "hellobench":
        profiles = dataset_config.get("output_profiles", [])
        selected = (
            {_normalise_length(value) for value in job.hellobench_lengths}
            if job.hellobench_lengths
            else {int(profile["target_words"]) for profile in profiles}
        )
        return int(dataset_config.get("samples_per_length", 1)) * len(selected)
    if dataset_name == "ruler":
        positions = dataset_config.get("positions", [])
        windows = dataset_config.get("dataset_kwargs", {}).get(
            "context_windows", [None]
        )
        per_position = int(
            dataset_config.get("samples_per_context_window_position", 1)
        )
        return len(positions) * len(windows) * per_position
    if dataset_config.get("protocol_type") == "capacity_diagnostic":
        return 1
    return None


def _expected_max_new_tokens(
    job: MatrixJob,
    dataset_config: dict[str, Any],
    sample_id: str,
    max_new_tokens_override: int | None,
) -> int:
    if max_new_tokens_override is not None:
        return max_new_tokens_override
    if dataset_config.get("dataset") != "hellobench":
        return job.max_new_tokens
    for profile in dataset_config.get("output_profiles", []):
        target_words = int(profile["target_words"])
        if f"-{target_words}-" in sample_id:
            return int(profile["max_new_tokens"])
    raise ValueError(f"cannot identify HelloBench output profile for {sample_id}")


def _variant_capabilities(job: MatrixJob, variant: str) -> tuple[bool, bool]:
    config = load_yaml(job.model_config)
    variant_config = config["configs"][variant]
    adapter = str(variant_config.get("adapter", ""))
    init_kwargs = variant_config.get("init_kwargs", {})
    if "gemma_dflash" in adapter:
        supports_trace = False
    elif "w1_api" in adapter:
        supports_trace = bool(init_kwargs.get("trace_available", False))
    else:
        supports_trace = True
    natively_measures_resources = "w1_api" in adapter
    return supports_trace, natively_measures_resources


def _sample_json_paths(directory: Path) -> dict[str, Path]:
    ignored = {"_meta.json", "oom_info.json", "summary.json"}
    return {
        path.stem: path
        for path in directory.glob("*.json")
        if path.name not in ignored
    }


def _compact_sample_errors(messages: list[str]) -> list[str]:
    """Collapse identical per-sample failures into one actionable line."""
    plain: list[str] = []
    grouped: dict[str, list[str]] = {}
    order: list[str] = []
    for message in messages:
        sample_id, separator, detail = message.partition(": ")
        if not separator:
            plain.append(message)
            continue
        if detail not in grouped:
            grouped[detail] = []
            order.append(detail)
        grouped[detail].append(sample_id)
    compacted = list(plain)
    for detail in order:
        sample_ids = grouped[detail]
        if len(sample_ids) == 1:
            compacted.append(f"{sample_ids[0]}: {detail}")
            continue
        examples = ", ".join(sample_ids[:3])
        compacted.append(
            f"{len(sample_ids)} samples: {detail} (e.g. {examples})"
        )
    return compacted


def _check_score(
    row: ArtifactCheck,
    *,
    output_root: str | Path,
    selected_ids: list[str],
    primary_metric: str | None,
) -> None:
    directory = resolve_score_output_dir(
        output_root, row.model, row.variant, row.dataset
    )
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        row.score_complete = False
        row.errors.append("score summary.json is missing")
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        row.score_complete = False
        row.errors.append(f"score summary is unreadable: {exc}")
        return

    score_ids = set(_sample_json_paths(directory))
    expected_ids = set(selected_ids)
    missing = expected_ids.difference(score_ids)
    extra = score_ids.difference(expected_ids)
    scoring = summary.get("scoring_metadata", {})
    if missing:
        row.errors.append(f"score files missing for {len(missing)} selected sample(s)")
    if extra:
        row.errors.append(f"score directory has {len(extra)} stale sample file(s)")
    if int(summary.get("n_samples", -1)) != len(selected_ids):
        row.errors.append(
            f"score summary n_samples={summary.get('n_samples')} != {len(selected_ids)}"
        )
    if int(scoring.get("actual_scored_sample_count", -1)) != len(selected_ids):
        row.errors.append("score fingerprint metadata has an incomplete sample count")
    if primary_metric and scoring.get("primary_metric") != primary_metric:
        row.errors.append(
            "score primary metric "
            f"{scoring.get('primary_metric')!r} != config {primary_metric!r}"
        )
    row.score_complete = not any(error.startswith("score") for error in row.errors)


def _check_visualization(
    row: ArtifactCheck,
    *,
    output_root: str | Path,
) -> None:
    directory = visualization_output_dir(
        output_root, row.model, row.variant, row.dataset
    )
    row.visualization_present = directory.is_dir() and any(directory.iterdir())
    if not row.visualization_present:
        row.errors.append("visualization output is missing")


def check_job_variant(
    job: MatrixJob,
    variant: str,
    *,
    output_root: str | Path,
    seed: int,
    stage: str = "generate",
    n_samples_override: int | None = None,
    max_new_tokens_override: int | None = None,
    optional: bool = False,
) -> ArtifactCheck:
    dataset_config = load_yaml(job.dataset_config)
    dataset_name = str(dataset_config.get("dataset", job.dataset_config.stem))
    directory = resolve_model_output_dir(
        output_root, job.model_name, variant, dataset_name
    )
    row = ArtifactCheck(
        model=job.model_name,
        variant=variant,
        dataset=dataset_name,
        optional=optional,
        generation_dir=str(directory),
        expected_samples=expected_sample_count(
            job, dataset_config, n_samples_override
        ),
    )
    meta_path = directory / "_meta.json"
    if not meta_path.is_file():
        message = "generation _meta.json is missing"
        (row.warnings if optional else row.errors).append(message)
        return row
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        row.errors.append(f"generation metadata is unreadable: {exc}")
        return row

    if (directory / "oom_info.json").exists() or meta.get("test_valid") is False:
        row.errors.append("dataset is invalid (OOM or infrastructure failure)")
    if meta.get("test_complete") is not True:
        row.errors.append("generation metadata says test_complete=false")
    if meta.get("config_name") != variant:
        row.errors.append(
            f"metadata variant {meta.get('config_name')!r} != {variant!r}"
        )
    if meta.get("dataset_name") != dataset_name:
        row.errors.append(
            f"metadata dataset {meta.get('dataset_name')!r} != {dataset_name!r}"
        )

    selected_ids = [str(value) for value in meta.get("selected_sample_ids", [])]
    declared = int(meta.get("selected_samples", -1))
    completed = int(meta.get("completed_samples", -1))
    sample_paths = _sample_json_paths(directory)
    row.actual_samples = len(sample_paths)
    if len(selected_ids) != declared:
        row.errors.append(
            f"selected_sample_ids={len(selected_ids)} != selected_samples={declared}"
        )
    if completed != declared:
        row.errors.append(f"completed_samples={completed} != selected_samples={declared}")
    if row.expected_samples is not None and declared != row.expected_samples:
        row.errors.append(
            f"selected_samples={declared} != config expectation {row.expected_samples}"
        )
    missing_ids = set(selected_ids).difference(sample_paths)
    extra_ids = set(sample_paths).difference(selected_ids)
    if missing_ids:
        row.errors.append(f"{len(missing_ids)} selected generation file(s) are missing")
    if extra_ids:
        row.errors.append(f"{len(extra_ids)} stale generation file(s) are present")

    run_metadata = meta.get("run_metadata", {})
    if int(run_metadata.get("seed", seed)) != seed:
        row.errors.append(
            f"run seed={run_metadata.get('seed')} != matrix seed={seed}"
        )
    if run_metadata.get("require_all_metrics") is not True:
        row.warnings.append("run did not require all formal metrics")
    if run_metadata.get("measure_compute") is True:
        row.warnings.append("run includes optional compute replay")

    expected_trace = str(dataset_config.get("trace_scope", "all_samples")) == "all_samples"
    supports_trace, native_resources = _variant_capabilities(job, variant)
    for sample_id in selected_ids:
        path = sample_paths.get(sample_id)
        if path is None:
            continue
        try:
            payload = _read_generation_prefix(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            row.errors.append(f"{sample_id}: unreadable generation prefix: {exc}")
            continue
        request = payload.get("request", {})
        status = str(payload.get("status", "missing"))
        row.status_counts[status] = row.status_counts.get(status, 0) + 1
        if status not in _VALID_GENERATION_STATUSES:
            row.errors.append(f"{sample_id}: invalid run status {status!r}")
        if request.get("sample_id") != sample_id:
            row.errors.append(
                f"{sample_id}: request sample_id={request.get('sample_id')!r}"
            )
        try:
            expected_max = _expected_max_new_tokens(
                job,
                dataset_config,
                sample_id,
                max_new_tokens_override,
            )
        except ValueError as exc:
            row.errors.append(str(exc))
        else:
            if int(request.get("max_new_tokens", -1)) != expected_max:
                row.errors.append(
                    f"{sample_id}: max_new_tokens={request.get('max_new_tokens')} "
                    f"!= config {expected_max}"
                )
        if int(request.get("seed", seed)) != seed:
            row.errors.append(
                f"{sample_id}: request seed={request.get('seed')} != {seed}"
            )
        capture_trace = request.get("config", {}).get("capture_trace")
        if capture_trace is not expected_trace:
            row.errors.append(
                f"{sample_id}: capture_trace={capture_trace!r} "
                f"!= config {expected_trace}"
            )
        if payload.get("_trace_nonempty"):
            row.trace_samples += 1
        elif expected_trace and supports_trace and status == "success":
            row.errors.append(f"{sample_id}: required trace is empty")

        timing = payload.get("timing")
        if not timing or float(timing.get("wall_clock_seconds", 0)) <= 0:
            row.errors.append(f"{sample_id}: timing is missing or non-positive")
        if not native_resources:
            if payload.get("energy_joules") is None:
                row.errors.append(f"{sample_id}: energy_joules is missing")
            if payload.get("peak_vram_gb") is None:
                row.errors.append(f"{sample_id}: peak_vram_gb is missing")
        if run_metadata.get("measure_compute") and payload.get("compute_tflops") is None:
            row.errors.append(f"{sample_id}: compute_tflops is missing")

    if stage in {"score", "all"} and not row.errors:
        _check_score(
            row,
            output_root=output_root,
            selected_ids=selected_ids,
            primary_metric=dataset_config.get("primary_metric"),
        )
    if stage in {"visualize", "all"} and not row.errors:
        _check_visualization(row, output_root=output_root)
    row.errors = _compact_sample_errors(row.errors)
    return row


def serialise_checks(rows: Iterable[ArtifactCheck]) -> list[dict[str, Any]]:
    return [row.to_dict() for row in rows]
