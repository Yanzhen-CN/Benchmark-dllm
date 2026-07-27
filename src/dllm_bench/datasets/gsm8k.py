"""GSM8K: Accuracy, plus valid-answer-rate and complete-output-rate (section 1).

Answer extraction follows the common GSM8K convention: a gold-style
``#### <number>`` marker if the model emits one, otherwise the last number
that appears in the response.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from .base import Dataset, Sample, ScoreResult
from ..data_paths import ensure_data_layout

_GOLD_MARKER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_NUMBER_RE = re.compile(r"-?\$?[\d,]+(?:\.\d+)?%?")

GSM8K_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
GSM8K_TEST_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    f"{GSM8K_REVISION}/grade_school_math/data/test.jsonl"
)


def extract_final_number(text: str) -> float | None:
    gold_match = _GOLD_MARKER_RE.search(text)
    if gold_match:
        return _to_float(gold_match.group(1))

    numbers = _NUMBER_RE.findall(text)
    if not numbers:
        return None
    return _to_float(numbers[-1])


def _to_float(token: str) -> float | None:
    cleaned = token.replace(",", "").replace("$", "")
    is_percent = cleaned.endswith("%")
    if is_percent:
        cleaned = cleaned[:-1]
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value / 100.0 if is_percent else value


def _looks_complete(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if "####" in stripped:
        return True
    return stripped[-1] in ".!?\"')" or stripped[-1].isdigit()


class GSM8KDataset(Dataset):
    name = "gsm8k"

    def __init__(
        self,
        samples: list[Sample] | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self._samples = list(samples) if samples is not None else None
        data_root = Path(
            cache_dir
            or os.environ.get("DLLM_DATA_CACHE", "")
            or ensure_data_layout()["datasets"]
        )
        self._cache_path = data_root / "gsm8k" / GSM8K_REVISION / "test.jsonl"

    def load_samples(self, n: int | None = None) -> list[Sample]:
        if self._samples is not None:
            samples = list(self._samples)
        else:
            samples = _load_official_test_samples(_ensure_official_test_file(self._cache_path))
        return samples[:n] if n is not None else samples

    def score(self, sample: Sample, output_text: str) -> ScoreResult:
        predicted = extract_final_number(output_text)
        valid = predicted is not None
        correct = valid and abs(predicted - float(sample.reference)) < 1e-4
        return ScoreResult(
            primary_score=1.0 if correct else 0.0,
            valid=valid,
            complete=_looks_complete(output_text),
        )


def _ensure_official_test_file(path: Path) -> Path:
    if path.exists() and _sha256(path.read_bytes()) == GSM8K_TEST_SHA256:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(GSM8K_TEST_URL, headers={"User-Agent": "dllm-bench/0.1"})
    try:
        with urlopen(request, timeout=60) as response:  # noqa: S310 - fixed HTTPS URL
            payload = response.read()
    except (OSError, URLError) as exc:
        raise RuntimeError(f"failed to download official GSM8K test data: {exc}") from exc

    digest = _sha256(payload)
    if digest != GSM8K_TEST_SHA256:
        raise RuntimeError(
            "official GSM8K test download failed checksum verification: "
            f"expected {GSM8K_TEST_SHA256}, got {digest}"
        )

    partial_path = path.with_suffix(path.suffix + ".part")
    try:
        partial_path.write_bytes(payload)
        os.replace(partial_path, path)
    finally:
        if partial_path.exists():
            partial_path.unlink()
    return path


def _load_official_test_samples(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open(encoding="utf-8") as dataset_file:
        for index, line in enumerate(dataset_file):
            if not line.strip():
                continue
            row = json.loads(line)
            reference = extract_final_number(row["answer"])
            if reference is None:
                raise ValueError(f"GSM8K test row {index} has no parseable gold answer")
            samples.append(
                Sample(
                    sample_id=f"gsm8k-test-{index:04d}",
                    prompt=row["question"],
                    reference=reference,
                    meta={
                        "source": "openai/grade-school-math",
                        "source_revision": GSM8K_REVISION,
                        "source_index": index,
                        "gold_solution": row["answer"],
                    },
                )
            )
    if len(samples) != 1319:
        raise ValueError(f"expected 1319 official GSM8K test rows, found {len(samples)}")
    return samples


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
