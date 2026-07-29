"""GSM8K 4-shot CoT with lm-eval's flexible final-number extraction."""

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

_FLEXIBLE_NUMBER_RE = re.compile(r"-?[$0-9.,]{2,}|-?[0-9]+")
_GENERATE_UNTIL = ("Q:", "</s>", "<|im_end|>")

GSM8K_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
GSM8K_TEST_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    f"{GSM8K_REVISION}/grade_school_math/data/test.jsonl"
)

# The first four fixed demonstrations from lm-evaluation-harness' gsm8k_cot
# task. Bertolani et al. explicitly run that task with num_fewshot=4 and the
# flexible-extract filter rather than the task YAML's default eight shots.
GSM8K_FOUR_SHOT = (
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the "
        "grove today. After they are done, there will be 21 trees. How many trees "
        "did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more "
        "were planted. So there must have been 21 - 15 = 6. The answer is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many "
        "cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer "
        "is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many "
        "pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they "
        "had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 "
        "lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to "
        "Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
)


def format_gsm8k_four_shot(question: str) -> str:
    turns = [f"Q: {q}\nA: {answer}" for q, answer in GSM8K_FOUR_SHOT]
    turns.append(f"Q: {question}\nA:")
    return "\n\n".join(turns)


def extract_final_number(text: str) -> float | None:
    """Port lm-eval ``gsm8k_cot``'s ``flexible-extract`` last match."""
    numbers = _FLEXIBLE_NUMBER_RE.findall(truncate_generate_until(text))
    if not numbers:
        return None
    return _to_float(numbers[-1])


def truncate_generate_until(text: str) -> str:
    """Apply ``gsm8k_cot``'s earliest generate-until delimiter.

    Autoregressive backends normally stop before these strings. Fixed-canvas
    diffusion backends may decode text after the answer, so applying the same
    truncation before the official regex keeps scoring backend-independent.
    """
    stops = [text.find(marker) for marker in _GENERATE_UNTIL]
    stops = [index for index in stops if index >= 0]
    return text[: min(stops)] if stops else text


def _to_float(token: str) -> float | None:
    cleaned = token.replace(",", "").replace("$", "")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value


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
        scored_text = truncate_generate_until(output_text)
        valid = predicted is not None
        correct = valid and abs(predicted - float(sample.reference)) < 1e-4
        return ScoreResult(
            primary_score=1.0 if correct else 0.0,
            valid=valid,
            complete=_looks_complete(scored_text),
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
                    prompt=format_gsm8k_four_shot(row["question"]),
                    reference=reference,
                    meta={
                        "source": "openai/grade-school-math",
                        "source_revision": GSM8K_REVISION,
                        "source_index": index,
                        "gold_solution": row["answer"],
                        "prompt_protocol": "lm-eval gsm8k_cot 4-shot first_n",
                    },
                )
            )
    if len(samples) != 1319:
        raise ValueError(f"expected 1319 official GSM8K test rows, found {len(samples)}")
    return samples


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
