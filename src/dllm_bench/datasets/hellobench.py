"""HelloBench: HelloEval Score, plus length-compliance rate, completion rate,
Seq-Rep-4, and quality-preservation retention (section 1 / 2.2).

The real HelloEval score is an LLM-judge rubric (not something this offline
harness can faithfully reproduce without wiring up an actual judge model/API).
``HelloBenchDataset`` accepts an optional ``judge_fn(prompt, output) -> float``
for that; without one it falls back to a repetition+length heuristic that is
explicitly **not** the real HelloEval metric — swap in a real judge before
using these numbers as the section-1 primary score. Seq-Rep-4 (n-gram
repetition) and length compliance are exact, judge-free metrics from the
HelloBench paper and are always computed for real.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json

from .base import Dataset, Sample, ScoreResult
from .remote import ensure_download

LENGTH_TOLERANCE = 0.1
HELLOBENCH_REVISION = "d403282968b0a61a4963a73c631d3fc1318f17d7"
HELLOBENCH_SOURCES = {
    2000: (
        "heuristic_text_generation_2k.jsonl",
        "0dfcf78d32e6d3762720883b53a4c8078036a7634853d2c21e50d25b73c5d6ed",
    ),
    4000: (
        "heuristic_text_generation_4k.jsonl",
        "60432969da1695ecbb72879ffe7a686e1ec7b93039311c09e2a2d5aa4e0f1d24",
    ),
}


@dataclass
class HelloBenchReference:
    target_length_words: int


def seq_rep_n(text: str, n: int = 4) -> float:
    """Fraction of repeated n-grams: 1 - unique_ngrams / total_ngrams."""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return 1 - len(set(ngrams)) / len(ngrams)


def length_compliant(word_count: int, target_words: int, tolerance: float = LENGTH_TOLERANCE) -> bool:
    if target_words <= 0:
        raise ValueError("target_words must be positive")
    return abs(word_count / target_words - 1.0) <= tolerance


class HelloBenchDataset(Dataset):
    name = "hellobench"

    def __init__(
        self,
        samples: list[Sample] | None = None,
        judge_fn: Callable[[str, str], float] | None = None,
    ) -> None:
        self._samples = list(samples) if samples is not None else None
        self._judge_fn = judge_fn

    def load_samples(self, n: int | None = None) -> list[Sample]:
        if self._samples is not None:
            samples = list(self._samples)
        else:
            samples = []
            for target_words, (filename, checksum) in HELLOBENCH_SOURCES.items():
                url = (
                    "https://raw.githubusercontent.com/Quehry/HelloBench/"
                    f"{HELLOBENCH_REVISION}/data/length_constrained_data/{filename}"
                )
                source = ensure_download(
                    "hellobench", filename, url=url, sha256=checksum
                )
                with source.open(encoding="utf-8") as input_file:
                    for line in input_file:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        samples.append(
                            Sample(
                                sample_id=f"hellobench-{target_words}-{row['id']}",
                                prompt=str(row["instruction"]),
                                reference=HelloBenchReference(target_words),
                                meta={
                                    "source": "Quehry/HelloBench",
                                    "source_revision": HELLOBENCH_REVISION,
                                    "category": row.get("category"),
                                    "checklists": row.get("checklists", []),
                                },
                            )
                        )
        return samples[:n] if n is not None else samples

    def score(self, sample: Sample, output_text: str) -> ScoreResult:
        ref: HelloBenchReference = sample.reference
        word_count = len(output_text.split())
        length_ratio = word_count / ref.target_length_words if ref.target_length_words else 0.0
        length_ok = length_compliant(word_count, ref.target_length_words)
        rep4 = seq_rep_n(output_text, 4)

        if self._judge_fn is not None:
            helloeval = self._judge_fn(sample.prompt, output_text)
        else:
            helloeval = max(0.0, 1 - rep4) * (1.0 if length_ok else 0.7)
        helloeval = min(1.0, max(0.0, helloeval))

        return ScoreResult(
            primary_score=helloeval,
            aux={
                "length_compliance_rate": 1.0 if length_ok else 0.0,
                "seq_rep_4": rep4,
                "length_ratio": length_ratio,
            },
            valid=True,
            complete=length_ok,
        )

    def aggregate_records(
        self, samples: list[Sample], results: list[ScoreResult]
    ) -> dict[str, float]:
        summary = super().aggregate_records(samples, results)
        targets = sorted({sample.reference.target_length_words for sample in samples})
        for target in targets:
            group = [
                result for sample, result in zip(samples, results)
                if sample.reference.target_length_words == target
            ]
            summary[f"helloeval_{target}_words"] = sum(r.primary_score for r in group) / len(group)
            summary[f"length_compliance_{target}_words"] = (
                sum(r.aux.get("length_compliance_rate", 0.0) for r in group) / len(group)
            )
            summary[f"seq_rep_4_{target}_words"] = (
                sum(r.aux.get("seq_rep_4", 0.0) for r in group) / len(group)
            )
        if 2000 in targets and 4000 in targets:
            score_2k = summary["helloeval_2000_words"]
            score_4k = summary["helloeval_4000_words"]
            summary["long_output_quality_retention"] = score_4k / score_2k if score_2k > 0 else 0.0
        return summary
