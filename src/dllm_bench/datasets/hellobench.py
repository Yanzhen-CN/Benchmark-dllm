"""HelloBench long-output diagnostics without an LLM judge.

Official HelloEval depends on checklist-based LLM judgments.  This offline
benchmark intentionally does not claim to reproduce it.  The primary score
here is therefore named ``objective_quality_score`` and combines only
auditable surface signals: target-length fidelity, Seq-Rep-4, repeated
segment rate, and explicit penalties for major observable failures.

Semantic correctness, factuality, instruction satisfaction, coherence, and
style are outside the scope of this no-judge score.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import statistics
import unicodedata

from .base import Dataset, Sample, ScoreResult
from ..interfaces import GenerationResult, RunStatus
from .remote import ensure_download

LENGTH_TOLERANCE = 0.1
SEVERE_LENGTH_RATIO_LOW = 0.5
SEVERE_LENGTH_RATIO_HIGH = 1.5
HIGH_SEQ_REP_4 = 0.25
REPEATED_SEGMENT_FRACTION = 0.20

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

_REFUSAL_PATTERNS = (
    "i cannot comply",
    "i can't comply",
    "i cannot fulfill",
    "i can't fulfill",
    "i am unable to",
    "i'm unable to",
    "as an ai language model",
    "i’m sorry, but i can’t",
    "i'm sorry, but i can't",
)
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SEGMENT_SPLIT_RE = re.compile(r"(?:\n\s*\n+)|(?<=[.!?。！？])\s+")


@dataclass
class HelloBenchReference:
    target_length_words: int


@dataclass(frozen=True)
class MajorIssueReport:
    empty_output: bool
    severe_underlength: bool
    severe_overlength: bool
    high_repetition: bool
    repeated_segment_loop: bool
    refusal: bool
    prompt_echo: bool
    corrupt_text: bool

    @property
    def count(self) -> int:
        return sum(
            (
                self.empty_output,
                self.severe_underlength,
                self.severe_overlength,
                self.high_repetition,
                self.repeated_segment_loop,
                self.refusal,
                self.prompt_echo,
                self.corrupt_text,
            )
        )

    def as_metrics(self) -> dict[str, float]:
        return {
            "empty_output_issue_rate": float(self.empty_output),
            "severe_underlength_issue_rate": float(self.severe_underlength),
            "severe_overlength_issue_rate": float(self.severe_overlength),
            "high_repetition_issue_rate": float(self.high_repetition),
            "repeated_segment_loop_issue_rate": float(self.repeated_segment_loop),
            "refusal_issue_rate": float(self.refusal),
            "prompt_echo_issue_rate": float(self.prompt_echo),
            "corrupt_text_issue_rate": float(self.corrupt_text),
            "major_issue_count": float(self.count),
            "major_issue_free_rate": 1.0 if self.count == 0 else 0.0,
        }


def seq_rep_n(text: str, n: int = 4) -> float:
    """Fraction of repeated whitespace-token n-grams."""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1 - len(set(ngrams)) / len(ngrams)


def length_compliant(
    word_count: int, target_words: int, tolerance: float = LENGTH_TOLERANCE
) -> bool:
    if target_words <= 0:
        raise ValueError("target_words must be positive")
    return abs(word_count / target_words - 1.0) <= tolerance


def repeated_segment_fraction(text: str) -> float:
    """Fraction of words in repeated, exact sentence/paragraph segments.

    Segments shorter than eight words are ignored so legitimate recurring
    headings and short dialogue do not dominate the signal.
    """
    normalized_segments: list[tuple[str, int]] = []
    for segment in _SEGMENT_SPLIT_RE.split(text):
        words = _WORD_RE.findall(segment.casefold())
        if len(words) >= 8:
            normalized_segments.append((" ".join(words), len(words)))
    total_words = sum(length for _, length in normalized_segments)
    if total_words == 0:
        return 0.0
    seen: set[str] = set()
    repeated_words = 0
    for normalized, length in normalized_segments:
        if normalized in seen:
            repeated_words += length
        else:
            seen.add(normalized)
    return repeated_words / total_words


def _contains_prompt_echo(prompt: str, output: str, window: int = 30) -> bool:
    prompt_words = _WORD_RE.findall(prompt.casefold())
    output_words = _WORD_RE.findall(output.casefold())
    if len(prompt_words) < window or len(output_words) < window:
        return False
    prompt_windows = {
        tuple(prompt_words[index : index + window])
        for index in range(len(prompt_words) - window + 1)
    }
    return any(
        tuple(output_words[index : index + window]) in prompt_windows
        for index in range(len(output_words) - window + 1)
    )


def _contains_corrupt_text(text: str) -> bool:
    if "\ufffd" in text or "\x00" in text:
        return True
    return any(
        unicodedata.category(character) == "Cc" and character not in "\n\r\t"
        for character in text
    )


def detect_major_issues(
    prompt: str,
    output: str,
    target_words: int,
    *,
    rep4: float | None = None,
    repeated_fraction: float | None = None,
) -> MajorIssueReport:
    word_count = len(output.split())
    length_ratio = word_count / target_words if target_words > 0 else 0.0
    rep4 = seq_rep_n(output, 4) if rep4 is None else rep4
    repeated_fraction = (
        repeated_segment_fraction(output)
        if repeated_fraction is None
        else repeated_fraction
    )
    lowered = output.casefold()
    return MajorIssueReport(
        empty_output=word_count == 0,
        severe_underlength=length_ratio < SEVERE_LENGTH_RATIO_LOW,
        severe_overlength=length_ratio > SEVERE_LENGTH_RATIO_HIGH,
        high_repetition=rep4 >= HIGH_SEQ_REP_4,
        repeated_segment_loop=repeated_fraction >= REPEATED_SEGMENT_FRACTION,
        refusal=any(pattern in lowered for pattern in _REFUSAL_PATTERNS),
        prompt_echo=_contains_prompt_echo(prompt, output),
        corrupt_text=_contains_corrupt_text(output),
    )


def objective_quality_score(
    length_ratio: float,
    rep4: float,
    repeated_fraction: float,
    issues: MajorIssueReport,
) -> tuple[float, float, float, float]:
    """Return the transparent no-judge score and its three components."""
    length_score = max(0.0, 1.0 - abs(length_ratio - 1.0))
    repetition_score = max(0.0, 1.0 - rep4 / 0.5)
    segment_score = max(0.0, 1.0 - repeated_fraction / 0.5)
    score = 0.50 * length_score + 0.35 * repetition_score + 0.15 * segment_score

    # Multiplicative penalties keep each failure independently visible and
    # make combinations of catastrophic problems substantially worse.
    if issues.empty_output:
        score = 0.0
    if issues.severe_underlength:
        score *= 0.35
    if issues.severe_overlength:
        score *= 0.75
    if issues.high_repetition:
        score *= 0.65
    if issues.repeated_segment_loop:
        score *= 0.60
    if issues.refusal:
        score *= 0.20
    if issues.prompt_echo:
        score *= 0.70
    if issues.corrupt_text:
        score *= 0.50
    return max(0.0, min(1.0, score)), length_score, repetition_score, segment_score


class HelloBenchDataset(Dataset):
    name = "hellobench"

    def __init__(self, samples: list[Sample] | None = None) -> None:
        self._samples = list(samples) if samples is not None else None

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
                                    # Retained as provenance only. No LLM judge
                                    # is invoked by this scorer.
                                    "checklists": row.get("checklists", []),
                                },
                            )
                        )
        return samples[:n] if n is not None else samples

    def score(self, sample: Sample, output_text: str) -> ScoreResult:
        ref: HelloBenchReference = sample.reference
        word_count = len(output_text.split())
        length_ratio = word_count / ref.target_length_words
        length_ok = length_compliant(word_count, ref.target_length_words)
        rep4 = seq_rep_n(output_text, 4)
        repeated_fraction = repeated_segment_fraction(output_text)
        issues = detect_major_issues(
            sample.prompt,
            output_text,
            ref.target_length_words,
            rep4=rep4,
            repeated_fraction=repeated_fraction,
        )
        score, length_score, repetition_score, segment_score = objective_quality_score(
            length_ratio, rep4, repeated_fraction, issues
        )
        return ScoreResult(
            primary_score=score,
            aux={
                "length_compliance_rate": 1.0 if length_ok else 0.0,
                "seq_rep_4": rep4,
                "length_ratio": length_ratio,
                "output_word_count": float(word_count),
                "objective_length_score": length_score,
                "objective_repetition_score": repetition_score,
                "objective_segment_score": segment_score,
                "repeated_segment_fraction": repeated_fraction,
                **issues.as_metrics(),
            },
            valid=word_count > 0 and not issues.corrupt_text,
            complete=length_ok,
        )

    def aggregate_records(
        self, samples: list[Sample], results: list[ScoreResult]
    ) -> dict[str, float]:
        summary = super().aggregate_records(samples, results)
        summary["objective_quality_score"] = summary["hellobench_score"]
        targets = sorted({sample.reference.target_length_words for sample in samples})
        for target in targets:
            group = [
                result
                for sample, result in zip(samples, results)
                if sample.reference.target_length_words == target
            ]
            summary[f"objective_quality_{target}_words"] = (
                sum(result.primary_score for result in group) / len(group)
            )
            summary[f"length_compliance_{target}_words"] = (
                sum(result.aux["length_compliance_rate"] for result in group) / len(group)
            )
            summary[f"seq_rep_4_{target}_words"] = (
                sum(result.aux["seq_rep_4"] for result in group) / len(group)
            )
            summary[f"major_issue_free_{target}_words"] = (
                sum(result.aux["major_issue_free_rate"] for result in group) / len(group)
            )
            summary[f"mean_output_words_{target}_words"] = (
                sum(result.aux["output_word_count"] for result in group) / len(group)
            )
        if 2000 in targets and 4000 in targets:
            score_2k = summary["objective_quality_2000_words"]
            score_4k = summary["objective_quality_4000_words"]
            summary["long_output_quality_retention"] = (
                score_4k / score_2k if score_2k > 0 else 0.0
            )
        return summary

    def aggregate_generation_records(
        self, samples: list[Sample], generations: list[GenerationResult]
    ) -> dict[str, float]:
        if len(samples) != len(generations):
            raise ValueError("samples and generations must have the same length")
        summary: dict[str, float] = {}
        targets = sorted({sample.reference.target_length_words for sample in samples})
        for target in targets:
            group = [
                generation
                for sample, generation in zip(samples, generations)
                if sample.reference.target_length_words == target
            ]
            successful = [
                generation for generation in group
                if generation.status == RunStatus.SUCCESS
            ]
            timed = [
                generation
                for generation in successful
                if generation.timing is not None
                and generation.timing.wall_clock_seconds > 0
            ]
            times = [generation.timing.wall_clock_seconds for generation in timed]
            summary[f"sample_count_{target}_words"] = float(len(group))
            summary[f"generation_success_rate_{target}_words"] = (
                len(successful) / len(group) if group else 0.0
            )
            summary[f"timed_sample_count_{target}_words"] = float(len(timed))
            if times:
                mean_seconds = statistics.fmean(times)
                summary[f"generation_time_mean_seconds_{target}_words"] = mean_seconds
                summary[f"generation_time_median_seconds_{target}_words"] = statistics.median(times)
                summary[f"generation_time_min_seconds_{target}_words"] = min(times)
                summary[f"generation_time_max_seconds_{target}_words"] = max(times)
                summary[f"generation_time_mean_hours_{target}_words"] = mean_seconds / 3600.0
                summary[f"mean_generated_tokens_{target}_words"] = statistics.fmean(
                    generation.final_valid_length for generation in timed
                )
        return summary
