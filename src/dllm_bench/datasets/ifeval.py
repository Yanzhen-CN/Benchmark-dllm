"""IFEval: Prompt-level Strict Accuracy, plus Instruction-level Strict and
Prompt/Instruction-level Loose (section 1), built from a checker registry
covering the observable-form-constraint + deterministic-content-requirement
combinations prioritized in Appendix A.3 (Section+Content, List+Keywords,
Heading+Items, Format+Limits+Content).

This reimplements a representative subset of the real IFEval instruction
checkers (keyword coverage/frequency, bullet/section/heading counts, case,
ending phrase, length bounds) rather than the full ~25-instruction registry —
the checker registry (``FORM_CHECKERS``/``CONTENT_CHECKERS``) is designed to
have more instruction kinds added the same way without touching scoring
logic. Termination-type constraints (length/ending phrase) are tracked but
kept out of the main ConstraintProgress score per section 4.3/Appendix A.3.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..interfaces import TraceStep
from .base import Dataset, Sample, ScoreResult
from .structeval_t import checkpoint_indices

Checker = Callable[[str, dict[str, Any]], bool]


@dataclass
class InstructionSpec:
    kind: str
    args: dict[str, Any] = field(default_factory=dict)
    terminal: bool = False
    """Termination-type constraint (length/ending phrase/exact sentence count):
    reported as auxiliary only, excluded from the main ConstraintProgress."""


@dataclass
class IFEvalSample:
    form_constraints: list[InstructionSpec] = field(default_factory=list)
    content_requirements: list[InstructionSpec] = field(default_factory=list)
    target_length_words: int | None = None


def _word_count(text: str) -> int:
    return len(text.split())


def _sentence_count(text: str) -> int:
    return len([s for s in re.split(r"[.!?]+", text) if s.strip()])


def _check_relation(actual: float, relation: str, target: float) -> bool:
    if relation == "at_least":
        return actual >= target
    if relation == "at_most":
        return actual <= target
    if relation == "around":
        return abs(actual - target) <= max(1.0, target * 0.1)
    if relation == "exactly":
        return actual == target
    raise ValueError(f"unknown relation: {relation}")


# --- form (structural) checkers -------------------------------------------------

def _check_number_words(text: str, args: dict[str, Any]) -> bool:
    return _check_relation(_word_count(text), args["relation"], args["count"])


def _check_number_sentences(text: str, args: dict[str, Any]) -> bool:
    return _check_relation(_sentence_count(text), args["relation"], args["count"])


def _check_number_bullets(text: str, args: dict[str, Any]) -> bool:
    bullet_count = len(re.findall(r"^\s*[-*]\s+", text, re.MULTILINE))
    return _check_relation(bullet_count, args.get("relation", "exactly"), args["count"])


def _check_number_sections(text: str, args: dict[str, Any]) -> bool:
    marker = args.get("marker", "Section")
    section_count = len(re.findall(rf"^\s*(#+\s*)?{re.escape(marker)}\b", text, re.MULTILINE))
    return _check_relation(section_count, args.get("relation", "exactly"), args["count"])


def _check_title(text: str, _args: dict[str, Any]) -> bool:
    return bool(re.search(r"<<[^<>]+>>", text))


def _check_all_uppercase(text: str, _args: dict[str, Any]) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _check_all_lowercase(text: str, _args: dict[str, Any]) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all(c.islower() for c in letters)


def _check_end_phrase(text: str, args: dict[str, Any]) -> bool:
    return text.strip().endswith(args["end_phrase"])


def _check_postscript(text: str, args: dict[str, Any]) -> bool:
    return args.get("marker", "P.S.") in text


FORM_CHECKERS: dict[str, Checker] = {
    "length:number_words": _check_number_words,
    "length:number_sentences": _check_number_sentences,
    "format:number_bullets": _check_number_bullets,
    "format:number_sections": _check_number_sections,
    "format:title": _check_title,
    "case:all_uppercase": _check_all_uppercase,
    "case:all_lowercase": _check_all_lowercase,
    "startend:end_phrase": _check_end_phrase,
    "content:postscript": _check_postscript,
}


# --- content checkers ------------------------------------------------------------

def _check_keyword_existence(text: str, args: dict[str, Any]) -> bool:
    lowered = text.lower()
    return all(kw.lower() in lowered for kw in args["keywords"])


def _check_keyword_frequency(text: str, args: dict[str, Any]) -> bool:
    count = text.lower().count(args["keyword"].lower())
    return _check_relation(count, args.get("relation", "at_least"), args["min_count"])


def _check_forbidden_words(text: str, args: dict[str, Any]) -> bool:
    lowered = text.lower()
    return all(w.lower() not in lowered for w in args["forbidden"])


def _check_phrase_coverage(text: str, args: dict[str, Any]) -> bool:
    lowered = text.lower()
    return all(p.lower() in lowered for p in args["phrases"])


def _check_item_coverage(text: str, args: dict[str, Any]) -> bool:
    lowered = text.lower()
    hits = sum(1 for item in args["items"] if item.lower() in lowered)
    return hits >= args.get("min_items", len(args["items"]))


CONTENT_CHECKERS: dict[str, Checker] = {
    "keywords:existence": _check_keyword_existence,
    "keywords:frequency": _check_keyword_frequency,
    "keywords:forbidden": _check_forbidden_words,
    "content:phrase_coverage": _check_phrase_coverage,
    "content:item_coverage": _check_item_coverage,
}


def _loose_variants(text: str) -> list[str]:
    stripped = text.strip()
    no_markdown = re.sub(r"[*_`#]", "", stripped)
    line_stripped = "\n".join(line.strip() for line in stripped.splitlines())
    variants = [text, stripped, no_markdown, line_stripped]
    seen: list[str] = []
    for v in variants:
        if v not in seen:
            seen.append(v)
    return seen


def check_instruction(text: str, spec: InstructionSpec, checkers: dict[str, Checker], *, strict: bool) -> bool:
    checker = checkers[spec.kind]
    if strict:
        return checker(text, spec.args)
    return any(checker(variant, spec.args) for variant in _loose_variants(text))


def evaluate_ifeval_progress(
    text: str, sample: IFEvalSample
) -> tuple[float, float, float | None]:
    """(ConstraintProgress, ContentProgress, LengthRatio) for one checkpoint —
    shared between the section-1 final score and section 4.3's per-checkpoint
    strategy-formation curves."""
    observable_form = [s for s in sample.form_constraints if not s.terminal]
    content = sample.content_requirements

    constraint_progress = (
        sum(check_instruction(text, s, FORM_CHECKERS, strict=True) for s in observable_form)
        / len(observable_form)
        if observable_form
        else 1.0
    )
    content_progress = (
        sum(check_instruction(text, s, CONTENT_CHECKERS, strict=True) for s in content)
        / len(content)
        if content
        else 1.0
    )
    length_ratio = (
        _word_count(text) / sample.target_length_words if sample.target_length_words else None
    )
    return constraint_progress, content_progress, length_ratio


def ifeval_checkpoint_scores(
    trace: list[TraceStep], sample: IFEvalSample, interval: int = 8
) -> tuple[list[float], list[float]]:
    """Section 4.2.2: score `trace`'s decoded_text at every 8th forward (plus
    always the final forward), returning (constraint_progress_scores,
    content_progress_scores) — the per-checkpoint curves
    :mod:`dllm_bench.metrics.strategy_score` turns into one sample's
    AUC/SFI. Shares `checkpoint_indices` with StructEval-T's
    version of this pipeline (only the interval and which detector runs
    differ: 4 vs 8, structure/content vs constraint/content).
    """
    if not trace:
        return [], []
    constraint_scores = []
    content_scores = []
    for i in checkpoint_indices(len(trace), interval):
        constraint_progress, content_progress, _length_ratio = evaluate_ifeval_progress(
            trace[i].decoded_text, sample
        )
        constraint_scores.append(constraint_progress)
        content_scores.append(content_progress)
    return constraint_scores, content_scores


class IFEvalDataset(Dataset):
    name = "ifeval"

    def __init__(self, samples: list[Sample] | None = None) -> None:
        self._samples = samples or []

    def load_samples(self, n: int | None = None) -> list[Sample]:
        return self._samples[:n] if n is not None else list(self._samples)

    def score(self, sample: Sample, output_text: str) -> ScoreResult:
        ref: IFEvalSample = sample.reference
        all_specs = [(s, FORM_CHECKERS) for s in ref.form_constraints] + [
            (s, CONTENT_CHECKERS) for s in ref.content_requirements
        ]

        strict_results = [check_instruction(output_text, s, checkers, strict=True) for s, checkers in all_specs]
        loose_results = [check_instruction(output_text, s, checkers, strict=False) for s, checkers in all_specs]

        prompt_level_strict = 1.0 if all(strict_results) else 0.0
        instruction_level_strict = sum(strict_results) / len(strict_results) if strict_results else 1.0
        prompt_level_loose = 1.0 if all(loose_results) else 0.0
        instruction_level_loose = sum(loose_results) / len(loose_results) if loose_results else 1.0

        constraint_progress, content_progress, length_ratio = evaluate_ifeval_progress(output_text, ref)

        aux = {
            "instruction_level_strict": instruction_level_strict,
            "prompt_level_loose": prompt_level_loose,
            "instruction_level_loose": instruction_level_loose,
            "constraint_progress": constraint_progress,
            "content_progress": content_progress,
        }
        if length_ratio is not None:
            aux["length_ratio"] = length_ratio

        return ScoreResult(
            primary_score=prompt_level_strict,
            aux=aux,
            valid=True,
            complete=bool(output_text.strip()),
        )
