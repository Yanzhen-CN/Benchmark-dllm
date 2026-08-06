from __future__ import annotations

import hashlib
import re
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from dllm_bench.datasets.base import ScoreResult
from dllm_bench.interfaces import TraceStep


_THINK_OPEN_TAGS = ("<think>", "<analysis>", "<reasoning>")
_THINK_CLOSE_TAGS = ("</think>", "</analysis>", "</reasoning>")
_FINAL_DIGIT_ANSWER_CUE = re.compile(
    r"(?im)(?:"
    r"final(?:\s+(?:verified|correct|actual))?\s+"
    r"(?:answer|string|grid|solution)"
    r"|correct\s+final\s+(?:answer|string|grid|solution)"
    r"|(?:actual|correct|completed|solved|verified)\s+"
    r"(?:solution|grid|answer|string)(?:\s+is)?"
    r"|the\s+(?:(?:actual|correct|completed|final)\s+)?"
    r"(?:solution|answer|grid)\s+is"
    r"|let(?:'|’)s\s+just\s+provide"
    r"|(?:最终答案|最终网格|答案是|解为)"
    r")\s*[:：]?\s*"
)
_REJECTED_ANSWER_TAIL = re.compile(
    r"(?is)(?:"
    r"\(\s*(?:no|wrong|incorrect)\s*\)"
    r"|\b(?:wait|mistake)\b"
    r"|\b(?:re[- ]?(?:solve|check|evaluate|calculate))\b"
    r"|\blet\s+me\s+(?:reconsider|try\s+again|correct)\b"
    r"|\b(?:that|this|it)\s+(?:is|was)\s+"
    r"(?:not\s+right|wrong|incorrect)\b"
    r"|(?:不对|错误|重新(?:求解|检查|计算))"
    r")"
)
_MASK_FRAGMENTS = ("▢", "[MASK]", "<mask>")


@dataclass(frozen=True)
class AnswerRegion:
    text: str
    start_char: int
    end_char: int
    detected: bool
    method: str
    marker_complete: bool | None = None
    reasoning_end_char: int = 0
    unclosed_thinking: bool = False


def thinking_boundary(text: str) -> tuple[int, bool]:
    """Return the end of the last closed thinking block and whether one is open."""
    lowered = text.lower()
    close_end = 0
    for tag in _THINK_CLOSE_TAGS:
        index = lowered.rfind(tag)
        if index >= 0:
            close_end = max(close_end, index + len(tag))

    last_open = max(lowered.rfind(tag) for tag in _THINK_OPEN_TAGS)
    unclosed = last_open >= close_end and last_open >= 0
    return close_end, unclosed


def empty_answer_region(text: str, method: str = "not_found") -> AnswerRegion:
    boundary, unclosed = thinking_boundary(text)
    return AnswerRegion(
        text="",
        start_char=len(text),
        end_char=len(text),
        detected=False,
        method=method,
        reasoning_end_char=boundary,
        unclosed_thinking=unclosed,
    )


def position_aux(region: AnswerRegion, output_text: str) -> dict[str, Any]:
    aux: dict[str, Any] = {
        "answer_region_detected_rate": float(region.detected),
        "unclosed_thinking_rate": float(region.unclosed_thinking),
        "answer_detection_method": region.method,
    }
    if region.marker_complete is not None:
        aux["answer_marker_complete_rate"] = float(region.marker_complete)
    if region.detected:
        denominator = max(1, len(output_text))
        aux.update(
            {
                "answer_start_char_index": float(region.start_char),
                "answer_end_char_index": float(region.end_char),
                "answer_start_char_ratio": region.start_char / denominator,
                "reasoning_char_count": float(max(0, region.start_char)),
                "answer_char_count": float(max(0, region.end_char - region.start_char)),
            }
        )
    return aux


def scored_payload_aux(payload: str) -> dict[str, str]:
    """Return the per-sample audit hash of the exact primary-score payload."""
    return {"scored_payload_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}


def aggregate_answer_position_metrics(results: list[ScoreResult]) -> dict[str, float]:
    if not results:
        return {}
    detected = [result for result in results if result.aux.get("answer_region_detected_rate", 0.0) > 0.5]
    metrics: dict[str, float] = {
        "answer_region_detected_rate": len(detected) / len(results),
        "answer_position_token_mapped_rate": statistics.fmean(
            float(result.aux.get("answer_position_token_mapped_rate", 0.0))
            for result in results
        ),
        "unclosed_thinking_rate": statistics.fmean(
            result.aux.get("unclosed_thinking_rate", 0.0) for result in results
        ),
    }
    marker_values = [
        float(result.aux["answer_marker_complete_rate"])
        for result in results
        if "answer_marker_complete_rate" in result.aux
    ]
    if marker_values:
        metrics["answer_marker_complete_rate"] = statistics.fmean(marker_values)
        if len(marker_values) != len(results):
            metrics["answer_marker_complete_rate_eligible_ratio"] = (
                len(marker_values) / len(results)
            )
    positions = [
        result.aux["answer_start_ratio"]
        for result in detected
        if "answer_start_ratio" in result.aux
    ]
    if positions:
        metrics["answer_start_ratio_mean"] = statistics.fmean(positions)
        metrics["answer_start_ratio_median"] = statistics.median(positions)
        ordered = sorted(positions)

        def percentile(fraction: float) -> float:
            if len(ordered) == 1:
                return ordered[0]
            location = (len(ordered) - 1) * fraction
            lower = int(location)
            upper = min(lower + 1, len(ordered) - 1)
            weight = location - lower
            return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

        p25 = percentile(0.25)
        p75 = percentile(0.75)
        metrics["answer_start_ratio_p25"] = p25
        metrics["answer_start_ratio_p75"] = p75
        metrics["answer_start_ratio_iqr"] = p75 - p25
    return metrics


def aggregate_direct_answer_only_score(
    results: list[ScoreResult],
) -> dict[str, float | None]:
    """Compute a diagnostic score only over strict direct-answer outputs.

    The benchmark's primary score keeps the full selected-sample denominator.
    These fields expose the changed denominator explicitly, and use ``None``
    when no sample followed the direct-answer instruction.
    """
    eligible = [
        result
        for result in results
        if float(result.aux.get("direct_answer_instruction_following_rate", 0.0))
        == 1.0
    ]
    count = len(eligible)
    return {
        "direct_answer_eligible_count": float(count),
        "direct_answer_excluded_count": float(len(results) - count),
        "direct_answer_only_score": (
            sum(result.primary_score for result in eligible) / count
            if count
            else None
        ),
    }


def _after_thinking(text: str) -> tuple[int, bool]:
    boundary, unclosed = thinking_boundary(text)
    # An unclosed final thinking block has no trustworthy final-answer region.
    return (len(text), True) if unclosed else (boundary, False)


def locate_mbpp_answer(text: str) -> AnswerRegion:
    """Locate MBPP's final code region after the last closed thinking block."""
    scan_start, unclosed = _after_thinking(text)
    if unclosed:
        return empty_answer_region(text, "unclosed_thinking")
    tail = text[scan_start:]
    lowered = tail.lower()

    begin = lowered.rfind("[begin]")
    if begin >= 0:
        content_start = begin + len("[begin]")
        done = lowered.find("[done]", content_start)
        content_end = len(tail) if done < 0 else done
        return AnswerRegion(
            text=tail[content_start:content_end].strip(),
            start_char=scan_start + content_start,
            end_char=scan_start + content_end,
            detected=True,
            method="begin_done_marker",
            marker_complete=done >= 0,
            reasoning_end_char=scan_start,
        )

    fence = re.search(r"```(?:python)?\s*\n", tail, re.IGNORECASE)
    if fence:
        content_start = fence.end()
        close = tail.find("```", content_start)
        content_end = len(tail) if close < 0 else close
        return AnswerRegion(
            text=tail[content_start:content_end].strip(),
            start_char=scan_start + content_start,
            end_char=scan_start + content_end,
            detected=True,
            method="python_fence",
            marker_complete=close >= 0,
            reasoning_end_char=scan_start,
        )

    code_start = re.search(
        r"(?m)^\s*(?:@|async\s+def\s+|def\s+|class\s+|from\s+\S+\s+import\s+|import\s+)",
        tail,
    )
    if code_start:
        start = code_start.start()
        done = lowered.find("[done]", start)
        end = len(tail) if done < 0 else done
        return AnswerRegion(
            text=tail[start:end].strip(),
            start_char=scan_start + start,
            end_char=scan_start + end,
            detected=True,
            method="python_code_start",
            marker_complete=done >= 0,
            reasoning_end_char=scan_start,
        )
    return empty_answer_region(text)


def locate_structeval_answer(text: str) -> AnswerRegion:
    """Locate the last explicit StructEval payload after final reasoning."""
    scan_start, unclosed = _after_thinking(text)
    if unclosed:
        return empty_answer_region(text, "unclosed_thinking")
    tail = text[scan_start:]
    lowered = tail.lower()
    begin_marker = "<|begin_code|>"
    end_marker = "<|end_code|>"
    begin = lowered.rfind(begin_marker)
    if begin >= 0:
        content_start = begin + len(begin_marker)
        end = lowered.find(end_marker, content_start)
        content_end = len(tail) if end < 0 else end
        return AnswerRegion(
            text=tail[content_start:content_end].strip(),
            start_char=scan_start + content_start,
            end_char=scan_start + content_end,
            detected=True,
            method="structeval_marker",
            marker_complete=end >= 0,
            reasoning_end_char=scan_start,
        )

    fence = re.search(r"```(?:\w+)?\s*\n", tail)
    if fence:
        content_start = fence.end()
        close = tail.find("```", content_start)
        content_end = len(tail) if close < 0 else close
        return AnswerRegion(
            text=tail[content_start:content_end].strip(),
            start_char=scan_start + content_start,
            end_char=scan_start + content_end,
            detected=True,
            method="code_fence",
            marker_complete=close >= 0,
            reasoning_end_char=scan_start,
        )

    # Official compatibility: raw text can still receive a task score, but
    # cannot enter answer-position or style aggregates.
    return AnswerRegion(
        text=tail.strip(),
        start_char=scan_start,
        end_char=len(text),
        detected=False,
        method="raw_fallback",
        reasoning_end_char=scan_start,
    )


def _token_span_for_chars(
    token_texts: Sequence[str],
    start_char: int,
    end_char: int,
) -> tuple[int, int] | None:
    if not token_texts or start_char < 0 or end_char < start_char:
        return None
    joined = "".join(token_texts)
    if end_char > len(joined):
        return None

    def token_index(offset: int, *, end: bool) -> int:
        consumed = 0
        for index, piece in enumerate(token_texts):
            next_consumed = consumed + len(piece)
            if offset < next_consumed or (end and offset == next_consumed):
                return index + int(end)
            consumed = next_consumed
        return len(token_texts)

    return token_index(start_char, end=False), token_index(end_char, end=True)


def trace_answer_span(region: AnswerRegion, trace: list[TraceStep]) -> tuple[int, int] | None:
    if not region.detected or not trace:
        return None
    final = trace[-1]
    token_texts = list(getattr(final, "token_texts", []) or [])
    decoded = getattr(final, "decoded_text", "")
    if "".join(token_texts) != decoded:
        return None
    return _token_span_for_chars(token_texts, region.start_char, region.end_char)


def trace_position_aux(region: AnswerRegion, trace: list[TraceStep]) -> dict[str, float]:
    span = trace_answer_span(region, trace)
    if span is None or not trace:
        return {"answer_position_token_mapped_rate": 0.0}
    token_count = max(1, len(getattr(trace[-1], "token_texts", []) or []))
    start, end = span
    return {
        "answer_position_token_mapped_rate": 1.0,
        "answer_start_token_index": float(start),
        "answer_end_token_index": float(end),
        "final_trace_token_count": float(token_count),
        "answer_start_ratio": start / token_count,
        "reasoning_tokens_before_answer": float(start),
        "answer_token_count": float(max(0, end - start)),
    }


def locate_strict_digit_output(
    text: str, *, expected_length: int, allowed_digits: str
) -> AnswerRegion:
    """Locate a marker-free direct answer only when the whole output is valid."""
    stripped = text.strip()
    if len(stripped) != expected_length or any(
        char not in allowed_digits for char in stripped
    ):
        return empty_answer_region(text, "strict_digit_output_not_found")
    start = len(text) - len(text.lstrip())
    return AnswerRegion(
        text=stripped,
        start_char=start,
        end_char=start + len(stripped),
        detected=True,
        method="strict_digit_output",
    )


def locate_full_output(text: str, *, method: str = "full_output") -> AnswerRegion:
    """Locate the output from its first non-whitespace character through EOS."""
    if not text.strip():
        return empty_answer_region(text, "empty_output")
    start = len(text) - len(text.lstrip())
    return AnswerRegion(
        text=text[start:],
        start_char=start,
        end_char=len(text),
        detected=True,
        method=method,
    )


def _clean_trace_text(text: str) -> str:
    for fragment in _MASK_FRAGMENTS:
        text = text.replace(fragment, "")
    return text.strip()


def answer_local_checkpoint_texts(
    trace: list[TraceStep],
    region: AnswerRegion,
    checkpoint_indices: Iterable[int],
) -> tuple[list[str], bool]:
    span = trace_answer_span(region, trace)
    if span is None:
        return [], False
    start, end = span
    texts: list[str] = []
    started = False
    for index in checkpoint_indices:
        token_texts = list(getattr(trace[index], "token_texts", []) or [])
        if len(token_texts) < end:
            return [], False
        candidate = _clean_trace_text("".join(token_texts[start:end]))
        if candidate:
            started = True
        if started:
            texts.append(candidate)
    return texts, bool(texts)


def locate_digit_answer(
    text: str,
    *,
    expected_length: int,
    allowed_digits: str,
    marker_pairs: Sequence[tuple[str, str]],
    minimum_partial_length: int,
    marker_minimum_partial_length: int | None = None,
) -> AnswerRegion:
    """Locate the final submitted digit answer after optional reasoning.

    Explicit answer blocks take precedence.  Within either an answer block or
    the unmarked fallback, the last complete candidate wins so a corrected
    final solution is not replaced by an earlier draft.  Partial candidates
    may be enabled independently for explicit markers; unmarked prose should
    normally require a complete answer to avoid scoring incidental digits.
    """
    boundary, unclosed = thinking_boundary(text)
    lowered = text.lower()
    if unclosed:
        return empty_answer_region(text, "unclosed_thinking")

    marked_blocks: list[tuple[int, str, str]] = []
    for begin_marker, end_marker in marker_pairs:
        begin = lowered.rfind(begin_marker.lower(), boundary)
        if begin >= 0:
            marked_blocks.append((begin, begin_marker, end_marker))
    if marked_blocks:
        begin, begin_marker, end_marker = max(marked_blocks, key=lambda item: item[0])
        content_start = begin + len(begin_marker)
        end = lowered.find(end_marker.lower(), content_start)
        content_end = len(text) if end < 0 else end
        content = text[content_start:content_end]
        marker_minimum = (
            minimum_partial_length
            if marker_minimum_partial_length is None
            else marker_minimum_partial_length
        )
        candidates = _digit_answer_candidates(
            content,
            allowed_digits=allowed_digits,
            minimum_length=marker_minimum,
        )
        selected_candidate = _select_digit_answer_candidate(
            candidates, expected_length=expected_length
        )
        if selected_candidate is not None:
            relative_start, relative_end, digits, method = selected_candidate
            selected = (
                digits[:expected_length]
                if len(digits) >= expected_length
                else digits
            )
            return AnswerRegion(
                text=selected,
                start_char=content_start + relative_start,
                end_char=content_start + relative_end,
                detected=True,
                method=f"answer_marker_{method}",
                marker_complete=end >= 0,
                reasoning_end_char=boundary,
                unclosed_thinking=unclosed,
            )
        return AnswerRegion(
            text="",
            start_char=content_start,
            end_char=content_end,
            detected=False,
            method="answer_marker_no_digit_answer",
            marker_complete=end >= 0,
            reasoning_end_char=boundary,
            unclosed_thinking=unclosed,
        )

    scan_start = boundary if boundary > 0 else 0
    scan_text = text[scan_start:]
    final_cues = list(_FINAL_DIGIT_ANSWER_CUE.finditer(scan_text))
    if final_cues:
        cue = final_cues[-1]
        final_text = scan_text[cue.end() :]
        final_candidates = _digit_answer_candidates(
            final_text,
            allowed_digits=allowed_digits,
            minimum_length=expected_length,
        )
        selected_final = _select_digit_answer_candidate(
            final_candidates,
            expected_length=expected_length,
            source_text=final_text,
            reject_revised_candidates=True,
        )
        if selected_final is not None:
            start, end, digits, method = selected_final
            return AnswerRegion(
                text=digits[:expected_length],
                start_char=scan_start + cue.end() + start,
                end_char=scan_start + cue.end() + end,
                detected=True,
                method=f"final_cue_{method}",
                reasoning_end_char=boundary,
                unclosed_thinking=unclosed,
            )
        return AnswerRegion(
            text="",
            start_char=scan_start + cue.end(),
            end_char=len(text),
            detected=False,
            method=(
                "final_cue_rejected"
                if final_candidates
                else "final_cue_incomplete"
            ),
            reasoning_end_char=boundary,
            unclosed_thinking=unclosed,
        )

    candidates = _digit_answer_candidates(
        scan_text,
        allowed_digits=allowed_digits,
        minimum_length=minimum_partial_length,
    )
    selected_candidate = _select_digit_answer_candidate(
        candidates,
        expected_length=expected_length,
        source_text=scan_text,
        reject_revised_candidates=True,
    )
    if selected_candidate is None:
        return empty_answer_region(
            text,
            "rejected_candidate_no_final_answer" if candidates else "not_found",
        )
    start, end, digits, method = selected_candidate
    selected = digits[:expected_length]
    return AnswerRegion(
        text=selected,
        start_char=scan_start + start,
        end_char=scan_start + end,
        detected=True,
        method=method,
        reasoning_end_char=boundary,
        unclosed_thinking=unclosed,
    )


def _digit_answer_candidates(
    text: str,
    *,
    allowed_digits: str,
    minimum_length: int,
) -> list[tuple[int, int, str, str]]:
    """Return ordered compact/separated digit candidates with source spans."""
    if minimum_length < 1:
        raise ValueError("minimum_length must be positive")
    escaped = re.escape(allowed_digits)
    candidates: list[tuple[int, int, str, str]] = []
    contiguous = re.compile(
        rf"(?<!\d)([{escaped}]{{{minimum_length},}})(?!\d)"
    )
    for match in contiguous.finditer(text):
        candidates.append(
            (match.start(1), match.end(1), match.group(1), "contiguous_digits")
        )

    separated = re.compile(
        rf"(?<!\d)(?:[{escaped}](?:[ \t,;|:/\\-]{{0,3}}|\r?\n))"
        rf"{{{minimum_length - 1},}}[{escaped}](?!\d)"
    )
    for match in separated.finditer(text):
        digits = "".join(
            char for char in match.group(0) if char in allowed_digits
        )
        candidates.append(
            (match.start(), match.end(), digits, "digit_grid")
        )
    ordered = sorted(candidates, key=lambda item: (item[0], item[1]))
    deduplicated: list[tuple[int, int, str, str]] = []
    seen: set[tuple[int, int, str]] = set()
    for candidate in ordered:
        identity = candidate[:3]
        if identity in seen:
            continue
        seen.add(identity)
        deduplicated.append(candidate)
    return deduplicated


def _select_digit_answer_candidate(
    candidates: Sequence[tuple[int, int, str, str]],
    *,
    expected_length: int,
    source_text: str | None = None,
    reject_revised_candidates: bool = False,
) -> tuple[int, int, str, str] | None:
    """Prefer the last complete, non-rejected submission.

    A full candidate followed by an explicit correction/rejection cue is a
    draft, not a final answer. The check is intentionally enabled only for
    cue/fallback extraction; an explicit answer marker remains authoritative.
    """
    full = [item for item in candidates if len(item[2]) >= expected_length]
    had_full_candidate = bool(full)
    if reject_revised_candidates and source_text is not None:
        accepted: list[tuple[int, int, str, str]] = []
        for index, item in enumerate(full):
            next_start = (
                full[index + 1][0]
                if index + 1 < len(full)
                else len(source_text)
            )
            following = source_text[item[1] : next_start]
            if not _REJECTED_ANSWER_TAIL.search(following):
                accepted.append(item)
        full = accepted
    if full:
        return full[-1]
    if had_full_candidate and reject_revised_candidates:
        return None
    if not candidates:
        return None
    return max(candidates, key=lambda item: (len(item[2]), item[1]))
