"""Instruction-model prompt formatting shared by local HF adapters."""

from __future__ import annotations

import re
from typing import Any


_RULER_FILLER_RUN = re.compile(r"\bbackground(?: background)*\b")


def _encoded_input_length(encoded) -> int:
    input_ids = encoded["input_ids"]
    shape = getattr(input_ids, "shape", None)
    if shape is not None:
        return int(shape[-1])
    if input_ids and isinstance(input_ids[0], (list, tuple)):
        return len(input_ids[0])
    return len(input_ids)


def _resize_ruler_filler(prompt: str, filler_count: int) -> str:
    """Resize generated RULER filler while preserving payload position."""
    matches = list(_RULER_FILLER_RUN.finditer(prompt))
    if not matches:
        return prompt

    original_counts = [match.group(0).count("background") for match in matches]
    total = sum(original_counts)
    if total <= 0:
        return prompt

    raw_allocations = [filler_count * count / total for count in original_counts]
    allocations = [int(value) for value in raw_allocations]
    remainder = filler_count - sum(allocations)
    order = sorted(
        range(len(matches)),
        key=lambda index: raw_allocations[index] - allocations[index],
        reverse=True,
    )
    for index in order[:remainder]:
        allocations[index] += 1

    pieces: list[str] = []
    cursor = 0
    for match, count in zip(matches, allocations):
        pieces.append(prompt[cursor : match.start()])
        pieces.append(" ".join(["background"] * count))
        cursor = match.end()
    pieces.append(prompt[cursor:])
    return "".join(pieces)


def fit_ruler_prompt_by_whitespace(prompt: str, target_words: int) -> str:
    """Best-effort RULER fitting for APIs that expose no local tokenizer.

    Local HF adapters must use :func:`tokenize_instruction_prompt` instead.
    This helper is intentionally labelled a whitespace proxy in persisted W1
    metadata and must not be reported as an observed tokenizer length.
    """
    if target_words <= 0:
        raise ValueError("target_words must be positive")
    without_filler = _resize_ruler_filler(prompt, 0)
    fixed_words = len(without_filler.split())
    if fixed_words > target_words:
        raise ValueError(
            f"RULER payload requires {fixed_words} words, exceeding the "
            f"{target_words}-word API proxy target"
        )
    return _resize_ruler_filler(prompt, target_words - fixed_words)


def tokenize_instruction_prompt(
    tokenizer,
    prompt: str,
    *,
    device: str,
    chat_template_kwargs: dict[str, Any] | None = None,
    target_input_tokens: int | None = None,
):
    """Tokenize one user turn with the checkpoint's own chat template.

    Minimal fake tokenizers used by sampler unit tests do not implement
    ``apply_chat_template``; retaining the plain-tokenization fallback keeps
    those algorithm tests focused on sampling rather than prompt formatting.
    Real instruction checkpoints in this benchmark all ship a template.
    """
    def encode(text: str):
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            return apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **(chat_template_kwargs or {}),
            )
        return tokenizer(text, return_tensors="pt")

    encoded = encode(prompt)
    if target_input_tokens is not None:
        if target_input_tokens <= 0:
            raise ValueError("target_input_tokens must be positive")
        current_length = _encoded_input_length(encoded)
        filler_count = prompt.count("background")
        if filler_count <= 0:
            if current_length != target_input_tokens:
                raise ValueError(
                    f"tokenized prompt has {current_length} tokens, not the required "
                    f"exact {target_input_tokens}-token target, and has no resizable "
                    "RULER filler"
                )
        else:
            # Prepared prompts use a whitespace proxy. Find the largest filler
            # that fits after this checkpoint's chat template and tokenizer,
            # expanding as well as shrinking when tokenization differs.
            high = max(filler_count, 1)
            high_candidate = encode(_resize_ruler_filler(prompt, high))
            expansion_limit = max(high, target_input_tokens * 4)
            while (
                _encoded_input_length(high_candidate) <= target_input_tokens
                and high < expansion_limit
            ):
                high = min(high * 2, expansion_limit)
                high_candidate = encode(_resize_ruler_filler(prompt, high))

            low = 0
            best = None
            while low <= high:
                middle = (low + high) // 2
                candidate = encode(_resize_ruler_filler(prompt, middle))
                if _encoded_input_length(candidate) <= target_input_tokens:
                    best = candidate
                    low = middle + 1
                else:
                    high = middle - 1
            if best is None:
                minimum = encode(_resize_ruler_filler(prompt, 0))
                raise ValueError(
                    f"RULER payload and chat template require {_encoded_input_length(minimum)} "
                    f"tokens, exceeding the {target_input_tokens}-token target"
                )
            fitted_length = _encoded_input_length(best)
            if fitted_length != target_input_tokens:
                raise ValueError(
                    f"RULER filler could only fit {fitted_length} encoded tokens, not the "
                    f"required exact {target_input_tokens}-token input"
                )
            encoded = best

    if hasattr(encoded, "to"):
        return encoded.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in encoded.items()
    }
