"""Instruction-model prompt formatting shared by local HF adapters."""

from __future__ import annotations

from typing import Any


def tokenize_instruction_prompt(
    tokenizer,
    prompt: str,
    *,
    device: str,
    chat_template_kwargs: dict[str, Any] | None = None,
):
    """Tokenize one user turn with the checkpoint's own chat template.

    Minimal fake tokenizers used by sampler unit tests do not implement
    ``apply_chat_template``; retaining the plain-tokenization fallback keeps
    those algorithm tests focused on sampling rather than prompt formatting.
    Real instruction checkpoints in this benchmark all ship a template.
    """
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        encoded = apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **(chat_template_kwargs or {}),
        )
    else:
        encoded = tokenizer(prompt, return_tensors="pt")

    if hasattr(encoded, "to"):
        return encoded.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in encoded.items()
    }
