from __future__ import annotations

import pytest

from dllm_bench.models.prompting import (
    fit_ruler_prompt_by_whitespace,
    tokenize_instruction_prompt,
)


class _MovableBatch(dict):
    def __init__(self, **values):
        super().__init__(values)
        self.device = None

    def to(self, device):
        self.device = device
        return self


class _ChatTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return _MovableBatch(input_ids="ids")


class _PlainTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return _MovableBatch(input_ids="ids")


class _Shape:
    def __init__(self, length):
        self.shape = (1, length)


class _LengthTokenizer:
    def __init__(self):
        self.prompts = []

    def apply_chat_template(self, messages, **kwargs):
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        # Two tokens stand in for checkpoint chat-template overhead.
        return _MovableBatch(input_ids=_Shape(len(prompt.split()) + 2))


def test_instruction_prompt_uses_checkpoint_chat_template():
    tokenizer = _ChatTokenizer()

    encoded = tokenize_instruction_prompt(
        tokenizer,
        "What is 2+3?",
        device="cuda",
        chat_template_kwargs={"enable_thinking": False},
    )

    messages, kwargs = tokenizer.calls[0]
    assert messages == [{"role": "user", "content": "What is 2+3?"}]
    assert kwargs == {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "enable_thinking": False,
    }
    assert encoded.device == "cuda"


def test_instruction_prompt_falls_back_for_minimal_test_tokenizer():
    tokenizer = _PlainTokenizer()

    encoded = tokenize_instruction_prompt(tokenizer, "prompt", device="cpu")

    assert tokenizer.calls == [("prompt", {"return_tensors": "pt"})]
    assert encoded.device == "cpu"


def test_ruler_prompt_is_fitted_after_chat_template_tokenization():
    tokenizer = _LengthTokenizer()
    prompt = (
        "background background background background\n"
        "The hidden access code is R123.\n"
        "background background background background\n\n"
        "What is the hidden access code?"
    )

    encoded = tokenize_instruction_prompt(
        tokenizer,
        prompt,
        device="cpu",
        target_input_tokens=17,
    )

    fitted = tokenizer.prompts[-1]
    assert encoded["input_ids"].shape[-1] == 17
    assert "The hidden access code is R123." in fitted
    assert "What is the hidden access code?" in fitted
    before, after = fitted.split("The hidden access code is R123.")
    assert abs(before.count("background") - after.count("background")) <= 1


def test_ruler_api_proxy_fits_whitespace_without_changing_payload():
    prompt = (
        "background background background background\n"
        "The hidden access code is R123.\n"
        "background background background background\n\n"
        "What is the hidden access code?"
    )

    fitted = fit_ruler_prompt_by_whitespace(prompt, 17)

    assert len(fitted.split()) == 17
    assert "The hidden access code is R123." in fitted
    assert "What is the hidden access code?" in fitted


def test_ruler_prompt_expands_filler_to_exact_token_target():
    tokenizer = _LengthTokenizer()
    prompt = (
        "background\n"
        "The hidden access code is R123.\n"
        "background\n\n"
        "What is the hidden access code?"
    )

    encoded = tokenize_instruction_prompt(
        tokenizer,
        prompt,
        device="cpu",
        target_input_tokens=25,
    )

    assert encoded["input_ids"].shape[-1] == 25
    assert tokenizer.prompts[-1].count("background") > 2


def test_exact_ruler_target_rejects_short_prompt_without_resizable_filler():
    tokenizer = _LengthTokenizer()

    with pytest.raises(ValueError, match="required exact 25-token target"):
        tokenize_instruction_prompt(
            tokenizer,
            "The hidden access code is R123.",
            device="cpu",
            target_input_tokens=25,
        )
