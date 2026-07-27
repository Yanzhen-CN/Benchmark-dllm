from __future__ import annotations

from dllm_bench.models.prompting import tokenize_instruction_prompt


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
