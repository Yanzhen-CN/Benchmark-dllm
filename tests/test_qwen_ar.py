from pathlib import Path

from dllm_bench.registry import build_model_adapter


ROOT = Path(__file__).resolve().parent.parent


def test_qwen3_8b_config_builds_distinct_ar_adapter():
    adapter = build_model_adapter(ROOT / "configs" / "models" / "qwen3_8b.yaml")

    assert adapter.name == "qwen3_8b"
    assert adapter.config_name == "ar-baseline"
    assert adapter._model_name == "Qwen/Qwen3-8B"
    assert adapter._enable_thinking is False


def test_qwen3_4b_default_adapter_name_is_unchanged():
    adapter = build_model_adapter(ROOT / "configs" / "models" / "qwen3_4b.yaml")

    assert adapter.name == "qwen3_4b"
