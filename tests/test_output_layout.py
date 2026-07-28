from dllm_bench.runner.output_layout import (
    model_output_dir,
    resolve_model_output_dir,
    run_id,
)


def test_qwen_ar_baseline_does_not_repeat_variant_in_run_id():
    assert run_id("qwen3_4b", "ar-baseline") == "qwen3_4b"
    assert run_id("qwen3_8b", "ar-baseline") == "qwen3_8b"
    assert run_id("illada", "best") == "illada_best"


def test_qwen_model_output_uses_unsuffixed_canonical_directory(tmp_path):
    path = model_output_dir(tmp_path, "qwen3_4b", "ar-baseline", "gsm8k")
    assert path == tmp_path / "model_output" / "qwen3_4b" / "gsm8k"

    path_8b = model_output_dir(tmp_path, "qwen3_8b", "ar-baseline", "gsm8k")
    assert path_8b == tmp_path / "model_output" / "qwen3_8b" / "gsm8k"


def test_dllm_architecture_and_sampling_axes_are_both_in_the_run_id():
    assert run_id("illada", "best") == "illada_best"


def test_qwen_reader_falls_back_to_legacy_suffixed_directory(tmp_path):
    legacy = tmp_path / "model_output" / "qwen3_4b_ar-baseline" / "gsm8k"
    legacy.mkdir(parents=True)

    assert (
        resolve_model_output_dir(tmp_path, "qwen3_4b", "ar-baseline", "gsm8k")
        == legacy
    )
