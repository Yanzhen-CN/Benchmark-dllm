from dllm_bench.runner.output_layout import (
    model_output_dir,
    model_profiling_dir,
    resolve_model_output_dir,
    resolve_model_profiling_dir,
    run_id,
)


def test_qwen_ar_baseline_does_not_repeat_variant_in_run_id():
    assert run_id("qwen3_4b", "ar-baseline") == "qwen3_4b"
    assert run_id("qwen3_8b", "ar-baseline") == "qwen3_8b"
    assert run_id("illada", "p1") == "illada_p1"


def test_qwen_model_output_uses_unsuffixed_canonical_directory(tmp_path):
    path = model_output_dir(tmp_path, "qwen3_4b", "ar-baseline", "gsm8k")
    assert path == tmp_path / "model_output" / "qwen3_4b" / "gsm8k"

    path_8b = model_output_dir(tmp_path, "qwen3_8b", "ar-baseline", "gsm8k")
    assert path_8b == tmp_path / "model_output" / "qwen3_8b" / "gsm8k"


def test_profiling_is_parallel_to_model_output(tmp_path):
    path = model_profiling_dir(
        tmp_path, "diffusiongemma", "official", "mbpp"
    )
    assert path == (
        tmp_path / "model_profiling" / "diffusiongemma_official" / "mbpp"
    )
    path.mkdir(parents=True)
    assert resolve_model_profiling_dir(
        tmp_path, "diffusiongemma", "official", "mbpp"
    ) == path


def test_dllm_architecture_and_sampling_axes_are_both_in_the_run_id():
    assert run_id("illada", "p1") == "illada_p1"


def test_dflash_generation_and_scoring_use_the_same_run_directory(tmp_path):
    generated = model_output_dir(tmp_path, "gemma", "dflash", "gsm8k")
    generated.mkdir(parents=True)

    assert generated == tmp_path / "model_output" / "gemma_dflash" / "gsm8k"
    assert (
        resolve_model_output_dir(
            tmp_path, "gemma_dflash", "dflash", "gsm8k"
        )
        == generated
    )


def test_qwen_reader_falls_back_to_legacy_suffixed_directory(tmp_path):
    legacy = tmp_path / "model_output" / "qwen3_4b_ar-baseline" / "gsm8k"
    legacy.mkdir(parents=True)

    assert (
        resolve_model_output_dir(tmp_path, "qwen3_4b", "ar-baseline", "gsm8k")
        == legacy
    )
