from pathlib import Path

from dllm_bench.runner.output_layout import (
    model_output_dir,
    model_profiling_dir,
    score_output_dir,
    resolve_model_output_dir,
    resolve_model_profiling_dir,
    run_id,
    visualization_output_dir,
)


def test_qwen_ar_baseline_does_not_repeat_variant_in_run_id():
    assert run_id("qwen3_4b", "ar-baseline") == "qwen3_4b"
    assert run_id("qwen3_8b", "ar-baseline") == "qwen3_8b"
    assert run_id("illada", "p1") == "illada_p1"


def test_qwen_model_output_uses_unsuffixed_canonical_directory(tmp_path):
    output_root = tmp_path / "model_output"
    path = model_output_dir(output_root, "qwen3_4b", "ar-baseline", "gsm8k")
    assert path == tmp_path / "model_output" / "qwen3_4b" / "gsm8k"

    path_8b = model_output_dir(output_root, "qwen3_8b", "ar-baseline", "gsm8k")
    assert path_8b == tmp_path / "model_output" / "qwen3_8b" / "gsm8k"


def test_profiling_is_parallel_to_model_output(tmp_path):
    path = model_profiling_dir(
        tmp_path / "model_profiling", "diffusiongemma", "official", "mbpp"
    )
    assert path == (
        tmp_path / "model_profiling" / "diffusiongemma_official" / "mbpp"
    )
    path.mkdir(parents=True)
    assert resolve_model_profiling_dir(
        tmp_path / "model_profiling", "diffusiongemma", "official", "mbpp"
    ) == path


def test_model_output_and_profiling_have_identical_substructure(tmp_path):
    generated = model_output_dir(
        tmp_path / "model_output", "illada", "p2", "structeval_t"
    )
    profiled = model_profiling_dir(
        tmp_path / "model_profiling", "illada", "p2", "structeval_t"
    )

    assert generated.relative_to(tmp_path / "model_output") == (
        profiled.relative_to(tmp_path / "model_profiling")
    )


def test_dllm_architecture_and_sampling_axes_are_both_in_the_run_id():
    assert run_id("illada", "p1") == "illada_p1"


def test_dflash_generation_and_scoring_use_the_same_run_directory(tmp_path):
    output_root = tmp_path / "model_output"
    generated = model_output_dir(output_root, "gemma", "dflash", "gsm8k")
    generated.mkdir(parents=True)

    assert generated == tmp_path / "model_output" / "gemma_dflash" / "gsm8k"
    assert (
        resolve_model_output_dir(
            output_root, "gemma_dflash", "dflash", "gsm8k"
        )
        == generated
    )


def test_qwen_reader_falls_back_to_legacy_suffixed_directory(tmp_path):
    legacy = tmp_path / "model_output" / "qwen3_4b_ar-baseline" / "gsm8k"
    legacy.mkdir(parents=True)

    assert (
        resolve_model_output_dir(
            tmp_path / "model_output", "qwen3_4b", "ar-baseline", "gsm8k"
        )
        == legacy
    )


def test_output_suffix_is_parallel_across_artifact_stages(tmp_path, monkeypatch):
    output_root = tmp_path / "model_output"
    monkeypatch.setenv("DLLM_BENCH_OUTPUT_SUFFIX", "l128")

    expected_tail = Path("diffusiongemma_official") / "sudoku9_1shot_l128"
    assert model_output_dir(
        output_root, "diffusiongemma", "official", "sudoku9_1shot"
    ) == output_root / expected_tail
    assert score_output_dir(
        output_root, "diffusiongemma", "official", "sudoku9_1shot"
    ) == tmp_path / "score_output" / expected_tail
    assert visualization_output_dir(
        output_root, "diffusiongemma", "official", "sudoku9_1shot"
    ) == tmp_path / "visualization_output" / expected_tail
