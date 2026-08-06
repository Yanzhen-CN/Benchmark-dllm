from pathlib import Path

from dllm_bench.runner.matrix import load_matrix_jobs


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "configs" / "experiments"


def test_full_matrix_declares_the_horizontal_benchmark_contract():
    jobs, seed = load_matrix_jobs(EXPERIMENTS / "full_matrix.yaml")

    assert seed == 42
    assert len(jobs) == 72
    assert {job.model_name for job in jobs} == {
        "qwen3_4b",
        "qwen3_8b",
        "illada",
        "illada_vargen",
        "dreamreasoner",
        "diffusiongemma",
        "gemma",
        "gemma_dflash",
    }
    assert {job.dataset_config.stem for job in jobs} == {
        "gsm8k",
        "mbpp",
        "structeval_t",
        "sudoku4",
        "sudoku9",
        "sudoku4_thinking",
        "sudoku9_thinking",
        "ruler",
        "hellobench",
    }

    parallel_models = {"illada", "illada_vargen", "dreamreasoner"}
    assert all(
        job.variants == ("p1", "p2")
        for job in jobs
        if job.model_name in parallel_models
    )
    assert all(
        job.max_new_tokens == 512
        for job in jobs
        if job.dataset_config.stem in {"gsm8k", "mbpp", "structeval_t"}
    )
    assert all(
        job.max_new_tokens == 2048 and job.n_samples == 1
        for job in jobs
        if job.dataset_config.stem in {"sudoku4_thinking", "sudoku9_thinking"}
    )
    sudoku4_samples = {
        job.model_name: job.n_samples
        for job in jobs
        if job.dataset_config.stem == "sudoku4"
    }
    assert sudoku4_samples["diffusiongemma"] == 100
    assert sudoku4_samples["gemma"] == 100
    assert sudoku4_samples["gemma_dflash"] == 100


def test_profiling_matrix_is_trace_free_and_uses_profiling_output():
    jobs, seed = load_matrix_jobs(EXPERIMENTS / "profiling_matrix.yaml")

    assert seed == 42
    assert len(jobs) == 12
    assert {job.model_name for job in jobs} == {
        "diffusiongemma",
        "illada",
        "illada_vargen",
        "dreamreasoner",
    }
    assert {job.dataset_config.stem for job in jobs} == {
        "mbpp",
        "gsm8k",
        "structeval_t",
    }
    assert all(job.n_samples == 1 for job in jobs)
    assert all(job.capture_trace is False for job in jobs)
    assert all(job.profiling_output is True for job in jobs)


def test_illada_entropy_matrix_is_a_focused_gsm8k_sweep():
    jobs, seed = load_matrix_jobs(EXPERIMENTS / "illada_entropy.yaml")

    assert seed == 42
    assert len(jobs) == 1
    job = jobs[0]
    assert job.model_name == "illada_entropy"
    assert job.dataset_config.stem == "gsm8k"
    assert job.variants == ("eb03", "eb05", "eb1")
    assert job.n_samples is None
    assert job.max_new_tokens == 512


def test_matrix_filters_models_datasets_and_sudoku_group():
    jobs, _ = load_matrix_jobs(
        EXPERIMENTS / "full_matrix.yaml",
        model_names=["illada", "dreamreasoner"],
        dataset_names=["ruler", "hellobench"],
    )
    assert len(jobs) == 4
    assert {job.model_name for job in jobs} == {"illada", "dreamreasoner"}
    assert {job.dataset_config.stem for job in jobs} == {"ruler", "hellobench"}

    sudoku_jobs, _ = load_matrix_jobs(
        EXPERIMENTS / "full_matrix.yaml",
        model_names=["qwen3_8b"],
        dataset_names=["sudoku"],
    )
    assert [job.dataset_config.stem for job in sudoku_jobs] == [
        "sudoku4",
        "sudoku9",
        "sudoku4_thinking",
        "sudoku9_thinking",
    ]


def test_dg_comparison_contains_only_matched_gemma_rows():
    jobs, seed = load_matrix_jobs(EXPERIMENTS / "dg_comparison.yaml")

    assert seed == 42
    assert len(jobs) == 9
    assert {job.model_name for job in jobs} == {
        "diffusiongemma",
        "gemma",
        "gemma_dflash",
    }
    assert {job.dataset_config.stem for job in jobs} == {
        "gsm8k",
        "mbpp",
        "structeval_t",
    }
