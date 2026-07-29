from pathlib import Path

from dllm_bench.runner.matrix import load_matrix_jobs


ROOT = Path(__file__).resolve().parent.parent


def test_full_matrix_contains_every_model_group_and_target_dataset():
    jobs, seed = load_matrix_jobs(ROOT / "configs" / "experiments" / "full_matrix.yaml")

    assert seed == 42
    assert len(jobs) == 7 * 6
    assert {job.model_name for job in jobs} == {
        "qwen3_4b",
        "qwen3_8b",
        "illada",
        "dreamreasoner",
        "w1",
        "diffusiongemma",
        "gemma4_26b_a4b",
    }
    assert {job.dataset_config.stem for job in jobs} == {
        "gsm8k",
        "mbpp",
        "structeval_t",
        "sudoku",
        "ruler",
        "hellobench",
    }
    illada_job = next(
        job for job in jobs
        if job.model_name == "illada" and job.dataset_config.stem == "gsm8k"
    )
    dream_job = next(
        job for job in jobs
        if job.model_name == "dreamreasoner" and job.dataset_config.stem == "gsm8k"
    )
    expected = ("best", "fast")
    assert illada_job.variants == expected
    assert dream_job.variants == expected
    sudoku_jobs = [job for job in jobs if job.dataset_config.stem == "sudoku"]
    assert all(job.max_new_tokens == 96 for job in sudoku_jobs)


def test_sudoku_long_output_probe_is_separate_and_uses_1024_tokens():
    jobs, seed = load_matrix_jobs(
        ROOT / "configs" / "diagnostics" / "sudoku_long_output_probe.yaml"
    )

    assert seed == 42
    assert {job.model_name for job in jobs} == {"illada", "dreamreasoner"}
    assert all(job.dataset_config.stem == "sudoku" for job in jobs)
    assert all(job.max_new_tokens == 1024 for job in jobs)


def test_matrix_can_filter_task2_datasets_for_selected_models():
    jobs, _ = load_matrix_jobs(
        ROOT / "configs" / "experiments" / "full_matrix.yaml",
        model_names=["illada", "dreamreasoner"],
        dataset_names=["ruler", "hellobench"],
    )

    assert len(jobs) == 4
    assert {job.model_name for job in jobs} == {"illada", "dreamreasoner"}
    assert {job.dataset_config.stem for job in jobs} == {"ruler", "hellobench"}


def test_dg_comparison_matrix_is_a_separate_matched_pair():
    jobs, seed = load_matrix_jobs(ROOT / "configs" / "experiments" / "dg_comparison.yaml")

    assert seed == 42
    assert len(jobs) == 2 * 7
    assert {job.model_name for job in jobs} == {
        "diffusiongemma",
        "gemma4_26b_a4b",
    }
    assert {job.dataset_config.stem for job in jobs} == {
        "gsm8k", "mbpp", "structeval_t", "sudoku", "sudoku_trace",
        "ruler", "hellobench"
    }
