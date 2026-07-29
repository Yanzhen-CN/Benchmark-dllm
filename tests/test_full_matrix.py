from pathlib import Path

from dllm_bench.runner.matrix import load_matrix_jobs


ROOT = Path(__file__).resolve().parent.parent


def test_full_matrix_contains_every_model_group_and_target_dataset():
    jobs, seed = load_matrix_jobs(ROOT / "configs" / "experiments" / "full_matrix.yaml")

    assert seed == 42
    assert len(jobs) == 9 * 8
    assert {job.model_name for job in jobs} == {
        "qwen3_4b",
        "qwen3_8b",
        "illada",
        "illada_vargen",
        "dreamreasoner",
        "w1",
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
        "ruler",
        "hellobench",
        "ruler_context_probe",
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
    vargen_job = next(
        job for job in jobs
        if job.model_name == "illada_vargen" and job.dataset_config.stem == "gsm8k"
    )
    assert vargen_job.variants == expected
    assert dream_job.variants == expected
    for dataset_name in ("gsm8k", "mbpp", "structeval_t"):
        assert all(
            job.max_new_tokens == 256
            for job in jobs
            if job.dataset_config.stem == dataset_name
        )
    sudoku4_jobs = [job for job in jobs if job.dataset_config.stem == "sudoku4"]
    sudoku9_jobs = [job for job in jobs if job.dataset_config.stem == "sudoku9"]
    assert all(job.max_new_tokens == 256 for job in sudoku4_jobs)
    assert all(job.max_new_tokens == 256 for job in sudoku9_jobs)
    assert {
        job.model_name: job.n_samples for job in sudoku9_jobs
        if job.model_name in {
            "qwen3_4b", "qwen3_8b", "illada", "illada_vargen",
            "dreamreasoner", "w1",
        }
    } == {
        "qwen3_4b": 10,
        "qwen3_8b": 10,
        "illada": 10,
        "illada_vargen": 10,
        "dreamreasoner": 10,
        "w1": 10,
    }
    assert {
        job.model_name: job.n_samples for job in sudoku4_jobs
        if job.model_name in {"diffusiongemma", "gemma", "gemma_dflash"}
    } == {"diffusiongemma": 10, "gemma": 10, "gemma_dflash": 10}
    for model_name in ("illada", "illada_vargen", "dreamreasoner"):
        hello_job = next(
            job for job in jobs
            if job.model_name == model_name and job.dataset_config.stem == "hellobench"
        )
        assert hello_job.hellobench_lengths == ("2k",)
        assert hello_job.n_samples == 1

    probe_jobs = [
        job for job in jobs
        if job.dataset_config.stem == "ruler_context_probe"
    ]
    assert len(probe_jobs) == 9
    assert all(job.max_new_tokens == 64 for job in probe_jobs)


def test_dg_comparison_uses_sudoku9_main_and_ten_sudoku4_probes():
    jobs, _ = load_matrix_jobs(
        ROOT / "configs" / "experiments" / "dg_comparison.yaml"
    )
    sudoku9_jobs = [job for job in jobs if job.dataset_config.stem == "sudoku9"]
    sudoku4_jobs = [job for job in jobs if job.dataset_config.stem == "sudoku4"]
    assert all(job.max_new_tokens == 256 and job.n_samples is None for job in sudoku9_jobs)
    assert all(job.max_new_tokens == 256 and job.n_samples == 10 for job in sudoku4_jobs)


def test_matrix_can_filter_task2_datasets_for_selected_models():
    jobs, _ = load_matrix_jobs(
        ROOT / "configs" / "experiments" / "full_matrix.yaml",
        model_names=["illada", "dreamreasoner"],
        dataset_names=["ruler", "hellobench"],
    )

    assert len(jobs) == 4
    assert {job.model_name for job in jobs} == {"illada", "dreamreasoner"}
    assert {job.dataset_config.stem for job in jobs} == {"ruler", "hellobench"}


def test_dg_comparison_matrix_contains_native_pair_and_dflash_deployment_row():
    jobs, seed = load_matrix_jobs(ROOT / "configs" / "experiments" / "dg_comparison.yaml")

    assert seed == 42
    assert len(jobs) == 3 * 8
    assert {job.model_name for job in jobs} == {
        "diffusiongemma",
        "gemma",
        "gemma_dflash",
    }
    assert {job.dataset_config.stem for job in jobs} == {
        "gsm8k", "mbpp", "structeval_t", "sudoku4", "sudoku9", "sudoku_trace",
        "ruler", "hellobench"
    }
