from pathlib import Path

from dllm_bench.runner.matrix import load_matrix_jobs


ROOT = Path(__file__).resolve().parent.parent


def test_full_matrix_contains_every_model_group_and_target_dataset():
    jobs, seed = load_matrix_jobs(ROOT / "configs" / "experiments" / "full_matrix.yaml")

    assert seed == 42
    assert len(jobs) == 11 * 10
    assert {job.model_name for job in jobs} == {
        "qwen3_4b",
        "qwen3_8b",
        "illada",
        "illada_vargen",
        "illada_entropy",
        "llada2_1",
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
        "sudoku4_thinking",
        "sudoku9_thinking",
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
    expected = ("p1", "p2")
    assert illada_job.variants == expected
    vargen_job = next(
        job for job in jobs
        if job.model_name == "illada_vargen" and job.dataset_config.stem == "gsm8k"
    )
    assert vargen_job.variants == expected
    assert dream_job.variants == expected
    for model_config in (
        "illada.yaml", "illada_vargen.yaml", "dreamreasoner.yaml"
    ):
        config_text = (ROOT / "configs" / "models" / model_config).read_text(
            encoding="utf-8"
        )
        assert all(f"  {variant}:" in config_text for variant in ("p1", "p2", "p4", "p8"))
    for dataset_name in ("gsm8k", "mbpp", "structeval_t"):
        assert all(
            job.max_new_tokens == 512
            for job in jobs
            if job.dataset_config.stem == dataset_name
        )
    sudoku4_jobs = [job for job in jobs if job.dataset_config.stem == "sudoku4"]
    sudoku9_jobs = [job for job in jobs if job.dataset_config.stem == "sudoku9"]
    assert all(job.max_new_tokens == 256 for job in sudoku4_jobs)
    assert all(job.max_new_tokens == 256 for job in sudoku9_jobs)
    thinking_jobs = [
        job for job in jobs
        if job.dataset_config.stem in {"sudoku4_thinking", "sudoku9_thinking"}
    ]
    assert len(thinking_jobs) == 20
    assert all(job.max_new_tokens == 2048 for job in thinking_jobs)
    assert all(job.n_samples == 1 for job in thinking_jobs)
    assert {
        job.model_name: job.n_samples for job in sudoku9_jobs
        if job.model_name in {
            "qwen3_4b", "qwen3_8b", "illada", "illada_vargen", "illada_entropy",
            "dreamreasoner", "w1",
        }
    } == {
        "qwen3_4b": 1,
        "qwen3_8b": 10,
        "illada": 1,
        "illada_vargen": 1,
        "illada_entropy": 1,
        "dreamreasoner": 1,
        "w1": 10,
    }
    assert {
        job.model_name: job.n_samples for job in sudoku4_jobs
        if job.model_name in {
            "qwen3_4b", "illada", "illada_vargen", "illada_entropy", "dreamreasoner"
        }
    } == {
        "qwen3_4b": 1,
        "illada": 1,
        "illada_vargen": 1,
        "illada_entropy": 1,
        "dreamreasoner": 1,
    }
    assert {
        job.model_name: job.n_samples for job in sudoku4_jobs
        if job.model_name in {"diffusiongemma", "gemma", "gemma_dflash"}
    } == {"diffusiongemma": 10, "gemma": 10, "gemma_dflash": 10}
    for model_name in ("illada", "illada_vargen", "illada_entropy", "dreamreasoner"):
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
    assert len(probe_jobs) == 10
    assert all(job.max_new_tokens == 64 for job in probe_jobs)


def test_profiling_matrix_is_trace_free_and_uses_profiling_output():
    jobs, seed = load_matrix_jobs(
        ROOT / "configs" / "experiments" / "profiling_matrix.yaml"
    )

    assert seed == 42
    assert len(jobs) == 9
    assert all(job.capture_trace is False for job in jobs)
    assert all(job.profiling_output is True for job in jobs)
    assert {job.dataset_config.stem for job in jobs} == {
        "mbpp",
        "gsm8k",
        "structeval_t",
    }


def test_illada_entropy_matrix_runs_three_full_primary_datasets():
    jobs, seed = load_matrix_jobs(
        ROOT / "configs" / "experiments" / "illada_entropy.yaml"
    )

    assert seed == 42
    assert len(jobs) == 3
    assert {job.model_name for job in jobs} == {"illada_entropy"}
    assert {job.variants for job in jobs} == {("eb01",)}
    assert {job.dataset_config.stem for job in jobs} == {
        "gsm8k", "mbpp", "structeval_t"
    }
    assert all(job.n_samples is None for job in jobs)
    assert all(job.max_new_tokens == 512 for job in jobs)


def test_dg_comparison_uses_main_and_reference_sudoku_rows():
    jobs, _ = load_matrix_jobs(
        ROOT / "configs" / "experiments" / "dg_comparison.yaml"
    )
    sudoku9_jobs = [job for job in jobs if job.dataset_config.stem == "sudoku9"]
    sudoku4_jobs = [job for job in jobs if job.dataset_config.stem == "sudoku4"]
    assert all(job.max_new_tokens == 256 and job.n_samples is None for job in sudoku9_jobs)
    assert all(job.max_new_tokens == 256 and job.n_samples == 10 for job in sudoku4_jobs)
    thinking_jobs = [
        job for job in jobs
        if job.dataset_config.stem in {"sudoku4_thinking", "sudoku9_thinking"}
    ]
    assert all(job.max_new_tokens == 2048 and job.n_samples == 1 for job in thinking_jobs)


def test_matrix_can_filter_task2_datasets_for_selected_models():
    jobs, _ = load_matrix_jobs(
        ROOT / "configs" / "experiments" / "full_matrix.yaml",
        model_names=["illada", "dreamreasoner"],
        dataset_names=["ruler", "hellobench"],
    )

    assert len(jobs) == 4
    assert {job.model_name for job in jobs} == {"illada", "dreamreasoner"}
    assert {job.dataset_config.stem for job in jobs} == {"ruler", "hellobench"}


def test_matrix_sudoku_group_expands_every_declared_variant():
    jobs, _ = load_matrix_jobs(
        ROOT / "configs" / "experiments" / "full_matrix.yaml",
        model_names=["qwen3_8b"],
        dataset_names=["sudoku"],
    )

    assert [job.dataset_config.stem for job in jobs] == [
        "sudoku4",
        "sudoku9",
        "sudoku4_thinking",
        "sudoku9_thinking",
    ]


def test_dg_comparison_matrix_contains_native_pair_and_dflash_deployment_row():
    jobs, seed = load_matrix_jobs(ROOT / "configs" / "experiments" / "dg_comparison.yaml")

    assert seed == 42
    assert len(jobs) == 3 * 9
    assert {job.model_name for job in jobs} == {
        "diffusiongemma",
        "gemma",
        "gemma_dflash",
    }
    assert {job.dataset_config.stem for job in jobs} == {
        "gsm8k", "mbpp", "structeval_t", "sudoku4", "sudoku9",
        "sudoku4_thinking", "sudoku9_thinking", "ruler", "hellobench"
    }
