from pathlib import Path

from dllm_bench.runner.matrix import load_matrix_jobs


ROOT = Path(__file__).resolve().parent.parent


def test_full_matrix_contains_every_model_group_and_target_dataset():
    jobs, seed = load_matrix_jobs(ROOT / "configs" / "experiments" / "full_matrix.yaml")

    assert seed == 42
    assert len(jobs) == 6 * 6
    assert {job.model_name for job in jobs} == {
        "qwen3_4b",
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
    assert len(jobs) == 2 * 6
    assert {job.model_name for job in jobs} == {"diffusiongemma", "gemma4_26b"}
    assert {job.dataset_config.stem for job in jobs} == {
        "gsm8k", "mbpp", "structeval_t", "sudoku", "ruler", "hellobench"
    }
