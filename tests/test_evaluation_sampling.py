from collections import Counter

import pytest

from dllm_bench.datasets.base import Sample
from dllm_bench.datasets.hellobench import HelloBenchReference
from dllm_bench.datasets.ruler import RulerReference
from dllm_bench.datasets.sudoku import SudokuReference
from dllm_bench.runner.evaluation_sampling import select_configured_samples


_SOLUTION = [[(row * 3 + row // 3 + col) % 9 + 1 for col in range(9)] for row in range(9)]


def _sudoku_sample(index: int, difficulty: str) -> Sample:
    return Sample(
        sample_id=f"{difficulty}-{index}",
        prompt="solve",
        reference=SudokuReference(
            puzzle=[row[:] for row in _SOLUTION],
            solution=[row[:] for row in _SOLUTION],
            difficulty=difficulty,
        ),
    )


def test_sudoku_uses_configured_difficulty_split():
    samples = [
        *[_sudoku_sample(index, "easy") for index in range(8)],
        *[_sudoku_sample(index, "hard") for index in range(8)],
    ]
    config = {"dataset": "sudoku", "difficulty_counts": {"easy": 5, "hard": 3}}

    selected = select_configured_samples(samples, config, {}, seed=7)

    assert Counter(sample.reference.difficulty for sample in selected) == {
        "easy": 5,
        "hard": 3,
    }
    assert [sample.sample_id for sample in selected] == [
        sample.sample_id
        for sample in select_configured_samples(samples, config, {}, seed=7)
    ]


def test_explicit_sudoku_count_stays_balanced():
    samples = [
        *[_sudoku_sample(index, "easy") for index in range(5)],
        *[_sudoku_sample(index, "hard") for index in range(5)],
    ]
    config = {"dataset": "sudoku", "difficulty_counts": {"easy": 50, "hard": 50}}

    selected = select_configured_samples(samples, config, {}, n_samples=3)

    assert Counter(sample.reference.difficulty for sample in selected) == {
        "easy": 2,
        "hard": 1,
    }


def test_hellobench_balances_profiles_and_attaches_generation_caps():
    samples = [
        Sample(
            sample_id=f"hello-{target}-{index}",
            prompt="write",
            reference=HelloBenchReference(target_length_words=target),
        )
        for target in (2000, 4000)
        for index in range(4)
    ]
    config = {
        "dataset": "hellobench",
        "samples_per_length": 3,
        "output_profiles": [
            {"target_words": 2000, "max_new_tokens": 3072},
            {"target_words": 4000, "max_new_tokens": 6144},
        ],
    }

    selected = select_configured_samples(samples, config, {}, seed=11)

    assert len(selected) == 6
    assert Counter(sample.reference.target_length_words for sample in selected) == {
        2000: 3,
        4000: 3,
    }
    assert {
        sample.reference.target_length_words: sample.meta["max_new_tokens"]
        for sample in selected
    } == {2000: 3072, 4000: 6144}


def _ruler_samples(windows: tuple[int, ...], per_position: int = 5) -> list[Sample]:
    samples = []
    for window in windows:
        for task in ("niah", "multi_hop", "aggregation"):
            for position in ("front", "middle", "back"):
                for index in range(per_position):
                    samples.append(
                        Sample(
                            sample_id=f"{window}-{task}-{position}-{index}",
                            prompt="context",
                            reference=RulerReference(
                                task_type=task,
                                position=position,
                                required_answers=["answer"],
                                context_length=window - 64,
                            ),
                        )
                    )
    return samples


def test_ruler_selects_tasks_windows_and_balanced_positions():
    samples = _ruler_samples((8192, 32768))
    config = {
        "dataset": "ruler",
        "task_types": ["niah", "multi_hop", "aggregation"],
        "positions": ["front", "middle", "back"],
        "samples_per_context_window_position": 10,
        "common_context_window_tokens": 8192,
        "include_model_max_context_window": True,
        "max_output_tokens": 64,
    }

    selected = select_configured_samples(
        samples, config, {"max_context_tokens": 32768}, seed=3
    )

    assert len(selected) == 60
    groups = Counter(
        (
            sample.meta["context_window_tokens"],
            sample.reference.task_type,
            sample.reference.position,
        )
        for sample in selected
    )
    for window in (8192, 32768):
        for position in ("front", "middle", "back"):
            assert sum(
                groups[(window, task, position)]
                for task in ("niah", "multi_hop", "aggregation")
            ) == 10
        for task in ("niah", "multi_hop", "aggregation"):
            assert sum(
                groups[(window, task, position)]
                for position in ("front", "middle", "back")
            ) == 10
    assert all(sample.meta["max_new_tokens"] == 64 for sample in selected)
    assert all(
        sample.meta["input_tokens"] == sample.meta["context_window_tokens"] - 64
        for sample in selected
    )
    assert all(sample.meta["target_input_tokens"] == sample.meta["input_tokens"] for sample in selected)


def test_ruler_deduplicates_equal_common_and_model_windows():
    selected = select_configured_samples(
        _ruler_samples((8192,)),
        {
            "dataset": "ruler",
            "task_types": ["niah", "multi_hop", "aggregation"],
            "positions": ["front", "middle", "back"],
            "samples_per_context_window_position": 10,
            "common_context_window_tokens": 8192,
            "include_model_max_context_window": True,
            "max_output_tokens": 64,
        },
        {"max_context_tokens": 8192},
    )

    assert len(selected) == 30


def test_ruler_reports_missing_stratum_clearly():
    with pytest.raises(ValueError, match="window=32768"):
        select_configured_samples(
            _ruler_samples((8192,)),
            {
                "dataset": "ruler",
                "task_types": ["niah"],
                "positions": ["front", "middle", "back"],
                "samples_per_context_window_position": 3,
                "common_context_window_tokens": 8192,
                "include_model_max_context_window": True,
                "max_output_tokens": 64,
            },
            {"max_context_tokens": 32768},
        )


def test_generic_selection_uses_configured_size_and_seed():
    samples = [Sample(str(index), "prompt", index) for index in range(10)]
    config = {"dataset": "gsm8k", "sample_size": 4}

    first = select_configured_samples(samples, config, {}, seed=9)
    second = select_configured_samples(list(reversed(samples)), config, {}, seed=9)

    assert len(first) == 4
    assert [sample.sample_id for sample in first] == [sample.sample_id for sample in second]
