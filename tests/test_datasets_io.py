"""Tests for datasets/io.py — reading prepared/raw JSON(L) samples back into
Sample objects. No prior coverage existed for this module before the JSONL
streaming fix (see _jsonl_records's docstring: reading a whole prepared file
into one string, then `.splitlines()`-ing it, meaningfully raised peak memory
for banks with very large individual records, e.g. RULER's 262144-token
context-window samples — a real, observed contributor to an OOM-kill several
matrix jobs later in the same long-running process).
"""

from __future__ import annotations

import inspect

import pytest

from dllm_bench.datasets.hellobench import HelloBenchReference
from dllm_bench.datasets.io import _records, load_samples_file
from dllm_bench.datasets.mbpp import MbppSample
from dllm_bench.datasets.ruler import RulerReference
from dllm_bench.datasets.sudoku import SudokuReference


def _write(path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_records_streams_jsonl_lazily_not_as_a_pre_built_list(tmp_path):
    """Locks in the memory fix itself, not just its output: a .jsonl file's
    records must come back as a lazy iterator, never a fully-materialized
    list — otherwise a future edit could silently reintroduce the
    read-whole-file-then-splitlines double-copy this was fixed to avoid."""
    path = tmp_path / "bank.jsonl"
    _write(path, '{"prompt": "a"}\n{"prompt": "b"}\n')

    records = _records(path)

    assert not isinstance(records, list)
    assert inspect.isgenerator(records) or hasattr(records, "__next__")
    assert list(records) == [{"prompt": "a"}, {"prompt": "b"}]


def test_load_samples_file_parses_jsonl_and_skips_blank_lines(tmp_path):
    path = tmp_path / "bank.jsonl"
    _write(
        path,
        '{"sample_id": "s1", "prompt": "hello"}\n'
        "\n"
        '{"sample_id": "s2", "prompt": "world", "meta": {"k": "v"}}\n'
        "   \n",
    )

    samples = load_samples_file(path, "generic")

    assert [s.sample_id for s in samples] == ["s1", "s2"]
    assert samples[0].prompt == "hello"
    assert samples[1].meta == {"k": "v"}


def test_load_samples_file_falls_back_to_a_bare_json_list(tmp_path):
    path = tmp_path / "bank.json"
    _write(path, '[{"prompt": "a"}, {"prompt": "b"}]')

    samples = load_samples_file(path, "generic")
    assert [s.prompt for s in samples] == ["a", "b"]


def test_load_samples_file_unwraps_a_samples_or_data_key(tmp_path):
    wrapped_samples = tmp_path / "wrapped_samples.json"
    _write(wrapped_samples, '{"samples": [{"prompt": "a"}]}')
    assert load_samples_file(wrapped_samples, "generic")[0].prompt == "a"

    wrapped_data = tmp_path / "wrapped_data.json"
    _write(wrapped_data, '{"data": [{"prompt": "b"}]}')
    assert load_samples_file(wrapped_data, "generic")[0].prompt == "b"


def test_load_samples_file_rejects_a_json_object_with_no_list(tmp_path):
    path = tmp_path / "bad.json"
    _write(path, '{"prompt": "not a list container"}')

    with pytest.raises(ValueError, match="must contain a JSON list"):
        load_samples_file(path, "generic")


def test_load_samples_file_accepts_question_or_text_as_the_prompt_field(tmp_path):
    path = tmp_path / "bank.jsonl"
    _write(path, '{"question": "q-form"}\n{"text": "text-form"}\n')

    samples = load_samples_file(path, "generic")
    assert [s.prompt for s in samples] == ["q-form", "text-form"]


def test_load_samples_file_raises_a_clear_error_when_no_prompt_field_exists(tmp_path):
    path = tmp_path / "bank.jsonl"
    _write(path, '{"sample_id": "s1"}\n')

    with pytest.raises(ValueError, match="no prompt/question/text"):
        load_samples_file(path, "generic")


def test_load_samples_file_falls_back_sample_id_to_id_task_id_then_index(tmp_path):
    path = tmp_path / "bank.jsonl"
    _write(
        path,
        '{"prompt": "a", "id": "from-id"}\n'
        '{"prompt": "b", "task_id": "from-task-id"}\n'
        '{"prompt": "c"}\n',
    )

    samples = load_samples_file(path, "generic")
    assert [s.sample_id for s in samples] == ["from-id", "from-task-id", "2"]


def test_load_samples_file_truncates_to_n(tmp_path):
    path = tmp_path / "bank.jsonl"
    _write(path, "".join(f'{{"prompt": "{i}"}}\n' for i in range(5)))

    samples = load_samples_file(path, "generic", n=2)
    assert [s.prompt for s in samples] == ["0", "1"]


def test_load_samples_file_reconstructs_the_ruler_reference_dataclass(tmp_path):
    path = tmp_path / "bank.jsonl"
    _write(
        path,
        '{"prompt": "p", "reference": {"task_type": "niah", "position": "front", '
        '"required_answers": ["42"], "context_length": 8192}}\n',
    )

    samples = load_samples_file(path, "ruler")
    reference = samples[0].reference
    assert isinstance(reference, RulerReference)
    assert reference.task_type == "niah"
    assert reference.required_answers == ["42"]


def test_load_samples_file_reconstructs_the_hellobench_reference_dataclass(tmp_path):
    path = tmp_path / "bank.jsonl"
    _write(path, '{"prompt": "p", "reference": {"target_length_words": 2000}}\n')

    samples = load_samples_file(path, "hellobench")
    assert isinstance(samples[0].reference, HelloBenchReference)
    assert samples[0].reference.target_length_words == 2000


def test_load_samples_file_reconstructs_the_sudoku_reference_dataclass(tmp_path):
    path = tmp_path / "bank.jsonl"
    _write(
        path,
        '{"prompt": "p", "reference": {"puzzle": [[1]], "solution": [[1]], '
        '"difficulty": "easy"}}\n',
    )

    samples = load_samples_file(path, "sudoku")
    reference = samples[0].reference
    assert isinstance(reference, SudokuReference)
    assert reference.difficulty == "easy"


def test_load_samples_file_reconstructs_instructed_sudoku_reference(tmp_path):
    path = tmp_path / "bank.jsonl"
    _write(
        path,
        '{"prompt": "solve", "reference": {"puzzle": [[0]], '
        '"solution": [[1]], "difficulty": "hard"}}\n',
    )

    samples = load_samples_file(path, "sudoku_trace")

    assert isinstance(samples[0].reference, SudokuReference)
    assert samples[0].reference.difficulty == "hard"


def test_load_samples_file_reconstructs_the_mbpp_reference_dataclass(tmp_path):
    path = tmp_path / "bank.jsonl"
    _write(
        path,
        '{"prompt": "p", "test_list": ["assert f(1) == 1"], '
        '"test_setup_code": "def f(x): return x"}\n',
    )

    samples = load_samples_file(path, "mbpp")
    reference = samples[0].reference
    assert isinstance(reference, MbppSample)
    assert reference.test_list == ["assert f(1) == 1"]


def test_load_samples_file_leaves_an_unrecognized_dataset_reference_as_raw(tmp_path):
    path = tmp_path / "bank.jsonl"
    _write(path, '{"prompt": "p", "reference": {"anything": "goes"}}\n')

    samples = load_samples_file(path, "some_future_dataset")
    assert samples[0].reference == {"anything": "goes"}
