import json
import zipfile

import pytest

from dllm_bench.datasets.base import Sample
from dllm_bench.datasets.gsm8k import GSM8K_REVISION, GSM8KDataset, _load_official_test_samples, extract_final_number
from dllm_bench.datasets.hellobench import (
    HelloBenchDataset,
    HelloBenchReference,
    detect_major_issues,
    repeated_segment_fraction,
    seq_rep_n,
)
from dllm_bench.datasets.mbpp import MBPPDataset, MbppSample, extract_code
from dllm_bench.datasets.ruler import (
    RulerDataset,
    RulerReference,
    generate_ruler_bank,
    position_robustness,
)
from dllm_bench.datasets.structeval_t import (
    StructEvalSchema,
    StructEvalTDataset,
    evaluate_struct_progress,
)
from dllm_bench.datasets.sudoku import (
    SudokuDataset,
    SudokuReference,
    _load_official_test_samples as _load_official_sudoku_test_samples,
    classify_difficulty,
    group_by_difficulty,
    naked_single_rounds,
    parse_grid,
)
from dllm_bench.interfaces import (
    GenerationRequest,
    GenerationResult,
    RunStatus,
    TimingResult,
)

# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

def test_gsm8k_extracts_gold_marker_over_other_numbers():
    text = "First we compute 12 + 3 = 15. Then double it. #### 30"
    assert extract_final_number(text) == 30.0


def test_gsm8k_falls_back_to_last_number_without_marker():
    text = "There were 5 apples, then 2 more were added, giving 7 apples."
    assert extract_final_number(text) == 7.0


def test_gsm8k_score_correct_answer():
    ds = GSM8KDataset()
    sample = Sample(sample_id="1", prompt="q", reference=42.0)
    result = ds.score(sample, "Some reasoning. #### 42")
    assert result.primary_score == 1.0
    assert result.valid is True


def test_gsm8k_score_wrong_answer():
    ds = GSM8KDataset()
    sample = Sample(sample_id="1", prompt="q", reference=42.0)
    result = ds.score(sample, "#### 41")
    assert result.primary_score == 0.0


def test_gsm8k_score_no_extractable_answer_is_invalid():
    ds = GSM8KDataset()
    sample = Sample(sample_id="1", prompt="q", reference=42.0)
    result = ds.score(sample, "I have no idea what to compute here at all.")
    assert result.valid is False
    assert result.primary_score == 0.0


def test_gsm8k_official_jsonl_loader_builds_stable_samples(tmp_path):
    dataset_path = tmp_path / "test.jsonl"
    rows = [
        {
            "question": f"Question {i}",
            "answer": f"Reasoning for {i}.\n#### {i + 10}",
        }
        for i in range(1319)
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    samples = _load_official_test_samples(dataset_path)

    assert len(samples) == 1319
    assert samples[0].sample_id == "gsm8k-test-0000"
    assert samples[-1].sample_id == "gsm8k-test-1318"
    assert samples[0].prompt == "Question 0"
    assert samples[0].reference == 10.0
    assert samples[0].meta["source_revision"] == GSM8K_REVISION
    assert samples[0].meta["gold_solution"] == "Reasoning for 0.\n#### 10"


# ---------------------------------------------------------------------------
# MBPP
# ---------------------------------------------------------------------------

def test_mbpp_extract_code_from_fence():
    text = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
    code = extract_code(text)
    assert code == "def add(a, b):\n    return a + b"


def test_mbpp_score_passing_solution():
    ds = MBPPDataset()
    sample = Sample(
        sample_id="1",
        prompt="write add(a, b)",
        reference=MbppSample(test_list=["assert add(2, 3) == 5", "assert add(-1, 1) == 0"]),
    )
    output = "```python\ndef add(a, b):\n    return a + b\n```"
    result = ds.score(sample, output)
    assert result.primary_score == 1.0
    assert result.valid is True
    assert result.complete is True


def test_mbpp_score_failing_solution_is_executable_but_not_passing():
    ds = MBPPDataset()
    sample = Sample(
        sample_id="1",
        prompt="write add(a, b)",
        reference=MbppSample(test_list=["assert add(2, 3) == 5"]),
    )
    output = "def add(a, b):\n    return a - b"
    result = ds.score(sample, output)
    assert result.primary_score == 0.0
    assert result.valid is True  # executed fine, assertion just failed
    assert result.aux["executable_rate"] == 1.0


def test_mbpp_score_syntax_error_is_not_executable():
    ds = MBPPDataset()
    sample = Sample(
        sample_id="1",
        prompt="write add(a, b)",
        reference=MbppSample(test_list=["assert add(2, 3) == 5"]),
    )
    output = "def add(a, b:\n    return a + b"
    result = ds.score(sample, output)
    assert result.primary_score == 0.0
    assert result.valid is False


def test_mbpp_aggregate_exposes_official_metric_name():
    ds = MBPPDataset()
    sample = Sample(
        sample_id="1",
        prompt="write add(a, b)",
        reference=MbppSample(test_list=["assert add(2, 3) == 5"]),
    )
    passing = ds.score(sample, "def add(a, b):\n    return a + b")
    failing = ds.score(sample, "def add(a, b):\n    return a - b")
    assert ds.aggregate([passing, failing])["pass_at_1"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# StructEval-T
# ---------------------------------------------------------------------------

def test_structeval_json_complete_and_correct():
    ds = StructEvalTDataset()
    schema = StructEvalSchema(
        format="json",
        required_keys=["name", "age"],
        critical_content=["Alice"],
    )
    sample = Sample(sample_id="1", prompt="p", reference=schema)
    output = '{"name": "Alice", "age": 30}'
    result = ds.score(sample, output)
    assert result.primary_score == 1.0
    assert result.aux["format_valid_rate"] == 1.0


def test_structeval_json_missing_key_is_incomplete():
    ds = StructEvalTDataset()
    schema = StructEvalSchema(format="json", required_keys=["name", "age"])
    sample = Sample(sample_id="1", prompt="p", reference=schema)
    result = ds.score(sample, '{"name": "Alice"}')
    # Official non-renderable StructEval: 20% strict parse + 80% path coverage.
    assert result.primary_score == pytest.approx(0.6)
    assert result.aux["official_render_score"] == 1.0
    assert result.aux["official_key_validation_score"] == pytest.approx(0.5)
    assert result.aux["complete_correct_rate"] == 0.0
    assert result.aux["field_completion_rate"] == pytest.approx(0.5)


def test_structeval_official_score_does_not_repair_malformed_output():
    ds = StructEvalTDataset()
    schema = StructEvalSchema(format="json", required_keys=["name"])
    sample = Sample(sample_id="1", prompt="p", reference=schema)
    result = ds.score(sample, '{"name": "Alice"')
    assert result.primary_score == 0.0
    assert result.aux["official_render_score"] == 0.0
    # The trace diagnostic remains deliberately fault-tolerant.
    assert evaluate_struct_progress('{"name": "Alice"', schema).parseability == 1.0


def test_structeval_malformed_yaml_is_strict_zero_instead_of_crashing():
    ds = StructEvalTDataset()
    schema = StructEvalSchema(format="yaml", required_keys=["name"])
    sample = Sample(sample_id="1", prompt="p", reference=schema)

    result = ds.score(sample, "Here is the result:\n\n```yaml\nname: Alice")

    assert result.primary_score == 0.0
    assert result.aux["official_render_score"] == 0.0
    assert result.aux["official_key_validation_score"] == 0.0


def test_structeval_json_tolerates_unclosed_structure():
    schema = StructEvalSchema(format="json", required_keys=["name"])
    progress = evaluate_struct_progress('{"name": "Alice", "tags": ["a", "b"', schema)
    assert progress.parseability == 1.0
    assert progress.key_coverage == 1.0


def test_structeval_unparseable_gives_zero_progress_but_no_crash():
    schema = StructEvalSchema(format="json", required_keys=["name"])
    progress = evaluate_struct_progress("this is not json at all !!!", schema)
    assert progress.parseability == 0.0
    assert progress.key_coverage == 0.0


def test_structeval_xml_basic():
    schema = StructEvalSchema(format="xml", required_keys=["root.name"])
    progress = evaluate_struct_progress("<root><name>Bob</name></root>", schema)
    assert progress.parseability == 1.0
    assert progress.key_coverage == 1.0


def test_structeval_csv_basic():
    schema = StructEvalSchema(format="csv", required_keys=["row0.name"])
    progress = evaluate_struct_progress("name,age\nAlice,30\n", schema)
    assert progress.parseability == 1.0
    assert progress.key_coverage == 1.0


def test_structeval_official_csv_uses_header_paths():
    ds = StructEvalTDataset()
    schema = StructEvalSchema(
        format="csv", required_keys=["csv::name", "csv::age", "csv::missing"]
    )
    sample = Sample(sample_id="1", prompt="p", reference=schema)
    result = ds.score(sample, "name,age\nAlice,30\n")
    assert result.aux["official_render_score"] == 1.0
    assert result.aux["official_key_validation_score"] == pytest.approx(2 / 3)
    assert result.primary_score == pytest.approx(0.73)


def test_structeval_official_xml_preserves_root_and_repeated_elements():
    ds = StructEvalTDataset()
    schema = StructEvalSchema(
        format="xml",
        required_keys=["root.name", "root.items.item[1].value"],
    )
    sample = Sample(sample_id="1", prompt="p", reference=schema)
    output = (
        "<root><name>Alice</name><items><item><value>1</value></item>"
        "<item><value>2</value></item></items></root>"
    )
    result = ds.score(sample, output)
    assert result.primary_score == 1.0
    assert result.aux["official_key_validation_score"] == 1.0
    assert ds.aggregate([result])["final_eval_score"] == 1.0


# ---------------------------------------------------------------------------
# Sudoku
# ---------------------------------------------------------------------------

_EASY_PUZZLE = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]


def _blank_copy(grid, positions):
    import copy

    g = copy.deepcopy(grid)
    for r, c in positions:
        g[r][c] = 0
    return g


def test_classify_difficulty_easy_when_one_cell_blank():
    puzzle = _blank_copy(_EASY_PUZZLE, [(0, 0)])
    assert classify_difficulty(puzzle) == "easy"
    assert naked_single_rounds(puzzle) == 1


def test_parse_grid_from_spaced_text():
    text = "\n".join(" ".join(str(v) for v in row) for row in _EASY_PUZZLE)
    parsed = parse_grid(text)
    assert parsed == _EASY_PUZZLE


def test_parse_grid_returns_none_for_prose():
    assert parse_grid("This response contains no grid at all, just words.") is None


def test_sudoku_score_correct_solution():
    ds = SudokuDataset()
    puzzle = _blank_copy(_EASY_PUZZLE, [(0, 0)])
    ref = SudokuReference(puzzle=puzzle, solution=_EASY_PUZZLE)
    sample = Sample(sample_id="1", prompt="solve", reference=ref)
    text = "".join(str(v) for row in _EASY_PUZZLE for v in row)
    result = ds.score(sample, text)
    assert result.primary_score == 1.0
    assert result.valid is True


def test_sudoku_official_score_rejects_non_official_answer_wrapping():
    ds = SudokuDataset()
    puzzle = _blank_copy(_EASY_PUZZLE, [(0, 0)])
    ref = SudokuReference(puzzle=puzzle, solution=_EASY_PUZZLE)
    sample = Sample(sample_id="1", prompt="solve", reference=ref)
    digits = "".join(str(v) for row in _EASY_PUZZLE for v in row)

    result = ds.score(sample, f"Answer:\n{digits}")

    assert result.primary_score == 0.0
    assert result.valid is False


def test_sudoku_score_unparseable_output():
    ds = SudokuDataset()
    ref = SudokuReference(puzzle=_EASY_PUZZLE, solution=_EASY_PUZZLE)
    sample = Sample(sample_id="1", prompt="solve", reference=ref)
    result = ds.score(sample, "I don't know the answer.")
    assert result.primary_score == 0.0
    assert result.valid is False


def test_load_official_sudoku_split_keeps_raw_sequence_protocol(tmp_path):
    solution = "".join(str(v) for row in _EASY_PUZZLE for v in row)
    easy = "0" + solution[1:]
    hard = "0" * 81
    archive_path = tmp_path / "sudoku.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "sudoku.csv",
            "quizzes,solutions\n"
            f"{easy},{solution}\n"
            f"{easy},{solution}\n"
            f"{easy},{solution}\n"
            f"{hard},{solution}\n",
        )

    samples = _load_official_sudoku_test_samples(
        archive_path, train_rows=2, test_rows=2
    )

    assert [sample.prompt for sample in samples] == [easy, hard]
    assert [sample.meta["source_index"] for sample in samples] == [2, 3]
    assert [sample.reference.difficulty for sample in samples] == ["easy", "hard"]
    assert all(sample.meta["max_new_tokens"] == 82 for sample in samples)


def test_group_by_difficulty():
    puzzle = _blank_copy(_EASY_PUZZLE, [(0, 0)])
    ref = SudokuReference(puzzle=puzzle, solution=_EASY_PUZZLE, difficulty="easy")
    samples = [Sample(sample_id="1", prompt="p", reference=ref)]
    results = [ScoreResultStub(1.0)]
    grouped = group_by_difficulty(samples, results)
    assert len(grouped["easy"]) == 1
    assert len(grouped["hard"]) == 0


class ScoreResultStub:
    """Avoids importing ScoreResult's [0,1] validation noise for this grouping test."""

    def __init__(self, primary_score):
        self.primary_score = primary_score


# ---------------------------------------------------------------------------
# RULER
# ---------------------------------------------------------------------------

def test_ruler_score_exact_match():
    ds = RulerDataset()
    ref = RulerReference(task_type="niah", position="front", required_answers=["42"])
    sample = Sample(sample_id="1", prompt="p", reference=ref)
    result = ds.score(sample, "The secret code is 42.")
    assert result.primary_score == 1.0


def test_ruler_official_style_prompt_ends_with_answer_prefix():
    samples = generate_ruler_bank(
        [256], samples_per_context_window_position=1, max_output_tokens=64
    )

    assert samples
    assert all(sample.prompt.endswith("\nAnswer:") for sample in samples)


def test_ruler_score_partial_match_for_multi_answer():
    ds = RulerDataset()
    ref = RulerReference(task_type="multi_hop", position="middle", required_answers=["Alice", "Bob"])
    sample = Sample(sample_id="1", prompt="p", reference=ref)
    result = ds.score(sample, "The answer involves Alice.")
    assert result.primary_score == pytest.approx(0.5)
    assert result.aux["all_answers_match"] == 0.0


def test_position_robustness_perfect_when_equal():
    assert position_robustness({"front": 0.8, "middle": 0.8, "back": 0.8}) == pytest.approx(1.0)


def test_position_robustness_penalizes_worst_position():
    assert position_robustness({"front": 1.0, "middle": 0.5, "back": 1.0}) == pytest.approx(0.5)


def test_position_robustness_rejects_empty():
    with pytest.raises(ValueError):
        position_robustness({})


# ---------------------------------------------------------------------------
# HelloBench
# ---------------------------------------------------------------------------

def test_seq_rep_4_no_repetition():
    text = " ".join(f"word{i}" for i in range(20))
    assert seq_rep_n(text, 4) == pytest.approx(0.0)


def test_seq_rep_4_full_repetition():
    text = " ".join(["a", "b", "c", "d"] * 5)
    rep = seq_rep_n(text, 4)
    assert rep > 0.5


def test_hellobench_score_within_length_tolerance():
    ds = HelloBenchDataset()
    ref = HelloBenchReference(target_length_words=100)
    sample = Sample(sample_id="1", prompt="p", reference=ref)
    output = " ".join(f"w{i}" for i in range(100))
    result = ds.score(sample, output)
    assert result.aux["length_compliance_rate"] == 1.0
    assert result.complete is True


def test_hellobench_score_outside_length_tolerance():
    ds = HelloBenchDataset()
    ref = HelloBenchReference(target_length_words=100)
    sample = Sample(sample_id="1", prompt="p", reference=ref)
    output = " ".join(f"w{i}" for i in range(10))
    result = ds.score(sample, output)
    assert result.aux["length_compliance_rate"] == 0.0
    assert result.aux["severe_underlength_issue_rate"] == 1.0
    assert result.complete is False


def test_hellobench_detects_repeated_segment_loop():
    segment = "This exact sentence contains enough words to count as a repeated segment."
    output = " ".join([segment] * 8)
    assert repeated_segment_fraction(output) > 0.5
    issues = detect_major_issues("write a long essay", output, target_words=100)
    assert issues.repeated_segment_loop is True
    assert issues.high_repetition is True


def test_hellobench_detects_refusal_and_prompt_echo():
    prompt = " ".join(f"instruction{index}" for index in range(40))
    output = f"I cannot comply with this request. {prompt}"
    issues = detect_major_issues(prompt, output, target_words=100)
    assert issues.refusal is True
    assert issues.prompt_echo is True


def test_hellobench_objective_score_is_not_named_helloeval():
    ds = HelloBenchDataset()
    ref = HelloBenchReference(target_length_words=10)
    sample = Sample(sample_id="1", prompt="write", reference=ref)
    result = ds.score(sample, "one two three four five six seven eight nine ten")
    assert result.primary_score == pytest.approx(1.0)
    assert result.aux["major_issue_free_rate"] == 1.0
    summary = ds.aggregate_records([sample], [result])
    assert summary["objective_quality_score"] == 1.0


def test_hellobench_aggregates_per_length_generation_time():
    ds = HelloBenchDataset()
    samples = [
        Sample(
            sample_id=f"sample-{index}",
            prompt="write",
            reference=HelloBenchReference(target_length_words=2000),
        )
        for index in range(3)
    ]

    def generation(index: int, seconds: float | None, status: RunStatus):
        return GenerationResult(
            request=GenerationRequest(
                prompt="write", max_new_tokens=3072, sample_id=f"sample-{index}"
            ),
            output_text="output" if status == RunStatus.SUCCESS else "",
            status=status,
            final_valid_length=1000 + index,
            timing=TimingResult(seconds) if seconds is not None else None,
        )

    generations = [
        generation(0, 3600.0, RunStatus.SUCCESS),
        generation(1, 7200.0, RunStatus.SUCCESS),
        generation(2, None, RunStatus.FAILED),
    ]
    summary = ds.aggregate_generation_records(samples, generations)
    assert summary["generation_success_rate_2000_words"] == pytest.approx(2 / 3)
    assert summary["timed_sample_count_2000_words"] == 2.0
    assert summary["generation_time_mean_seconds_2000_words"] == 5400.0
    assert summary["generation_time_median_seconds_2000_words"] == 5400.0
    assert summary["generation_time_min_seconds_2000_words"] == 3600.0
    assert summary["generation_time_max_seconds_2000_words"] == 7200.0
    assert summary["generation_time_mean_hours_2000_words"] == pytest.approx(1.5)
