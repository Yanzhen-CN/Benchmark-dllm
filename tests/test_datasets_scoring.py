import json
import zipfile

import pytest

from dllm_bench.datasets.base import Sample, ScoreResult
from dllm_bench.datasets.answer_region import aggregate_direct_answer_only_score
from dllm_bench.datasets.gsm8k import (
    GSM8K_REVISION,
    GSM8KDataset,
    _load_official_test_samples,
    extract_final_number,
    format_gsm8k_four_shot,
)
from dllm_bench.datasets.hellobench import (
    HelloBenchDataset,
    HelloBenchReference,
    detect_major_issues,
    repeated_segment_fraction,
    seq_rep_n,
)
from dllm_bench.datasets.mbpp import (
    MBPPDataset,
    MbppSample,
    _format_official_prompt,
    extract_code,
)
from dllm_bench.datasets.ruler import (
    RulerContextProbeDataset,
    RulerDataset,
    RulerReference,
    generate_ruler_bank,
    position_robustness,
)
from dllm_bench.datasets.structeval_t import (
    STRUCTEVAL_OUTPUT_INSTRUCTION,
    StructEvalSchema,
    StructEvalTDataset,
    _official_structeval_sample,
    evaluate_struct_progress,
)
from dllm_bench.datasets.sudoku9 import (
    Sudoku9Dataset,
    SudokuTraceDataset,
    SudokuReference,
    SUDOKU_ANSWER_BEGIN,
    SUDOKU_ANSWER_END,
    _build_prompt,
    extract_final_grid,
    is_valid_solution,
    _load_official_test_samples as _load_official_sudoku_test_samples,
    _select_formal_subset,
    classify_difficulty,
    group_by_difficulty,
    naked_single_rounds,
    parse_grid,
)
from dllm_bench.datasets.sudoku4 import (
    Sudoku4Dataset,
    Sudoku4Reference,
    Sudoku4ThinkingDataset,
    _load_d1_samples,
    extract_sudoku4_answer,
    is_valid_sudoku4,
)
from dllm_bench.interfaces import (
    GenerationRequest,
    GenerationResult,
    PositionState,
    RunStatus,
    TimingResult,
    TraceStep,
)

# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

def test_gsm8k_flexible_extract_uses_last_number():
    text = "First we compute 12 + 3 = 15. The answer is 30."
    assert extract_final_number(text) == 30.0


def test_gsm8k_flexible_extract_matches_number_after_marker_if_present():
    assert extract_final_number("#### 30\nVerification: 5 * 6 = 30.") == 30.0


def test_gsm8k_flexible_extract_stops_before_next_question():
    assert extract_final_number("The answer is 6.\nQ: unrelated 999") == 6.0


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
    assert samples[0].prompt == format_gsm8k_four_shot("Question 0")
    assert samples[0].prompt.count("Q: ") == 5
    assert samples[0].prompt.endswith("Q: Question 0\nA:")
    # Keep the exact lm-eval reference token; numeric normalization belongs to
    # the official flexible-extract scorer rather than dataset preparation.
    assert samples[0].reference == "10"
    assert samples[0].meta["source_revision"] == GSM8K_REVISION
    assert samples[0].meta["gold_solution"] == "Reasoning for 0.\n#### 10"


# ---------------------------------------------------------------------------
# MBPP
# ---------------------------------------------------------------------------

def test_mbpp_extract_code_from_fence():
    text = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
    code = extract_code(text)
    assert code == "def add(a, b):\n    return a + b"


def test_mbpp_extract_code_from_completion_only_closing_marker():
    text = "def add(a, b):\n    return a + b\n[DONE]"
    code = extract_code(text)
    assert code == "def add(a, b):\n    return a + b"


def test_mbpp_official_prompt_uses_begin_done_delimiters():
    row = {
        "prompt": "Write add.",
        "test_list": ["assert add(1, 2) == 3"],
        "code": "def add(a, b): return a + b",
    }
    demonstration = _format_official_prompt(row, include_solution=True)
    candidate = _format_official_prompt(row, include_solution=False)

    assert demonstration.endswith("def add(a, b): return a + b\n[DONE]")
    assert candidate.endswith("[BEGIN]\n")
    assert "assert add(1, 2) == 3" in candidate


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


def test_structeval_empty_parsed_structure_has_zero_official_render_score():
    ds = StructEvalTDataset()
    schema = StructEvalSchema(format="json", required_keys=["name"])
    sample = Sample(sample_id="1", prompt="p", reference=schema)

    result = ds.score(sample, "{}")

    assert result.primary_score == 0.0
    assert result.aux["official_render_score"] == 0.0
    assert result.aux["official_key_validation_score"] == 0.0


def test_structeval_official_extractor_accepts_unclosed_fence_to_eos():
    ds = StructEvalTDataset()
    schema = StructEvalSchema(format="yaml", required_keys=["name"])
    sample = Sample(sample_id="1", prompt="p", reference=schema)

    result = ds.score(sample, "Here is the result:\n\n```yaml\nname: Alice")

    assert result.primary_score == 1.0
    assert result.aux["official_render_score"] == 1.0
    assert result.aux["official_key_validation_score"] == 1.0


def test_structeval_sample_appends_official_marker_instruction():
    sample = _official_structeval_sample(
        {
            "task_id": 1,
            "query": "Output JSON.",
            "output_type": "json",
            "raw_output_metric": ["name"],
        }
    )

    assert sample.prompt == f"Output JSON.\n\n{STRUCTEVAL_OUTPUT_INSTRUCTION}"
    assert "<|BEGIN_CODE|>" in sample.prompt
    assert "<|END_CODE|>" in sample.prompt


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
# Sudoku 4x4 (d1)
# ---------------------------------------------------------------------------

def test_sudoku4_extracts_last_tagged_answer_and_tolerates_reasoning():
    text = (
        "<reasoning>try 1234</reasoning>\n"
        "<answer>3142243142131324</answer>"
    )

    answer, marker_present, marker_complete = extract_sudoku4_answer(text)

    assert answer == "3142243142131324"
    assert marker_present is True
    assert marker_complete is True


def test_sudoku4_reports_d1_cell_accuracy_and_strict_puzzle_success():
    dataset = Sudoku4Dataset()
    reference = Sudoku4Reference(
        puzzle="3102200002100320",
        solution="3142243142131324",
    )
    sample = Sample("s4", "prompt", reference)

    correct = dataset.score(sample, "<answer>3142243142131324</answer>")
    one_blank_wrong = dataset.score(sample, "<answer>3142243142131321</answer>")
    direct = dataset.score(sample, "3142243142131324")

    assert correct.primary_score == 1.0
    assert direct.primary_score == 1.0
    assert correct.aux["puzzle_success_rate"] == 1.0
    assert one_blank_wrong.primary_score == pytest.approx(7 / 8)
    assert one_blank_wrong.aux["puzzle_success_rate"] == 0.0
    assert is_valid_sudoku4(reference.solution, reference.puzzle)


def test_sudoku4_blank_score_does_not_hide_changed_given():
    dataset = Sudoku4Dataset()
    reference = Sudoku4Reference(
        puzzle="3102200002100320",
        solution="3142243142131324",
    )
    sample = Sample("s4", "prompt", reference)

    result = dataset.score(sample, "4142243142131324")

    assert result.primary_score == 1.0
    assert result.aux["blank_cell_accuracy"] == 1.0
    assert result.aux["given_preservation_rate"] < 1.0
    assert result.aux["given_mismatch_count"] == 1.0
    assert result.aux["constraint_valid"] == 0.0
    assert result.aux["legal_completion"] == 0.0


def test_sudoku4_primary_metric_mirrors_d1_extraction_and_padding_rules():
    dataset = Sudoku4Dataset()
    reference = Sudoku4Reference(
        puzzle="3102200002100320",
        solution="3142243142131324",
    )
    sample = Sample("s4", "prompt", reference)

    shortened = dataset.score(sample, "<answer>314224314213</answer>")
    overlong = dataset.score(sample, "<answer>31422431421313241111</answer>")
    tolerant_only = dataset.score(sample, "final: 3142243142131324")

    assert shortened.primary_score == pytest.approx(6 / 8)
    assert shortened.aux["puzzle_success_rate"] == 0.0
    assert overlong.primary_score == 1.0
    assert tolerant_only.primary_score == 1.0
    assert tolerant_only.aux["puzzle_success_rate"] == 1.0
    assert tolerant_only.aux["direct_answer_instruction_following_rate"] == 0.0


def test_sudoku4_scores_the_last_complete_submission_not_the_copied_puzzle():
    dataset = Sudoku4Dataset()
    reference = Sudoku4Reference(
        puzzle="3102200002100320",
        solution="3142243142131324",
    )
    sample = Sample("s4", "prompt", reference)

    result = dataset.score(
        sample,
        "thought\nCopied puzzle: 3102200002100320\n"
        "Earlier guess: 3142243142131321\n"
        "Final Answer: 3142243142131324",
    )

    assert result.primary_score == 1.0
    assert result.aux["puzzle_success_rate"] == 1.0
    assert result.aux["direct_answer_instruction_following_rate"] == 0.0


def test_sudoku4_unmarked_incomplete_reasoning_digits_get_no_partial_credit():
    dataset = Sudoku4Dataset()
    reference = Sudoku4Reference(
        puzzle="3102200002100320",
        solution="3142243142131324",
    )
    sample = Sample("s4", "prompt", reference)

    result = dataset.score(sample, "thought\npartial row: 314224314213")

    assert result.primary_score == 0.0
    assert result.valid is False


def test_sudoku4_thinking_aggregate_uses_its_dataset_score_key():
    dataset = Sudoku4ThinkingDataset()
    reference = Sudoku4Reference(
        puzzle="3102200002100320",
        solution="3142243142131324",
    )
    result = dataset.score(
        Sample("s4", "prompt", reference),
        "<answer>3142243142131324</answer>",
    )

    summary = dataset.aggregate([result])

    assert summary["sudoku4_thinking_score"] == 1.0
    assert summary["d1_blank_cell_accuracy"] == 1.0


def test_sudoku4_loader_validates_official_shape_and_prompt(tmp_path):
    source = tmp_path / "d1.csv"
    source.write_text(
        "Puzzle,Solution\n"
        + "\n".join(
            f"3102200002100320,3142243142131324" for _ in range(500)
        )
        + "\n",
        encoding="utf-8",
    )

    samples = _load_d1_samples(source)

    assert len(samples) == 500
    assert samples[0].meta["blank_count"] == 8
    assert samples[0].meta["difficulty_stratified"] is False
    assert "Every non-zero digit is a fixed clue" in samples[0].prompt
    assert "COMPLETE 16-character string answer in row-major order" in samples[0].prompt
    assert "using only digits 1-4 and nothing else" in samples[0].prompt
    assert "<reasoning>" not in samples[0].prompt
    assert "<answer>" not in samples[0].prompt
    assert "3102200002100320" in samples[0].prompt

    reasoning_sample = _load_d1_samples(
        source, enable_reasoning=True
    )[0]
    assert "<reasoning>" in reasoning_sample.prompt
    assert reasoning_sample.meta["enable_reasoning"] is True


# Sudoku 9x9 (Park/Ye)
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
    ds = Sudoku9Dataset()
    puzzle = _blank_copy(_EASY_PUZZLE, [(0, 0)])
    ref = SudokuReference(puzzle=puzzle, solution=_EASY_PUZZLE)
    sample = Sample(sample_id="1", prompt="solve", reference=ref)
    text = "".join(str(v) for row in _EASY_PUZZLE for v in row)
    result = ds.score(sample, text)
    assert result.primary_score == 1.0
    assert result.valid is True
    assert result.aux["strict_reference_exact_match"] == 1.0
    assert result.aux["strict_81_digit_format_rate"] == 1.0
    assert result.aux["direct_answer_instruction_following_rate"] == 1.0
    assert result.aux["exact_solve_rate"] == 1.0
    assert result.aux["blank_cell_accuracy"] == 1.0
    assert result.aux["cell_accuracy"] == 1.0
    assert result.aux["given_preservation_rate"] == 1.0
    assert result.aux["conflict_rate"] == 0.0


def test_sudoku_reports_direct_answer_only_score_with_explicit_denominator():
    compliant = ScoreResult(
        primary_score=0.75,
        aux={"direct_answer_instruction_following_rate": 1.0},
    )
    noncompliant = ScoreResult(
        primary_score=0.25,
        aux={"direct_answer_instruction_following_rate": 0.0},
    )

    summary = aggregate_direct_answer_only_score([compliant, noncompliant])

    assert summary["direct_answer_only_score"] == 0.75
    assert summary["direct_answer_eligible_count"] == 1.0
    assert summary["direct_answer_excluded_count"] == 1.0


def test_sudoku9_direct_answer_only_score_is_none_when_cohort_is_empty():
    result = ScoreResult(
        primary_score=0.0,
        aux={"direct_answer_instruction_following_rate": 0.0},
    )

    summary = aggregate_direct_answer_only_score([result])

    assert summary["direct_answer_only_score"] is None
    assert summary["direct_answer_eligible_count"] == 0.0
    assert summary["direct_answer_excluded_count"] == 1.0


def test_sudoku_score_tolerates_reasoning_with_marked_final_answer():
    ds = Sudoku9Dataset()
    puzzle = _blank_copy(_EASY_PUZZLE, [(0, 0)])
    ref = SudokuReference(puzzle=puzzle, solution=_EASY_PUZZLE)
    sample = Sample(sample_id="1", prompt="solve", reference=ref)
    digits = "".join(str(v) for row in _EASY_PUZZLE for v in row)

    result = ds.score(
        sample,
        f"I solved it by checking every row.\n{SUDOKU_ANSWER_BEGIN}\n"
        f"{digits}\n{SUDOKU_ANSWER_END}",
    )

    assert result.primary_score == 1.0
    assert result.valid is True
    assert result.aux["answer_marker_present"] == 1.0
    assert result.aux["answer_marker_complete_rate"] == 1.0
    assert result.aux["direct_answer_instruction_following_rate"] == 0.0
    assert result.aux["reference_exact_match"] == 1.0
    assert result.aux["blank_cell_accuracy"] == 1.0


def test_sudoku_score_tolerates_prose_wrapped_complete_grid_without_marker():
    ds = Sudoku9Dataset()
    puzzle = _blank_copy(_EASY_PUZZLE, [(0, 0)])
    ref = SudokuReference(puzzle=puzzle, solution=_EASY_PUZZLE)
    sample = Sample(sample_id="1", prompt="solve", reference=ref)
    digits = "".join(str(v) for row in _EASY_PUZZLE for v in row)

    result = ds.score(sample, f"Reasoning omitted.\n{digits}")

    assert result.primary_score == 1.0
    assert result.valid is True
    assert result.aux["answer_marker_present"] == 0.0
    assert result.aux["strict_81_digit_format_rate"] == 1.0
    assert result.aux["direct_answer_instruction_following_rate"] == 0.0


def test_sudoku_score_uses_last_complete_candidate_after_self_correction():
    ds = Sudoku9Dataset()
    puzzle = _blank_copy(_EASY_PUZZLE, [(0, 0)])
    ref = SudokuReference(puzzle=puzzle, solution=_EASY_PUZZLE)
    sample = Sample(sample_id="1", prompt="solve", reference=ref)
    correct = "".join(str(v) for row in _EASY_PUZZLE for v in row)
    wrong = "1" * 81

    result = ds.score(
        sample,
        f"thought\nEarlier guess: {wrong}\nWait, that is wrong.\n"
        f"Final Answer: {correct}",
    )

    assert result.primary_score == 1.0
    assert result.aux["reference_exact_match"] == 1.0
    assert result.aux["strict_reference_exact_match"] == 1.0
    assert result.aux["strict_81_digit_format_rate"] == 1.0
    assert result.aux["direct_answer_instruction_following_rate"] == 0.0


def test_sudoku_incomplete_final_cue_does_not_fall_back_to_rejected_draft():
    ds = Sudoku9Dataset()
    puzzle = _blank_copy(_EASY_PUZZLE, [(0, 0)])
    ref = SudokuReference(puzzle=puzzle, solution=_EASY_PUZZLE)
    sample = Sample(sample_id="1", prompt="solve", reference=ref)
    correct = "".join(str(v) for row in _EASY_PUZZLE for v in row)

    result = ds.score(
        sample,
        f"Earlier candidate: {correct} (No)\n"
        "Actually, the solution is: 534678912672195",
    )

    assert result.primary_score == 0.0
    assert result.valid is False
    assert result.aux["answer_detection_method"] == "final_cue_incomplete"


def test_sudoku_rejected_complete_draft_is_not_a_final_answer():
    ds = Sudoku9Dataset()
    puzzle = _blank_copy(_EASY_PUZZLE, [(0, 0)])
    ref = SudokuReference(puzzle=puzzle, solution=_EASY_PUZZLE)
    sample = Sample(sample_id="1", prompt="solve", reference=ref)
    correct = "".join(str(v) for row in _EASY_PUZZLE for v in row)

    result = ds.score(
        sample,
        f"thought\n{correct}\n\nWait, that is wrong. Let me re-solve it.",
    )

    assert result.primary_score == 0.0
    assert result.valid is False
    assert result.aux["answer_detection_method"] == (
        "rejected_candidate_no_final_answer"
    )


def test_sudoku_unclosed_reasoning_grid_is_not_a_final_answer():
    ds = Sudoku9Dataset()
    puzzle = _blank_copy(_EASY_PUZZLE, [(0, 0)])
    ref = SudokuReference(puzzle=puzzle, solution=_EASY_PUZZLE)
    sample = Sample(sample_id="1", prompt="solve", reference=ref)
    correct = "".join(str(v) for row in _EASY_PUZZLE for v in row)

    result = ds.score(sample, f"<reasoning>Draft grid: {correct}")

    assert result.primary_score == 0.0
    assert result.valid is False
    assert result.aux["answer_detection_method"] == "unclosed_thinking"


def test_sudoku_completed_grid_cue_locates_the_final_submission():
    ds = Sudoku9Dataset()
    puzzle = _blank_copy(_EASY_PUZZLE, [(0, 0)])
    ref = SudokuReference(puzzle=puzzle, solution=_EASY_PUZZLE)
    sample = Sample(sample_id="1", prompt="solve", reference=ref)
    rows = "\n".join(" ".join(str(v) for v in row) for row in _EASY_PUZZLE)

    result = ds.score(sample, f"After solving, the completed grid is:\n{rows}")

    assert result.primary_score == 1.0
    assert result.aux["answer_detection_method"].startswith("final_cue_")


def test_sudoku_marker_prevents_reasoning_digits_from_being_scored():
    digits = "".join(str(v) for row in _EASY_PUZZLE for v in row)

    parsed, marker_present = extract_final_grid(
        f"Earlier guess {digits}\n{SUDOKU_ANSWER_BEGIN}\nnot finished"
    )

    assert parsed is None
    assert marker_present is True


def test_sudoku_constraint_validation_rejects_clue_change():
    puzzle = _blank_copy(_EASY_PUZZLE, [(0, 0)])
    changed = [row[:] for row in _EASY_PUZZLE]
    changed[0][1], changed[0][2] = changed[0][2], changed[0][1]

    assert is_valid_solution(_EASY_PUZZLE, puzzle) is True
    assert is_valid_solution(changed, puzzle) is False


def test_sudoku9_official_primary_is_reference_sequence_not_legality():
    ds = Sudoku9Dataset()
    puzzle = [[0] * 9 for _ in range(9)]
    alternative = [
        [2 if value == 1 else 1 if value == 2 else value for value in row]
        for row in _EASY_PUZZLE
    ]
    sample = Sample(
        sample_id="1",
        prompt="solve",
        reference=SudokuReference(puzzle=puzzle, solution=_EASY_PUZZLE),
    )
    text = "".join(str(value) for row in alternative for value in row)

    result = ds.score(sample, text)

    assert is_valid_solution(alternative, puzzle) is True
    assert result.primary_score == 0.0
    assert result.aux["constraint_valid"] == 1.0
    assert result.aux["legal_completion"] == 1.0
    assert result.aux["reference_exact_match"] == 0.0


def test_sudoku9_legal_board_with_changed_givens_is_not_the_puzzle_solution():
    ds = Sudoku9Dataset()
    alternative = [
        [2 if value == 1 else 1 if value == 2 else value for value in row]
        for row in _EASY_PUZZLE
    ]
    sample = Sample(
        sample_id="1",
        prompt="solve",
        reference=SudokuReference(
            puzzle=_EASY_PUZZLE,
            solution=_EASY_PUZZLE,
        ),
    )
    text = "".join(str(value) for row in alternative for value in row)

    result = ds.score(sample, text)

    assert result.primary_score == 0.0
    assert result.aux["constraint_valid"] == 1.0
    assert result.aux["given_preservation_rate"] < 1.0
    assert result.aux["given_mismatch_count"] > 0.0
    assert result.aux["legal_completion"] == 0.0


def test_sudoku_copied_puzzle_gets_no_solving_credit():
    ds = Sudoku9Dataset()
    puzzle = _blank_copy(_EASY_PUZZLE, [(0, 0), (0, 1)])
    sample = Sample(
        sample_id="1",
        prompt="solve",
        reference=SudokuReference(puzzle=puzzle, solution=_EASY_PUZZLE),
    )
    text = "".join(str(v) for row in puzzle for v in row)

    result = ds.score(sample, text)

    assert result.primary_score == 0.0
    assert result.aux["blank_cell_accuracy"] == 0.0
    assert result.aux["given_preservation_rate"] == 1.0
    assert result.aux["exact_solve_rate"] == 0.0
    assert result.aux["strict_81_digit_format_rate"] == 0.0
    assert result.aux["direct_answer_instruction_following_rate"] == 0.0


def test_sudoku_partial_solution_gets_proportional_credit():
    ds = Sudoku9Dataset()
    puzzle = _blank_copy(_EASY_PUZZLE, [(0, 0), (0, 1)])
    partial = _blank_copy(_EASY_PUZZLE, [(0, 1)])
    sample = Sample(
        sample_id="1",
        prompt="solve",
        reference=SudokuReference(puzzle=puzzle, solution=_EASY_PUZZLE),
    )
    text = "".join(str(v) for row in partial for v in row)

    result = ds.score(sample, text)

    assert result.primary_score == 0.0
    assert result.aux["blank_cell_accuracy"] == 0.5
    assert result.aux["given_preservation_rate"] == 1.0
    assert result.aux["exact_solve_rate"] == 0.0
    assert result.aux["strict_81_digit_format_rate"] == 0.0


def test_sudoku_score_unparseable_output():
    ds = Sudoku9Dataset()
    ref = SudokuReference(puzzle=_EASY_PUZZLE, solution=_EASY_PUZZLE)
    sample = Sample(sample_id="1", prompt="solve", reference=ref)
    result = ds.score(sample, "I don't know the answer.")
    assert result.primary_score == 0.0
    assert result.aux["exact_solve_rate"] == 0.0
    assert result.aux["blank_cell_accuracy"] == 0.0
    assert result.aux["given_preservation_rate"] == 0.0
    assert result.valid is False


def test_load_official_sudoku_split_wraps_raw_puzzle_in_minimal_instruction(tmp_path):
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

    assert [sample.prompt for sample in samples] == [
        _build_prompt(easy),
        _build_prompt(hard),
    ]
    assert all("Solve this 9x9 Sudoku puzzle" in sample.prompt for sample in samples)
    assert all(
        "Every non-zero digit is a fixed clue" in sample.prompt for sample in samples
    )
    assert all(
        "COMPLETE 81-character string answer in row-major order" in sample.prompt
        for sample in samples
    )
    assert all(
        "using only digits 1-9 and nothing else" in sample.prompt
        for sample in samples
    )
    assert all("You may reason" not in sample.prompt for sample in samples)
    assert all(SUDOKU_ANSWER_BEGIN not in sample.prompt for sample in samples)
    assert all(SUDOKU_ANSWER_END not in sample.prompt for sample in samples)
    assert [sample.meta["source_index"] for sample in samples] == [2, 3]
    assert [sample.reference.difficulty for sample in samples] == ["easy", "hard"]
    assert all("max_new_tokens" not in sample.meta for sample in samples)

    reasoning_prompt = _build_prompt(easy, enable_reasoning=True)
    assert "You may reason before the final answer" in reasoning_prompt
    assert SUDOKU_ANSWER_BEGIN in reasoning_prompt
    assert SUDOKU_ANSWER_END in reasoning_prompt


def test_instructed_sudoku_wraps_the_same_puzzle_and_reference():
    puzzle = _blank_copy(_EASY_PUZZLE, [(0, 0)])
    puzzle_digits = "".join(str(value) for row in puzzle for value in row)
    reference = SudokuReference(puzzle=puzzle, solution=_EASY_PUZZLE, difficulty="easy")
    raw = Sample(
        sample_id="sudoku-test-0001",
        prompt=puzzle_digits,
        reference=reference,
        meta={"max_new_tokens": 82},
    )

    sample = SudokuTraceDataset(samples=[raw]).load_samples()[0]

    assert sample.sample_id == raw.sample_id
    assert sample.reference is reference
    assert "0 marks a blank cell" in sample.prompt
    assert "Return exactly the completed 81 digits" in sample.prompt
    assert sample.prompt.endswith(puzzle_digits)
    assert sample.meta["prompt_protocol"] == "compact-trace-81-digit-v1"
    assert sample.meta["max_new_tokens"] == 128


def test_instructed_sudoku_persists_decoded_canvas_revision_metrics():
    puzzle = _blank_copy(_EASY_PUZZLE, [(0, 0)])
    reference = SudokuReference(puzzle=puzzle, solution=_EASY_PUZZLE, difficulty="easy")
    sample = Sample("trace", "solve", reference)
    solution = "".join(str(value) for row in _EASY_PUZZLE for value in row)
    wrong_digit = "9" if solution[0] != "9" else "8"
    trace = [
        TraceStep(
            forward_index=index,
            token_ids=list(range(256)),
            position_states=[PositionState.VISIBLE] * 256,
            committed_positions=[],
            decoded_text=digits,
        )
        for index, digits in enumerate((wrong_digit + solution[1:], solution))
    ]
    dataset = SudokuTraceDataset(samples=[sample])

    metrics = dataset.trace_aux_metrics(sample, trace)
    score = dataset.score(sample, solution)
    score.aux.update(metrics)
    summary = dataset.aggregate_records([sample], [score])

    assert metrics["trace_revision_count"] == 1.0
    assert metrics["trace_parseable_step_rate"] == 1.0
    assert metrics["trace_error_then_correct_count"] == 1.0
    assert summary["trace_correction_opportunity_count"] == 1.0
    assert summary["trace_correction_success_rate"] == 1.0


def test_sudoku_preparation_freezes_fifty_easy_and_fifty_hard():
    easy_puzzle = _blank_copy(_EASY_PUZZLE, [(0, 0)])
    hard_puzzle = [[0] * 9 for _ in range(9)]
    official = [
        Sample(
            sample_id=f"sudoku-test-easy-{index:03d}",
            prompt="0" * 81,
            reference=SudokuReference(easy_puzzle, _EASY_PUZZLE, "easy"),
        )
        for index in range(60)
    ] + [
        Sample(
            sample_id=f"sudoku-test-hard-{index:03d}",
            prompt="0" * 81,
            reference=SudokuReference(hard_puzzle, _EASY_PUZZLE, "hard"),
        )
        for index in range(60)
    ]

    selected = _select_formal_subset(
        official, easy_count=50, hard_count=50, seed=42
    )
    repeated = _select_formal_subset(
        list(reversed(official)), easy_count=50, hard_count=50, seed=42
    )

    assert [sample.sample_id for sample in selected] == [
        sample.sample_id for sample in repeated
    ]
    assert sum(sample.reference.difficulty == "easy" for sample in selected) == 50
    assert sum(sample.reference.difficulty == "hard" for sample in selected) == 50
    assert all(sample.meta["formal_subset"] for sample in selected)


def test_sudoku_aggregation_separates_partial_credit_and_exact_solve_rate():
    ds = Sudoku9Dataset()
    puzzle = _blank_copy(_EASY_PUZZLE, [(0, 0), (0, 1)])
    partial = _blank_copy(_EASY_PUZZLE, [(0, 1)])
    samples = [
        Sample(
            sample_id="easy",
            prompt="solve",
            reference=SudokuReference(puzzle, _EASY_PUZZLE, "easy"),
        ),
        Sample(
            sample_id="hard",
            prompt="solve",
            reference=SudokuReference(puzzle, _EASY_PUZZLE, "hard"),
        ),
    ]
    results = [
        ds.score(
            samples[0],
            "".join(str(v) for row in partial for v in row),
        ),
        ds.score(
            samples[1],
            "".join(str(v) for row in _EASY_PUZZLE for v in row),
        ),
    ]

    summary = ds.aggregate_records(samples, results)

    assert summary["sudoku9_score"] == 0.5
    assert summary["blank_cell_accuracy"] == 0.75
    assert summary["blank_cell_accuracy_easy"] == 0.5
    assert summary["blank_cell_accuracy_hard"] == 1.0
    assert summary["exact_solve_rate_easy"] == 0.0
    assert summary["exact_solve_rate_hard"] == 1.0
    assert summary["accuracy_easy"] == summary["exact_solve_rate_easy"]
    assert summary["accuracy_hard"] == summary["exact_solve_rate_hard"]


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


def test_ruler_context_probe_aggregate_uses_its_dataset_score_key():
    dataset = RulerContextProbeDataset()
    reference = RulerReference(
        task_type="niah",
        position="middle",
        required_answers=["P123"],
        context_length=4096,
    )
    sample = Sample(
        sample_id="probe",
        prompt="p",
        reference=reference,
        meta={"context_window_tokens": 4096},
    )
    result = dataset.score(sample, "P123")

    summary = dataset.aggregate_records([sample], [result])

    assert summary["ruler_context_probe_score"] == 1.0
    assert summary["ruler_string_match_all"] == 1.0


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


def test_hellobench_minimum_length_is_complete_and_overlength_is_not_penalized():
    ds = HelloBenchDataset()
    ref = HelloBenchReference(target_length_words=100)
    sample = Sample(sample_id="1", prompt="p", reference=ref)
    output = " ".join(f"w{i}" for i in range(100))
    result = ds.score(sample, output)
    assert result.aux["length_compliance_rate"] == 1.0
    assert result.complete is True
    longer = ds.score(sample, " ".join(f"w{i}" for i in range(150)))
    assert longer.aux["length_attainment"] == 1.0
    assert longer.aux["severe_overlength_issue_rate"] == 0.0
    assert longer.complete is True


def test_hellobench_score_below_minimum_length():
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


def test_hellobench_integrity_score_is_not_named_helloeval():
    ds = HelloBenchDataset()
    ref = HelloBenchReference(target_length_words=10)
    sample = Sample(sample_id="1", prompt="write", reference=ref)
    result = ds.score(sample, "one two three four five six seven eight nine ten")
    assert result.primary_score == pytest.approx(1.0)
    assert result.aux["major_issue_free_rate"] == 1.0
    summary = ds.aggregate_records([sample], [result])
    assert summary["long_output_integrity_score"] == 1.0


def test_hellobench_scores_only_the_article_after_thinking():
    ds = HelloBenchDataset()
    ref = HelloBenchReference(target_length_words=4)
    sample = Sample(sample_id="1", prompt="write", reference=ref)
    output = "<think>private plan with repeated repeated text</think>\n\n# Essay\none two three four"
    result = ds.score(sample, output)
    assert result.aux["answer_region_detected_rate"] == 1.0
    assert result.aux["answer_word_count"] == 5.0
    assert result.aux["reasoning_word_count"] > 0


def test_hellobench_rejects_unclosed_thinking_as_final_article():
    ds = HelloBenchDataset()
    ref = HelloBenchReference(target_length_words=4)
    sample = Sample(sample_id="1", prompt="write", reference=ref)
    result = ds.score(sample, "<think>one two three four")
    assert result.primary_score == 0.0
    assert result.aux["answer_region_detected_rate"] == 0.0
    assert result.complete is False


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
