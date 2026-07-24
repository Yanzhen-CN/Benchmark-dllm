import pytest

from dllm_bench.datasets.base import Sample
from dllm_bench.datasets.gsm8k import GSM8KDataset, extract_final_number
from dllm_bench.datasets.hellobench import HelloBenchDataset, HelloBenchReference, seq_rep_n
from dllm_bench.datasets.ifeval import (
    IFEvalDataset,
    IFEvalSample,
    InstructionSpec,
)
from dllm_bench.datasets.mbpp import MBPPDataset, MbppSample, extract_code
from dllm_bench.datasets.ruler import RulerDataset, RulerReference, position_robustness
from dllm_bench.datasets.structeval_t import (
    StructEvalSchema,
    StructEvalTDataset,
    evaluate_struct_progress,
)
from dllm_bench.datasets.sudoku import (
    SudokuDataset,
    SudokuReference,
    classify_difficulty,
    group_by_difficulty,
    parse_grid,
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
    assert result.primary_score == 0.0
    assert result.aux["field_completion_rate"] == pytest.approx(0.5)


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


# ---------------------------------------------------------------------------
# IFEval
# ---------------------------------------------------------------------------

def test_ifeval_all_constraints_satisfied():
    ds = IFEvalDataset()
    ref = IFEvalSample(
        form_constraints=[InstructionSpec("format:number_bullets", {"count": 2, "relation": "at_least"})],
        content_requirements=[InstructionSpec("keywords:existence", {"keywords": ["python"]})],
    )
    sample = Sample(sample_id="1", prompt="p", reference=ref)
    output = "- I like python.\n- It is great."
    result = ds.score(sample, output)
    assert result.primary_score == 1.0
    assert result.aux["instruction_level_strict"] == 1.0


def test_ifeval_partial_failure_lowers_instruction_level_but_not_prompt_level():
    ds = IFEvalDataset()
    ref = IFEvalSample(
        form_constraints=[InstructionSpec("case:all_uppercase", {})],
        content_requirements=[InstructionSpec("keywords:existence", {"keywords": ["python"]})],
    )
    sample = Sample(sample_id="1", prompt="p", reference=ref)
    output = "I like PYTHON but this is not all uppercase."
    result = ds.score(sample, output)
    assert result.primary_score == 0.0
    assert result.aux["instruction_level_strict"] == pytest.approx(0.5)


def test_ifeval_terminal_constraint_excluded_from_constraint_progress():
    ref = IFEvalSample(
        form_constraints=[
            InstructionSpec("format:title", {}),
            InstructionSpec("startend:end_phrase", {"end_phrase": "The End"}, terminal=True),
        ],
        content_requirements=[],
    )
    from dllm_bench.datasets.ifeval import evaluate_ifeval_progress

    constraint_progress, content_progress, _ = evaluate_ifeval_progress("<<My Title>> some body without ending", ref)
    # only the non-terminal title constraint counts toward constraint_progress
    assert constraint_progress == 1.0


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
    text = "\n".join(" ".join(str(v) for v in row) for row in _EASY_PUZZLE)
    result = ds.score(sample, text)
    assert result.primary_score == 1.0
    assert result.aux["cell_accuracy"] == 1.0
    assert result.aux["conflict_rate"] == 0.0


def test_sudoku_score_unparseable_output():
    ds = SudokuDataset()
    ref = SudokuReference(puzzle=_EASY_PUZZLE, solution=_EASY_PUZZLE)
    sample = Sample(sample_id="1", prompt="solve", reference=ref)
    result = ds.score(sample, "I don't know the answer.")
    assert result.primary_score == 0.0
    assert result.valid is False


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


def test_ruler_score_partial_match_for_multi_answer():
    ds = RulerDataset()
    ref = RulerReference(task_type="multi_hop", position="middle", required_answers=["Alice", "Bob"])
    sample = Sample(sample_id="1", prompt="p", reference=ref)
    result = ds.score(sample, "The answer involves Alice.")
    assert result.primary_score == 0.0
    assert result.aux["partial_match_rate"] == pytest.approx(0.5)


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
    assert result.complete is False


def test_hellobench_uses_custom_judge_fn_when_provided():
    ds = HelloBenchDataset(judge_fn=lambda prompt, output: 0.42)
    ref = HelloBenchReference(target_length_words=5)
    sample = Sample(sample_id="1", prompt="p", reference=ref)
    result = ds.score(sample, "one two three four five")
    assert result.primary_score == pytest.approx(0.42)
