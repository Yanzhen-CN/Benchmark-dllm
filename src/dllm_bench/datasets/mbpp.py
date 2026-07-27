"""MBPP-Sanitized: Pass@1, plus test-pass-rate, executable-rate, complete-code-rate.

Candidate code is executed against the sample's test asserts in a *separate
subprocess* (own interpreter, timeout, no shared state with this process) —
the standard approach for code-generation benchmarks (HumanEval/MBPP-style
execution eval). This still runs arbitrary model-generated code locally;
only run it in an environment where that is acceptable (a sandboxed CI
runner/VM), same as upstream MBPP/HumanEval harnesses.
"""

from __future__ import annotations

import io
import keyword
import re
import subprocess
import sys
import tempfile
import tokenize
from dataclasses import dataclass
from pathlib import Path

from .base import Dataset, Sample, ScoreResult
from ..interfaces import TraceStep

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_TRAILING_CONTINUATION_RE = re.compile(r"(\\|[+\-*/,(\[{]|\band\b|\bor\b)\s*$")


@dataclass
class MbppSample:
    """Convenience constructor: pass this as ``Sample.reference``."""

    test_list: list[str]
    test_setup_code: str = ""


def extract_code(output_text: str) -> str:
    fence_match = _CODE_FENCE_RE.search(output_text)
    if fence_match:
        return fence_match.group(1).strip()

    def_index = output_text.find("def ")
    if def_index != -1:
        return output_text[def_index:].strip()
    return output_text.strip()


def _looks_complete(code: str) -> bool:
    if "def " not in code:
        return False
    lines = [ln for ln in code.splitlines() if ln.strip()]
    if not lines:
        return False
    last_line = lines[-1]
    if _TRAILING_CONTINUATION_RE.search(last_line):
        return False
    open_count = sum(code.count(c) for c in "([{")
    close_count = sum(code.count(c) for c in ")]}")
    return open_count == close_count


def run_tests(code: str, test_list: list[str], test_setup_code: str = "", timeout_s: float = 10.0) -> tuple[bool, bool]:
    """Returns (executable, all_tests_passed)."""
    program = "\n".join([test_setup_code, code, *test_list])
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = Path(tmp_dir) / "candidate.py"
        script_path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                timeout=timeout_s,
                cwd=tmp_dir,
            )
        except subprocess.TimeoutExpired:
            return True, False

    executable = proc.returncode == 0 or _is_assertion_only_failure(proc.stderr.decode("utf-8", "replace"))
    all_passed = proc.returncode == 0
    return executable, all_passed


def _is_assertion_only_failure(stderr_text: str) -> bool:
    """A raised AssertionError still means the *code itself* executed fine —
    only the test outcome was a failure, which is not the same as unexecutable
    (SyntaxError/NameError/etc.)."""
    return "AssertionError" in stderr_text and "SyntaxError" not in stderr_text


class MBPPDataset(Dataset):
    name = "mbpp"

    def __init__(self, samples: list[Sample] | None = None, timeout_s: float = 10.0) -> None:
        self._samples = samples or []
        self._timeout_s = timeout_s

    def load_samples(self, n: int | None = None) -> list[Sample]:
        return self._samples[:n] if n is not None else list(self._samples)

    def score(self, sample: Sample, output_text: str) -> ScoreResult:
        ref: MbppSample = sample.reference
        code = extract_code(output_text)
        complete = _looks_complete(code)

        executable, all_passed = run_tests(
            code, ref.test_list, ref.test_setup_code, timeout_s=self._timeout_s
        )
        return ScoreResult(
            primary_score=1.0 if all_passed else 0.0,
            aux={"executable_rate": 1.0 if executable else 0.0},
            valid=executable,
            complete=complete,
        )


def _python_feature_sets(code: str) -> tuple[list[set[str]], list[set[str]]]:
    """Appendix A.3's four structure and three content feature groups."""
    brackets: set[str] = set()
    indents: set[str] = set()
    keywords: set[str] = set()
    signatures: set[str] = set()
    identifiers: set[str] = set()
    literals: set[str] = set()
    expressions: set[str] = set()

    tokens = []
    generator = tokenize.generate_tokens(io.StringIO(code).readline)
    while True:
        try:
            tokens.append(next(generator))
        except (StopIteration, tokenize.TokenError, IndentationError):
            break

    for tok in tokens:
        value = tok.string
        location = f"{tok.start[0]}:{tok.start[1]}"
        if tok.type == tokenize.OP:
            if value in "()[]{}":
                brackets.add(f"{value}@{location}")
            if value not in {",", ":", ";", "."}:
                expressions.add(f"{value}@{location}")
        elif tok.type == tokenize.INDENT:
            indents.add(f"indent@{tok.start[0]}")
        elif tok.type == tokenize.NAME:
            if keyword.iskeyword(value):
                keywords.add(f"{value}@{location}")
            else:
                identifiers.add(f"{value}@{location}")
        elif tok.type in {tokenize.NUMBER, tokenize.STRING}:
            literals.add(f"{value}@{location}")

    for match in re.finditer(r"(?m)^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", code):
        signatures.add(match.group(1))
    return [brackets, indents, keywords, signatures], [identifiers, literals, expressions]


def _mean_feature_coverage(current: list[set[str]], final: list[set[str]]) -> float:
    values = [len(now & target) / len(target) for now, target in zip(current, final) if target]
    return sum(values) / len(values) if values else 0.0


def mbpp_checkpoint_scores(
    trace: list[TraceStep], interval: int = 4
) -> tuple[list[float], list[float]]:
    """StructureProgress/ContentProgress every four forwards plus the final.

    Coverage is measured against the final checkpoint's feature inventory,
    giving the per-sample final-state normalization required by Appendix A.4.
    """
    if not trace:
        return [], []
    from .structeval_t import checkpoint_indices

    final_structure, final_content = _python_feature_sets(trace[-1].decoded_text)
    structure_scores: list[float] = []
    content_scores: list[float] = []
    for index in checkpoint_indices(len(trace), interval):
        structure, content = _python_feature_sets(trace[index].decoded_text)
        structure_scores.append(_mean_feature_coverage(structure, final_structure))
        content_scores.append(_mean_feature_coverage(content, final_content))
    return structure_scores, content_scores
