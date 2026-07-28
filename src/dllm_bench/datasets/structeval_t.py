"""StructEval-T official final score plus diagnostic formation metrics.

The final-output primary score follows the upstream non-renderable evaluator:
``round(0.2 * render_score + 0.8 * key_validation_score, 2)``.  Here
``render_score`` is strict syntax/parse validity and ``key_validation_score``
is the fraction of ``raw_output_metric`` paths present in the parsed object.

The fault-tolerant, partial-structure-aware detectors remain deliberately
separate.  They are diagnostics for generation-process analysis and must not
silently replace the official final-output score.

:func:`evaluate_struct_progress` is the reusable building block: the dataset
scorer calls it once on the final output for the section-1 Complete Correct
Rate, and section 4.3's Structure/Content-formation-strategy analysis calls
the *same* function on the decoded text at every scoring checkpoint (every 4
forwards) to get the ``StructureProgress``/``ContentProgress`` curves that
feed :mod:`dllm_bench.metrics.strategy_score`.
"""

from __future__ import annotations

import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Literal

from ..interfaces import TraceStep
from .base import Dataset, Sample, ScoreResult
from .remote import ensure_download

STRUCTEVAL_REVISION = "788a40c0bf41aa7b2cbc6a480015c842353a2492"
STRUCTEVAL_T_SHA256 = "a87f46dcafba1e5bde8a85c9f99f106089ca397cb345cd07a8fea1fe03b4ffab"
STRUCTEVAL_T_URL = (
    "https://raw.githubusercontent.com/TIGER-AI-Lab/StructEval/"
    f"{STRUCTEVAL_REVISION}/dataset/nonrenderable.json"
)

Format = Literal["json", "yaml", "xml", "toml", "csv"]

_CODE_FENCE_RE = re.compile(r"```(?:\w+)?\s*\n(.*?)```", re.DOTALL)


@dataclass
class StructEvalSchema:
    """One StructEval-T sample's expected structure, per Appendix A.2."""

    format: Format
    required_keys: list[str] = field(default_factory=list)
    """Dotted flattened paths (see :func:`flatten` conventions) that must be present."""
    nesting_paths: list[str] = field(default_factory=list)
    """Dotted path prefixes that must exist as a nested object/array (HierarchyCoverage)."""
    critical_content: list[str] = field(default_factory=list)
    """Substrings that must appear somewhere in the raw text."""
    numeric_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    """required_key -> (min, max) inclusive bound."""
    item_count_path: str | None = None
    expected_item_count: int | None = None
    target_length_words: int | None = None


@dataclass
class StructProgressResult:
    symbol_coverage: float
    key_coverage: float
    hierarchy_coverage: float
    value_coverage: float
    critical_content_coverage: float
    parseability: float
    item_count_progress: float | None
    numeric_bound_satisfaction: float | None
    length_ratio: float | None

    @property
    def structure_progress(self) -> float:
        return (self.symbol_coverage + self.key_coverage + self.hierarchy_coverage) / 3

    @property
    def content_progress(self) -> float:
        return (self.value_coverage + self.critical_content_coverage) / 2


def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _auto_close_json(text: str) -> str | None:
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    if not stack:
        return None
    closing = "".join("}" if c == "{" else "]" for c in reversed(stack))
    return text + closing


def _parse_json_partial(text: str) -> Any | None:
    text = _strip_code_fence(text)
    for candidate in (text, _auto_close_json(text)):
        if candidate is None:
            continue
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _parse_yaml_partial(text: str) -> Any | None:
    import yaml

    text = _strip_code_fence(text)
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        lines = text.splitlines()
        while lines:
            lines.pop()
            try:
                return yaml.safe_load("\n".join(lines))
            except yaml.YAMLError:
                continue
        return None


def _parse_xml_partial(text: str) -> ET.Element | None:
    text = _strip_code_fence(text)
    try:
        return ET.fromstring(text)
    except ET.ParseError:
        pass

    open_tags: list[str] = []
    for m in re.finditer(r"<(/?)([A-Za-z_][\w\-.]*)[^>]*?(/?)>", text):
        closing, tag, self_close = m.groups()
        if self_close:
            continue
        if closing:
            if open_tags and open_tags[-1] == tag:
                open_tags.pop()
        else:
            open_tags.append(tag)
    if not open_tags:
        return None
    patched = text + "".join(f"</{t}>" for t in reversed(open_tags))
    try:
        return ET.fromstring(patched)
    except ET.ParseError:
        return None


def _parse_toml_partial(text: str) -> Any | None:
    text = _strip_code_fence(text)
    tomllib = None
    try:
        import tomllib as _tomllib

        tomllib = _tomllib
    except ImportError:
        try:
            import tomli as _tomllib

            tomllib = _tomllib
        except ImportError:
            tomllib = None

    if tomllib is not None:
        try:
            return tomllib.loads(text)
        except Exception:
            pass

    result: dict[str, str] = {}
    for m in re.finditer(r'^\s*([A-Za-z_][\w\-]*)\s*=\s*(.+)$', text, re.MULTILINE):
        key, value = m.groups()
        result[key] = value.strip().strip("\"'")
    return result or None


def _parse_csv_partial(text: str) -> dict[str, Any] | None:
    text = _strip_code_fence(text)
    reader = csv.reader(io.StringIO(text))
    rows = [r for r in reader if r]
    if not rows:
        return None
    header, *data_rows = rows
    return {"header": header, "rows": data_rows}


_PARSERS = {
    "json": _parse_json_partial,
    "yaml": _parse_yaml_partial,
    "xml": _parse_xml_partial,
    "toml": _parse_toml_partial,
    "csv": _parse_csv_partial,
}


def _xml_element_to_official_dict(element: ET.Element) -> dict[str, Any]:
    """Small ``xmltodict``-compatible conversion for official path checks."""
    node: dict[str, Any] = {f"@{key}": value for key, value in element.attrib.items()}
    children: dict[str, list[Any]] = {}
    for child in element:
        converted = _xml_element_value(child)
        children.setdefault(child.tag, []).append(converted)
    for tag, values in children.items():
        node[tag] = values[0] if len(values) == 1 else values
    text = (element.text or "").strip()
    if text:
        if node:
            node["#text"] = text
        else:
            return {element.tag: text}
    return {element.tag: node}


def _xml_element_value(element: ET.Element) -> Any:
    wrapped = _xml_element_to_official_dict(element)
    return wrapped[element.tag]


def _parse_official_strict(text: str, fmt: Format) -> Any | None:
    """Parse exactly enough for StructEval's strict non-renderable evaluator.

    Unlike :func:`evaluate_struct_progress`, this never repairs or truncates a
    malformed output.  A fenced payload is extracted because StructEval's
    render stage likewise extracts the generated code before saving it.
    """
    payload = _strip_code_fence(text)
    try:
        if fmt == "json":
            return json.loads(payload)
        if fmt == "yaml":
            import yaml

            try:
                return yaml.safe_load(payload)
            except yaml.YAMLError:
                return None
        if fmt == "xml":
            return _xml_element_to_official_dict(ET.fromstring(payload))
        if fmt == "toml":
            try:
                import tomllib
            except ImportError:  # pragma: no cover - Python 3.10 only
                import tomli as tomllib
            return tomllib.loads(payload)
        if fmt == "csv":
            reader = csv.DictReader(io.StringIO(payload))
            return {"csv_headers": reader.fieldnames, "csv_rows": list(reader)}
    except (ValueError, TypeError, ET.ParseError):
        return None
    return None


def _tokenize_official_path(path: str) -> list[str]:
    """Port of StructEval ``eval_utils.tokenize_path``."""
    if path.startswith("csv::"):
        return [path]
    tokens: list[str] = []
    buffer = ""
    in_backticks = False
    index = 0
    while index < len(path):
        char = path[index]
        if char == "`":
            in_backticks = not in_backticks
            index += 1
            continue
        if char == "." and not in_backticks:
            if buffer:
                tokens.append(buffer)
                buffer = ""
            index += 1
            continue
        if char == "[" and not in_backticks:
            if buffer:
                tokens.append(buffer)
                buffer = ""
            close = path.find("]", index)
            if close == -1:
                raise ValueError(f"unclosed '[' in StructEval path: {path}")
            tokens.append(path[index : close + 1])
            index = close + 1
            continue
        buffer += char
        index += 1
    if buffer:
        tokens.append(buffer)
    return tokens


def _official_path_exists(data: Any, path: str) -> bool:
    """Port of StructEval ``eval_utils.path_exists``."""
    try:
        tokens = _tokenize_official_path(path)
    except ValueError:
        return False

    def walk(node: Any, remaining: list[str]) -> bool:
        if not remaining:
            return True
        token, *rest = remaining
        if isinstance(node, dict) and "csv_headers" in node and token.startswith("csv::"):
            header = token[5:]
            return header in (node.get("csv_headers") or []) and not rest
        if token == "*":
            return isinstance(node, list) and any(walk(item, rest) for item in node)
        if token.startswith("[") and token.endswith("]"):
            try:
                item_index = int(token[1:-1])
            except ValueError:
                return False
            return (
                isinstance(node, list)
                and 0 <= item_index < len(node)
                and walk(node[item_index], rest)
            )
        if isinstance(node, dict):
            if token in node:
                return walk(node[token], rest)
            if token.startswith("@") and token[1:] in node:
                return walk(node[token[1:]], rest)
        return False

    return walk(data, tokens)


def official_structeval_nonrenderable_score(
    text: str, schema: StructEvalSchema
) -> tuple[float, float, float]:
    """Return upstream ``(final, render, key_validation)`` scores."""
    parsed = _parse_official_strict(text, schema.format)
    render_score = 1.0 if parsed is not None else 0.0
    if parsed is None or not schema.required_keys:
        key_validation_score = 0.0
    else:
        key_validation_score = sum(
            1.0 for path in schema.required_keys if _official_path_exists(parsed, path)
        ) / len(schema.required_keys)
    final_score = round(0.2 * render_score + 0.8 * key_validation_score, 2)
    return final_score, render_score, key_validation_score


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten JSON/YAML/TOML-like nested dict/list into dotted-path -> leaf value."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(value, path))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            path = f"{prefix}[{index}]"
            out.update(flatten(value, path))
    else:
        out[prefix] = obj
    return out


def _flatten_xml(element: ET.Element, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    path = f"{prefix}.{element.tag}" if prefix else element.tag
    for attr_name, attr_value in element.attrib.items():
        out[f"{path}@{attr_name}"] = attr_value
    children = list(element)
    if not children:
        out[path] = (element.text or "").strip()
    for child in children:
        out.update(_flatten_xml(child, path))
    return out


def _flatten_csv(parsed: dict[str, Any]) -> dict[str, Any]:
    header = parsed["header"]
    out: dict[str, Any] = {}
    for row_index, row in enumerate(parsed["rows"]):
        for col_index, value in enumerate(row):
            col_name = header[col_index] if col_index < len(header) else str(col_index)
            out[f"row{row_index}.{col_name}"] = value
    return out


def _flatten_by_format(parsed: Any, fmt: Format) -> dict[str, Any]:
    if fmt == "xml":
        return _flatten_xml(parsed) if isinstance(parsed, ET.Element) else {}
    if fmt == "csv":
        return _flatten_csv(parsed) if isinstance(parsed, dict) else {}
    return flatten(parsed)


def _lookup_path(flattened: dict[str, Any], target_path: str) -> Any | None:
    """Exact match, or (for list-indexed paths like ``items[0].name``) fall
    back to matching any index at that position — required keys are usually
    schema-level ("items[*].name" semantics), not tied to one specific index."""
    if target_path in flattened:
        return flattened[target_path]
    pattern = re.compile("^" + re.escape(target_path).replace(r"\[\*\]", r"\[\d+\]") + "$")
    for key, value in flattened.items():
        if pattern.match(key):
            return value
    return None


def _count_list_items(flattened: dict[str, Any], list_path: str) -> int:
    prefix = f"{list_path}["
    indices = set()
    for key in flattened:
        if key.startswith(prefix):
            rest = key[len(prefix):]
            end = rest.find("]")
            if end != -1:
                indices.add(rest[:end])
    return len(indices)


def evaluate_struct_progress(text: str, schema: StructEvalSchema) -> StructProgressResult:
    parser = _PARSERS[schema.format]
    parsed = parser(text)
    parseability = 1.0 if parsed is not None else 0.0
    symbol_coverage = parseability  # successfully recognizing the format's structural symbols

    if parsed is None:
        return StructProgressResult(
            symbol_coverage=0.0,
            key_coverage=0.0,
            hierarchy_coverage=0.0,
            value_coverage=0.0,
            critical_content_coverage=_critical_content_coverage(text, schema),
            parseability=0.0,
            item_count_progress=0.0 if schema.expected_item_count else None,
            numeric_bound_satisfaction=0.0 if schema.numeric_ranges else None,
            length_ratio=_length_ratio(text, schema),
        )

    flattened = _flatten_by_format(parsed, schema.format)

    key_coverage = _ratio_present(schema.required_keys, flattened)
    hierarchy_coverage = _hierarchy_coverage(schema.nesting_paths, flattened)
    value_coverage = _value_coverage(schema.required_keys, flattened)
    critical_content_coverage = _critical_content_coverage(text, schema)

    item_count_progress = None
    if schema.expected_item_count and schema.item_count_path:
        actual = _count_list_items(flattened, schema.item_count_path)
        item_count_progress = min(actual / schema.expected_item_count, 1.0)

    numeric_bound_satisfaction = None
    if schema.numeric_ranges:
        satisfied = 0
        for key, (low, high) in schema.numeric_ranges.items():
            value = _lookup_path(flattened, key)
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if low <= numeric_value <= high:
                satisfied += 1
        numeric_bound_satisfaction = satisfied / len(schema.numeric_ranges)

    return StructProgressResult(
        symbol_coverage=symbol_coverage,
        key_coverage=key_coverage,
        hierarchy_coverage=hierarchy_coverage,
        value_coverage=value_coverage,
        critical_content_coverage=critical_content_coverage,
        parseability=parseability,
        item_count_progress=item_count_progress,
        numeric_bound_satisfaction=numeric_bound_satisfaction,
        length_ratio=_length_ratio(text, schema),
    )


def checkpoint_indices(num_steps: int, interval: int) -> list[int]:
    """Section 4.2.2: score every `interval`-th forward, always including the
    final forward regardless of whether it lands on the interval."""
    if num_steps <= 0:
        return []
    indices = list(range(0, num_steps, interval))
    if indices[-1] != num_steps - 1:
        indices.append(num_steps - 1)
    return indices


def struct_eval_t_checkpoint_scores(
    trace: list[TraceStep], schema: StructEvalSchema, interval: int = 4
) -> tuple[list[float], list[float]]:
    """Section 4.2.2: score `trace`'s decoded_text at every 4th forward (plus
    always the final forward) using StructEval-T's own detectors, returning
    (structure_progress_scores, content_progress_scores) — the per-checkpoint
    curves :mod:`dllm_bench.metrics.strategy_score` turns into one sample's
    pairwise Structure-First Score.
    """
    if not trace:
        return [], []
    structure_scores = []
    content_scores = []
    for i in checkpoint_indices(len(trace), interval):
        progress = evaluate_struct_progress(trace[i].decoded_text, schema)
        structure_scores.append(progress.structure_progress)
        content_scores.append(progress.content_progress)
    return structure_scores, content_scores


def _ratio_present(keys: list[str], flattened: dict[str, Any]) -> float:
    if not keys:
        return 1.0
    present = sum(1 for k in keys if _lookup_path(flattened, k) is not None)
    return present / len(keys)


def _value_coverage(keys: list[str], flattened: dict[str, Any]) -> float:
    if not keys:
        return 1.0
    non_empty = 0
    for k in keys:
        value = _lookup_path(flattened, k)
        if value is not None and str(value).strip() != "":
            non_empty += 1
    return non_empty / len(keys)


def _hierarchy_coverage(nesting_paths: list[str], flattened: dict[str, Any]) -> float:
    if not nesting_paths:
        return 1.0
    present = sum(
        1 for path in nesting_paths if any(k.startswith(path) for k in flattened)
    )
    return present / len(nesting_paths)


def _critical_content_coverage(text: str, schema: StructEvalSchema) -> float:
    if not schema.critical_content:
        return 1.0
    lowered = text.lower()
    hits = sum(1 for phrase in schema.critical_content if phrase.lower() in lowered)
    return hits / len(schema.critical_content)


def _length_ratio(text: str, schema: StructEvalSchema) -> float | None:
    if not schema.target_length_words:
        return None
    word_count = len(text.split())
    return word_count / schema.target_length_words


class StructEvalTDataset(Dataset):
    name = "structeval_t"

    def __init__(self, samples: list[Sample] | None = None) -> None:
        self._samples = list(samples) if samples is not None else None

    def load_samples(self, n: int | None = None) -> list[Sample]:
        if self._samples is not None:
            samples = list(self._samples)
        else:
            source = ensure_download(
                "structeval_t", "nonrenderable.json",
                url=STRUCTEVAL_T_URL, sha256=STRUCTEVAL_T_SHA256,
            )
            rows = json.loads(source.read_text(encoding="utf-8"))
            samples = [_official_structeval_sample(row) for row in rows]
        return samples[:n] if n is not None else samples

    def score(self, sample: Sample, output_text: str) -> ScoreResult:
        schema: StructEvalSchema = sample.reference
        official_score, render_score, key_validation_score = (
            official_structeval_nonrenderable_score(output_text, schema)
        )
        progress = evaluate_struct_progress(output_text, schema)

        complete_correct = (
            progress.parseability == 1.0
            and progress.key_coverage == 1.0
            and progress.value_coverage == 1.0
            and progress.critical_content_coverage == 1.0
            and (progress.item_count_progress is None or progress.item_count_progress == 1.0)
            and (
                progress.numeric_bound_satisfaction is None
                or progress.numeric_bound_satisfaction == 1.0
            )
        )
        return ScoreResult(
            primary_score=official_score,
            aux={
                "official_render_score": render_score,
                "official_key_validation_score": key_validation_score,
                "complete_correct_rate": 1.0 if complete_correct else 0.0,
                "format_valid_rate": render_score,
                "structure_progress": progress.structure_progress,
                "content_progress": progress.content_progress,
                "field_completion_rate": key_validation_score,
            },
            valid=render_score == 1.0,
            complete=complete_correct or progress.value_coverage > 0,
        )

    def aggregate(self, results: list[ScoreResult]) -> dict[str, float]:
        summary = super().aggregate(results)
        summary["final_eval_score"] = summary["structeval_t_score"]
        structure_first = [
            result.aux["structure_first_score"]
            for result in results
            if "structure_first_score" in result.aux
        ]
        summary["structure_first_eligible_ratio"] = len(structure_first) / len(results)
        if structure_first:
            summary["structure_first_score"] = sum(structure_first) / len(structure_first)
        return summary

    def trace_aux_metrics(
        self, sample: Sample, trace: list[TraceStep]
    ) -> dict[str, float]:
        if not trace:
            return {"structure_first_eligible_rate": 0.0}
        from ..metrics.strategy_score import strategy_score

        schema: StructEvalSchema = sample.reference
        structure, content = struct_eval_t_checkpoint_scores(trace, schema)
        score = strategy_score(structure, content)
        if score is None:
            return {"structure_first_eligible_rate": 0.0}
        return {
            "structure_first_score": score,
            "structure_first_eligible_rate": 1.0,
        }


def _nesting_prefixes(paths: list[str]) -> list[str]:
    prefixes: set[str] = set()
    for path in paths:
        normalized = re.sub(r"\[\d+\]", "", path)
        parts = normalized.split(".")
        for end in range(1, len(parts)):
            prefixes.add(".".join(parts[:end]))
    return sorted(prefixes)


def _official_structeval_sample(row: dict[str, Any]) -> Sample:
    fmt = str(row["output_type"]).lower()
    if fmt not in _PARSERS:
        raise ValueError(f"unexpected StructEval-T output type: {row['output_type']!r}")
    required = [str(value) for value in row.get("raw_output_metric", [])]
    schema = StructEvalSchema(
        format=fmt,
        required_keys=required,
        nesting_paths=_nesting_prefixes(required),
    )
    return Sample(
        sample_id=f"structeval-t-{row['task_id']}",
        prompt=str(row["query"]),
        reference=schema,
        meta={
            "source": "TIGER-AI-Lab/StructEval",
            "source_revision": STRUCTEVAL_REVISION,
            "task_name": row.get("task_name"),
            "input_type": row.get("input_type"),
            "output_type": row.get("output_type"),
        },
    )
