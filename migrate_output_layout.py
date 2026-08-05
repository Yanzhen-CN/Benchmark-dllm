#!/usr/bin/env python3
"""Safely migrate flat run directories to model/config/dataset directories."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil


STAGES = (
    "model_output",
    "score_output",
    "model_profiling",
    "visualization_output",
)


@dataclass(frozen=True)
class Move:
    stage: str
    model: str
    config: str
    dataset: str
    source: str
    target: str


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _safe_component(value: object) -> str | None:
    text = str(value or "").strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        return None
    return text


def _pair(data: dict) -> tuple[str, str] | None:
    model = _safe_component(data.get("model_name") or data.get("model"))
    config = _safe_component(data.get("config_name") or data.get("config"))
    return (model, config) if model and config else None


def discover_run_mapping(output_root: Path) -> dict[str, tuple[str, str]]:
    mapping: dict[str, tuple[str, str]] = {}
    conflicts: set[str] = set()
    for stage, marker_name in (
        ("model_output", "_meta.json"),
        ("model_profiling", "_meta.json"),
        ("score_output", "summary.json"),
    ):
        root = output_root / stage
        if not root.is_dir():
            continue
        for marker in root.glob(f"*/*/{marker_name}"):
            pair = _pair(_read_json(marker))
            if pair is None:
                continue
            run_name = marker.parent.parent.name
            previous = mapping.get(run_name)
            if previous is not None and previous != pair:
                conflicts.add(run_name)
            else:
                mapping[run_name] = pair
    for run_name in conflicts:
        mapping.pop(run_name, None)
        print(f"AMBIGUOUS run directory, skipped everywhere: {run_name}")
    return mapping


def _looks_like_legacy_dataset(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        return any(child.is_file() for child in path.iterdir())
    except OSError:
        return False


def plan_moves(
    output_root: Path,
    stages: tuple[str, ...],
) -> tuple[list[Move], list[str]]:
    mapping = discover_run_mapping(output_root)
    moves: list[Move] = []
    notes: list[str] = []
    for stage in stages:
        stage_root = output_root / stage
        if not stage_root.is_dir():
            continue
        for run_name, (model, config) in sorted(mapping.items()):
            run_root = stage_root / run_name
            if not run_root.is_dir():
                continue
            for dataset_dir in sorted(run_root.iterdir()):
                if not _looks_like_legacy_dataset(dataset_dir):
                    continue
                target = stage_root / model / config / dataset_dir.name
                moves.append(
                    Move(
                        stage=stage,
                        model=model,
                        config=config,
                        dataset=dataset_dir.name,
                        source=str(dataset_dir.resolve()),
                        target=str(target.resolve()),
                    )
                )
    if not mapping:
        notes.append("No legacy run metadata was found.")
    known_roots = set(mapping)
    for stage in stages:
        root = output_root / stage
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if not child.is_dir() or child.name in known_roots:
                continue
            try:
                looks_flat = any(
                    _looks_like_legacy_dataset(item) for item in child.iterdir()
                )
            except OSError:
                looks_flat = False
            if looks_flat:
                notes.append(
                    f"{stage}/{child.name}: no unambiguous model/config metadata; left unchanged"
                )
    return moves, notes


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def apply_moves(output_root: Path, moves: list[Move]) -> tuple[list[Move], list[str]]:
    completed: list[Move] = []
    skipped: list[str] = []
    for move in moves:
        source = Path(move.source)
        target = Path(move.target)
        if not _inside(source, output_root) or not _inside(target, output_root):
            skipped.append(f"outside output root: {source} -> {target}")
            continue
        if not source.exists():
            skipped.append(f"source missing: {source}")
            continue
        if target.exists():
            skipped.append(f"target exists, no merge attempted: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        completed.append(move)
        try:
            source.parent.rmdir()
        except OSError:
            pass
    return completed, skipped


def write_manifest(
    output_root: Path,
    completed: list[Move],
    skipped: list[str],
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = output_root / "_layout_migrations"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"migration_{timestamp}.json"
    payload = {
        "schema": "model-config-dataset-v1",
        "created_at_utc": timestamp,
        "moves": [asdict(move) for move in completed],
        "skipped": skipped,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def rollback(manifest_path: Path) -> int:
    data = _read_json(manifest_path)
    moves = [Move(**item) for item in data.get("moves", [])]
    if not moves:
        print("Manifest contains no completed moves.")
        return 1
    failed = 0
    for move in reversed(moves):
        source = Path(move.target)
        target = Path(move.source)
        if not source.exists():
            print(f"SKIP missing migrated directory: {source}")
            failed += 1
            continue
        if target.exists():
            print(f"SKIP original path already exists: {target}")
            failed += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        print(f"ROLLBACK {source} -> {target}")
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--stage", action="append", choices=STAGES)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--rollback", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rollback:
        return rollback(args.rollback.expanduser().resolve())
    output_root = Path(args.output_root).expanduser().resolve()
    stages = tuple(args.stage or STAGES)
    moves, notes = plan_moves(output_root, stages)
    for move in moves:
        print(f"{'MOVE' if args.apply else 'PLAN'} {move.source} -> {move.target}")
    for note in notes:
        print(f"NOTE {note}")
    if not args.apply:
        print(f"Dry run: {len(moves)} row(s) planned. Add --apply to migrate.")
        return 0
    completed, skipped = apply_moves(output_root, moves)
    manifest = write_manifest(output_root, completed, skipped)
    for note in skipped:
        print(f"SKIP {note}")
    print(f"Migrated {len(completed)}/{len(moves)} row(s). Manifest: {manifest}")
    return 1 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
