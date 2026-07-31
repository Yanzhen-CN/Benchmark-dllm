"""Migrate legacy best/fast run names to explicit P1/P2 names.

The benchmark now names fixed-block operating points by planned parallelism:
``p1`` means one scheduled token per denoising step and ``p2`` means two.
This utility migrates already-generated artifacts without touching model
outputs or traces.  It is a dry run unless ``--apply`` is supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Migration:
    model: str
    old_variant: str
    new_variant: str

    @property
    def old_run_id(self) -> str:
        return f"{self.model}_{self.old_variant}"

    @property
    def new_run_id(self) -> str:
        return f"{self.model}_{self.new_variant}"


MIGRATIONS = (
    Migration("illada", "best", "p1"),
    Migration("illada", "fast", "p2"),
    Migration("illada_vargen", "best", "p1"),
    Migration("illada_vargen", "fast", "p2"),
    Migration("dreamreasoner", "best", "p1"),
    Migration("dreamreasoner", "fast", "p2"),
)

VARIANT_FIELDS = {"config", "config_name", "variant"}
MODEL_FIELDS = {"model", "model_name"}
STRUCTURED_JSON_NAMES = {"_meta.json", "summary.json", "oom_info.json"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
    description="Rename legacy iLLaDA/Dream/VarGen best/fast artifacts to P1/P2 safely."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Benchmark output root (default: output).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the migration. Without this flag only a preview is printed.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse P1/P2 back to historical best/fast (requires --apply).",
    )
    return parser.parse_args(argv)


def selected_migrations(reverse: bool) -> tuple[Migration, ...]:
    if not reverse:
        return MIGRATIONS
    return tuple(
        Migration(item.model, item.new_variant, item.old_variant)
        for item in MIGRATIONS
    )


def _composite_replacements(value: str, migrations: Iterable[Migration]) -> str:
    result = value
    for item in migrations:
        replacements = (
            (item.old_run_id, item.new_run_id),
            (f"{item.model}/{item.old_variant}", f"{item.model}/{item.new_variant}"),
            (f"{item.model}-{item.old_variant}", f"{item.model}-{item.new_variant}"),
        )
        for old, new in replacements:
            result = result.replace(old, new)
    return result


def _migration_for_model_variant(
    model: str | None,
    variant: str | None,
    migrations: Iterable[Migration],
) -> Migration | None:
    if model is None or variant is None:
        return None
    normalized_model = model.lower()
    normalized_variant = variant.lower()
    return next(
        (
            item
            for item in migrations
            if item.model == normalized_model and item.old_variant == normalized_variant
        ),
        None,
    )


def migrate_json_value(
    value: Any,
    migrations: tuple[Migration, ...],
    *,
    inherited_model: str | None = None,
    forced: Migration | None = None,
) -> tuple[Any, bool]:
    """Return a migrated JSON-compatible value and whether it changed."""

    if isinstance(value, dict):
        local_model = inherited_model
        for field in MODEL_FIELDS:
            candidate = value.get(field)
            if isinstance(candidate, str):
                local_model = candidate.lower()
                break
        local_variant = next(
            (
                value.get(field)
                for field in VARIANT_FIELDS
                if isinstance(value.get(field), str)
            ),
            None,
        )
        active = forced or _migration_for_model_variant(local_model, local_variant, migrations)
        changed = False
        migrated: dict[str, Any] = {}
        for key, item_value in value.items():
            if (
                active is not None
                and key in VARIANT_FIELDS
                and isinstance(item_value, str)
                and item_value == active.old_variant
            ):
                migrated[key] = active.new_variant
                changed = True
                continue
            new_value, item_changed = migrate_json_value(
                item_value,
                migrations,
                inherited_model=local_model,
                forced=active,
            )
            migrated[key] = new_value
            changed = changed or item_changed
        return migrated, changed

    if isinstance(value, list):
        changed = False
        migrated_items = []
        for item in value:
            new_item, item_changed = migrate_json_value(
                item,
                migrations,
                inherited_model=inherited_model,
                forced=forced,
            )
            migrated_items.append(new_item)
            changed = changed or item_changed
        return migrated_items, changed

    if isinstance(value, str):
        migrated = _composite_replacements(value, migrations)
        return migrated, migrated != value

    return value, False


def _forced_migration(path: Path, migrations: Iterable[Migration]) -> Migration | None:
    parts = set(path.parts)
    return next(
        (
            item
            for item in migrations
            if item.old_run_id in parts or item.new_run_id in parts
        ),
        None,
    )


def _json_candidates(output_root: Path) -> list[Path]:
    candidates: set[Path] = set()
    for name in STRUCTURED_JSON_NAMES:
        candidates.update(output_root.rglob(name))
    return sorted(candidates)


def _tabular_candidates(output_root: Path) -> list[Path]:
    # Aggregate tables are small; traces and per-sample JSONL are deliberately
    # excluded because their payload does not own model/config identity.
    return sorted(path for path in output_root.rglob("*.csv") if path.stat().st_size <= 20_000_000)


def _preview_json_changes(path: Path, migrations: tuple[Migration, ...]) -> tuple[Any, bool]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return migrate_json_value(payload, migrations, forced=_forced_migration(path, migrations))


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.p-variant.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _migrate_csv(path: Path, migrations: tuple[Migration, ...], apply: bool) -> bool:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return False
        rows = list(reader)

    changed = False
    field_lookup = {field.lower(): field for field in reader.fieldnames}
    model_field = next((field_lookup[name] for name in MODEL_FIELDS if name in field_lookup), None)
    variant_field = next((field_lookup[name] for name in VARIANT_FIELDS if name in field_lookup), None)
    for row in rows:
        active = _migration_for_model_variant(
            row.get(model_field) if model_field else None,
            row.get(variant_field) if variant_field else None,
            migrations,
        )
        if active and variant_field and row.get(variant_field) == active.old_variant:
            row[variant_field] = active.new_variant
            changed = True
        for field, value in row.items():
            if value is None:
                continue
            migrated = _composite_replacements(value, migrations)
            if migrated != value:
                row[field] = migrated
                changed = True

    if changed and apply:
        temporary = path.with_name(f".{path.name}.p-variant.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    return changed


def _directory_moves(output_root: Path, migrations: tuple[Migration, ...]) -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    for path in output_root.rglob("*"):
        if not path.is_dir():
            continue
        item = next((entry for entry in migrations if path.name == entry.old_run_id), None)
        if item is not None:
            moves.append((path, path.with_name(item.new_run_id)))
    return sorted(moves, key=lambda pair: len(pair[0].parts), reverse=True)


def migrate(output_root: Path, *, apply: bool, reverse: bool = False) -> int:
    output_root = output_root.resolve()
    if not output_root.is_dir():
        raise SystemExit(f"output root does not exist: {output_root}")

    migrations = selected_migrations(reverse)
    moves = _directory_moves(output_root, migrations)
    conflicts = [(source, target) for source, target in moves if target.exists()]
    if conflicts:
        details = "\n".join(f"- {source} -> {target}" for source, target in conflicts)
        raise SystemExit(f"refusing to overwrite existing migration target(s):\n{details}")

    json_changes: list[tuple[Path, Any]] = []
    for path in _json_candidates(output_root):
        try:
            payload, changed = _preview_json_changes(path, migrations)
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot safely migrate {path}: {exc}") from exc
        if changed:
            json_changes.append((path, payload))

    csv_changes = [
        path for path in _tabular_candidates(output_root) if _migrate_csv(path, migrations, apply=False)
    ]

    mode = "APPLY" if apply else "DRY-RUN"
    print(f"[{mode}] output root: {output_root}")
    for source, target in moves:
        print(f"[directory] {source} -> {target}")
    for path, _ in json_changes:
        print(f"[metadata]  {path}")
    for path in csv_changes:
        print(f"[table]     {path}")
    print(
        f"Planned: {len(moves)} directorie(s), "
        f"{len(json_changes)} JSON metadata file(s), {len(csv_changes)} table(s)."
    )

    if not apply:
        print("No files changed. Re-run with --apply after reviewing this preview.")
        return 0

    for path, payload in json_changes:
        _write_json(path, payload)
    for path in csv_changes:
        _migrate_csv(path, migrations, apply=True)
    for source, target in moves:
        source.rename(target)

    manifest = output_root / "parallelism_name_migration.json"
    manifest.write_text(
        json.dumps(
            {
                "direction": "p-to-legacy" if reverse else "legacy-to-p",
                "directories": [
                    {"from": str(source), "to": str(target)} for source, target in moves
                ],
                "json_metadata_files": [str(path) for path, _ in json_changes],
                "tables": [str(path) for path in csv_changes],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Migration complete. Manifest: {manifest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return migrate(args.output_root, apply=args.apply, reverse=args.reverse)


if __name__ == "__main__":
    raise SystemExit(main())
