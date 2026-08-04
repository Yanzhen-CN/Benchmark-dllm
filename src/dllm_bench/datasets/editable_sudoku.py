from __future__ import annotations

from dataclasses import replace

from .sudoku4 import Sudoku4Dataset, valid_sudoku4_solutions
from .sudoku9 import Sudoku9Dataset
from ..metrics.sudoku_editing import compute_sudoku_editing_metrics


def _digits(text: str) -> str:
    return "".join(char for char in str(text) if char.isdigit())


def _controlled_seed(puzzle: str, error_count: int) -> tuple[str, list[int], list[str]]:
    puzzle = _digits(puzzle)
    solutions = valid_sudoku4_solutions(puzzle)
    accepted = [{solution[index] for solution in solutions} for index in range(16)]
    clues = [(index, value) for index, value in enumerate(puzzle) if value != "0"]
    blanks = [index for index, value in enumerate(puzzle) if value == "0"]
    seed, targets, labels = list(puzzle), [], []

    def kinds(cell: int, other: int) -> set[str]:
        row, col = divmod(cell, 4)
        other_row, other_col = divmod(other, 4)
        result = set()
        if row == other_row: result.add("row")
        if col == other_col: result.add("column")
        if row // 2 == other_row // 2 and col // 2 == other_col // 2: result.add("box")
        return result

    for wanted in ("row", "column", "box", "latent"):
        if len(targets) >= error_count: break
        for cell in blanks:
            if cell in targets: continue
            if wanted == "latent":
                conflicting = {value for other, value in clues if kinds(cell, other)}
                values = [str(value) for value in range(1, 5)
                          if str(value) not in accepted[cell] and str(value) not in conflicting]
            else:
                values = [value for other, value in clues
                          if wanted in kinds(cell, other) and value not in accepted[cell]]
            if values:
                seed[cell] = sorted(values)[0]
                targets.append(cell)
                labels.append(wanted)
                break
    for cell in blanks:
        if len(targets) >= error_count: break
        if cell in targets: continue
        values = [str(value) for value in range(1, 5) if str(value) not in accepted[cell]]
        if values:
            seed[cell], targets, labels = values[0], targets + [cell], labels + ["fallback"]
    return "".join(seed), targets, labels


class EditableSudoku4Dataset(Sudoku4Dataset):
    name = "editable_sudoku4"

    def load_samples(self, n_samples: int | None = None):
        expanded = []
        for sample in super().load_samples(None):
            puzzle = _digits(sample.reference.puzzle)
            solution = _digits(sample.reference.solution)
            clues = [index for index, value in enumerate(puzzle) if value != "0"]
            meta = dict(sample.meta)
            meta["editable_sudoku"] = {"track": "natural", "answer_cells": 16,
                "puzzle": puzzle, "solution": solution, "immutable_cells": [],
                "target_error_cells": [], "error_count": 0, "error_types": []}
            expanded.append(replace(sample, sample_id=f"{sample.sample_id}-natural", meta=meta))
            for count in (1, 2, 4):
                grid, targets, labels = _controlled_seed(puzzle, count)
                meta = dict(sample.meta)
                meta["editable_sudoku"] = {"track": "controlled_repair", "answer_cells": 16,
                    "puzzle": puzzle, "solution": solution, "seeded_grid": grid,
                    "immutable_cells": clues, "target_error_cells": targets,
                    "error_count": len(targets), "error_types": labels}
                expanded.append(replace(sample, sample_id=f"{sample.sample_id}-repair{count}", meta=meta))
        return expanded if n_samples is None else expanded[:n_samples]

    def score_generation(self, sample, generation):
        score = super().score_generation(sample, generation)
        score.aux.update(compute_sudoku_editing_metrics(sample, generation, size=4))
        return score


class Llada21Sudoku9Dataset(Sudoku9Dataset):
    name = "llada2_1_sudoku9"

    def load_samples(self, n_samples: int | None = None):
        result = []
        for sample in super().load_samples(n_samples):
            puzzle = _digits(sample.reference.puzzle)
            meta = dict(sample.meta)
            meta["editable_sudoku"] = {"track": "natural", "answer_cells": 81,
                "puzzle": puzzle, "solution": _digits(sample.reference.solution),
                "immutable_cells": [],
                "target_error_cells": [], "error_count": 0, "error_types": []}
            result.append(replace(sample, meta=meta))
        return result
