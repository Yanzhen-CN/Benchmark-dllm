from __future__ import annotations

from statistics import mean


def _digit(text: str, size: int) -> str | None:
    value = str(text).strip()
    return value if len(value) == 1 and value in {str(number) for number in range(1, size + 1)} else None


def _conflicts(grid: dict[int, str], size: int) -> int:
    box = int(size**0.5)
    groups = [[row * size + col for col in range(size)] for row in range(size)]
    groups += [[row * size + col for row in range(size)] for col in range(size)]
    groups += [[(br + dr) * size + bc + dc for dr in range(box) for dc in range(box)]
               for br in range(0, size, box) for bc in range(0, size, box)]
    conflicts = 0
    for group in groups:
        values = [grid[cell] for cell in group if cell in grid]
        conflicts += len(values) - len(set(values))
    return conflicts


def compute_sudoku_editing_metrics(sample, generation, size: int = 4) -> dict[str, object]:
    trace = list(getattr(generation, "editing_trace", []) or [])
    if not trace:
        return {"editing_trace_available": 0.0, "editing_metric_status": "N/A: no editing trace"}
    spec = dict(sample.meta.get("editable_sudoku") or {})
    puzzle = "".join(char for char in str(spec.get("puzzle", "")) if char.isdigit())
    solution = "".join(char for char in str(spec.get("solution", "")) if char.isdigit())
    accepted = [{solution[index]} if index < len(solution) else set() for index in range(size * size)]
    if size == 4 and puzzle:
        from ..datasets.sudoku4 import valid_sudoku4_solutions
        valid = valid_sudoku4_solutions(puzzle)
        if valid: accepted = [{candidate[index] for candidate in valid} for index in range(16)]

    target_cells = {int(value) for value in spec.get("target_error_cells", [])}
    grid, opportunities, corrected_at = {}, {}, {}
    first_provisional, mapped_cells = set(), set()
    wrong_first = replacements = corrections = harmful = collateral = conflict_reduction = 0
    phase_corrections = {"mask_filling": 0, "post_edit": 0}
    for step in trace:
        mapping = {int(key): int(value) for key, value in step.position_to_cell_map.items()}
        mapped_cells.update(mapping.values())
        for local, cell in mapping.items():
            old = _digit(step.old_block_token_texts[local], size)
            if old is not None:
                grid[cell] = old
                if cell not in opportunities and old not in accepted[cell] and local in step.editable_positions:
                    opportunities[cell] = step.forward_index
        before = _conflicts(grid, size)
        for local in step.mask_transfer_positions:
            if local not in mapping: continue
            cell, value = mapping[local], _digit(step.new_block_token_texts[local], size)
            if value is None: continue
            if cell not in first_provisional:
                first_provisional.add(cell)
                if value not in accepted[cell]:
                    wrong_first += 1
                    opportunities.setdefault(cell, step.forward_index)
            grid[cell] = value
        for local in step.editing_transfer_positions:
            if local not in mapping: continue
            cell = mapping[local]
            old, new = _digit(step.old_block_token_texts[local], size), _digit(step.new_block_token_texts[local], size)
            if old is None or new is None or old == new: continue
            replacements += 1
            if old not in accepted[cell] and new in accepted[cell]:
                corrections += 1
                corrected_at.setdefault(cell, step.forward_index)
                phase_corrections[step.phase] = phase_corrections.get(step.phase, 0) + 1
            if old in accepted[cell] and new not in accepted[cell]:
                harmful += 1
                if cell not in target_cells: collateral += 1
            grid[cell] = new
        conflict_reduction += before - _conflicts(grid, size)
    latencies = [corrected_at[cell] - start for cell, start in opportunities.items() if cell in corrected_at]
    result: dict[str, object] = {
        "editing_trace_available": 1.0, "editing_mapping_coverage": len(mapped_cells) / float(size * size),
        "editing_opportunities": float(len(opportunities)), "editing_replacements": float(replacements),
        "editing_corrections": float(corrections), "editing_harmful_replacements": float(harmful),
        "editing_collateral_damage": float(collateral), "editing_wrong_first_provisional": float(wrong_first),
        "editing_first_provisional_count": float(len(first_provisional)),
        "editing_constraint_conflict_reduction": float(conflict_reduction),
        "editing_remaining_errors": float(sum(1 for cell, values in enumerate(accepted)
                                               if cell in grid and grid[cell] not in values)),
        "editing_mask_phase_corrections": float(phase_corrections.get("mask_filling", 0)),
        "editing_post_phase_corrections": float(phase_corrections.get("post_edit", 0)),
    }
    if opportunities: result["editing_correction_rate"] = corrections / float(len(opportunities))
    else: result["editing_correction_rate_status"] = "N/A: no wrong editable token appeared"
    if replacements: result["editing_harmful_rate"] = harmful / float(replacements)
    else: result["editing_harmful_rate_status"] = "N/A: no T2T replacement occurred"
    if first_provisional: result["editing_wrong_first_rate"] = wrong_first / float(len(first_provisional))
    if latencies: result["editing_repair_latency_forwards"] = mean(latencies)
    else: result["editing_repair_latency_status"] = "N/A: no observed opportunity was repaired"
    return result

