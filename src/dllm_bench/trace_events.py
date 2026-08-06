"""Pure trace-event extraction shared by previews and publication figures."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _value(item: Any, name: str, default: Any) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _is_accepted(state: object) -> bool:
    value = getattr(state, "value", state)
    return str(value).lower().rsplit(".", 1)[-1] == "accepted"


def extract_acceptance_events(trace: Sequence[Any]) -> dict[str, Any]:
    last_accepted_token: dict[int, int] = {}
    accept_count_by_position: dict[int, int] = {}
    accept_positions: list[int] = []
    accept_steps: list[int] = []
    accept_ranks: list[int] = []
    renoise_positions: list[int] = []
    renoise_steps: list[int] = []
    reaccept_positions: list[int] = []
    revision_positions: list[int] = []
    revision_accept_indices: list[int] = []
    previous_accepted: set[int] = set()

    for step in trace:
        states = list(_value(step, "position_states", []) or [])
        token_ids = list(_value(step, "token_ids", []) or [])
        forward_index = int(_value(step, "forward_index", 0))
        current_accepted = {
            position
            for position, state in enumerate(states)
            if _is_accepted(state)
        }
        for position in sorted(previous_accepted - current_accepted):
            renoise_positions.append(position)
            renoise_steps.append(forward_index)
        explicit_commits = {
            int(position)
            for position in (_value(step, "committed_positions", []) or [])
        }
        newly_accepted = current_accepted - previous_accepted
        changed_while_accepted = {
            position
            for position in current_accepted & previous_accepted
            if position < len(token_ids)
            and position in last_accepted_token
            and int(token_ids[position]) != last_accepted_token[position]
        }
        event_candidates = (
            explicit_commits | newly_accepted | changed_while_accepted
        ) & current_accepted
        for position in sorted(event_candidates):
            if position < 0 or position >= len(token_ids):
                continue
            token_id = int(token_ids[position])
            if token_id < 0:
                continue
            previous_token = last_accepted_token.get(position)
            is_accept_event = (
                previous_token is None
                or token_id != previous_token
                or position not in previous_accepted
            )
            if is_accept_event:
                accept_rank = accept_count_by_position.get(position, 0) + 1
                accept_count_by_position[position] = accept_rank
                accept_positions.append(position)
                accept_steps.append(forward_index)
                accept_ranks.append(accept_rank)
            if previous_token is None:
                pass
            elif token_id != previous_token:
                revision_positions.append(position)
                revision_accept_indices.append(len(accept_positions) - 1)
            elif position not in previous_accepted:
                reaccept_positions.append(position)
            last_accepted_token[position] = token_id
        previous_accepted = current_accepted

    return {
        "accept_positions": accept_positions,
        "accept_steps": accept_steps,
        "accept_ranks": accept_ranks,
        "renoise_positions": renoise_positions,
        "renoise_steps": renoise_steps,
        "reaccept_positions": reaccept_positions,
        "revision_positions": revision_positions,
        "revision_accept_indices": revision_accept_indices,
        "accept_count_by_position": accept_count_by_position,
    }
