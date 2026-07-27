"""Shared trace-construction helper for models that only expose raw per-step
canvas snapshots (a full token-id sequence per forward pass) rather than an
explicit "which positions changed and why" signal.

iLLaDA and DiffusionGemma both expose the richer signal directly from their
own reference samplers (``selected_positions``/confidences for iLLaDA,
``accepted_token_mask``/entropy for DG) and build :class:`TraceStep`
instances more precisely from that — see ``models/illada.py``/``dg.py``.
This module is the general fallback for a model whose public interface only
gives you snapshots (e.g. Dream's ``output_history``): diff consecutive
snapshots to recover ``committed_positions`` generically, without assuming
whether the model ever revises a position once committed.
"""

from __future__ import annotations

from ..interfaces import PositionState, TraceStep

MASK_DISPLAY = "▢"


def trace_steps_from_snapshots(
    snapshots: list[list[int]],
    mask_token_id: int,
    tokenizer,
) -> list[TraceStep]:
    """``snapshots[t]`` is the full generation-region token-id list after
    forward pass t (all snapshots must have the same length). A position is
    MASKED while its id equals ``mask_token_id``, ACCEPTED otherwise.
    ``committed_positions`` at step t is whatever changed since step t-1
    (covers both first-time unmasking and any later revision generically —
    this makes no assumption either way about whether the model revises).
    """
    if not snapshots:
        return []

    trace: list[TraceStep] = []
    previous: list[int] | None = None
    for step_index, snapshot in enumerate(snapshots):
        if previous is None:
            committed = [i for i, tok in enumerate(snapshot) if tok != mask_token_id]
        else:
            committed = [i for i, tok in enumerate(snapshot) if tok != previous[i]]

        position_states = [
            PositionState.MASKED if tok == mask_token_id else PositionState.ACCEPTED for tok in snapshot
        ]
        token_texts = [
            tokenizer.decode([tok]) if tok != mask_token_id else MASK_DISPLAY for tok in snapshot
        ]

        trace.append(
            TraceStep(
                forward_index=step_index,
                token_ids=list(snapshot),
                position_states=position_states,
                committed_positions=committed,
                decoded_text="".join(token_texts),
                token_texts=token_texts,
            )
        )
        previous = snapshot
    return trace
