"""Outcome-blind candidate selection shared by the strong Open3D baseline."""

from __future__ import annotations

from typing import Any

import numpy as np


def select_candidate(
    states: list[dict[str, Any]], anchor_cap_mm: float
) -> tuple[dict[str, Any], int, bool]:
    if not states:
        raise ValueError("No candidate states")
    if not np.isfinite(anchor_cap_mm) or anchor_cap_mm <= 0.0:
        raise ValueError("anchor_cap_mm must be positive and finite")
    upright = [state for state in states if not bool(state["upside_down"])]
    pool = upright if upright else states
    eligible = [
        state
        for state in pool
        if float(state["nose_anchor_point_mm"]) <= anchor_cap_mm
    ]
    fallback = not bool(eligible)
    if eligible:
        pool = eligible
    selected = min(
        pool,
        key=lambda state: (
            float(state["selection_score_mm"]),
            float(state["nose_anchor_point_mm"]),
            int(state["candidate_index"]),
        ),
    )
    return selected, len(eligible), fallback
