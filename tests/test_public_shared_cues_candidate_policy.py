from __future__ import annotations

from revision_tools.shared_cues_candidate_policy import select_candidate


def state(index: int, score: float, anchor: float, upside_down: int = 0):
    return {
        "candidate_index": index,
        "selection_score_mm": score,
        "nose_anchor_point_mm": anchor,
        "upside_down": upside_down,
    }


def test_prefers_best_score_inside_anchor_cap() -> None:
    selected, eligible, fallback = select_candidate(
        [state(1, 1.0, 30.0), state(2, 4.0, 8.0), state(3, 3.0, 9.0)],
        10.0,
    )
    assert selected["candidate_index"] == 3
    assert eligible == 2
    assert fallback is False


def test_falls_back_to_surface_score_when_no_anchor_candidate_is_eligible() -> None:
    selected, eligible, fallback = select_candidate(
        [state(1, 2.0, 30.0), state(2, 1.0, 20.0)],
        10.0,
    )
    assert selected["candidate_index"] == 2
    assert eligible == 0
    assert fallback is True


def test_excludes_upside_down_candidate_when_upright_exists() -> None:
    selected, _, _ = select_candidate(
        [state(1, 0.1, 1.0, upside_down=1), state(2, 2.0, 5.0)],
        10.0,
    )
    assert selected["candidate_index"] == 2
