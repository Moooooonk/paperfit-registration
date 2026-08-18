from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "revision_tools"
sys.path.insert(0, str(TOOLS))
SCRIPT = TOOLS / "run_mm_s8_from_rigid.py"
SPEC = importlib.util.spec_from_file_location("run_mm_s8_from_rigid", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reference_s8_has_16_scheduled_passes() -> None:
    assert MODULE.expected_scheduled_passes(8) == 16


def test_sensitivity_stage_counts_include_same_eight_followup_updates() -> None:
    assert MODULE.expected_scheduled_passes(4) == 12
    assert MODULE.expected_scheduled_passes(12) == 20


def test_single_vertex_semantic_anchor_does_not_replace_tip_alar_pass() -> None:
    pilot = MODULE.s8

    class FakeNonrigid:
        @staticmethod
        def build_edges(_faces: np.ndarray) -> np.ndarray:
            return np.array([[0, 1]], dtype=np.int64)

        @staticmethod
        def depth_contours(
            vertices: np.ndarray,
            _masks: dict[str, np.ndarray],
            contours: tuple[float, ...],
        ) -> tuple[list[tuple[float, np.ndarray]], None]:
            active = np.ones(len(vertices), dtype=bool)
            return [(value, active.copy()) for value in contours], None

        @staticmethod
        def solve(
            current: np.ndarray,
            _edges: np.ndarray,
            _target_idx: np.ndarray,
            _target_pos: np.ndarray,
            _weights: np.ndarray,
            _fixed_idx: np.ndarray,
            _fixed_pos: np.ndarray,
            **_kwargs: float,
        ) -> np.ndarray:
            return current.copy()

    vertices = np.column_stack(
        [np.linspace(0.0, 31.0, 32), np.zeros(32), np.zeros(32)]
    )
    all_vertices = np.ones(32, dtype=bool)
    no_vertices = np.zeros(32, dtype=bool)
    tip = np.zeros(32, dtype=bool)
    tip[4:16] = True
    alar = np.zeros(32, dtype=bool)
    alar[16:28] = True
    source_anchor = np.zeros(32, dtype=bool)
    source_anchor[10] = True
    masks = {
        "eye": no_vertices.copy(),
        "eye_soft": no_vertices.copy(),
        "full_no_eye": all_vertices.copy(),
        "nasal_bridge": all_vertices.copy(),
        "nasal_dorsum": all_vertices.copy(),
        "nose_tip": tip,
        "alar": alar,
        "subnasal": all_vertices.copy(),
        "philtrum": all_vertices.copy(),
        "source_anchor": source_anchor,
    }
    variant = {
        "contours": pilot.S8_CONTOURS,
        "step_multiplier": 1.0,
        "gain_multiplier": 1.0,
        "anchor_weight": 0.0,
        "anchor_step_mm": 0.0,
    }
    _, history = pilot.run_variant(
        vertices,
        np.array([[0, 1, 2]], dtype=np.int64),
        vertices.copy(),
        masks,
        vertices[10].copy(),
        FakeNonrigid(),
        variant,
    )

    assert len(history) == 16
    assert [item["name"] for item in history if item["phase"] == "local"][-5:] == [
        "bridge",
        "dorsum",
        "tip_alar",
        "subnasal",
        "philtrum",
    ]


def test_empty_local_active_set_is_recorded_as_a_scheduled_noop() -> None:
    pilot = MODULE.s8
    solve_calls = 0

    class FakeNonrigid:
        @staticmethod
        def build_edges(_faces: np.ndarray) -> np.ndarray:
            return np.array([[0, 1]], dtype=np.int64)

        @staticmethod
        def depth_contours(
            vertices: np.ndarray,
            _masks: dict[str, np.ndarray],
            contours: tuple[float, ...],
        ) -> tuple[list[tuple[float, np.ndarray]], None]:
            active = np.ones(len(vertices), dtype=bool)
            return [(value, active.copy()) for value in contours], None

        @staticmethod
        def solve(
            current: np.ndarray,
            _edges: np.ndarray,
            _target_idx: np.ndarray,
            _target_pos: np.ndarray,
            _weights: np.ndarray,
            _fixed_idx: np.ndarray,
            _fixed_pos: np.ndarray,
            **_kwargs: float,
        ) -> np.ndarray:
            nonlocal solve_calls
            solve_calls += 1
            return current.copy()

    vertices = np.column_stack(
        [np.linspace(0.0, 31.0, 32), np.zeros(32), np.zeros(32)]
    )
    all_vertices = np.ones(32, dtype=bool)
    no_vertices = np.zeros(32, dtype=bool)
    source_anchor = np.zeros(32, dtype=bool)
    source_anchor[10] = True
    masks = {
        "eye": no_vertices.copy(),
        "eye_soft": no_vertices.copy(),
        "full_no_eye": all_vertices.copy(),
        "nasal_bridge": no_vertices.copy(),
        "nasal_dorsum": all_vertices.copy(),
        "nose_tip": all_vertices.copy(),
        "alar": no_vertices.copy(),
        "subnasal": all_vertices.copy(),
        "philtrum": all_vertices.copy(),
        "source_anchor": source_anchor,
    }
    variant = {
        "contours": pilot.S8_CONTOURS,
        "step_multiplier": 1.0,
        "gain_multiplier": 1.0,
        "anchor_weight": 0.0,
        "anchor_step_mm": 0.0,
    }
    _, history = pilot.run_variant(
        vertices,
        np.array([[0, 1, 2]], dtype=np.int64),
        vertices.copy(),
        masks,
        vertices[10].copy(),
        FakeNonrigid(),
        variant,
    )

    assert len(history) == 16
    skipped = [item for item in history if not item["executed_constrained_solve"]]
    assert len(skipped) == 1
    assert skipped[0]["name"] == "bridge"
    assert skipped[0]["active_vertices"] == 0
    assert solve_calls == 15
