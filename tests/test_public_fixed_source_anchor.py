from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "revision_tools"
    / "run_pairwise_mm_rigid.py"
)
SPEC = importlib.util.spec_from_file_location("run_pairwise_mm_rigid", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
RIGID = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RIGID)


def native_source() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    count = 100
    vertices = np.zeros((count, 3), dtype=np.float64)
    vertices[:, 0] = np.linspace(-2.0, 2.0, count)
    vertices[:, 1] = np.linspace(100.0, 1.0, count)
    vertices[:, 2] = np.linspace(0.0, 99.0, count)
    masks = {
        "nose_tip": np.ones(count, dtype=bool),
        "alar": np.zeros(count, dtype=bool),
    }
    return vertices, masks


def test_fixed_hrn_source_anchor_uses_canonical_depth_axis() -> None:
    vertices, masks = native_source()
    anchor_mask = RIGID.fixed_source_anchor_mask(vertices, masks)
    expected = vertices[:, 2] >= np.quantile(vertices[:, 2], 0.80)

    np.testing.assert_array_equal(anchor_mask, expected)
    assert not bool(anchor_mask[0])
    assert bool(anchor_mask[-1])


def test_3ddfa_source_anchor_uses_official_nose_tip_landmark() -> None:
    landmarks = np.arange(68, dtype=np.int64) + 10
    observed = RIGID.fixed_3ddfa_source_anchor_mask(100, landmarks)

    assert int(observed.sum()) == 1
    assert bool(observed[40])


def test_source_anchor_is_similarity_transform_equivariant() -> None:
    vertices, masks = native_source()
    masks["source_anchor"] = RIGID.fixed_source_anchor_mask(vertices, masks)
    angle = np.deg2rad(37.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    scale = 2.75
    translation = np.asarray([13.0, -7.0, 4.0], dtype=np.float64)

    anchor_native = RIGID.source_anchor(vertices, masks)
    transformed = RIGID.transform(vertices, scale, rotation, translation)
    anchor_after = RIGID.source_anchor(transformed, masks)
    expected = RIGID.transform(
        anchor_native[None, :], scale, rotation, translation
    )[0]

    np.testing.assert_allclose(anchor_after, expected, rtol=0.0, atol=1e-12)
