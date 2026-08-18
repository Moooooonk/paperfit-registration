from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "revision_tools"
    / "run_pairwise_mm_rigid.py"
)
SPEC = importlib.util.spec_from_file_location("run_pairwise_mm_rigid", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
RIGID = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RIGID)


def test_bidirectional_overlap_penalizes_source_shrinkage() -> None:
    axis = np.linspace(-10.0, 10.0, 21)
    xx, zz = np.meshgrid(axis, axis)
    target = np.column_stack((xx.ravel(), np.zeros(xx.size), zz.ravel()))
    target_tree = cKDTree(target)
    center_index = int(np.argmin(np.linalg.norm(target, axis=1)))
    mouth_index = int(np.argmin(np.linalg.norm(target - [0.0, 0.0, -5.0], axis=1)))

    fit_weight = np.ones(len(target), dtype=np.float64)
    nose_mask = np.linalg.norm(target[:, [0, 2]], axis=1) <= 3.0
    masks = {
        "nose": nose_mask,
        "nose_tip": nose_mask,
        "alar": np.zeros(len(target), dtype=bool),
        "source_anchor": np.arange(len(target)) == center_index,
        "mouth_downweighted": np.arange(len(target)) == mouth_index,
    }
    target_anchor = target[center_index]

    matched = RIGID.geometric_metrics(
        target, target_tree, target, fit_weight, masks, target_anchor
    )
    shrunken = RIGID.geometric_metrics(
        target * 0.5, target_tree, target, fit_weight, masks, target_anchor
    )

    assert shrunken["target_coverage_median_mm"] > matched[
        "target_coverage_median_mm"
    ]
    assert shrunken["symmetric_full_median_mm"] > matched[
        "symmetric_full_median_mm"
    ]
    assert shrunken["selection_score_mm"] > matched["selection_score_mm"]
