from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "revision_tools"
    / "run_pairwise_mm_rigid.py"
)
SPEC = importlib.util.spec_from_file_location("revision_rigid_scale", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_interocular_scale_reproduces_target_anatomical_length() -> None:
    source_interocular = 0.731
    target_interocular_mm = 72.87
    scale = MODULE.anatomical_scale_from_interocular(
        source_interocular, target_interocular_mm
    )

    assert scale * source_interocular == pytest.approx(target_interocular_mm)
    assert np.array_equal(MODULE.SCALE_RATIOS, np.asarray([1.0]))


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_interocular_scale_rejects_invalid_lengths(value: float) -> None:
    with pytest.raises(ValueError):
        MODULE.anatomical_scale_from_interocular(value, 72.87)
    with pytest.raises(ValueError):
        MODULE.anatomical_scale_from_interocular(0.731, value)
