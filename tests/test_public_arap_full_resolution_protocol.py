import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CASE_SCRIPT = ROOT / "revision_tools" / "run_arap_full_resolution_case.py"
RESULTS = ROOT / "results" / "aggregate"


def load_case_module():
    spec = importlib.util.spec_from_file_location("arap_full_resolution_case", CASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_arap_runner_is_full_resolution_only():
    source = CASE_SCRIPT.read_text(encoding="utf-8")
    assert "simplify_quadric_decimation" not in source
    assert "transfer_idw4" not in source
    assert "transfer_barycentric" not in source
    assert '"mesh_mode": "full"' in source
    assert '"eye_constraints": "all"' in source
    assert '"transfer": "direct"' in source
    assert 'checked_output(args.output_json, root, "JSON output")' in source
    assert 'checked_output(args.output_npz, root, "NPZ output")' in source


def test_arap_strain_is_zero_without_deformation():
    pytest.importorskip("open3d")
    module = load_case_module()
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)
    eye = np.asarray([True, False, False])
    summary = module.strain_regions(vertices, vertices.copy(), faces, eye)
    assert summary["all"]["max"] == 0.0
    assert summary["eye_boundary"]["max"] == 0.0
    assert summary["non_eye_non_eye"]["max"] == 0.0


def test_public_arap_aggregate_matches_reaudited_result():
    selection = json.loads(
        (RESULTS / "arap_full_resolution_development_selection.json").read_text(
            encoding="utf-8"
        )
    )
    assert selection["selected_setting_id"] == "FULL_C300_R1_S05"
    assert selection["selected_configuration"] == {
        "controls": 300,
        "rounds": 1,
        "step": 0.5,
    }

    with (RESULTS / "arap_full_resolution_paired_metrics.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = {row["metric"]: row for row in csv.DictReader(handle)}
    assert int(rows["post_full_median_mm"]["pairs"]) == 190
    assert int(rows["post_full_median_mm"]["subjects"]) == 10
    assert np.isclose(
        float(rows["post_full_median_mm"]["paired_mean_subject_difference"]),
        0.6652585435416342,
    )
    assert np.isclose(
        float(rows["post_nose_median_mm"]["paired_mean_subject_difference"]),
        1.5133204579733825,
    )
    assert np.isclose(
        float(rows["edge_strain_p99"]["paired_mean_subject_difference"]),
        -0.11636067632174185,
    )
