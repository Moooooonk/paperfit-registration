from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_06_final_decision.py"
SPEC = spec_from_file_location("final_decision", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(case, full_before, full_after, nose_before, nose_after, nose_p90, eye=0.0):
    return {
        "case": case,
        "full_median_before": str(full_before),
        "full_median_after": str(full_after),
        "nose_median_before": str(nose_before),
        "nose_median_after": str(nose_after),
        "nose_p90_after": str(nose_p90),
        "eye_fixed_max": str(eye),
    }


def test_branch_specific_acceptance_rules():
    rigid = [row("001_a", 0.01, 0.009, 0.02, 0.01, 0.02)]
    anchor = [
        row("001_b", 0.02, 0.019, 0.03, 0.02, 0.04),
        row("001_c", 0.02, 0.021, 0.03, 0.02, 0.04),
    ]
    broad = [
        row("001_d", 0.04, 0.020, 0.05, 0.030, 0.060),
        row("001_e", 0.04, 0.020, 0.05, 0.030, 0.060),
    ]
    selection = [
        {"case": "001_d", "upside_down": "0"},
        {"case": "001_e", "upside_down": "1"},
    ]

    output = {item["case"]: item for item in MODULE.final_rows(rigid, anchor, broad, selection)}
    assert output["001_a"]["accepted"] == 1
    assert output["001_b"]["accepted"] == 1
    assert output["001_c"]["accepted"] == 0
    assert output["001_d"]["accepted"] == 1
    assert output["001_e"]["accepted"] == 0


def test_posthoc_facescape_metric_conversion_does_not_change_acceptance():
    rows = [
        {
            "case": "001_a",
            "accepted": 1,
            "full_median_after": 0.01,
            "nose_median_after": 0.02,
            "nose_p90_after": 0.03,
        }
    ]
    scales = {"1": {"18": [200.0, []]}}
    converted = MODULE.attach_metric_distances(rows, scales)
    assert converted[0]["accepted"] == 1
    assert converted[0]["full_median_after_mm"] == 2.0
    assert converted[0]["nose_median_after_mm"] == 4.0
    assert converted[0]["nose_p90_after_mm"] == 6.0
