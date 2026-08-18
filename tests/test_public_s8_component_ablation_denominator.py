from __future__ import annotations

import json
import sys
from pathlib import Path


REVISION_TOOLS = Path(__file__).resolve().parents[1] / "revision_tools"
sys.path.insert(0, str(REVISION_TOOLS))

import run_s8_component_ablation as ablation  # noqa: E402


def test_failure_rows_preserve_all_four_conditions() -> None:
    rows = ablation.failure_rows(
        "901_01_synthetic",
        "hrn",
        "broad_failure",
        "s8_ablation_execution_failure",
        FloatingPointError("non-finite output"),
    )
    assert len(rows) == len(ablation.CONDITIONS) == 4
    assert {row["condition"] for row in rows} == {
        condition["condition"] for condition in ablation.CONDITIONS
    }
    assert all(row["completed"] == 0 for row in rows)
    assert all(
        row["execution_exception_type"] == "FloatingPointError" for row in rows
    )


def test_condition_failure_does_not_replace_other_conditions() -> None:
    condition = ablation.CONDITIONS[2]
    row = ablation.condition_failure_row(
        "901_01_synthetic",
        "hrn",
        "broad_failure",
        condition,
        "s8_ablation_condition_failure",
        RuntimeError("condition-specific failure"),
    )
    assert row["condition"] == "no_nasal_depth_contours"
    assert row["completed"] == 0
    assert row["execution_exception_type"] == "RuntimeError"


def test_empty_completed_condition_has_null_metric_summary(tmp_path: Path) -> None:
    rows = ablation.failure_rows(
        "901_01_synthetic",
        "hrn",
        "residual_invalid_evidence",
        "invalid_pre_s8_evidence_not_processed",
    )
    ablation.write_summary(tmp_path, rows, {"case_count": 1})
    summary = json.loads(
        (tmp_path / "s8_component_ablation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["recorded_rows"] == 4
    assert summary["completed_rows"] == 0
    assert summary["invalid_pre_s8_evidence_rows"] == 4
    assert summary["s8_execution_failure_rows"] == 0
    assert all(
        item["mean_full_median_mm"] is None
        for item in summary["conditions"].values()
    )
