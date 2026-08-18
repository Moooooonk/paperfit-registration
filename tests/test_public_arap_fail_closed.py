from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REVISION_TOOLS = Path(__file__).resolve().parents[1] / "revision_tools"
sys.path.insert(0, str(REVISION_TOOLS))

open3d = pytest.importorskip("open3d")

import run_arap_baseline_from_rigid as arap  # noqa: E402


def test_failed_arap_row_is_explicit_and_counted(tmp_path: Path) -> None:
    row = arap.failed_arap_row(
        "901_01_synthetic",
        "hrn",
        {
            "pre_s8_branch": "broad_failure",
            "routing_failure_reasons": "nasal",
        },
        "arap_execution_failure",
        RuntimeError("solver failed"),
    )
    assert row["completed"] == 0
    assert row["execution_exception_type"] == "RuntimeError"
    arap.write_summary(tmp_path, [row], {"setting_id": "ARAP-A"})
    summary = json.loads(
        (tmp_path / "arap_baseline_summary.json").read_text(encoding="utf-8")
    )
    assert summary["attempted_cases"] == 1
    assert summary["completed_cases"] == 0
    assert summary["unprocessed_invalid_evidence_cases"] == 0
    assert summary["arap_execution_failure_cases"] == 1
