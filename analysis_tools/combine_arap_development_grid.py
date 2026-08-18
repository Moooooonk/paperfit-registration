#!/usr/bin/env python3
"""Validate and combine collected ARAP development settings for common QC."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SETTING_ORDER = ("ARAP-A", "ARAP-B", "ARAP-C", "ARAP-D")
SOURCE_EXPRESSION_INDICES = frozenset((*range(1, 18), 19, 20))
METRIC_FIELDS = (
    "post_orientation_pass",
    "eye_fixed_max_mm",
    "post_full_median_mm",
    "post_full_p90_mm",
    "post_nose_median_mm",
    "post_nose_p90_mm",
    "post_anchor_mm",
    "edge_strain_p99",
)
SUMMARY_TO_ROW = {
    "control_count": "setting_control_count",
    "rounds": "setting_rounds",
    "max_step_mm": "setting_max_step_mm",
    "arap_iterations": "setting_arap_iterations",
    "decimated_triangles": "setting_decimated_triangles",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def case_components(case: str) -> tuple[str, int, str]:
    try:
        subject_text, expression = case.split("_", 1)
        expression_index_text, expression_name = expression.split("_", 1)
        subject = f"{int(subject_text):03d}"
        expression_index = int(expression_index_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid ARAP case identifier: {case!r}") from exc
    if not expression_name:
        raise ValueError(f"Missing expression name in ARAP case: {case}")
    return subject, expression_index, expression_name


def validate_rows(
    rows: list[dict[str, str]], development_subjects: set[str], setting_id: str
) -> set[str]:
    if not rows:
        raise ValueError(f"{setting_id} contains no rows")
    required = {
        "case",
        "subject",
        "source_method",
        "completed",
        "execution_failure_reason",
        *METRIC_FIELDS,
    }
    if not required.issubset(rows[0]):
        raise KeyError(f"{setting_id} missing fields: {sorted(required - set(rows[0]))}")
    cases: set[str] = set()
    by_subject: dict[str, set[int]] = {subject: set() for subject in development_subjects}
    for row in rows:
        case = str(row["case"])
        if case in cases:
            raise ValueError(f"Duplicate {setting_id} case: {case}")
        cases.add(case)
        subject, expression_index, expression_name = case_components(case)
        row_subject = f"{int(row['subject']):03d}"
        if row_subject != subject or subject not in development_subjects:
            raise ValueError(f"{setting_id} case/subject mismatch: {case}")
        if expression_index not in SOURCE_EXPRESSION_INDICES:
            raise ValueError(f"{setting_id} has invalid source expression: {case}")
        if expression_index in by_subject[subject]:
            raise ValueError(f"{setting_id} repeats expression {expression_index} for {subject}")
        by_subject[subject].add(expression_index)
        if str(row.get("expression_index", expression_index)) not in {
            "",
            str(expression_index),
        }:
            raise ValueError(f"{setting_id} expression-index mismatch: {case}")
        if str(row.get("expression_name", expression_name)) not in {
            "",
            expression_name,
        }:
            raise ValueError(f"{setting_id} expression-name mismatch: {case}")
        if str(row["source_method"]).strip().lower() != "hrn":
            raise ValueError(f"{setting_id} must contain HRN rows only")
        try:
            completed_value = float(row["completed"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Non-binary completed value in {setting_id}: {case}") from exc
        if completed_value not in (0.0, 1.0):
            raise ValueError(f"Non-binary completed value in {setting_id}: {case}")
        completed = int(completed_value)
        if completed:
            for field in METRIC_FIELDS:
                try:
                    value = float(row[field])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Completed {setting_id} case {case} lacks {field}"
                    ) from exc
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        f"Completed {setting_id} case {case} has invalid {field}"
                    )
            if float(row["post_orientation_pass"]) not in (0.0, 1.0):
                raise ValueError(f"Non-binary orientation value in {setting_id}: {case}")
        elif row["execution_failure_reason"] != "arap_execution_failure":
            raise ValueError(
                f"Incomplete {setting_id} case lacks explicit ARAP failure: {case}"
            )
    expected_cases = len(development_subjects) * len(SOURCE_EXPRESSION_INDICES)
    if len(rows) != expected_cases:
        raise ValueError(
            f"{setting_id} expected {expected_cases} rows, found {len(rows)}"
        )
    for subject, expressions in by_subject.items():
        if expressions != SOURCE_EXPRESSION_INDICES:
            raise ValueError(
                f"{setting_id} subject {subject} has an incomplete expression grid"
            )
    return cases


def combine_collection(
    collection_dir: Path, split_json: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    collection_dir = collection_dir.resolve(strict=True)
    manifest_path = collection_dir / "collection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "arap":
        raise ValueError("Collection manifest is not ARAP evidence")
    if not manifest.get("development_only") or manifest.get("heldout_inspected"):
        raise ValueError("ARAP collection must be uninspected development evidence")
    split_json = split_json.resolve(strict=True)
    split = json.loads(split_json.read_text(encoding="utf-8"))
    development_subjects = {
        f"{int(value):03d}" for value in split["development_subjects"]
    }
    if manifest.get("split_sha256") != sha256(split_json):
        raise ValueError("Collection and supplied split hashes differ")
    jobs = {str(job["job_identity"]): job for job in manifest.get("jobs", [])}
    if set(jobs) != set(SETTING_ORDER):
        raise ValueError("Collection must contain exactly ARAP-A/B/C/D")

    combined: list[dict[str, Any]] = []
    reference_cases: set[str] | None = None
    settings_summary: list[dict[str, Any]] = []
    for setting_id in SETTING_ORDER:
        matches = [path for path in collection_dir.iterdir() if path.is_dir() and path.name.endswith(f"_{setting_id}")]
        if len(matches) != 1:
            raise ValueError(f"Expected one collected directory for {setting_id}")
        setting_dir = matches[0]
        rows_path = setting_dir / "arap_baseline_rows.csv"
        summary_path = setting_dir / "arap_baseline_summary.json"
        job = jobs[setting_id]
        if sha256(rows_path) != job["rows_sha256"]:
            raise ValueError(f"{setting_id} row hash differs from collection manifest")
        if sha256(summary_path) != job["summary_sha256"]:
            raise ValueError(f"{setting_id} summary hash differs from collection manifest")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = read_csv(rows_path)
        cases = validate_rows(rows, development_subjects, setting_id)
        if reference_cases is None:
            reference_cases = cases
        elif cases != reference_cases:
            raise ValueError(f"{setting_id} case grid differs from the other settings")
        if int(summary.get("attempted_cases", -1)) != len(rows):
            raise ValueError(f"{setting_id} summary attempted denominator differs")
        completed = sum(int(float(row["completed"])) for row in rows)
        if int(summary.get("completed_cases", -1)) != completed:
            raise ValueError(f"{setting_id} summary completed denominator differs")
        if int(summary.get("arap_execution_failure_cases", -1)) != len(rows) - completed:
            raise ValueError(f"{setting_id} summary failure denominator differs")
        setting_values = {
            output_name: summary[input_name]
            for input_name, output_name in SUMMARY_TO_ROW.items()
        }
        for row in rows:
            combined.append(
                {
                    "setting_id": setting_id,
                    **setting_values,
                    **row,
                }
            )
        settings_summary.append(
            {
                "setting_id": setting_id,
                **setting_values,
                "attempted_cases": len(rows),
                "completed_cases": completed,
                "execution_failure_cases": len(rows) - completed,
                "rows_sha256": sha256(rows_path),
                "summary_sha256": sha256(summary_path),
            }
        )

    expected_total = len(SETTING_ORDER) * len(development_subjects) * len(
        SOURCE_EXPRESSION_INDICES
    )
    if len(combined) != expected_total:
        raise ValueError(f"Combined ARAP grid expected {expected_total} rows")
    summary = {
        "development_only": True,
        "heldout_inspected": False,
        "settings": settings_summary,
        "setting_count": len(SETTING_ORDER),
        "cases_per_setting": len(reference_cases or ()),
        "combined_rows": len(combined),
        "split_sha256": sha256(split_json),
        "collection_manifest_sha256": sha256(manifest_path),
        "selection_performed": False,
        "selection_blocked_until_common_qc_is_frozen": True,
    }
    return combined, summary


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    rows, summary = combine_collection(args.collection_dir, args.split_json)
    args.output_dir.mkdir(parents=True)
    fields = list(rows[0])
    with (args.output_dir / "arap_grid_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary["combined_rows_sha256"] = sha256(args.output_dir / "arap_grid_rows.csv")
    (args.output_dir / "arap_grid_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
