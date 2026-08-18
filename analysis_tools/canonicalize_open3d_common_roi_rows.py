#!/usr/bin/env python3
"""Canonicalize common-ROI Open3D rows without modifying raw results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    "post_full_median_mm",
    "post_full_p90_mm",
    "post_nose_median_mm",
    "post_nose_p90_mm",
    "post_anchor_mm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=190)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field!r} for {row.get('case', '?')}") from exc


def canonicalize(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    canonical: list[dict[str, Any]] = []
    for row in rows:
        completed_value = row.get("completed")
        is_completed = (
            True
            if completed_value in (None, "")
            else int(float(completed_value)) == 1
        )
        surface_anchor_field = (
            "post_anchor_surface_mm"
            if row.get("post_anchor_surface_mm") not in (None, "")
            else "post_anchor_mm"
        )
        point_anchor = (
            number(row, "post_anchor_point_mm") if is_completed else None
        )
        surface_anchor = (
            number(row, surface_anchor_field) if is_completed else None
        )
        canonical.append(
            {
                **row,
                "post_anchor_surface_mm": surface_anchor,
                "post_anchor_mm": point_anchor,
                "post_anchor_point_mm": point_anchor,
                "nose_anchor_metric_definition": (
                    "transformed source semantic nose-tip point to target nose-tip anchor"
                ),
                "surface_anchor_is_diagnostic_only": 1,
                "surface_anchor_source_column": surface_anchor_field,
            }
        )
    return canonical


def validate(rows: list[dict[str, Any]], expected_cases: int) -> dict[str, list[str]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["baseline_method"])].append(row)
    if not by_method:
        raise ValueError("No baseline rows")
    case_sets: dict[str, list[str]] = {}
    reference: set[str] | None = None
    for method, method_rows in sorted(by_method.items()):
        cases = [str(row["case"]) for row in method_rows]
        if len(cases) != len(set(cases)):
            raise ValueError(f"Duplicate cases for {method}")
        if len(cases) != expected_cases:
            raise ValueError(
                f"Expected {expected_cases} cases for {method}, found {len(cases)}"
            )
        case_set = set(cases)
        if reference is None:
            reference = case_set
        elif case_set != reference:
            raise ValueError(f"Case denominator differs for {method}")
        for row in method_rows:
            if int(float(row["attempted"])) != 1:
                raise ValueError(f"Unattempted case for {method}: {row['case']}")
        case_sets[method] = sorted(cases)
    return case_sets


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["baseline_method"])].append(row)
    table: list[dict[str, Any]] = []
    for method, method_rows in sorted(by_method.items()):
        completed = [row for row in method_rows if int(float(row["completed"])) == 1]
        item: dict[str, Any] = {
            "baseline_method": method,
            "attempted_pairs": len(method_rows),
            "completed_pairs": len(completed),
            "error_pairs": len(method_rows) - len(completed),
            "subjects": len({str(row["subject"]) for row in method_rows}),
            "orientation_pass_pairs": sum(
                int(float(row["post_orientation_pass"])) for row in completed
            ),
        }
        item["orientation_pass_rate"] = (
            item["orientation_pass_pairs"] / len(method_rows)
        )
        for field in METRICS:
            values = np.asarray([number(row, field) for row in completed], dtype=float)
            item[f"{field}_mean"] = float(np.mean(values)) if len(values) else None
            item[f"{field}_median"] = (
                float(np.median(values)) if len(values) else None
            )
            item[f"{field}_p90_across_pairs"] = (
                float(np.quantile(values, 0.90)) if len(values) else None
            )
        table.append(item)
    summary = {
        "methods": len(table),
        "rows": len(rows),
        "anchor_column_used_for_qc": "post_anchor_mm",
        "anchor_definition": (
            "Euclidean distance between the transformed source semantic nose-tip "
            "point and the target semantic nose-tip anchor"
        ),
        "surface_anchor_column": "post_anchor_surface_mm",
        "surface_anchor_role": "diagnostic only",
        "acceptance_threshold_applied": False,
        "denominator_policy": "all attempted pairs remain in each method denominator",
        "descriptive_table": table,
    }
    return table, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    raw = read_rows(args.input_csv)
    rows = canonicalize(raw)
    validate(rows, args.expected_cases)
    table, summary = summarize(rows)
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "open3d_common_roi_rows_canonical.csv", rows)
    write_csv(args.output_dir / "open3d_threshold_free_summary.csv", table)
    (args.output_dir / "open3d_canonicalization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
