#!/usr/bin/env python3
"""Verify that the ablation's proposed condition reproduces frozen S8."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


EXPRESSIONS_PER_SUBJECT = 19
STRING_FIELDS = (
    "source_method",
    "pre_s8_branch",
    "source_anchor_definition",
    "nose_anchor_metric_definition",
)
INTEGER_FIELDS = (
    "completed",
    "post_orientation_pass",
    "total_scheduled_passes",
    "expected_scheduled_passes",
    "executed_constrained_solves",
    "skipped_scheduled_passes",
)
FLOAT_FIELDS = (
    "post_full_median_mm",
    "post_full_p90_mm",
    "post_nose_median_mm",
    "post_nose_p90_mm",
    "post_anchor_mm",
    "post_anchor_point_mm",
    "post_anchor_surface_mm",
    "eye_fixed_max_mm",
    "displacement_p90_mm",
    "displacement_max_mm",
    "edge_strain_median",
    "edge_strain_p90",
    "edge_strain_p99",
    "edge_strain_max",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--ablation-csv", type=Path, required=True)
    parser.add_argument("--condition", default="proposed_s8")
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument(
        "--partition", choices=("development", "test"), required=True
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-10)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.resolve(strict=True).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_rows(
    rows: list[dict[str, str]], subjects: set[str], label: str
) -> dict[str, dict[str, str]]:
    required = {"case", "subject", *STRING_FIELDS, *INTEGER_FIELDS, *FLOAT_FIELDS}
    if not rows or not required.issubset(rows[0]):
        raise KeyError(f"{label} is missing fields: {sorted(required - set(rows[0] if rows else ())) }")
    indexed: dict[str, dict[str, str]] = {}
    counts: Counter[str] = Counter()
    for row in rows:
        case = str(row["case"])
        subject = f"{int(row['subject']):03d}"
        case_subject = f"{int(case.split('_', 1)[0]):03d}"
        if subject != case_subject or subject not in subjects:
            raise ValueError(f"{label} case/subject is outside the frozen partition: {case}")
        if case in indexed:
            raise ValueError(f"{label} contains duplicate case {case}")
        indexed[case] = row
        counts[subject] += 1
    bad = {
        subject: counts.get(subject, 0)
        for subject in sorted(subjects | set(counts))
        if counts.get(subject, 0) != EXPRESSIONS_PER_SUBJECT
    }
    expected = len(subjects) * EXPRESSIONS_PER_SUBJECT
    if set(counts) != subjects or len(indexed) != expected or bad:
        raise ValueError(
            f"{label} does not match the frozen partition: "
            f"cases={len(indexed)}/{expected}, counts={bad}"
        )
    return indexed


def verify_equivalence(
    reference: dict[str, dict[str, str]],
    proposed: dict[str, dict[str, str]],
    tolerance: float,
) -> dict[str, Any]:
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Absolute tolerance must be finite and nonnegative")
    if set(reference) != set(proposed):
        raise ValueError("Reference and proposed-S8 case sets differ")
    string_mismatches: list[dict[str, str]] = []
    integer_mismatches: list[dict[str, Any]] = []
    maximum_absolute_difference = {field: 0.0 for field in FLOAT_FIELDS}
    for case in sorted(reference):
        left = reference[case]
        right = proposed[case]
        for field in STRING_FIELDS:
            if str(left[field]) != str(right[field]):
                string_mismatches.append(
                    {"case": case, "field": field, "reference": str(left[field]), "proposed": str(right[field])}
                )
        for field in INTEGER_FIELDS:
            left_value = int(float(left[field]))
            right_value = int(float(right[field]))
            if left_value != right_value:
                integer_mismatches.append(
                    {"case": case, "field": field, "reference": left_value, "proposed": right_value}
                )
        for field in FLOAT_FIELDS:
            left_value = float(left[field])
            right_value = float(right[field])
            if not math.isfinite(left_value) or not math.isfinite(right_value):
                raise ValueError(f"Non-finite {field} for {case}")
            difference = abs(right_value - left_value)
            maximum_absolute_difference[field] = max(
                maximum_absolute_difference[field], difference
            )
    numeric_failures = {
        field: value
        for field, value in maximum_absolute_difference.items()
        if value > tolerance
    }
    if string_mismatches or integer_mismatches or numeric_failures:
        raise ValueError(
            "Ablation proposed_s8 does not reproduce the frozen reference: "
            f"string={string_mismatches[:3]}, integer={integer_mismatches[:3]}, "
            f"numeric={numeric_failures}"
        )
    return {
        "cases": len(reference),
        "string_fields_verified": list(STRING_FIELDS),
        "integer_fields_verified": list(INTEGER_FIELDS),
        "float_fields_verified": list(FLOAT_FIELDS),
        "maximum_absolute_difference": maximum_absolute_difference,
        "absolute_tolerance": tolerance,
        "equivalent": True,
    }


def main() -> None:
    args = parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_json}")
    reference_path = args.reference_csv.resolve(strict=True)
    ablation_path = args.ablation_csv.resolve(strict=True)
    split_path = args.split_json.resolve(strict=True)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split_key = "development_subjects" if args.partition == "development" else "test_subjects"
    subjects = {f"{int(value):03d}" for value in split[split_key]}
    reference = index_rows(read_csv(reference_path), subjects, "reference")
    all_ablation_rows = read_csv(ablation_path)
    proposed_rows = [
        row for row in all_ablation_rows if str(row.get("condition", "")) == args.condition
    ]
    proposed = index_rows(proposed_rows, subjects, args.condition)
    audit = verify_equivalence(reference, proposed, args.absolute_tolerance)
    payload = {
        "partition": args.partition,
        "development_only": args.partition == "development",
        "heldout_inspected": args.partition == "test",
        "condition": args.condition,
        "reference_csv_sha256": sha256(reference_path),
        "ablation_csv_sha256": sha256(ablation_path),
        "split_sha256": sha256(split_path),
        **audit,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
