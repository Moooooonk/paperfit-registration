#!/usr/bin/env python3
"""Build a fail-closed analysis view without modifying raw metric rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_REASON = "invalid_target_anchor_direct_intersection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--invalid-subject", action="append", required=True)
    parser.add_argument("--group-column")
    parser.add_argument("--expected-cases-per-group", type=int, default=190)
    parser.add_argument("--expected-expressions-per-subject", type=int, default=19)
    parser.add_argument("--failure-reason", default=DEFAULT_REASON)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_subject(value: object) -> str:
    return f"{int(str(value)):03d}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_rows(
    rows: list[dict[str, str]],
    invalid_subjects: set[str],
    group_column: str | None,
    expected_cases_per_group: int,
    expected_expressions_per_subject: int,
) -> dict[str, list[dict[str, str]]]:
    required = {"case", "subject", "completed"}
    if group_column:
        required.add(group_column)
    if not rows or not required.issubset(rows[0]):
        raise KeyError(f"Missing columns: {required - set(rows[0]) if rows else required}")
    if expected_cases_per_group <= 0 or expected_expressions_per_subject <= 0:
        raise ValueError("Expected denominators must be positive")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        subject = normalized_subject(row["subject"])
        case_subject = normalized_subject(str(row["case"]).split("_", 1)[0])
        if subject != case_subject:
            raise ValueError(f"Case/subject mismatch: {row['case']}/{subject}")
        group = str(row[group_column]) if group_column else "__single__"
        grouped[group].append(row)

    reference_cases: set[str] | None = None
    observed_subjects: set[str] = set()
    for group, values in sorted(grouped.items()):
        cases = [str(row["case"]) for row in values]
        if len(cases) != len(set(cases)):
            raise ValueError(f"Duplicate cases in group {group}")
        if len(cases) != expected_cases_per_group:
            raise ValueError(
                f"Expected {expected_cases_per_group} cases in group {group}, "
                f"found {len(cases)}"
            )
        case_set = set(cases)
        if reference_cases is None:
            reference_cases = case_set
        elif case_set != reference_cases:
            raise ValueError(f"Case denominator differs for group {group}")
        counts = Counter(normalized_subject(row["subject"]) for row in values)
        bad = {
            subject: count
            for subject, count in sorted(counts.items())
            if count != expected_expressions_per_subject
        }
        if bad:
            raise ValueError(f"Incomplete subject grid in group {group}: {bad}")
        observed_subjects.update(counts)

    missing_invalid = sorted(invalid_subjects - observed_subjects)
    if missing_invalid:
        raise ValueError(f"Invalid-evidence subjects absent from input: {missing_invalid}")
    return dict(grouped)


def build_view(
    rows: list[dict[str, str]],
    invalid_subjects: set[str],
    failure_reason: str,
) -> tuple[list[dict[str, Any]], int]:
    output: list[dict[str, Any]] = []
    modified = 0
    for row in rows:
        subject = normalized_subject(row["subject"])
        invalid = subject in invalid_subjects
        record: dict[str, Any] = {
            **row,
            "primary_input_completed_original": row.get("completed", ""),
            "primary_input_failure_reason_original": row.get(
                "execution_failure_reason", ""
            ),
            "target_anchor_direct_intersection_valid": int(not invalid),
            "primary_fail_closed_due_to_input_evidence": int(invalid),
        }
        if invalid:
            record["completed"] = 0
            record["execution_failure_reason"] = failure_reason
            modified += 1
        output.append(record)
    return output, modified


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    input_path = args.input_csv.resolve(strict=True)
    invalid_subjects = {
        normalized_subject(value) for value in args.invalid_subject
    }
    rows = read_csv(input_path)
    grouped = validate_rows(
        rows,
        invalid_subjects,
        args.group_column,
        args.expected_cases_per_group,
        args.expected_expressions_per_subject,
    )
    output_rows, modified = build_view(
        rows, invalid_subjects, args.failure_reason
    )
    expected_modified = (
        len(invalid_subjects)
        * args.expected_expressions_per_subject
        * len(grouped)
    )
    if modified != expected_modified:
        raise RuntimeError(
            f"Fail-closed row count mismatch: {modified}/{expected_modified}"
        )

    args.output_dir.mkdir(parents=True)
    output_csv = args.output_dir / "primary_fail_closed_rows.csv"
    write_csv(output_csv, output_rows)
    report = {
        "input_csv": str(input_path),
        "input_sha256": sha256(input_path),
        "output_csv": str(output_csv.resolve()),
        "output_sha256": sha256(output_csv),
        "raw_input_preserved": True,
        "policy": (
            "All rows for identities without a protocol-valid direct target-anchor "
            "intersection remain in the denominator and are marked incomplete in the "
            "primary confirmatory view. Original completion and failure fields are "
            "retained in dedicated audit columns."
        ),
        "failure_reason": args.failure_reason,
        "invalid_subjects_internal_do_not_publish": sorted(invalid_subjects),
        "group_column": args.group_column,
        "groups": sorted(grouped),
        "expected_cases_per_group": args.expected_cases_per_group,
        "expected_expressions_per_subject": args.expected_expressions_per_subject,
        "input_rows": len(rows),
        "fail_closed_rows": modified,
    }
    (args.output_dir / "fail_closed_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
