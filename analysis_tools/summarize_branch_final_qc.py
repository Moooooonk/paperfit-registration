#!/usr/bin/env python3
"""Summarize pre-S8 strata under the branch-independent final QC."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BRANCHES = ("rigid_pass", "anchor_only", "broad_failure", "residual")
GATES = (
    "orientation",
    "full_median",
    "full_p90",
    "nose_median",
    "nose_p90",
    "nose_anchor",
    "edge_strain",
    "eye_constraint",
)
EXPRESSIONS_PER_SUBJECT = 19


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument(
        "--partition", choices=("development", "test", "full"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-label", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def selected_subjects(split: dict[str, Any], partition: str) -> list[str]:
    development = [f"{int(value):03d}" for value in split["development_subjects"]]
    heldout = [f"{int(value):03d}" for value in split["test_subjects"]]
    if partition == "development":
        return sorted(development)
    if partition == "test":
        return sorted(heldout)
    return sorted(development + heldout)


def integer(row: dict[str, str], field: str) -> int:
    value = int(float(row[field]))
    if value not in (0, 1):
        raise ValueError(f"{row['case']}: {field} must be binary")
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    rows = read_csv(args.input_csv.resolve(strict=True))
    required = {
        "case",
        "subject",
        "pre_s8_branch",
        "completed",
        "final_accepted",
        "failure_reasons",
        *(f"gate_{gate}_pass" for gate in GATES),
    }
    if not rows or not required.issubset(rows[0]):
        raise KeyError(f"Missing columns: {sorted(required - set(rows[0] if rows else ())) }")

    split = json.loads(args.split_json.resolve(strict=True).read_text(encoding="utf-8"))
    subjects = selected_subjects(split, args.partition)
    subject_set = set(subjects)
    selected: list[dict[str, str]] = []
    case_ids: set[str] = set()
    counts: Counter[str] = Counter()
    for row in rows:
        subject = f"{int(row['subject']):03d}"
        if subject not in subject_set:
            continue
        case = str(row["case"])
        if f"{int(case.split('_', 1)[0]):03d}" != subject:
            raise ValueError(f"Case/subject mismatch: {case}/{subject}")
        if case in case_ids:
            raise ValueError(f"Duplicate case: {case}")
        case_ids.add(case)
        counts[subject] += 1
        selected.append({**row, "subject": subject})

    expected = len(subjects) * EXPRESSIONS_PER_SUBJECT
    bad_counts = {
        subject: counts.get(subject, 0)
        for subject in subjects
        if counts.get(subject, 0) != EXPRESSIONS_PER_SUBJECT
    }
    if len(selected) != expected or bad_counts:
        raise ValueError(
            f"Incomplete partition: rows={len(selected)}/{expected}, counts={bad_counts}"
        )

    by_branch: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        branch = str(row["pre_s8_branch"])
        if branch not in BRANCHES:
            raise ValueError(f"Unsupported branch: {branch}")
        accepted = integer(row, "final_accepted")
        completed = integer(row, "completed")
        if not completed:
            if accepted:
                raise ValueError(f"Incomplete case cannot be accepted: {row['case']}")
            by_branch[branch].append(row)
            continue
        gate_values = {
            gate: integer(row, f"gate_{gate}_pass") for gate in GATES
        }
        if accepted and not all(gate_values.values()):
            raise ValueError(f"Accepted case lacks complete passing evidence: {row['case']}")
        if branch == "anchor_only" and accepted and not gate_values["nose_anchor"]:
            raise ValueError(f"Accepted anchor-only case did not repair anchor: {row['case']}")
        by_branch[branch].append(row)

    branch_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for branch in BRANCHES:
        values = by_branch.get(branch, [])
        completed_count = sum(integer(row, "completed") for row in values)
        accepted_count = sum(integer(row, "final_accepted") for row in values)
        branch_rows.append(
            {
                "pre_s8_branch": branch,
                "attempted": len(values),
                "completed": completed_count,
                "invalid_or_incomplete_evidence": len(values) - completed_count,
                "accepted_after_common_final_qc": accepted_count,
                "rejected_after_common_final_qc": len(values) - accepted_count,
                "acceptance_rate": accepted_count / len(values) if values else None,
            }
        )
        failures: Counter[str] = Counter()
        for row in values:
            if not integer(row, "completed"):
                failures["invalid_or_incomplete_evidence"] += 1
                continue
            for gate in GATES:
                if not integer(row, f"gate_{gate}_pass"):
                    failures[gate] += 1
        for gate in (*GATES, "invalid_or_incomplete_evidence"):
            gate_rows.append(
                {
                    "pre_s8_branch": branch,
                    "failed_gate": gate,
                    "failed_cases": failures[gate],
                }
            )

    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "branch_final_qc_summary.csv", branch_rows)
    write_csv(args.output_dir / "branch_failed_gate_counts.csv", gate_rows)
    summary = {
        "analysis_label": args.analysis_label,
        "partition": args.partition,
        "subjects": len(subjects),
        "attempted_pairs": len(selected),
        "common_final_qc_applied_to_every_completed_branch": True,
        "fixed_eye_is_implementation_audit_only": True,
        "anchor_only_acceptance_requires_repaired_anchor": True,
        "branch_summaries": branch_rows,
    }
    (args.output_dir / "branch_final_qc_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
