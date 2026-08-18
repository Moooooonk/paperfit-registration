#!/usr/bin/env python3
"""Apply one frozen common-QC rule to equal-denominator method groups."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import analyze_subject_clustered_results as clustered
import apply_frozen_common_qc as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument(
        "--subset", choices=("development", "heldout", "all"), required=True
    )
    parser.add_argument("--group-column", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-label", required=True)
    parser.add_argument("--expected-cases-per-group", type=int, default=190)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.resolve(strict=True).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_groups(
    rows: list[dict[str, str]],
    group_column: str,
    expected_cases: int,
    split: dict[str, object] | None = None,
    subset: str | None = None,
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_column])].append(row)
    if not grouped:
        raise ValueError("No method groups")
    reference: set[str] | None = None
    for name, values in sorted(grouped.items()):
        cases = [str(row["case"]) for row in values]
        if len(cases) != len(set(cases)):
            raise ValueError(f"Duplicate cases in group {name}")
        if len(cases) != expected_cases:
            raise ValueError(
                f"Expected {expected_cases} cases in group {name}, found {len(cases)}"
            )
        if split is not None and subset is not None:
            common.validate_subset_rows(values, split, subset)
        case_set = set(cases)
        if reference is None:
            reference = case_set
        elif case_set != reference:
            raise ValueError(f"Case denominator differs for group {name}")
    return dict(grouped)


def safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return result or "group"


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
    rows = read_csv(args.input_csv)
    required = {
        "case",
        "subject",
        args.group_column,
        "post_orientation_pass",
        "eye_fixed_max_mm",
        "post_full_median_mm",
        "post_full_p90_mm",
        "post_nose_median_mm",
        "post_nose_p90_mm",
        "post_anchor_mm",
        "edge_strain_p99",
    }
    if not rows or not required.issubset(rows[0]):
        raise KeyError(f"Missing fields: {required - set(rows[0])}")
    split_path = args.split_json.resolve(strict=True)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    grouped = validate_groups(
        rows,
        args.group_column,
        args.expected_cases_per_group,
        split,
        args.subset,
    )
    calibration = json.loads(
        args.calibration_json.resolve(strict=True).read_text(encoding="utf-8")
    )
    rule = common.validate_calibration_artifact(
        calibration, split, common.sha256(split_path)
    )

    decisions: list[dict[str, Any]] = []
    summaries = []
    for index, (name, values) in enumerate(sorted(grouped.items()), start=1):
        group_decisions = []
        for row in values:
            if common.completed(row):
                gates, failed = common.evaluate_gates(row, rule)
                gate_columns = {
                    f"gate_{gate}_pass": int(gates[gate])
                    for gate in common.GATE_NAMES
                }
            else:
                failure = row.get("execution_failure_reason") or "execution_failure"
                failed = [failure]
                gate_columns = {
                    f"gate_{gate}_pass": "" for gate in common.GATE_NAMES
                }
            decision = {
                **row,
                **gate_columns,
                "final_accepted": int(not failed),
                "failure_reasons": ";".join(failed),
            }
            group_decisions.append(decision)
            decisions.append(decision)
        cluster_summary = clustered.group_summary(
            group_decisions, "final_accepted", seed_offset=1000 + index
        )
        summaries.append(
            {
                args.group_column: name,
                "attempted_pairs": len(group_decisions),
                "completed_pairs": sum(common.completed(row) for row in values),
                **cluster_summary,
            }
        )

    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "common_qc_grouped_cases.csv", decisions)
    write_csv(args.output_dir / "common_qc_grouped_summary.csv", summaries)
    for name, values in sorted(grouped.items()):
        selected = [
            row for row in decisions if str(row[args.group_column]) == name
        ]
        write_csv(args.output_dir / f"cases_{safe_name(name)}.csv", selected)
    report = {
        "analysis_label": args.analysis_label,
        "group_column": args.group_column,
        "subset": args.subset,
        "groups": len(grouped),
        "expected_cases_per_group": args.expected_cases_per_group,
        "equal_case_denominators_verified": True,
        "execution_failures_remain_in_denominator": True,
        "frozen_rule": rule,
        "development_calibration_target_reached": bool(
            calibration.get("final_target_reached")
        ),
        "summaries": summaries,
    }
    (args.output_dir / "common_qc_grouped_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
