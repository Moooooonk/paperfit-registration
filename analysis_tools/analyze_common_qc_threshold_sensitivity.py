#!/usr/bin/env python3
"""Describe common-QC sensitivity without selecting or changing its rule."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import analyze_subject_clustered_results as clustered
import apply_frozen_common_qc as common


MULTIPLIERS = (0.8, 0.9, 1.0, 1.1, 1.2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument(
        "--subset", choices=("development", "heldout"), required=True
    )
    parser.add_argument("--ratings-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-label", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.resolve(strict=True).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def scaled_rule(rule: dict[str, float], multiplier: float) -> dict[str, float]:
    if multiplier <= 0.0:
        raise ValueError("Threshold multiplier must be positive")
    return {name: float(value) * multiplier for name, value in rule.items()}


def evaluate_rows(
    rows: list[dict[str, str]],
    frozen_rule: dict[str, float],
    multipliers: tuple[float, ...] = MULTIPLIERS,
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for multiplier in multipliers:
        rule = scaled_rule(frozen_rule, multiplier)
        for row in rows:
            if common.completed(row):
                gates, failed = common.evaluate_gates(row, rule)
                gate_columns = {
                    f"gate_{name}_pass": int(gates[name])
                    for name in common.GATE_NAMES
                }
            else:
                failed = [
                    row.get("execution_failure_reason") or "execution_failure"
                ]
                gate_columns = {
                    f"gate_{name}_pass": "" for name in common.GATE_NAMES
                }
            decisions.append(
                {
                    **row,
                    **gate_columns,
                    "threshold_multiplier": multiplier,
                    "final_accepted": int(not failed),
                    "failure_reasons": ";".join(failed),
                }
            )
    return decisions


def rating_labels(path: Path, expected_cases: set[str]) -> dict[str, bool]:
    rows = read_csv(path)
    required = {"case", "consensus_resolved", "consensus_usable"}
    if not rows or not required.issubset(rows[0]):
        raise KeyError(f"Rating fields missing: {required - set(rows[0] if rows else ())}")
    labels: dict[str, bool] = {}
    for row in rows:
        case = str(row["case"])
        if case in labels:
            raise ValueError(f"Duplicate rating case: {case}")
        if int(float(row["consensus_resolved"])) != 1:
            raise ValueError("Threshold sensitivity requires resolved consensus")
        usable = int(float(row["consensus_usable"]))
        if usable not in (0, 1):
            raise ValueError("consensus_usable must be binary")
        labels[case] = bool(usable)
    if set(labels) != expected_cases:
        raise ValueError("Rating and metric case sets differ")
    return labels


def diagnostic_confusion(
    rows: list[dict[str, Any]], labels: dict[str, bool]
) -> dict[str, Any]:
    tp = sum(int(row["final_accepted"]) == 1 and labels[str(row["case"])] for row in rows)
    fp = sum(int(row["final_accepted"]) == 1 and not labels[str(row["case"])] for row in rows)
    tn = sum(int(row["final_accepted"]) == 0 and not labels[str(row["case"])] for row in rows)
    fn = sum(int(row["final_accepted"]) == 0 and labels[str(row["case"])] for row in rows)

    def divide(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    sensitivity = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "ppv": divide(tp, tp + fp),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (
            (sensitivity + specificity) / 2.0
            if sensitivity is not None and specificity is not None
            else None
        ),
    }


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
    metrics_path = args.metrics_csv.resolve(strict=True)
    calibration_path = args.calibration_json.resolve(strict=True)
    split_path = args.split_json.resolve(strict=True)
    rows = read_csv(metrics_path)
    required = {
        "case",
        "subject",
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
        raise KeyError(f"Metric fields missing: {required - set(rows[0] if rows else ())}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    common.validate_subset_rows(rows, split, args.subset)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    frozen_rule = common.validate_calibration_artifact(
        calibration, split, common.sha256(split_path)
    )
    decisions = evaluate_rows(rows, frozen_rule)
    expected_cases = {str(row["case"]) for row in rows}
    labels = rating_labels(args.ratings_csv, expected_cases) if args.ratings_csv else None

    summaries: list[dict[str, Any]] = []
    for index, multiplier in enumerate(MULTIPLIERS, start=1):
        subset = [
            row
            for row in decisions
            if float(row["threshold_multiplier"]) == multiplier
        ]
        summary: dict[str, Any] = {
            "threshold_multiplier": multiplier,
            "attempted_pairs": len(subset),
            "completed_pairs": sum(common.completed(row) for row in subset),
            **clustered.group_summary(
                subset, "final_accepted", seed_offset=3000 + index
            ),
        }
        if labels is not None:
            summary.update(diagnostic_confusion(subset, labels))
        summaries.append(summary)

    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "threshold_sensitivity_cases.csv", decisions)
    write_csv(args.output_dir / "threshold_sensitivity_summary.csv", summaries)
    report = {
        "analysis_label": args.analysis_label,
        "subset": args.subset,
        "descriptive_only": True,
        "changes_frozen_rule": False,
        "selection_permitted": False,
        "multipliers": MULTIPLIERS,
        "scaled_thresholds": list(common.RULE_KEYS),
        "fixed_gates": ["orientation", "eye_constraint"],
        "execution_failures_remain_in_every_denominator": True,
        "calibration_sha256": sha256(calibration_path),
        "metrics_sha256": sha256(metrics_path),
        "ratings_sha256": (
            sha256(args.ratings_csv.resolve(strict=True)) if args.ratings_csv else None
        ),
        "summaries": summaries,
    }
    (args.output_dir / "threshold_sensitivity_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
