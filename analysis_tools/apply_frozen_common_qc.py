#!/usr/bin/env python3
"""Apply one frozen branch-independent post-S8 decision rule."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


EYE_TOLERANCE_MM = 1e-6
EXPRESSIONS_PER_SUBJECT = 19
EXPECTED_EXPRESSION_INDICES = {*range(1, 18), 19, 20}
GATE_NAMES = (
    "orientation",
    "eye_constraint",
    "full_median",
    "full_p90",
    "nose_median",
    "nose_p90",
    "nose_anchor",
    "edge_strain",
)
RULE_KEYS = (
    "full_median_mm",
    "full_p90_mm",
    "nose_median_mm",
    "nose_p90_mm",
    "anchor_mm",
    "edge_strain_p99",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument(
        "--subset", choices=("development", "heldout", "all"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-label", required=True)
    return parser.parse_args()


def finite_value(row: dict[str, str], name: str) -> float | None:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def completed(row: dict[str, str]) -> bool:
    value = row.get("completed")
    return True if value in (None, "") else int(float(value)) == 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_calibration_artifact(
    calibration: dict[str, object],
    split: dict[str, object],
    split_sha256: str | None = None,
) -> dict[str, float]:
    if (
        calibration.get("development_only") is not True
        or calibration.get("heldout_inspected") is not False
        or calibration.get("heldout_test_inspected") is not False
    ):
        raise ValueError(
            "Calibration JSON is not a pre-heldout development-only artifact"
        )
    if split_sha256 is not None and calibration.get("split_sha256") != split_sha256:
        raise ValueError("Calibration split hash does not match the application split")
    if str(calibration.get("calibration_source_method", "")).lower() != "hrn":
        raise ValueError("Common-QC calibration must originate from HRN development")
    if (
        calibration.get("final_target_reached") is not True
        or calibration.get("common_qc_frozen") is not True
    ):
        raise ValueError(
            "Common QC was not frozen because the prespecified development target "
            "was not reached"
        )
    development = {f"{int(value):03d}" for value in split["development_subjects"]}
    calibrated_subjects = {
        f"{int(value):03d}" for value in calibration.get("subjects", [])
    }
    expected_cases = len(development) * EXPRESSIONS_PER_SUBJECT
    if calibrated_subjects != development or int(calibration.get("cases", -1)) != expected_cases:
        raise ValueError(
            "Calibration subjects or denominator do not match the frozen development split"
        )
    eye_tolerance = float(calibration.get("eye_tolerance_mm", float("nan")))
    if not math.isfinite(eye_tolerance) or eye_tolerance != EYE_TOLERANCE_MM:
        raise ValueError("Calibration eye tolerance does not match the frozen QC rule")
    raw_rule = calibration.get("final_frozen_rule")
    if not isinstance(raw_rule, dict) or set(raw_rule) != set(RULE_KEYS):
        raise ValueError("Calibration has an invalid frozen-rule schema")
    rule = {name: float(raw_rule[name]) for name in RULE_KEYS}
    if any(not math.isfinite(value) or value <= 0.0 for value in rule.values()):
        raise ValueError("Calibration thresholds must be finite and positive")
    return rule


def validate_subset_rows(
    rows: list[dict[str, str]], split: dict[str, object], subset: str
) -> set[str]:
    cases = [str(row["case"]) for row in rows]
    if len(cases) != len(set(cases)):
        raise ValueError("Duplicate cases in common-QC input")
    if subset == "all":
        expected_subjects = set(split["development_subjects"]) | set(
            split["test_subjects"]
        )
    elif subset == "heldout":
        expected_subjects = set(split["test_subjects"])
    else:
        expected_subjects = set(split["development_subjects"])

    counts: Counter[str] = Counter()
    for row in rows:
        subject = f"{int(row['subject']):03d}"
        case_subject = f"{int(str(row['case']).split('_', 1)[0]):03d}"
        if subject != case_subject:
            raise ValueError(
                f"Case/subject mismatch: case={row['case']}, subject={subject}"
            )
        counts[subject] += 1
    observed_subjects = set(counts)
    expected_cases = len(expected_subjects) * EXPRESSIONS_PER_SUBJECT
    bad_counts = {
        subject: counts.get(subject, 0)
        for subject in sorted(expected_subjects | observed_subjects)
        if counts.get(subject, 0) != EXPRESSIONS_PER_SUBJECT
    }
    if (
        observed_subjects != expected_subjects
        or len(rows) != expected_cases
        or bad_counts
    ):
        raise ValueError(
            f"Common-QC rows do not match frozen {subset} split: "
            f"cases={len(rows)}/{expected_cases}, "
            f"subjects={sorted(observed_subjects)}/{sorted(expected_subjects)}, "
            f"per_subject_counts={bad_counts}"
        )
    expression_indices: dict[str, list[int]] = {
        subject: [] for subject in expected_subjects
    }
    for row in rows:
        subject = f"{int(row['subject']):03d}"
        try:
            expression_index = int(str(row["case"]).split("_", 2)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(
                f"Case lacks a numeric expression index: {row['case']}"
            ) from error
        expression_indices[subject].append(expression_index)
    for subject, indices in sorted(expression_indices.items()):
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate expression index for subject {subject}")
        if set(indices) != EXPECTED_EXPRESSION_INDICES:
            raise ValueError(
                f"Incomplete expression grid for subject {subject}: "
                f"observed={sorted(indices)}"
            )
    return expected_subjects


def evaluate_gates(
    row: dict[str, str], rule: dict[str, float]
) -> tuple[dict[str, bool], list[str]]:
    values = {
        name: finite_value(row, name)
        for name in (
            "post_orientation_pass",
            "eye_fixed_max_mm",
            "post_full_median_mm",
            "post_full_p90_mm",
            "post_nose_median_mm",
            "post_nose_p90_mm",
            "post_anchor_mm",
            "edge_strain_p99",
        )
    }
    gates = {
        "orientation": values["post_orientation_pass"] == 1.0,
        "eye_constraint": values["eye_fixed_max_mm"] is not None
        and 0.0 <= values["eye_fixed_max_mm"] <= EYE_TOLERANCE_MM,
        "full_median": values["post_full_median_mm"] is not None
        and 0.0 <= values["post_full_median_mm"] <= float(rule["full_median_mm"]),
        "full_p90": values["post_full_p90_mm"] is not None
        and 0.0 <= values["post_full_p90_mm"] <= float(rule["full_p90_mm"]),
        "nose_median": values["post_nose_median_mm"] is not None
        and 0.0 <= values["post_nose_median_mm"] <= float(rule["nose_median_mm"]),
        "nose_p90": values["post_nose_p90_mm"] is not None
        and 0.0 <= values["post_nose_p90_mm"] <= float(rule["nose_p90_mm"]),
        "nose_anchor": values["post_anchor_mm"] is not None
        and 0.0 <= values["post_anchor_mm"] <= float(rule["anchor_mm"]),
        "edge_strain": values["edge_strain_p99"] is not None
        and 0.0 <= values["edge_strain_p99"] <= float(rule["edge_strain_p99"]),
    }
    return gates, [name for name, passed in gates.items() if not passed]


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    with args.metrics_csv.resolve(strict=True).open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    calibration = json.loads(
        args.calibration_json.resolve(strict=True).read_text(encoding="utf-8")
    )
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
        raise KeyError(f"Missing metric columns: {required - set(rows[0])}")
    split_path = args.split_json.resolve(strict=True)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    rule = validate_calibration_artifact(calibration, split, sha256(split_path))
    target_reached = calibration["final_target_reached"]
    validate_subset_rows(rows, split, args.subset)

    decisions = []
    failures = Counter()
    for row in rows:
        if completed(row):
            gates, failed = evaluate_gates(row, rule)
            gate_columns = {
                f"gate_{name}_pass": int(gates[name]) for name in GATE_NAMES
            }
        else:
            failure = row.get("execution_failure_reason") or "execution_failure"
            failed = [failure]
            gate_columns = {f"gate_{name}_pass": "" for name in GATE_NAMES}
        failures.update(failed)
        decisions.append(
            {
                **row,
                **gate_columns,
                "final_accepted": int(not failed),
                "failure_reasons": ";".join(failed),
            }
        )

    summary = {
        "analysis_label": args.analysis_label,
        "subset": args.subset,
        "case_count": len(decisions),
        "subject_count": len({f"{int(row['subject']):03d}" for row in decisions}),
        "accepted": sum(int(row["final_accepted"]) for row in decisions),
        "coverage": sum(int(row["final_accepted"]) for row in decisions)
        / len(decisions),
        "failure_gate_counts": dict(failures),
        "frozen_rule": rule,
        "development_calibration_target_reached": target_reached,
        "independently_validated_label_permitted": target_reached,
        "calibration_interpretation": (
            "The prespecified development PPV target was reached."
            if target_reached
            else "The prespecified development PPV target was not reached; the "
            "protocol-defined fallback rule was applied without describing its "
            "accepted outputs as independently validated."
        ),
        "eye_tolerance_mm": EYE_TOLERANCE_MM,
        "branch_independent": True,
        "execution_failures_remain_in_denominator": True,
    }
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "common_qc_cases.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = sorted({field for row in decisions for field in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(decisions)
    (args.output_dir / "common_qc_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
