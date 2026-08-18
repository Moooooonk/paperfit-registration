#!/usr/bin/env python3
"""Select the frozen ARAP setting from equal-denominator development results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import apply_frozen_common_qc as common
except ModuleNotFoundError:  # Imported as tools.select_arap_development_setting.
    from tools import apply_frozen_common_qc as common


SETTING_ORDER = ("ARAP-A", "ARAP-B", "ARAP-C", "ARAP-D")
EXPRESSIONS_PER_SUBJECT = 19
EXPECTED_EXPRESSION_INDICES = {*range(1, 18), 19, 20}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions-csv", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.resolve(strict=True).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} for {row.get('case', '<unknown>')}") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"Non-finite or negative {field}: {value}")
    return value


def completed(row: dict[str, str]) -> bool:
    value = row.get("completed")
    return True if value in (None, "") else int(float(value)) == 1


def summarize_settings(
    rows: list[dict[str, str]], development_subjects: set[str]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["setting_id"])].append(row)
    if set(grouped) != set(SETTING_ORDER):
        raise ValueError(
            f"Expected settings {SETTING_ORDER}, found {tuple(sorted(grouped))}"
        )

    expected_cases = len(development_subjects) * EXPRESSIONS_PER_SUBJECT
    reference_cases: set[str] | None = None
    summaries: list[dict[str, Any]] = []
    for setting_id in SETTING_ORDER:
        values = grouped[setting_id]
        cases = [str(row["case"]) for row in values]
        if len(cases) != len(set(cases)) or len(cases) != expected_cases:
            raise ValueError(
                f"{setting_id} must contain {expected_cases} unique cases; "
                f"found {len(cases)}/{len(set(cases))}"
            )
        case_set = set(cases)
        if reference_cases is None:
            reference_cases = case_set
        elif case_set != reference_cases:
            raise ValueError(f"Case denominator differs for {setting_id}")

        by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in values:
            subject = f"{int(row['subject']):03d}"
            case_subject = f"{int(str(row['case']).split('_', 1)[0]):03d}"
            if subject != case_subject:
                raise ValueError(f"Case/subject mismatch in {setting_id}: {row['case']}")
            accepted = int(float(row["final_accepted"]))
            if accepted not in (0, 1):
                raise ValueError(f"Non-binary final_accepted in {setting_id}")
            by_subject[subject].append(row)
        counts = Counter({subject: len(items) for subject, items in by_subject.items()})
        if set(by_subject) != development_subjects or set(counts.values()) != {
            EXPRESSIONS_PER_SUBJECT
        }:
            raise ValueError(
                f"{setting_id} does not match the frozen development partition: "
                f"{dict(sorted(counts.items()))}"
            )
        for subject, items in sorted(by_subject.items()):
            try:
                indices = [
                    int(str(row["case"]).split("_", 2)[1]) for row in items
                ]
            except (IndexError, ValueError) as error:
                raise ValueError(
                    f"{setting_id} contains a case without a numeric expression index"
                ) from error
            if len(indices) != len(set(indices)):
                raise ValueError(
                    f"{setting_id} duplicates an expression index for {subject}"
                )
            if set(indices) != EXPECTED_EXPRESSION_INDICES:
                raise ValueError(
                    f"{setting_id} has an incomplete expression grid for {subject}: "
                    f"{sorted(indices)}"
                )

        subject_acceptance = []
        subject_edge = []
        subject_displacement = []
        completed_pairs = 0
        for subject in sorted(by_subject):
            items = by_subject[subject]
            subject_acceptance.append(
                sum(int(float(row["final_accepted"])) for row in items) / len(items)
            )
            completed_items = [row for row in items if completed(row)]
            completed_pairs += len(completed_items)
            if completed_items:
                subject_edge.append(
                    sum(finite(row, "edge_strain_p99") for row in completed_items)
                    / len(completed_items)
                )
                subject_displacement.append(
                    sum(
                        finite(row, "displacement_p90_mm")
                        for row in completed_items
                    )
                    / len(completed_items)
                )
        if not subject_edge or not subject_displacement:
            raise ValueError(f"{setting_id} has no completed ARAP metric evidence")
        summaries.append(
            {
                "setting_id": setting_id,
                "attempted_pairs": len(values),
                "completed_pairs": completed_pairs,
                "invalid_evidence_pairs": len(values) - completed_pairs,
                "subjects": len(by_subject),
                "subjects_with_completed_metrics": len(subject_edge),
                "accepted_pairs": sum(
                    int(float(row["final_accepted"])) for row in values
                ),
                "mean_subject_acceptance": sum(subject_acceptance)
                / len(subject_acceptance),
                "mean_subject_edge_strain_p99": sum(subject_edge) / len(subject_edge),
                "mean_subject_displacement_p90_mm": sum(subject_displacement)
                / len(subject_displacement),
                "capacity_order": SETTING_ORDER.index(setting_id),
            }
        )
    return summaries


def select_setting(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if {str(row["setting_id"]) for row in summaries} != set(SETTING_ORDER):
        raise ValueError("ARAP summaries are incomplete")
    return min(
        summaries,
        key=lambda row: (
            -float(row["mean_subject_acceptance"]),
            float(row["mean_subject_edge_strain_p99"]),
            float(row["mean_subject_displacement_p90_mm"]),
            int(row["capacity_order"]),
        ),
    )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    rows = read_csv(args.decisions_csv)
    required = {
        "case",
        "subject",
        "setting_id",
        "final_accepted",
        "completed",
        "edge_strain_p99",
        "displacement_p90_mm",
    }
    if not rows or not required.issubset(rows[0]):
        raise KeyError(f"Missing fields: {required - set(rows[0] if rows else ())}")
    split_path = args.split_json.resolve(strict=True)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    development = {f"{int(subject):03d}" for subject in split["development_subjects"]}
    if len(development) != 10:
        raise ValueError("Expected exactly 10 frozen development subjects")
    calibration_path = args.calibration_json.resolve(strict=True)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    common.validate_calibration_artifact(
        calibration, split, common.sha256(split_path)
    )

    summaries = summarize_settings(rows, development)
    selected = select_setting(summaries)
    payload = {
        "development_only": True,
        "heldout_inspected": False,
        "selection_rule": (
            "highest mean subject-level acceptance; ties use lower mean subject-level "
            "p99 edge strain, lower mean subject-level p90 displacement, then lower "
            "capacity in the prespecified ARAP-A/B/C/D order"
        ),
        "selected_setting_id": selected["setting_id"],
        "selected_summary": selected,
        "candidate_summaries": summaries,
        "decisions_csv_sha256": sha256(args.decisions_csv.resolve(strict=True)),
        "split_sha256": sha256(split_path),
        "common_qc_calibration_sha256": sha256(calibration_path),
    }
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "arap_development_setting_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    (args.output_dir / "arap_selected_setting.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
