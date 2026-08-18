#!/usr/bin/env python3
"""LOSO development-only calibration of the frozen common final-QC grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


# Frozen before rating labels were collected. The round values span the
# observed HRN development distribution instead of inheriting target-unit gates.
FULL_MEDIAN = (1.5, 2.0, 2.5, 3.0, 4.0, 6.0)
FULL_P90 = (6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0)
NOSE_MEDIAN = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
NOSE_P90 = (1.5, 2.0, 3.0, 4.0, 6.0, 10.0)
ANCHOR = (5.0, 10.0, 15.0, 20.0, 30.0)
EDGE_P99 = (0.10, 0.15, 0.20, 0.25, 0.35)
EYE_TOLERANCE_MM = 1e-6
MEAN_SUBJECT_PPV_TARGET = 0.90
MIN_SUBJECT_PPV_TARGET = 0.80
MIN_ACCEPTED_SUBJECT_FRACTION = 0.70
Z_95 = 1.959963984540054
EXPRESSIONS_PER_SUBJECT = 19
EXPECTED_EXPRESSION_INDICES = {*range(1, 18), 19, 20}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--ratings-csv", type=Path, required=True)
    parser.add_argument("--rating-summary-json", type=Path, required=True)
    parser.add_argument("--rater-metadata-csv", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.resolve(strict=True).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_rating_provenance(
    summary: dict[str, Any], ratings_path: Path, metadata_path: Path, cases: int
) -> None:
    if summary.get("qualified_rater_metadata_complete") is not True:
        raise ValueError("Qualified rater metadata is not complete")
    if int(summary.get("raters", 0)) not in (2, 3):
        raise ValueError("Rating summary must contain two or three raters")
    if int(summary.get("primary_independent_raters", 0)) < 2:
        raise ValueError("Rating summary must contain at least two full independent raters")
    qualification = summary.get("rater_qualification_summary")
    if not isinstance(qualification, list) or len(qualification) != int(
        summary.get("raters", 0)
    ):
        raise ValueError("Rating summary lacks the complete rater qualification record")
    if any(
        str(row.get("registration_method_development_involvement", "")).strip().lower()
        not in {"no", "n", "0", "false"}
        for row in qualification
    ):
        raise ValueError("A consensus rater was involved in method development")
    if int(summary.get("cases", -1)) != cases or int(
        summary.get("unresolved_cases", -1)
    ) != 0:
        raise ValueError("Rating summary is incomplete or unresolved")
    if summary.get("consensus_ratings_sha256") != sha256(ratings_path):
        raise ValueError("Consensus rating hash does not match the rating summary")
    if summary.get("rater_metadata_sha256") != sha256(metadata_path):
        raise ValueError("Rater metadata hash does not match the rating summary")


def validate_development_rating_partition(
    summary: dict[str, Any], split_path: Path, development: set[str]
) -> None:
    if (
        summary.get("subset") != "development"
        or summary.get("development_only") is not True
        or summary.get("heldout_inspected") is not False
    ):
        raise ValueError("Rating summary is not a development-only artifact")
    if summary.get("split_sha256") != sha256(split_path):
        raise ValueError("Rating summary split hash does not match calibration split")
    subjects = {f"{int(value):03d}" for value in summary.get("partition_subjects", [])}
    if subjects != development:
        raise ValueError("Rating summary subjects do not match development split")


def completed(row: dict[str, str]) -> bool:
    value = row.get("completed")
    if value in (None, ""):
        return True
    numeric = float(value)
    if (
        not math.isfinite(numeric)
        or not numeric.is_integer()
        or int(numeric) not in (0, 1)
    ):
        raise ValueError(f"completed must be binary, found {value!r}")
    return int(numeric) == 1


def validate_development_case_rows(
    rows: list[dict[str, str]], development: set[str]
) -> None:
    cases = [str(row["case"]) for row in rows]
    if len(cases) != len(set(cases)):
        raise ValueError("Duplicate metric cases")
    counts: Counter[str] = Counter()
    for row in rows:
        subject = f"{int(row['subject']):03d}"
        case_subject = f"{int(str(row['case']).split('_', 1)[0]):03d}"
        if subject != case_subject:
            raise ValueError(
                f"Case/subject mismatch: case={row['case']}, subject={subject}"
            )
        counts[subject] += 1
    expected_cases = len(development) * EXPRESSIONS_PER_SUBJECT
    bad_counts = {
        subject: counts.get(subject, 0)
        for subject in sorted(development | set(counts))
        if counts.get(subject, 0) != EXPRESSIONS_PER_SUBJECT
    }
    if set(counts) != development or len(rows) != expected_cases or bad_counts:
        raise ValueError(
            "Calibration rows do not match the frozen development split: "
            f"cases={len(rows)}/{expected_cases}, per_subject_counts={bad_counts}"
        )
    indices_by_subject: dict[str, list[int]] = {
        subject: [] for subject in development
    }
    for row in rows:
        subject = f"{int(row['subject']):03d}"
        try:
            expression_index = int(str(row["case"]).split("_", 2)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(
                f"Case lacks a numeric expression index: {row['case']}"
            ) from error
        indices_by_subject[subject].append(expression_index)
    for subject, indices in sorted(indices_by_subject.items()):
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate expression index for subject {subject}")
        if set(indices) != EXPECTED_EXPRESSION_INDICES:
            raise ValueError(
                f"Incomplete expression grid for subject {subject}: "
                f"observed={sorted(indices)}"
            )


def wilson_lower(successes: int, total: int) -> float:
    if total <= 0:
        return 0.0
    proportion = successes / total
    z2 = Z_95 * Z_95
    center = proportion + z2 / (2.0 * total)
    radius = Z_95 * math.sqrt(
        proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total)
    )
    return (center - radius) / (1.0 + z2 / total)


def confusion(predicted: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    tp = int(np.sum(predicted & labels))
    fp = int(np.sum(predicted & ~labels))
    tn = int(np.sum(~predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    accepted = tp + fp
    positives = tp + fn
    negatives = tn + fp
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accepted": accepted,
        "coverage": accepted / len(labels),
        "ppv": tp / accepted if accepted else None,
        "ppv_wilson_lower_95": wilson_lower(tp, accepted),
        "sensitivity": tp / positives if positives else None,
        "specificity": tn / negatives if negatives else None,
    }


def threshold_index_sum(rule: tuple[float, ...]) -> int:
    grids = (FULL_MEDIAN, FULL_P90, NOSE_MEDIAN, NOSE_P90, ANCHOR, EDGE_P99)
    return sum(grid.index(value) for grid, value in zip(grids, rule))


def subject_precision(
    predicted: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    ppv_values = []
    subject_count = 0
    accepted_subjects = 0
    for subject in sorted(set(subjects[mask])):
        subject_count += 1
        subject_mask = mask & (subjects == subject)
        accepted = int(np.sum(predicted[subject_mask]))
        if accepted == 0:
            continue
        accepted_subjects += 1
        true_accepted = int(np.sum(predicted[subject_mask] & labels[subject_mask]))
        ppv_values.append(true_accepted / accepted)
    return {
        "subject_count": subject_count,
        "accepted_subjects": accepted_subjects,
        "accepted_subject_fraction": accepted_subjects / subject_count,
        "mean_subject_ppv": float(np.mean(ppv_values)) if ppv_values else 0.0,
        "min_subject_ppv": float(np.min(ppv_values)) if ppv_values else 0.0,
    }


def predict(matrix: dict[str, np.ndarray], rule: tuple[float, ...]) -> np.ndarray:
    full_median, full_p90, nose_median, nose_p90, anchor, edge_p99 = rule
    return (
        (matrix["completed"] == 1)
        & (matrix["post_orientation_pass"] == 1)
        & (matrix["eye_fixed_max_mm"] <= EYE_TOLERANCE_MM)
        & (matrix["post_full_median_mm"] <= full_median)
        & (matrix["post_full_p90_mm"] <= full_p90)
        & (matrix["post_nose_median_mm"] <= nose_median)
        & (matrix["post_nose_p90_mm"] <= nose_p90)
        & (matrix["post_anchor_mm"] <= anchor)
        & (matrix["edge_strain_p99"] <= edge_p99)
    )


def select_rule(
    matrix: dict[str, np.ndarray],
    labels: np.ndarray,
    subjects: np.ndarray,
    mask: np.ndarray,
) -> tuple[tuple[float, ...], dict[str, Any], bool]:
    candidates = []
    for rule in itertools.product(
        FULL_MEDIAN, FULL_P90, NOSE_MEDIAN, NOSE_P90, ANCHOR, EDGE_P99
    ):
        all_predictions = predict(matrix, rule)
        stats = {
            **confusion(all_predictions[mask], labels[mask]),
            **subject_precision(all_predictions, labels, subjects, mask),
        }
        candidates.append((rule, stats))
    qualified = [
        item
        for item in candidates
        if item[1]["mean_subject_ppv"] >= MEAN_SUBJECT_PPV_TARGET
        and item[1]["min_subject_ppv"] >= MIN_SUBJECT_PPV_TARGET
        and item[1]["accepted_subject_fraction"] >= MIN_ACCEPTED_SUBJECT_FRACTION
    ]
    target_reached = bool(qualified)
    pool = qualified if qualified else candidates

    def key(item: tuple[tuple[float, ...], dict[str, Any]]) -> tuple[float, ...]:
        rule, stats = item
        sensitivity = -1.0 if stats["sensitivity"] is None else stats["sensitivity"]
        if target_reached:
            return (
                stats["coverage"],
                -stats["fp"],
                sensitivity,
                -threshold_index_sum(rule),
            )
        return (
            stats["min_subject_ppv"],
            stats["mean_subject_ppv"],
            stats["accepted_subject_fraction"],
            stats["coverage"],
            -stats["fp"],
            sensitivity,
            -threshold_index_sum(rule),
        )

    selected_rule, selected_stats = max(pool, key=key)
    return selected_rule, selected_stats, target_reached


def rule_dict(rule: tuple[float, ...]) -> dict[str, float]:
    names = (
        "full_median_mm",
        "full_p90_mm",
        "nose_median_mm",
        "nose_p90_mm",
        "anchor_mm",
        "edge_strain_p99",
    )
    return dict(zip(names, rule))


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    metric_rows = read_csv(args.metrics_csv)
    rating_rows = read_csv(args.ratings_csv)
    rating_summary_path = args.rating_summary_json.resolve(strict=True)
    metadata_path = args.rater_metadata_csv.resolve(strict=True)
    rating_summary = json.loads(rating_summary_path.read_text(encoding="utf-8"))
    required_metrics = {
        "case",
        "subject",
        "source_method",
        "completed",
        "post_orientation_pass",
        "eye_fixed_max_mm",
        "post_full_median_mm",
        "post_full_p90_mm",
        "post_nose_median_mm",
        "post_nose_p90_mm",
        "post_anchor_mm",
        "edge_strain_p99",
    }
    if not metric_rows or not required_metrics.issubset(metric_rows[0]):
        raise KeyError(f"Missing metric fields: {required_metrics - set(metric_rows[0])}")
    observed_methods = {str(row["source_method"]).strip().lower() for row in metric_rows}
    if observed_methods != {"hrn"}:
        raise ValueError(
            "Common-QC calibration is frozen to HRN development outputs only; "
            f"observed source methods: {sorted(observed_methods)}"
        )
    required_ratings = {"case", "consensus_resolved", "consensus_usable"}
    if not rating_rows or not required_ratings.issubset(rating_rows[0]):
        raise KeyError(f"Missing rating fields: {required_ratings - set(rating_rows[0])}")
    unresolved_cases = [
        row["case"]
        for row in rating_rows
        if int(float(row["consensus_resolved"])) != 1
    ]
    if unresolved_cases:
        raise ValueError(
            "Development calibration requires adjudicated consensus for every case; "
            f"unresolved cases: {len(unresolved_cases)}"
        )
    ratings = {row["case"]: int(float(row["consensus_usable"])) for row in rating_rows}
    if len(ratings) != len(rating_rows):
        raise ValueError("Duplicate rating cases")
    if {row["case"] for row in metric_rows} != set(ratings):
        raise ValueError("Metric and rating case sets differ")
    if any(value not in (0, 1) for value in ratings.values()):
        raise ValueError("consensus_usable must be binary")

    split_path = args.split_json.resolve(strict=True)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    development = {
        f"{int(value):03d}" for value in split["development_subjects"]
    }
    validate_rating_provenance(
        rating_summary,
        args.ratings_csv.resolve(strict=True),
        metadata_path,
        len(rating_rows),
    )
    validate_development_rating_partition(
        rating_summary, split_path, development
    )
    validate_development_case_rows(metric_rows, development)
    subjects = np.asarray([f"{int(row['subject']):03d}" for row in metric_rows])
    if set(subjects) != development:
        raise ValueError(
            "Calibration input must contain exactly the frozen development identities"
        )
    cases = np.asarray([row["case"] for row in metric_rows])
    labels = np.asarray([bool(ratings[case]) for case in cases], dtype=bool)
    completed_mask = np.asarray(
        [completed(row) for row in metric_rows], dtype=bool
    )
    matrix: dict[str, np.ndarray] = {
        "completed": completed_mask.astype(np.float64)
    }
    for key in required_metrics - {"case", "subject", "source_method", "completed"}:
        values = []
        for row, is_completed in zip(metric_rows, completed_mask):
            raw = row.get(key, "")
            if raw in (None, ""):
                if is_completed:
                    raise ValueError(
                        f"Completed case {row['case']} is missing metric {key}"
                    )
                values.append(float("nan"))
            else:
                values.append(float(raw))
        matrix[key] = np.asarray(values, dtype=np.float64)
    nonfinite_fields = [
        key
        for key, values in matrix.items()
        if key != "completed" and not np.all(np.isfinite(values[completed_mask]))
    ]
    negative_fields = [
        key
        for key, values in matrix.items()
        if key not in {"completed", "post_orientation_pass"}
        and np.any(values[completed_mask] < 0.0)
    ]
    if nonfinite_fields or negative_fields:
        raise ValueError(
            "Calibration metrics require finite nonnegative evidence: "
            f"nonfinite={nonfinite_fields}, negative={negative_fields}"
        )
    if not np.all(
        np.isin(matrix["post_orientation_pass"][completed_mask], (0.0, 1.0))
    ):
        raise ValueError("post_orientation_pass must be binary")

    fold_rows = []
    out_of_fold = []
    for held_subject in sorted(development):
        train_mask = subjects != held_subject
        validation_mask = subjects == held_subject
        rule, train_stats, target_reached = select_rule(
            matrix, labels, subjects, train_mask
        )
        validation_prediction = predict(matrix, rule)[validation_mask]
        validation_stats = confusion(validation_prediction, labels[validation_mask])
        fold_rows.append(
            {
                "held_out_development_subject": held_subject,
                **rule_dict(rule),
                "training_target_reached": int(target_reached),
                "training_mean_subject_ppv": train_stats["mean_subject_ppv"],
                "training_min_subject_ppv": train_stats["min_subject_ppv"],
                "training_accepted_subject_fraction": train_stats[
                    "accepted_subject_fraction"
                ],
                "validation_coverage": validation_stats["coverage"],
                "validation_ppv": validation_stats["ppv"],
                "validation_sensitivity": validation_stats["sensitivity"],
                "validation_specificity": validation_stats["specificity"],
            }
        )
        for case, label, prediction in zip(
            cases[validation_mask], labels[validation_mask], validation_prediction
        ):
            out_of_fold.append(
                {
                    "case": str(case),
                    "subject": held_subject,
                    "consensus_usable": int(label),
                    "predicted_accepted": int(prediction),
                }
            )

    full_mask = np.ones(len(labels), dtype=bool)
    final_rule, final_stats, final_target_reached = select_rule(
        matrix, labels, subjects, full_mask
    )
    oof_labels = np.asarray([bool(row["consensus_usable"]) for row in out_of_fold])
    oof_predictions = np.asarray([bool(row["predicted_accepted"]) for row in out_of_fold])
    summary = {
        "development_only": True,
        "heldout_inspected": False,
        "calibration_source_method": "hrn",
        "subjects": sorted(development),
        "cases": len(metric_rows),
        "completed_cases": int(np.sum(completed_mask)),
        "invalid_evidence_cases": int(np.sum(~completed_mask)),
        "invalid_evidence_policy": (
            "Retained in every denominator and predicted rejected by every rule; "
            "missing post-S8 metrics were not imputed."
        ),
        "candidate_grid": {
            "full_median_mm": FULL_MEDIAN,
            "full_p90_mm": FULL_P90,
            "nose_median_mm": NOSE_MEDIAN,
            "nose_p90_mm": NOSE_P90,
            "anchor_mm": ANCHOR,
            "edge_strain_p99": EDGE_P99,
        },
        "mean_subject_ppv_target": MEAN_SUBJECT_PPV_TARGET,
        "min_subject_ppv_target": MIN_SUBJECT_PPV_TARGET,
        "min_accepted_subject_fraction": MIN_ACCEPTED_SUBJECT_FRACTION,
        "eye_tolerance_mm": EYE_TOLERANCE_MM,
        "loso_out_of_fold": confusion(oof_predictions, oof_labels),
        "loso_out_of_fold_subject_precision": subject_precision(
            oof_predictions,
            oof_labels,
            np.asarray([str(row["subject"]) for row in out_of_fold]),
            np.ones(len(out_of_fold), dtype=bool),
        ),
        "loso_training_folds_reaching_target": sum(
            int(row["training_target_reached"]) for row in fold_rows
        ),
        "loso_training_fold_count": len(fold_rows),
        "final_candidate_rule": rule_dict(final_rule),
        "final_frozen_rule": (
            rule_dict(final_rule) if final_target_reached else None
        ),
        "final_development_fit": final_stats,
        "final_target_reached": final_target_reached,
        "common_qc_frozen": bool(final_target_reached),
        "freeze_policy": (
            "The all-development refit is eligible for freezing only when the "
            "prespecified subject-level PPV and accepted-subject constraints are met. "
            "Otherwise the candidate remains diagnostic and downstream application "
            "must stop."
        ),
        "heldout_test_inspected": False,
        "split_sha256": sha256(split_path),
        "rating_summary_sha256": sha256(rating_summary_path),
        "consensus_ratings_sha256": sha256(args.ratings_csv.resolve(strict=True)),
        "rater_metadata_sha256": sha256(metadata_path),
    }

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "loso_folds.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fold_rows)
    with (args.output_dir / "loso_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_of_fold[0].keys()))
        writer.writeheader()
        writer.writerows(out_of_fold)
    (args.output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
