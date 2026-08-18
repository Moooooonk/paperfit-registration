#!/usr/bin/env python3
"""Compare frozen automatic QC with blinded anatomical consensus by subject."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.subject_anonymization import build_anonymized_subject_labels
except ModuleNotFoundError:
    from subject_anonymization import build_anonymized_subject_labels


METRIC_NAMES = (
    "accuracy",
    "ppv",
    "npv",
    "sensitivity",
    "specificity",
    "balanced_accuracy",
    "cohen_kappa",
)
EXPRESSIONS_PER_SUBJECT = 19
EXPECTED_EXPRESSION_INDICES = {*range(1, 18), 19, 20}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-csv", type=Path, required=True)
    parser.add_argument("--ratings-csv", type=Path, required=True)
    parser.add_argument("--rating-summary-json", type=Path, required=True)
    parser.add_argument("--rater-metadata-csv", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--partition", choices=("development", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def read(path: Path) -> list[dict[str, str]]:
    with path.resolve(strict=True).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_rating_provenance(
    summary: dict[str, Any],
    ratings_path: Path,
    metadata_path: Path,
    split_path: Path,
    partition: str,
    selected_subjects: set[str],
) -> None:
    expected_subset = "development" if partition == "development" else "heldout"
    expected_development_only = partition == "development"
    expected_heldout_inspected = partition == "test"
    summary_subjects = {
        f"{int(subject):03d}" for subject in summary.get("partition_subjects", [])
    }
    expected_cases = len(selected_subjects) * EXPRESSIONS_PER_SUBJECT
    if (
        summary.get("subset") != expected_subset
        or summary.get("development_only") is not expected_development_only
        or summary.get("heldout_inspected") is not expected_heldout_inspected
        or summary_subjects != selected_subjects
        or int(summary.get("cases", -1)) != expected_cases
        or int(summary.get("unresolved_cases", -1)) != 0
    ):
        raise ValueError("Rating summary does not match the selected frozen partition")
    if summary.get("qualified_rater_metadata_complete") is not True or int(
        summary.get("raters", 0)
    ) not in (2, 3):
        raise ValueError("Rating summary lacks complete qualified-rater evidence")
    if int(summary.get("primary_independent_raters", 0)) < 2:
        raise ValueError("Rating summary lacks two full independent raters")
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
    expected_hashes = {
        "consensus_ratings_sha256": sha256(ratings_path),
        "rater_metadata_sha256": sha256(metadata_path),
        "split_sha256": sha256(split_path),
    }
    mismatches = {
        key: {"recorded": summary.get(key), "observed": value}
        for key, value in expected_hashes.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Rating provenance hash mismatch: {mismatches}")


def divide(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def kappa(predicted: np.ndarray, observed: np.ndarray) -> float | None:
    if len(predicted) == 0:
        return None
    agreement = float(np.mean(predicted == observed))
    p_pred = float(np.mean(predicted))
    p_obs = float(np.mean(observed))
    expected = p_pred * p_obs + (1.0 - p_pred) * (1.0 - p_obs)
    if abs(1.0 - expected) <= 1e-12:
        return 1.0 if agreement >= 1.0 - 1e-12 else None
    return (agreement - expected) / (1.0 - expected)


def confusion(predicted: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    predicted = np.asarray(predicted, dtype=bool)
    observed = np.asarray(observed, dtype=bool)
    tp = int(np.sum(predicted & observed))
    fp = int(np.sum(predicted & ~observed))
    tn = int(np.sum(~predicted & ~observed))
    fn = int(np.sum(~predicted & observed))
    sensitivity = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    return {
        "cases": len(predicted),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": divide(tp + tn, len(predicted)),
        "ppv": divide(tp, tp + fp),
        "npv": divide(tn, tn + fn),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (
            (sensitivity + specificity) / 2.0
            if sensitivity is not None and specificity is not None
            else None
        ),
        "cohen_kappa": kappa(predicted, observed),
    }


def validate_complete_partition(
    rows: list[dict[str, Any]], selected_subjects: set[str]
) -> None:
    counts = Counter(str(row["subject"]) for row in rows)
    bad_counts = {
        subject: counts.get(subject, 0)
        for subject in sorted(selected_subjects | set(counts))
        if counts.get(subject, 0) != EXPRESSIONS_PER_SUBJECT
    }
    if set(counts) != selected_subjects or bad_counts:
        raise ValueError(
            "Independent rating analysis does not contain the complete frozen "
            f"partition: per_subject_counts={bad_counts}"
        )
    by_subject: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        try:
            expression_index = int(str(row["case"]).split("_", 2)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(
                f"Rating case lacks a numeric expression index: {row['case']}"
            ) from error
        by_subject[str(row["subject"])].append(expression_index)
    for subject, indices in sorted(by_subject.items()):
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate expression index for subject {subject}")
        if set(indices) != EXPECTED_EXPRESSION_INDICES:
            raise ValueError(
                f"Incomplete expression grid for subject {subject}: "
                f"observed={sorted(indices)}"
            )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    if args.bootstrap_repetitions < 1000:
        raise ValueError("At least 1000 bootstrap repetitions are required")
    decision_path = args.decision_csv.resolve(strict=True)
    ratings_path = args.ratings_csv.resolve(strict=True)
    summary_path = args.rating_summary_json.resolve(strict=True)
    metadata_path = args.rater_metadata_csv.resolve(strict=True)
    split_path = args.split_json.resolve(strict=True)
    decisions = read(decision_path)
    ratings = read(ratings_path)
    required_decision = {"case", "subject", "final_accepted"}
    required_rating = {"case", "consensus_resolved", "consensus_usable"}
    if not decisions or not required_decision.issubset(decisions[0]):
        raise KeyError(f"Decision fields missing: {required_decision - set(decisions[0])}")
    if not ratings or not required_rating.issubset(ratings[0]):
        raise KeyError(f"Rating fields missing: {required_rating - set(ratings[0])}")
    decision_by_case = {row["case"]: row for row in decisions}
    rating_by_case = {row["case"]: row for row in ratings}
    if len(decision_by_case) != len(decisions) or len(rating_by_case) != len(ratings):
        raise ValueError("Duplicate case in decisions or ratings")
    if set(decision_by_case) != set(rating_by_case):
        raise ValueError("Decision and rating case sets differ")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    selected_subjects = {
        f"{int(subject):03d}" for subject in split[f"{args.partition}_subjects"]
    }
    anonymized = build_anonymized_subject_labels(split)
    rating_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validate_rating_provenance(
        rating_summary,
        ratings_path,
        metadata_path,
        split_path,
        args.partition,
        selected_subjects,
    )

    joined = []
    unresolved_cases = []
    for case in sorted(decision_by_case):
        decision = decision_by_case[case]
        subject = f"{int(decision['subject']):03d}"
        case_subject = f"{int(case.split('_', 1)[0]):03d}"
        if subject != case_subject:
            raise ValueError(f"Case/subject mismatch: {case}/{subject}")
        if subject not in selected_subjects:
            continue
        rating = rating_by_case[case]
        if int(float(rating["consensus_resolved"])) != 1:
            unresolved_cases.append(case)
            continue
        automatic_accepted = int(float(decision["final_accepted"]))
        consensus_usable = int(float(rating["consensus_usable"]))
        if automatic_accepted not in (0, 1) or consensus_usable not in (0, 1):
            raise ValueError(f"Non-binary decision or rating for {case}")
        joined.append(
            {
                "case": case,
                "subject": subject,
                "automatic_accepted": automatic_accepted,
                "consensus_usable": consensus_usable,
            }
        )
    if unresolved_cases:
        raise ValueError(
            "Independent anatomical validation requires resolved consensus for "
            f"every selected case; unresolved={len(unresolved_cases)}, "
            f"examples={unresolved_cases[:5]}"
        )
    validate_complete_partition(joined, selected_subjects)

    predicted = np.asarray([row["automatic_accepted"] for row in joined], dtype=bool)
    observed = np.asarray([row["consensus_usable"] for row in joined], dtype=bool)
    overall = confusion(predicted, observed)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        grouped[row["subject"]].append(row)
    subject_rows = []
    for subject in sorted(grouped):
        rows = grouped[subject]
        subject_stats = confusion(
            np.asarray([row["automatic_accepted"] for row in rows]),
            np.asarray([row["consensus_usable"] for row in rows]),
        )
        subject_rows.append(
            {"anonymized_subject": anonymized[subject], **subject_stats}
        )

    subjects = sorted(grouped)
    rng = np.random.default_rng(args.seed)
    bootstrap_values: dict[str, list[float]] = defaultdict(list)
    for _ in range(args.bootstrap_repetitions):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        rows = [row for subject in sampled for row in grouped[str(subject)]]
        stats = confusion(
            np.asarray([row["automatic_accepted"] for row in rows]),
            np.asarray([row["consensus_usable"] for row in rows]),
        )
        for name in METRIC_NAMES:
            if stats[name] is not None and np.isfinite(float(stats[name])):
                bootstrap_values[name].append(float(stats[name]))
    intervals = {}
    for name in METRIC_NAMES:
        values = np.asarray(bootstrap_values[name], dtype=float)
        intervals[name] = (
            [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
            if len(values)
            else [None, None]
        )

    summary = {
        "partition": args.partition,
        "subjects": len(subjects),
        "resolved_cases": len(joined),
        "unresolved_cases": 0,
        "automatic_vs_consensus": overall,
        "subject_cluster_bootstrap_95_ci": intervals,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.seed,
        "decision_sha256": sha256(decision_path),
        "consensus_ratings_sha256": sha256(ratings_path),
        "rating_summary_sha256": sha256(summary_path),
        "rater_metadata_sha256": sha256(metadata_path),
        "split_sha256": sha256(split_path),
        "interpretation": (
            "Confidence intervals resample whole subjects. Every selected case "
            "has resolved blinded consensus; unresolved cases are not excluded."
        ),
    }
    subject_rows.sort(key=lambda row: str(row["anonymized_subject"]))
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "qc_rating_joined_cases.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        public_joined = []
        for row in joined:
            expression_index = int(str(row["case"]).split("_", 2)[1])
            public_joined.append(
                {
                    "anonymized_case": (
                        f"{anonymized[str(row['subject'])]}_E{expression_index:02d}"
                    ),
                    "anonymized_subject": anonymized[str(row["subject"])],
                    "expression_index": expression_index,
                    "automatic_accepted": row["automatic_accepted"],
                    "consensus_usable": row["consensus_usable"],
                }
            )
        public_joined.sort(key=lambda row: str(row["anonymized_case"]))
        writer = csv.DictWriter(handle, fieldnames=list(public_joined[0].keys()))
        writer.writeheader()
        writer.writerows(public_joined)
    with (args.output_dir / "qc_rating_subject_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(subject_rows[0].keys()))
        writer.writeheader()
        writer.writerows(subject_rows)
    with (args.output_dir / "case_id_key_internal_do_not_publish.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("anonymized_case", "internal_case_id")
        )
        writer.writeheader()
        writer.writerows(
            {
                "anonymized_case": (
                    f"{anonymized[str(row['subject'])]}_"
                    f"E{int(str(row['case']).split('_', 2)[1]):02d}"
                ),
                "internal_case_id": row["case"],
            }
            for row in joined
        )
    (args.output_dir / "qc_rating_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
