#!/usr/bin/env python3
"""Summarize continuous metrics with subjects as the independent unit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.subject_anonymization import build_anonymized_subject_labels
except ModuleNotFoundError:
    from subject_anonymization import build_anonymized_subject_labels


EXPRESSIONS_PER_SUBJECT = 19
EXPECTED_EXPRESSION_INDICES = {*range(1, 18), 19, 20}
BOOTSTRAP_REPETITIONS = 20_000
BOOTSTRAP_SEED = 20260816


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument(
        "--partition", choices=("development", "test", "full"), required=True
    )
    parser.add_argument("--metric", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-label", required=True)
    parser.add_argument("--completed-column", default="completed")
    parser.add_argument("--filter-column")
    parser.add_argument("--filter-value", default="1")
    parser.add_argument("--bootstrap-repetitions", type=int, default=BOOTSTRAP_REPETITIONS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.resolve(strict=True).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def completed(row: dict[str, str], column: str) -> bool:
    raw = row.get(column)
    return True if raw in (None, "") else int(float(raw)) == 1


def bootstrap_subject_mean(
    subject_means: list[float], repetitions: int, seed: int
) -> tuple[float | None, float | None]:
    if not subject_means:
        return None, None
    values = np.asarray(subject_means, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    samples = values[indices].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summarize_metric(
    rows: list[dict[str, Any]],
    metric: str,
    subjects: list[str],
    repetitions: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["subject"])].append(float(row[metric]))
    subject_rows: list[dict[str, Any]] = []
    subject_means: list[float] = []
    for subject in subjects:
        values = np.asarray(grouped.get(subject, []), dtype=np.float64)
        if len(values):
            mean = float(np.mean(values))
            median = float(np.median(values))
            subject_means.append(mean)
        else:
            mean = None
            median = None
        subject_rows.append(
            {
                "subject": subject,
                "metric": metric,
                "available_pairs": int(len(values)),
                "subject_mean": mean,
                "subject_median": median,
            }
        )
    all_values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    low, high = bootstrap_subject_mean(subject_means, repetitions, seed)
    summary = {
        "metric": metric,
        "available_pairs": int(len(all_values)),
        "subjects_with_values": len(subject_means),
        "subjects_without_values": len(subjects) - len(subject_means),
        "pair_mean": float(np.mean(all_values)) if len(all_values) else None,
        "pair_median": float(np.median(all_values)) if len(all_values) else None,
        "pair_standard_deviation": (
            float(np.std(all_values, ddof=1)) if len(all_values) > 1 else None
        ),
        "pair_iqr": (
            [
                float(np.quantile(all_values, 0.25)),
                float(np.quantile(all_values, 0.75)),
            ]
            if len(all_values)
            else [None, None]
        ),
        "mean_subject_mean": (
            float(np.mean(subject_means)) if subject_means else None
        ),
        "subject_mean_standard_deviation": (
            float(np.std(subject_means, ddof=1)) if len(subject_means) > 1 else None
        ),
        "subject_mean_range": (
            [float(np.min(subject_means)), float(np.max(subject_means))]
            if subject_means
            else [None, None]
        ),
        "subject_cluster_bootstrap_95_ci_for_mean_subject_mean": [low, high],
    }
    return summary, subject_rows


def selected_subjects(split: dict[str, Any], partition: str) -> list[str]:
    development = {f"{int(value):03d}" for value in split["development_subjects"]}
    test = {f"{int(value):03d}" for value in split["test_subjects"]}
    if partition == "development":
        return sorted(development)
    if partition == "test":
        return sorted(test)
    return sorted(development | test)


def validate_expression_grid(rows: list[dict[str, str]], subjects: list[str]) -> None:
    """Require one consistent row for each of the 19 expressions per subject."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    names_by_index: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        grouped[str(row["subject"])].append(row)
        index = int(float(row["expression_index"]))
        names_by_index[index].add(str(row["expression_name"]).strip())
    for subject in subjects:
        indices = [
            int(float(row["expression_index"])) for row in grouped.get(subject, [])
        ]
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate expression index for subject {subject}")
        if set(indices) != EXPECTED_EXPRESSION_INDICES:
            raise ValueError(
                f"Incomplete expression grid for subject {subject}: "
                f"observed={sorted(indices)}"
            )
    inconsistent_names = {
        index: sorted(names)
        for index, names in sorted(names_by_index.items())
        if len(names) != 1
    }
    if inconsistent_names:
        raise ValueError(
            f"Expression names differ across subjects: {inconsistent_names}"
        )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    if args.bootstrap_repetitions < 1000:
        raise ValueError("At least 1000 bootstrap repetitions are required")
    rows = read_csv(args.input_csv)
    if not rows:
        raise ValueError("Metric input is empty")
    required = {"case", "subject", "expression_index", "expression_name"}
    required.update(args.metric)
    if args.filter_column:
        required.add(args.filter_column)
    missing = required - set(rows[0])
    if missing:
        raise KeyError(f"Missing input fields: {sorted(missing)}")

    split = json.loads(args.split_json.resolve(strict=True).read_text(encoding="utf-8"))
    subjects = selected_subjects(split, args.partition)
    subject_set = set(subjects)
    partition_rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    seen_cases: set[str] = set()
    for row in rows:
        subject = f"{int(row['subject']):03d}"
        if subject not in subject_set:
            continue
        case = str(row["case"])
        if case in seen_cases:
            raise ValueError(f"Duplicate case: {case}")
        seen_cases.add(case)
        if f"{int(case.split('_', 1)[0]):03d}" != subject:
            raise ValueError(f"Case/subject mismatch: {case}/{subject}")
        counts[subject] += 1
        partition_rows.append({**row, "subject": subject})
    expected = len(subjects) * EXPRESSIONS_PER_SUBJECT
    bad_counts = {
        subject: counts.get(subject, 0)
        for subject in subjects
        if counts.get(subject, 0) != EXPRESSIONS_PER_SUBJECT
    }
    if len(partition_rows) != expected or bad_counts:
        raise ValueError(
            "Rows do not contain one complete frozen partition: "
            f"rows={len(partition_rows)}/{expected}, counts={bad_counts}"
        )
    validate_expression_grid(partition_rows, subjects)

    completed_rows = [
        row for row in partition_rows if completed(row, args.completed_column)
    ]
    if args.filter_column:
        analysis_rows = [
            row
            for row in completed_rows
            if str(row[args.filter_column]).strip() == args.filter_value
        ]
    else:
        analysis_rows = completed_rows

    numeric_rows: list[dict[str, Any]] = []
    for row in analysis_rows:
        converted: dict[str, Any] = dict(row)
        for metric in args.metric:
            raw = row.get(metric, "")
            if raw in (None, ""):
                raise ValueError(f"Selected case {row['case']} lacks {metric}")
            value = float(raw)
            if not np.isfinite(value):
                raise ValueError(f"Selected case {row['case']} has non-finite {metric}")
            converted[metric] = value
        numeric_rows.append(converted)

    anonymized = build_anonymized_subject_labels(split)
    summaries: list[dict[str, Any]] = []
    subject_output: list[dict[str, Any]] = []
    for index, metric in enumerate(args.metric):
        summary, metric_subject_rows = summarize_metric(
            numeric_rows,
            metric,
            subjects,
            args.bootstrap_repetitions,
            args.seed + index,
        )
        summaries.append(summary)
        for row in metric_subject_rows:
            subject_output.append(
                {**row, "anonymized_subject": anonymized[row["subject"]]}
            )

    expression_output: list[dict[str, Any]] = []
    for metric in args.metric:
        grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
        for row in numeric_rows:
            key = (int(float(row["expression_index"])), str(row["expression_name"]))
            grouped[key].append(float(row[metric]))
        for (expression_index, expression_name), values in sorted(grouped.items()):
            expression_output.append(
                {
                    "metric": metric,
                    "expression_index": expression_index,
                    "expression_name": expression_name,
                    "available_subjects": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "standard_deviation": (
                        float(np.std(values, ddof=1)) if len(values) > 1 else None
                    ),
                }
            )

    subject_output.sort(
        key=lambda row: (str(row["metric"]), str(row["anonymized_subject"]))
    )

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "subject_metric_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "anonymized_subject",
                "metric",
                "available_pairs",
                "subject_mean",
                "subject_median",
            ),
        )
        writer.writeheader()
        writer.writerows(
            {key: row[key] for key in writer.fieldnames} for row in subject_output
        )
    with (args.output_dir / "subject_id_key_internal_do_not_publish.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("anonymized_subject", "internal_subject_id")
        )
        writer.writeheader()
        writer.writerows(
            {
                "anonymized_subject": anonymized[subject],
                "internal_subject_id": subject,
            }
            for subject in subjects
        )
    with (args.output_dir / "expression_metric_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = (
            "metric",
            "expression_index",
            "expression_name",
            "available_subjects",
            "mean",
            "median",
            "standard_deviation",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(expression_output)
    payload = {
        "analysis_label": args.analysis_label,
        "partition": args.partition,
        "attempted_pairs": len(partition_rows),
        "completed_pairs": len(completed_rows),
        "failed_or_invalid_pairs": len(partition_rows) - len(completed_rows),
        "filter": (
            {"column": args.filter_column, "value": args.filter_value}
            if args.filter_column
            else None
        ),
        "pairs_in_metric_summary": len(numeric_rows),
        "subjects": len(subjects),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.seed,
        "metrics": summaries,
        "interpretation": (
            "The confidence interval resamples subject-level means, not individual "
            "expression pairs. Failed, invalid, or filtered cases are never assigned "
            "fabricated metric values; their counts are reported explicitly."
        ),
    }
    (args.output_dir / "metric_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
