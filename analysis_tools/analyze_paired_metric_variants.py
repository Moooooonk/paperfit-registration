#!/usr/bin/env python3
"""Compare equal-denominator S8 variants with subjects as independent units."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.subject_anonymization import build_anonymized_subject_labels
except ModuleNotFoundError:
    from subject_anonymization import build_anonymized_subject_labels


EXPRESSIONS_PER_SUBJECT = 19
DEFAULT_BOOTSTRAP_REPETITIONS = 20_000
DEFAULT_SEED = 20260816


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        action="append",
        help="Repeat LABEL=CSV for one case row per label and case.",
    )
    source.add_argument("--grouped-csv", type=Path)
    parser.add_argument("--group-column")
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument(
        "--partition", choices=("development", "test"), required=True
    )
    parser.add_argument("--metric", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-label", required=True)
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPETITIONS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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


def completed(row: dict[str, str]) -> bool:
    raw = row.get("completed")
    return True if raw in (None, "") else int(float(raw)) == 1


def selected_subjects(split: dict[str, Any], partition: str) -> list[str]:
    key = "development_subjects" if partition == "development" else "test_subjects"
    return sorted(f"{int(value):03d}" for value in split[key])


def validate_group(
    rows: list[dict[str, str]],
    subjects: list[str],
    metrics: list[str],
    label: str,
) -> dict[str, dict[str, str]]:
    required = {"case", "subject", "expression_index", "expression_name", *metrics}
    if not rows or not required.issubset(rows[0]):
        missing = required - set(rows[0] if rows else ())
        raise KeyError(f"{label}: missing fields {sorted(missing)}")
    subject_set = set(subjects)
    expected = len(subjects) * EXPRESSIONS_PER_SUBJECT
    indexed: dict[str, dict[str, str]] = {}
    counts: Counter[str] = Counter()
    for row in rows:
        subject = f"{int(row['subject']):03d}"
        if subject not in subject_set:
            raise ValueError(f"{label}: subject outside frozen partition: {subject}")
        case = str(row["case"])
        if case in indexed:
            raise ValueError(f"{label}: duplicate case {case}")
        if f"{int(case.split('_', 1)[0]):03d}" != subject:
            raise ValueError(f"{label}: case/subject mismatch {case}/{subject}")
        indexed[case] = {**row, "subject": subject}
        counts[subject] += 1
    bad_counts = {
        subject: counts.get(subject, 0)
        for subject in subjects
        if counts.get(subject, 0) != EXPRESSIONS_PER_SUBJECT
    }
    if len(indexed) != expected or bad_counts:
        raise ValueError(
            f"{label}: incomplete frozen partition rows={len(indexed)}/{expected}, "
            f"counts={bad_counts}"
        )
    return indexed


def parse_sources(args: argparse.Namespace) -> tuple[dict[str, list[dict[str, str]]], dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = {}
    provenance: dict[str, str] = {}
    if args.grouped_csv is not None:
        if not args.group_column:
            raise ValueError("--group-column is required with --grouped-csv")
        path = args.grouped_csv.resolve(strict=True)
        rows = read_csv(path)
        if not rows or args.group_column not in rows[0]:
            raise KeyError(f"Grouped CSV lacks {args.group_column}")
        for row in rows:
            label = str(row[args.group_column]).strip()
            if not label:
                raise ValueError("Grouped CSV contains an empty label")
            groups.setdefault(label, []).append(row)
        provenance[str(path)] = sha256(path)
    else:
        if not args.input or len(args.input) < 2:
            raise ValueError("At least two --input LABEL=CSV values are required")
        for specification in args.input:
            if "=" not in specification:
                raise ValueError(f"Invalid labeled input: {specification}")
            label, raw_path = specification.split("=", 1)
            label = label.strip()
            if not label or label in groups:
                raise ValueError(f"Duplicate or empty input label: {label}")
            path = Path(raw_path).resolve(strict=True)
            groups[label] = read_csv(path)
            provenance[str(path)] = sha256(path)
    if args.reference_label not in groups:
        raise ValueError(f"Reference label not found: {args.reference_label}")
    if len(groups) < 2:
        raise ValueError("At least two variant groups are required")
    return groups, provenance


def finite_metric(row: dict[str, str], metric: str) -> float | None:
    try:
        value = float(row[metric])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def subject_bootstrap_ci(
    values: list[float], repetitions: int, seed: int
) -> tuple[float, float]:
    if not values:
        raise ValueError("Subject bootstrap requires at least one value")
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(repetitions, len(array)))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def exact_sign_flip_pvalue(values: list[float]) -> float:
    nonzero = np.asarray(
        [value for value in values if abs(value) > 1e-15], dtype=np.float64
    )
    if not len(nonzero):
        return 1.0
    observed = abs(float(np.mean(nonzero)))
    if len(nonzero) > 20:
        raise ValueError("Exact sign-flip implementation supports at most 20 subjects")
    null = [
        abs(float(np.mean(nonzero * np.asarray(signs, dtype=np.float64))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path.name}")
    fields = list(rows[0])
    extra = sorted({key for row in rows for key in row} - set(fields))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + extra)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    if args.bootstrap_repetitions < 1000:
        raise ValueError("At least 1000 bootstrap repetitions are required")
    if len(args.metric) != len(set(args.metric)):
        raise ValueError("Metric names must be unique")
    groups, provenance = parse_sources(args)
    split_path = args.split_json.resolve(strict=True)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    subjects = selected_subjects(split, args.partition)
    indexed = {
        label: validate_group(rows, subjects, args.metric, label)
        for label, rows in groups.items()
    }
    reference_cases = set(indexed[args.reference_label])
    for label, values in indexed.items():
        if set(values) != reference_cases:
            raise ValueError(f"{label}: case denominator differs from reference")

    anonymized = build_anonymized_subject_labels(split)
    variant_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    reference = indexed[args.reference_label]
    seed_offset = 0
    for label in sorted(indexed):
        values = indexed[label]
        completion = [completed(row) for row in values.values()]
        for metric in args.metric:
            available = [
                finite_metric(row, metric)
                for row in values.values()
                if completed(row)
            ]
            numeric = [value for value in available if value is not None]
            grouped_values: dict[str, list[float]] = defaultdict(list)
            for row in values.values():
                value = finite_metric(row, metric) if completed(row) else None
                if value is not None:
                    grouped_values[str(row["subject"])].append(value)
            subject_means = [
                float(np.mean(grouped_values[subject]))
                for subject in subjects
                if grouped_values[subject]
            ]
            variant_rows.append(
                {
                    "variant": label,
                    "metric": metric,
                    "attempted_pairs": len(values),
                    "completed_pairs": sum(completion),
                    "pairs_with_finite_metric": len(numeric),
                    "subjects_with_metric": len(subject_means),
                    "pair_mean": float(np.mean(numeric)) if numeric else None,
                    "pair_median": float(np.median(numeric)) if numeric else None,
                    "mean_subject_mean": (
                        float(np.mean(subject_means)) if subject_means else None
                    ),
                }
            )
            for subject in subjects:
                subject_rows.append(
                    {
                        "variant": label,
                        "metric": metric,
                        "anonymized_subject": anonymized[subject],
                        "available_pairs": len(grouped_values[subject]),
                        "subject_mean": (
                            float(np.mean(grouped_values[subject]))
                            if grouped_values[subject]
                            else None
                        ),
                        "subject_median": (
                            float(np.median(grouped_values[subject]))
                            if grouped_values[subject]
                            else None
                        ),
                    }
                )
            if label == args.reference_label:
                continue
            differences: dict[str, list[float]] = defaultdict(list)
            pair_differences: list[float] = []
            excluded = 0
            for case in sorted(reference_cases):
                reference_row = reference[case]
                variant_row = values[case]
                reference_value = (
                    finite_metric(reference_row, metric)
                    if completed(reference_row)
                    else None
                )
                variant_value = (
                    finite_metric(variant_row, metric)
                    if completed(variant_row)
                    else None
                )
                if reference_value is None or variant_value is None:
                    excluded += 1
                    continue
                difference = variant_value - reference_value
                pair_differences.append(difference)
                differences[str(reference_row["subject"])].append(difference)
            subject_differences = [
                float(np.mean(differences[subject]))
                for subject in subjects
                if differences[subject]
            ]
            seed_offset += 1
            low, high = subject_bootstrap_ci(
                subject_differences,
                args.bootstrap_repetitions,
                args.seed + seed_offset,
            )
            paired_rows.append(
                {
                    "reference": args.reference_label,
                    "variant": label,
                    "metric": metric,
                    "difference_definition": "variant_minus_reference",
                    "paired_available_pairs": len(pair_differences),
                    "excluded_pairs_without_both_metrics": excluded,
                    "subjects_with_paired_metrics": len(subject_differences),
                    "pair_mean_difference": float(np.mean(pair_differences)),
                    "pair_median_difference": float(np.median(pair_differences)),
                    "mean_subject_difference": float(np.mean(subject_differences)),
                    "subject_cluster_bootstrap_95_ci_low": low,
                    "subject_cluster_bootstrap_95_ci_high": high,
                    "exact_subject_sign_flip_two_sided_p": exact_sign_flip_pvalue(
                        subject_differences
                    ),
                }
            )

    subject_rows.sort(
        key=lambda row: (
            str(row["variant"]),
            str(row["metric"]),
            str(row["anonymized_subject"]),
        )
    )

    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "variant_metric_summary.csv", variant_rows)
    write_csv(args.output_dir / "paired_metric_comparisons.csv", paired_rows)
    write_csv(args.output_dir / "subject_metric_summary.csv", subject_rows)
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
    report = {
        "analysis_label": args.analysis_label,
        "partition": args.partition,
        "development_only": args.partition == "development",
        "heldout_inspected": args.partition == "test",
        "descriptive_only": True,
        "selection_permitted": False,
        "reference_label": args.reference_label,
        "variants": sorted(indexed),
        "metrics": args.metric,
        "subjects": len(subjects),
        "cases_per_variant": len(reference_cases),
        "equal_case_denominators_verified": True,
        "difference_definition": "variant_minus_reference",
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.seed,
        "split_sha256": sha256(split_path),
        "input_sha256": provenance,
        "paired_comparisons": paired_rows,
    }
    (args.output_dir / "paired_metric_analysis_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
