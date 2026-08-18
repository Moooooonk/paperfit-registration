"""Generate subject- and expression-aware summaries for any final QC column."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.subject_anonymization import build_anonymized_subject_labels
except ModuleNotFoundError:
    from subject_anonymization import build_anonymized_subject_labels


BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260816
EXPRESSIONS_PER_SUBJECT = 19
EXPECTED_EXPRESSION_INDICES = {*range(1, 18), 19, 20}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def subject_cluster_interval(
    rows: list[dict[str, Any]], accept_key: str, seed_offset: int = 0
) -> tuple[float | None, float | None]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["subject"])].append(row)
    subjects = sorted(grouped)
    if not subjects:
        return None, None
    accepted = np.asarray(
        [sum(int(item[accept_key]) for item in grouped[subject]) for subject in subjects],
        dtype=np.float64,
    )
    totals = np.asarray([len(grouped[subject]) for subject in subjects], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    indices = rng.integers(0, len(subjects), size=(BOOTSTRAP_REPLICATES, len(subjects)))
    rates = accepted[indices].sum(axis=1) / totals[indices].sum(axis=1)
    return float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))


def group_summary(
    rows: list[dict[str, Any]], accept_key: str, seed_offset: int = 0
) -> dict[str, Any]:
    accepted = sum(int(row[accept_key]) for row in rows)
    low, high = subject_cluster_interval(rows, accept_key, seed_offset)
    subjects = sorted({str(row["subject"]) for row in rows})
    subject_rates = []
    for subject in subjects:
        items = [row for row in rows if str(row["subject"]) == subject]
        subject_rates.append(sum(int(item[accept_key]) for item in items) / len(items))
    return {
        "pairs": len(rows),
        "accepted": accepted,
        "pair_coverage": accepted / len(rows) if rows else None,
        "subjects": len(subjects),
        "mean_subject_success_rate": float(np.mean(subject_rates)) if subject_rates else None,
        "median_subject_success_rate": float(np.median(subject_rates)) if subject_rates else None,
        "subject_success_rate_std": (
            float(np.std(subject_rates, ddof=1)) if len(subject_rates) > 1 else None
        ),
        "subject_success_rate_min": float(np.min(subject_rates)) if subject_rates else None,
        "subject_success_rate_max": float(np.max(subject_rates)) if subject_rates else None,
        "subject_success_rate_iqr": (
            [
                float(np.quantile(subject_rates, 0.25)),
                float(np.quantile(subject_rates, 0.75)),
            ]
            if subject_rates
            else [None, None]
        ),
        "subject_cluster_bootstrap_95_ci": [low, high],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED + seed_offset,
    }


def validate_expression_grid(rows: list[dict[str, Any]]) -> None:
    """Require the same 19 uniquely indexed expressions for every subject."""
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    names_by_index: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        by_subject[str(row["subject"])].append(row)
        names_by_index[int(row["expression_index"])].add(
            str(row["expression_name"]).strip()
        )
    for subject, items in sorted(by_subject.items()):
        indices = [int(item["expression_index"]) for item in items]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--accept-column", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-label", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    rows = read_csv(args.input_csv)
    if not rows or args.accept_column not in rows[0]:
        raise KeyError(f"Missing acceptance column: {args.accept_column}")
    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    development = {f"{int(value):03d}" for value in split["development_subjects"]}
    test = {f"{int(value):03d}" for value in split["test_subjects"]}
    anonymized_subjects = build_anonymized_subject_labels(split)

    cases = [str(row["case"]) for row in rows]
    if len(cases) != len(set(cases)):
        raise ValueError("Duplicate cases in clustered-statistics input")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        subject = f"{int(row['subject']):03d}"
        case_subject = f"{int(str(row['case']).split('_', 1)[0]):03d}"
        if subject != case_subject:
            raise ValueError(
                f"Case/subject mismatch: case={row['case']}, subject={subject}"
            )
        if subject in development:
            partition = "development"
        elif subject in test:
            partition = "test"
        else:
            raise ValueError(f"Subject missing from frozen split: {subject}")
        accepted = int(float(row[args.accept_column]))
        if accepted not in (0, 1):
            raise ValueError(f"Acceptance must be binary for {row['case']}")
        normalized.append(
            {
                **row,
                "subject": subject,
                "partition": partition,
                args.accept_column: accepted,
                "expression_index": int(float(row["expression_index"])),
            }
        )

    subject_rows = []
    subject_id_key = []
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        by_subject[row["subject"]].append(row)
    bad_counts = {
        subject: len(items)
        for subject, items in sorted(by_subject.items())
        if len(items) != EXPRESSIONS_PER_SUBJECT
    }
    observed_subjects = set(by_subject)
    allowed_sets = (development, test, development | test)
    if observed_subjects not in allowed_sets or bad_counts:
        raise ValueError(
            "Clustered-statistics rows do not match a complete frozen partition: "
            f"subjects={sorted(observed_subjects)}, per_subject_counts={bad_counts}"
        )
    validate_expression_grid(normalized)
    for subject in sorted(by_subject):
        items = by_subject[subject]
        accepted = sum(int(item[args.accept_column]) for item in items)
        subject_rows.append(
            {
                "anonymized_subject": anonymized_subjects[subject],
                "partition": items[0]["partition"],
                "pairs": len(items),
                "accepted": accepted,
                "success_rate": accepted / len(items),
            }
        )
        subject_id_key.append(
            {
                "anonymized_subject": anonymized_subjects[subject],
                "internal_subject_id": subject,
                "partition": items[0]["partition"],
                "publication_status": "internal_key_do_not_publish",
            }
        )
    subject_rows.sort(key=lambda row: str(row["anonymized_subject"]))
    subject_id_key.sort(key=lambda row: str(row["anonymized_subject"]))

    expression_rows = []
    partition_rows = {
        "development": [row for row in normalized if row["partition"] == "development"],
        "test": [row for row in normalized if row["partition"] == "test"],
        "full": normalized,
    }
    offset = 100
    for partition, partition_items in partition_rows.items():
        by_expression: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in partition_items:
            key = (int(row["expression_index"]), str(row["expression_name"]))
            by_expression[key].append(row)
        for (expression_index, expression_name), items in sorted(by_expression.items()):
            offset += 1
            accepted = sum(int(item[args.accept_column]) for item in items)
            low, high = subject_cluster_interval(items, args.accept_column, offset)
            expression_rows.append(
                {
                    "partition": partition,
                    "expression_index": expression_index,
                    "expression_name": expression_name,
                    "subjects": len({item["subject"] for item in items}),
                    "pairs": len(items),
                    "accepted": accepted,
                    "success_rate": accepted / len(items),
                    "subject_bootstrap_95_ci_low": low,
                    "subject_bootstrap_95_ci_high": high,
                }
            )

    development_rows = [row for row in normalized if row["partition"] == "development"]
    test_rows = [row for row in normalized if row["partition"] == "test"]
    summary = {
        "analysis_label": args.analysis_label,
        "input_csv": str(args.input_csv.resolve()),
        "acceptance_column": args.accept_column,
        "frozen_split": str(args.split_json.resolve()),
        "full": group_summary(normalized, args.accept_column, 1),
        "development": group_summary(development_rows, args.accept_column, 2),
        "test": group_summary(test_rows, args.accept_column, 3),
        "interpretation": (
            "Pair-level coverage is descriptive. Confidence intervals resample whole "
            "subjects and therefore preserve within-subject dependence among expressions."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "subject_level_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(subject_rows[0].keys()))
        writer.writeheader()
        writer.writerows(subject_rows)
    with (args.output_dir / "subject_id_key_internal_do_not_publish.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(subject_id_key[0].keys()))
        writer.writeheader()
        writer.writerows(subject_id_key)
    with (args.output_dir / "expression_level_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(expression_rows[0].keys()))
        writer.writeheader()
        writer.writerows(expression_rows)
    (args.output_dir / "clustered_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
