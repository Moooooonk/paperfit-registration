#!/usr/bin/env python3
"""Paired subject-level comparison for dependent multi-expression results."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from tools.subject_anonymization import build_anonymized_subject_labels
except ModuleNotFoundError:
    from subject_anonymization import build_anonymized_subject_labels


EXPRESSIONS_PER_SUBJECT = 19
EXPECTED_EXPRESSION_INDICES = {*range(1, 18), 19, 20}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposed-csv", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--proposed-accept-column", required=True)
    parser.add_argument("--baseline-accept-column", required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument(
        "--partition", choices=("development", "test", "full"), required=True
    )
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def read(path: Path, accept_column: str) -> dict[str, dict[str, object]]:
    with path.resolve(strict=True).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or accept_column not in rows[0]:
        raise KeyError(f"Missing acceptance column {accept_column} in {path}")
    result = {}
    for row in rows:
        case = str(row["case"])
        if case in result:
            raise ValueError(f"Duplicate case in {path}: {case}")
        subject = f"{int(row['subject']):03d}"
        case_subject = f"{int(case.split('_', 1)[0]):03d}"
        if subject != case_subject:
            raise ValueError(f"Case/subject mismatch in {path}: {case}/{subject}")
        accepted = int(float(row[accept_column]))
        if accepted not in (0, 1):
            raise ValueError(f"Acceptance must be binary in {path}: {case}")
        result[case] = {
            "case": case,
            "subject": subject,
            "accepted": accepted,
        }
    return result


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    nonzero = differences[np.abs(differences) > 1e-15]
    if len(nonzero) == 0:
        return 1.0
    observed = abs(float(np.mean(nonzero)))
    if len(nonzero) <= 20:
        values = []
        for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
            values.append(abs(float(np.mean(nonzero * np.asarray(signs)))))
        return float(np.mean(np.asarray(values) >= observed - 1e-15))
    rng = np.random.default_rng(20260816)
    signs = rng.choice((-1.0, 1.0), size=(1_000_000, len(nonzero)))
    values = np.abs(np.mean(signs * nonzero, axis=1))
    return float(np.mean(values >= observed - 1e-15))


def validate_case_grid(
    grouped: dict[str, list[dict[str, object]]], subjects: set[str]
) -> None:
    """Require every subject to contribute the same 19 expression suffixes."""
    expected_suffixes: set[str] | None = None
    expected_indices = EXPECTED_EXPRESSION_INDICES
    for subject in sorted(subjects):
        suffixes = [str(row["case"]).split("_", 1)[1] for row in grouped[subject]]
        try:
            indices = [int(suffix.split("_", 1)[0]) for suffix in suffixes]
        except ValueError as error:
            raise ValueError(
                f"Case lacks a numeric expression index for subject {subject}"
            ) from error
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate expression index for subject {subject}")
        if set(indices) != expected_indices:
            raise ValueError(
                f"Incomplete expression grid for subject {subject}: "
                f"observed={sorted(indices)}"
            )
        if len(suffixes) != len(set(suffixes)):
            raise ValueError(f"Duplicate expression case for subject {subject}")
        current = set(suffixes)
        if expected_suffixes is None:
            expected_suffixes = current
        elif current != expected_suffixes:
            raise ValueError(
                f"Expression case grid differs for subject {subject}: "
                f"observed={sorted(current)}"
            )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    proposed = read(args.proposed_csv, args.proposed_accept_column)
    baseline = read(args.baseline_csv, args.baseline_accept_column)
    if set(proposed) != set(baseline):
        only_proposed = sorted(set(proposed) - set(baseline))
        only_baseline = sorted(set(baseline) - set(proposed))
        raise ValueError(
            f"Case sets differ: proposed-only={only_proposed[:5]}, "
            f"baseline-only={only_baseline[:5]}"
        )
    split = json.loads(args.split_json.resolve(strict=True).read_text(encoding="utf-8"))
    if args.partition == "full":
        subjects = set(split["development_subjects"]) | set(split["test_subjects"])
    else:
        subjects = set(split[f"{args.partition}_subjects"])

    case_rows = []
    for case in sorted(proposed):
        p = proposed[case]
        b = baseline[case]
        if p["subject"] != b["subject"]:
            raise ValueError(f"Subject mismatch for {case}")
        if p["subject"] not in subjects:
            continue
        case_rows.append(
            {
                "case": case,
                "subject": p["subject"],
                "proposed_accepted": p["accepted"],
                "baseline_accepted": b["accepted"],
                "paired_difference": int(p["accepted"]) - int(b["accepted"]),
            }
        )

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in case_rows:
        grouped[str(row["subject"])].append(row)
    bad_counts = {
        subject: len(values)
        for subject, values in sorted(grouped.items())
        if len(values) != EXPRESSIONS_PER_SUBJECT
    }
    if set(grouped) != subjects or bad_counts:
        raise ValueError(
            f"Paired comparison does not match complete {args.partition} split: "
            f"subjects={sorted(grouped)}/{sorted(subjects)}, "
            f"per_subject_counts={bad_counts}"
        )
    validate_case_grid(grouped, subjects)
    anonymized = build_anonymized_subject_labels(split)
    subject_rows = []
    for subject in sorted(grouped):
        rows = grouped[subject]
        p_rate = float(np.mean([int(row["proposed_accepted"]) for row in rows]))
        b_rate = float(np.mean([int(row["baseline_accepted"]) for row in rows]))
        subject_rows.append(
            {
                "anonymized_subject": anonymized[subject],
                "pairs": len(rows),
                "proposed_success_rate": p_rate,
                "baseline_success_rate": b_rate,
                "paired_success_rate_difference": p_rate - b_rate,
            }
        )
    differences = np.asarray(
        [float(row["paired_success_rate_difference"]) for row in subject_rows],
        dtype=np.float64,
    )
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(
        0, len(differences), size=(args.bootstrap_repetitions, len(differences))
    )
    bootstrap = differences[indices].mean(axis=1)
    summary = {
        "baseline": args.baseline_name,
        "partition": args.partition,
        "subjects": len(subject_rows),
        "pairs": len(case_rows),
        "proposed_pair_coverage": float(
            np.mean([int(row["proposed_accepted"]) for row in case_rows])
        ),
        "baseline_pair_coverage": float(
            np.mean([int(row["baseline_accepted"]) for row in case_rows])
        ),
        "mean_subject_rate_difference": float(np.mean(differences)),
        "subject_cluster_bootstrap_95_ci_difference": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "exact_subject_sign_flip_two_sided_p": exact_sign_flip_pvalue(differences),
        "interpretation": (
            "The paired test treats each subject, not each expression pair, as the "
            "independent unit. Pair coverage is secondary descriptive information."
        ),
    }
    subject_rows.sort(key=lambda row: str(row["anonymized_subject"]))

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "subject_paired_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(subject_rows[0].keys()))
        writer.writeheader()
        writer.writerows(subject_rows)
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
            for subject in sorted(subjects)
        )
    with (args.output_dir / "case_paired_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(case_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
