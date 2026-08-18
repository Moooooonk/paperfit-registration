#!/usr/bin/env python3
"""Combine paired subject-level comparisons and apply Holm correction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison-summary", type=Path, action="append", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--family-label", required=True)
    return parser.parse_args()


def holm_adjust(pvalues: list[float]) -> list[float]:
    if any(not 0.0 <= value <= 1.0 for value in pvalues):
        raise ValueError("P-values must be in [0, 1]")
    count = len(pvalues)
    order = sorted(range(count), key=lambda index: (pvalues[index], index))
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * pvalues[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    rows: list[dict[str, Any]] = []
    for path in args.comparison_summary:
        payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
        required = {
            "baseline",
            "partition",
            "subjects",
            "pairs",
            "mean_subject_rate_difference",
            "subject_cluster_bootstrap_95_ci_difference",
            "exact_subject_sign_flip_two_sided_p",
        }
        if not required.issubset(payload):
            raise KeyError(f"Missing paired-comparison fields in {path}")
        rows.append({**payload, "source_summary": str(path.resolve())})
    partitions = {str(row["partition"]) for row in rows}
    subject_counts = {int(row["subjects"]) for row in rows}
    pair_counts = {int(row["pairs"]) for row in rows}
    if len(partitions) != 1 or len(subject_counts) != 1 or len(pair_counts) != 1:
        raise ValueError("Comparison summaries do not form one statistical family")
    raw = [float(row["exact_subject_sign_flip_two_sided_p"]) for row in rows]
    adjusted = holm_adjust(raw)
    output_rows = []
    for row, corrected in zip(rows, adjusted, strict=True):
        output_rows.append(
            {
                "baseline": row["baseline"],
                "partition": row["partition"],
                "subjects": row["subjects"],
                "pairs": row["pairs"],
                "mean_subject_rate_difference": row[
                    "mean_subject_rate_difference"
                ],
                "subject_cluster_bootstrap_95_ci_difference": json.dumps(
                    row["subject_cluster_bootstrap_95_ci_difference"]
                ),
                "raw_exact_sign_flip_p": row[
                    "exact_subject_sign_flip_two_sided_p"
                ],
                "holm_adjusted_p": corrected,
                "source_summary": row["source_summary"],
            }
        )
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "paired_comparisons_holm.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    report = {
        "family_label": args.family_label,
        "partition": next(iter(partitions)),
        "comparisons": len(output_rows),
        "independent_unit": "subject",
        "multiple_comparison_control": "Holm family-wise error correction",
        "rows": output_rows,
    }
    (args.output_dir / "paired_comparisons_holm.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
