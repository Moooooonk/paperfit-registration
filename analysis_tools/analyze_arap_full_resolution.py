#!/usr/bin/env python3
"""Paired subject-level analysis of frozen full-resolution ARAP versus S8."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    ("post_full_median_mm", "fit", False),
    ("post_full_p90_mm", "fit", False),
    ("post_nose_median_mm", "fit", False),
    ("post_nose_p90_mm", "fit", False),
    ("post_anchor_mm", "fit", True),
    ("edge_strain_p99", "deformation", False),
    ("eye_fixed_max_mm", "deformation", False),
    ("displacement_p90_mm", "deformation", False),
)
BOOTSTRAP_REPLICATES = 20_000
BASE_SEED = 20260819


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s8-rows", type=Path, required=True)
    parser.add_argument("--arap-rows", type=Path, required=True)
    parser.add_argument("--target-anchors", type=Path, required=True)
    parser.add_argument(
        "--analysis-plan",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs"
        / "ARAP_FULL_RESOLUTION_PROTOCOL.md",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.resolve(strict=True).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def finite(row: dict[str, str], metric: str) -> float | None:
    raw = row.get(metric, "")
    if raw in (None, ""):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def validate_case_structure(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    by_case = {row["case"]: row for row in rows}
    if len(rows) != 190 or len(by_case) != 190:
        raise ValueError(f"{label}: expected 190 unique attempts")
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[f"{int(row['subject']):03d}"] += 1
    if len(counts) != 10 or set(counts.values()) != {19}:
        raise ValueError(f"{label}: expected ten identities with 19 expressions each")
    return by_case


def direct_anchor_subjects(path: Path) -> set[str]:
    payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    result = {
        f"{int(subject):03d}"
        for subject, record in payload["anchors"].items()
        if record.get("diagnostics", {}).get("anchor_surface_method")
        == "perspective-correct triangle-ray intersection"
    }
    if len(result) != 9:
        raise ValueError(f"Expected nine direct-anchor evaluation identities, found {len(result)}")
    return result


def bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    estimates = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def exact_sign_flip_p(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    exceed = 0
    total = 0
    tolerance = 1e-15
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        estimate = abs(float(np.mean(values * np.asarray(signs, dtype=np.float64))))
        exceed += int(estimate + tolerance >= observed)
        total += 1
    return exceed / total


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=lambda key: (raw[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, key in enumerate(ordered):
        candidate = min(1.0, (count - index) * raw[key])
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def main() -> None:
    args = parse_args()
    s8_path = args.s8_rows.resolve(strict=True)
    arap_path = args.arap_rows.resolve(strict=True)
    anchor_path = args.target_anchors.resolve(strict=True)
    plan_path = args.analysis_plan.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    s8_rows = read_rows(s8_path)
    arap_rows = read_rows(arap_path)
    s8 = validate_case_structure(s8_rows, "S8")
    arap = validate_case_structure(arap_rows, "ARAP")
    if set(s8) != set(arap):
        raise ValueError("S8 and ARAP case sets differ")
    direct_subjects = direct_anchor_subjects(anchor_path)

    execution = {
        "attempted_pairs": 190,
        "s8_completed": sum(int(row.get("completed") or 0) for row in s8_rows),
        "arap_completed": sum(int(row.get("completed") or 0) for row in arap_rows),
        "arap_orientation_pass": sum(
            int(row.get("post_orientation_pass") or 0)
            for row in arap_rows
            if int(row.get("completed") or 0) == 1
        ),
    }

    results: list[dict[str, object]] = []
    raw_p_by_domain: dict[str, dict[str, float]] = defaultdict(dict)
    for metric_index, (metric, domain, direct_only) in enumerate(METRICS):
        paired: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for case in sorted(s8):
            s8_row = s8[case]
            arap_row = arap[case]
            subject = f"{int(s8_row['subject']):03d}"
            if direct_only and subject not in direct_subjects:
                continue
            if int(s8_row.get("completed") or 0) != 1:
                continue
            if int(arap_row.get("completed") or 0) != 1:
                continue
            s8_value = finite(s8_row, metric)
            arap_value = finite(arap_row, metric)
            if s8_value is None or arap_value is None:
                continue
            paired[subject].append((s8_value, arap_value))

        subjects = sorted(paired)
        if direct_only and execution["arap_completed"] == 190:
            if set(subjects) != direct_subjects or set(map(len, paired.values())) != {19}:
                raise ValueError(f"{metric}: direct-anchor pairing is incomplete")
        elif not direct_only and execution["arap_completed"] == 190:
            if len(subjects) != 10 or set(map(len, paired.values())) != {19}:
                raise ValueError(f"{metric}: complete-output pairing is incomplete")
        if not subjects:
            raise ValueError(f"{metric}: no paired values")

        s8_subject = np.asarray(
            [np.mean([pair[0] for pair in paired[subject]]) for subject in subjects],
            dtype=np.float64,
        )
        arap_subject = np.asarray(
            [np.mean([pair[1] for pair in paired[subject]]) for subject in subjects],
            dtype=np.float64,
        )
        difference = arap_subject - s8_subject
        raw_p = exact_sign_flip_p(difference)
        raw_p_by_domain[domain][metric] = raw_p
        results.append(
            {
                "metric": metric,
                "domain": domain,
                "difference_definition": "ARAP minus S8",
                "pairs": int(sum(len(paired[subject]) for subject in subjects)),
                "subjects": len(subjects),
                "s8_mean_of_subject_means": float(s8_subject.mean()),
                "arap_mean_of_subject_means": float(arap_subject.mean()),
                "paired_mean_subject_difference": float(difference.mean()),
                "paired_whole_subject_bootstrap_95_ci": bootstrap_ci(
                    difference, BASE_SEED + 100 + metric_index
                ),
                "exact_subject_sign_flip_p": raw_p,
                "direct_anchor_only": direct_only,
                "subject_pair_counts": {
                    subject: len(paired[subject]) for subject in subjects
                },
            }
        )

    adjusted = {
        domain: holm_adjust(raw) for domain, raw in raw_p_by_domain.items()
    }
    by_metric = {str(row["metric"]): row for row in results}
    for domain, values in adjusted.items():
        for metric, value in values.items():
            by_metric[metric]["holm_adjusted_p_within_domain"] = value

    payload = {
        "analysis_scope": "frozen revision-stage evaluation identities",
        "execution": execution,
        "direct_anchor_subjects": sorted(direct_subjects),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "base_seed": BASE_SEED,
        "binary_qc_used_for_method_comparison": False,
        "metrics": results,
        "input_hashes": {
            "s8_rows": sha256(s8_path),
            "arap_rows": sha256(arap_path),
            "target_anchors": sha256(anchor_path),
            "frozen_analysis_plan": sha256(plan_path),
            "analysis_script": sha256(Path(__file__).resolve()),
        },
    }
    (output / "arap_vs_s8_paired_analysis.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    csv_fields = [
        "metric",
        "domain",
        "pairs",
        "subjects",
        "s8_mean_of_subject_means",
        "arap_mean_of_subject_means",
        "paired_mean_subject_difference",
        "ci_low",
        "ci_high",
        "exact_subject_sign_flip_p",
        "holm_adjusted_p_within_domain",
        "direct_anchor_only",
    ]
    with (output / "arap_vs_s8_paired_analysis.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in results:
            flat = {key: row[key] for key in csv_fields if key in row}
            low, high = row["paired_whole_subject_bootstrap_95_ci"]
            flat["ci_low"] = low
            flat["ci_high"] = high
            writer.writerow(flat)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
