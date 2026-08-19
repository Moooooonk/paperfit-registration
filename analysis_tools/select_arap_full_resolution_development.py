#!/usr/bin/env python3
"""Validate and select one full-resolution ARAP setting from development only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


SETTINGS = {
    "FULL_C300_R1_S05": {"controls": 300, "rounds": 1, "step": 0.5},
    "FULL_C300_R1_S10": {"controls": 300, "rounds": 1, "step": 1.0},
    "FULL_C600_R1_S10": {"controls": 600, "rounds": 1, "step": 1.0},
    "FULL_C600_R2_S10": {"controls": 600, "rounds": 2, "step": 1.0},
}
FIT_METRICS = (
    "post_full_median_mm",
    "post_full_p90_mm",
    "post_nose_median_mm",
    "post_nose_p90_mm",
    "post_anchor_mm",
)
DEFORMATION_METRICS = (
    "non_eye_strain_p99",
    "eye_boundary_strain_p99",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="SETTING_ID=ROWS_CSV",
        help="Repeat once for each of the four frozen development settings.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def candidate_paths(specifications: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(f"Invalid candidate specification: {specification}")
        setting, raw_path = specification.split("=", 1)
        if setting in result:
            raise ValueError(f"Duplicate candidate setting: {setting}")
        result[setting] = Path(raw_path).resolve(strict=True)
    if set(result) != set(SETTINGS):
        raise ValueError(
            f"Expected candidate IDs {sorted(SETTINGS)}, received {sorted(result)}"
        )
    return result


def finite(value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite value: {value!r}")
    return result


def load_setting(
    setting: str, path: Path,
) -> tuple[list[dict[str, str]], dict[str, float] | None, list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    cases = [row["case"] for row in rows]
    ineligible_reasons: list[str] = []
    if len(rows) != 190 or len(set(cases)) != 190:
        ineligible_reasons.append("did not contain 190 unique attempts")
    subjects = sorted({row["subject"] for row in rows})
    if len(subjects) != 10:
        ineligible_reasons.append("did not contain ten development subjects")
    counts = {subject: sum(row["subject"] == subject for row in rows) for subject in subjects}
    if set(counts.values()) != {19}:
        ineligible_reasons.append("did not contain 19 expressions per subject")
    completed_rows = [row for row in rows if int(row["completed"]) == 1]
    if len(completed_rows) != 190:
        ineligible_reasons.append(f"completed {len(completed_rows)}/190 attempts")
    orientation_passes = sum(
        int(row.get("post_orientation_pass") or 0) for row in completed_rows
    )
    if orientation_passes != 190:
        ineligible_reasons.append(
            f"passed orientation for {orientation_passes}/190 attempts"
        )
    if ineligible_reasons:
        return rows, None, ineligible_reasons

    values: dict[str, float] = {}
    for metric in FIT_METRICS + DEFORMATION_METRICS:
        by_subject: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            by_subject[row["subject"]].append(finite(row[metric]))
        subject_means = [statistics.fmean(by_subject[subject]) for subject in subjects]
        values[metric] = statistics.median(subject_means)
    return rows, values, []


def normalize(values: dict[str, float]) -> dict[str, float]:
    low = min(values.values())
    high = max(values.values())
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-12):
        return {key: 0.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def main() -> None:
    args = parse_args()
    paths = candidate_paths(args.candidate)
    rows_by_setting: dict[str, list[dict[str, str]]] = {}
    aggregate: dict[str, dict[str, float]] = {}
    eligibility: dict[str, dict[str, object]] = {}
    reference_cases: set[str] | None = None
    for setting in SETTINGS:
        rows, values, reasons = load_setting(setting, paths[setting])
        cases = {row["case"] for row in rows}
        if reference_cases is None:
            reference_cases = cases
        elif cases != reference_cases:
            raise ValueError("Candidate case sets differ")
        rows_by_setting[setting] = rows
        eligibility[setting] = {
            "eligible": not reasons,
            "ineligible_reasons": reasons,
        }
        if values is not None:
            aggregate[setting] = values

    eligible_settings = [
        setting for setting in SETTINGS if bool(eligibility[setting]["eligible"])
    ]
    if not eligible_settings:
        raise ValueError("No development candidate satisfied the frozen eligibility rule")

    normalized: dict[str, dict[str, float]] = {
        setting: {} for setting in eligible_settings
    }
    for metric in FIT_METRICS + DEFORMATION_METRICS:
        metric_norm = normalize(
            {setting: aggregate[setting][metric] for setting in eligible_settings}
        )
        for setting, value in metric_norm.items():
            normalized[setting][metric] = value

    scores: dict[str, dict[str, float]] = {}
    for setting in eligible_settings:
        fit_score = statistics.fmean(normalized[setting][metric] for metric in FIT_METRICS)
        deformation_score = statistics.fmean(
            normalized[setting][metric] for metric in DEFORMATION_METRICS
        )
        scores[setting] = {
            "fit_score": fit_score,
            "deformation_score": deformation_score,
            "balanced_score": (fit_score + deformation_score) / 2.0,
        }

    def selection_key(setting: str) -> tuple[float, float, int, int, float, str]:
        config = SETTINGS[setting]
        capacity = config["controls"] * config["rounds"] * config["step"]
        return (
            scores[setting]["balanced_score"],
            capacity,
            config["controls"],
            config["rounds"],
            config["step"],
            setting,
        )

    selected = min(eligible_settings, key=selection_key)
    payload = {
        "scope": "revision-stage development identities only",
        "attempted_cases_per_setting": 190,
        "subject_count": 10,
        "selection_rule": (
            "Equal weight for normalized subject-level fit and deformation domains; "
            "no S8-derived threshold and no evaluation result used"
        ),
        "fit_metrics": list(FIT_METRICS),
        "deformation_metrics": list(DEFORMATION_METRICS),
        "settings": SETTINGS,
        "eligibility": eligibility,
        "eligible_setting_ids": eligible_settings,
        "subject_level_aggregate_values": aggregate,
        "normalized_values": normalized,
        "scores": scores,
        "selected_setting_id": selected,
        "selected_configuration": SETTINGS[selected],
        "input_hashes": {
            setting: sha256(paths[setting])
            for setting in SETTINGS
        },
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
