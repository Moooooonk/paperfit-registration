#!/usr/bin/env python3
"""Apply the manuscript's branch-wise final acceptance rules.

This stage does not rerun registration. It combines the compact S8 outputs
from the rigid-pass, anchor-only, and broad-failure branches and writes one
auditable final status per source-target pair.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path


DEFAULT_ROOT = Path(os.environ.get("PAPERFIT_ROOT", "/path/to/prepared/facescape_pipeline"))
FULL_MEDIAN_MAX = 0.023
NOSE_MEDIAN_MAX = 0.040
NOSE_P90_MAX = 0.070
EYE_EPS = 1e-12


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def by_case(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["case"]: row for row in rows}


def value(row: dict[str, str], key: str) -> float:
    return float(row[key])


def eye_is_fixed(row: dict[str, str]) -> bool:
    return abs(value(row, "eye_fixed_max")) <= EYE_EPS


def load_facescape_scales(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def attach_metric_distances(
    rows: list[dict[str, object]],
    scale_dict: dict[str, object],
    target_expression: str = "18",
) -> list[dict[str, object]]:
    """Add post-hoc millimeter fields without changing QC decisions."""
    for row in rows:
        subject = str(int(str(row["case"]).split("_", 1)[0]))
        mm_per_unit = float(scale_dict[subject][target_expression][0])
        row["target_mm_per_unit"] = mm_per_unit
        for key in ("full_median_after", "nose_median_after", "nose_p90_after"):
            row[f"{key}_mm"] = float(row[key]) * mm_per_unit
    return rows


def final_rows(
    rigid_rows: list[dict[str, str]],
    anchor_rows: list[dict[str, str]],
    broad_rows: list[dict[str, str]],
    broad_selection_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    selection = by_case(broad_selection_rows)
    output: list[dict[str, object]] = []

    for row in rigid_rows:
        accepted = eye_is_fixed(row)
        output.append(
            {
                "case": row["case"],
                "input_branch": "rigid_pass",
                "final_stage": "rigid_effective_pass" if accepted else "remaining_fail",
                "accepted": int(accepted),
                "orientation_pass": 1,
                "fixed_eye_pass": int(eye_is_fixed(row)),
                "full_median_after": value(row, "full_median_after"),
                "nose_median_after": value(row, "nose_median_after"),
                "nose_p90_after": value(row, "nose_p90_after"),
            }
        )

    for row in anchor_rows:
        nondegrading = (
            value(row, "full_median_after") <= value(row, "full_median_before")
            and value(row, "nose_median_after") <= value(row, "nose_median_before")
        )
        accepted = nondegrading and eye_is_fixed(row)
        output.append(
            {
                "case": row["case"],
                "input_branch": "anchor_only",
                "final_stage": "anchor_only_nonrigid_recovered" if accepted else "remaining_fail",
                "accepted": int(accepted),
                "orientation_pass": 1,
                "fixed_eye_pass": int(eye_is_fixed(row)),
                "full_median_after": value(row, "full_median_after"),
                "nose_median_after": value(row, "nose_median_after"),
                "nose_p90_after": value(row, "nose_p90_after"),
            }
        )

    for row in broad_rows:
        selected = selection.get(row["case"])
        if selected is None:
            raise ValueError(f"Missing broad-failure selection evidence for {row['case']}")
        orientation_pass = int(float(selected.get("upside_down", "1")) == 0)
        thresholds_pass = (
            value(row, "full_median_after") <= FULL_MEDIAN_MAX
            and value(row, "nose_median_after") <= NOSE_MEDIAN_MAX
            and value(row, "nose_p90_after") <= NOSE_P90_MAX
        )
        accepted = bool(orientation_pass and thresholds_pass and eye_is_fixed(row))
        output.append(
            {
                "case": row["case"],
                "input_branch": "broad_failure",
                "final_stage": "broadfail_nonrigid_threshold_accept" if accepted else "remaining_fail",
                "accepted": int(accepted),
                "orientation_pass": orientation_pass,
                "fixed_eye_pass": int(eye_is_fixed(row)),
                "full_median_after": value(row, "full_median_after"),
                "nose_median_after": value(row, "nose_median_after"),
                "nose_p90_after": value(row, "nose_p90_after"),
            }
        )

    cases = [str(row["case"]) for row in output]
    if len(cases) != len(set(cases)):
        duplicates = sorted(case for case, count in Counter(cases).items() if count > 1)
        raise ValueError(f"Cases occur in more than one branch: {duplicates}")
    return sorted(output, key=lambda row: str(row["case"]))


def write_outputs(rows: list[dict[str, object]], out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    status_csv = out_dir / "final_case_status.csv"
    with status_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(str(row["final_stage"]) for row in rows)
    accepted = sum(int(row["accepted"]) for row in rows)
    summary = {
        "total_pairs": len(rows),
        "main_and_recovered_rigid_pass": counts["rigid_effective_pass"],
        "anchor_only_s8_accept": counts["anchor_only_nonrigid_recovered"],
        "broad_failure_s8_accept": counts["broadfail_nonrigid_threshold_accept"],
        "final_accepted": accepted,
        "coverage": accepted / len(rows) if rows else None,
        "residual_failures": counts["remaining_fail"],
        "thresholds_in_registration_units": {
            "full_median": FULL_MEDIAN_MAX,
            "nose_median": NOSE_MEDIAN_MAX,
            "nose_p90": NOSE_P90_MAX,
        },
        "status_csv": str(status_csv),
    }
    if "target_mm_per_unit" in rows[0]:
        summary["millimeter_reporting"] = {
            "formula": "distance_mm = target_mm_per_unit * distance_registration_unit",
            "decision_policy": "reporting only; QC decisions remain in evaluated registration units",
        }
    (out_dir / "final_experiment_status.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--expect-paper-counts", action="store_true")
    parser.add_argument(
        "--facescape-scale-dict",
        type=Path,
        help="Official FaceScape toolkit/predef/Rt_scale_dict.json for post-hoc mm columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    compact_name = "nonrigid_proposed_s8_hardgate_full40_pass32_compact_summary.csv"
    rows = final_rows(
        read_rows(root / "research_nonrigid_proposed_s8_allpairs_20260622" / compact_name),
        read_rows(root / "research_nonrigid_proposed_s8_allpairs_rough_anchorfail_20260623" / compact_name),
        read_rows(root / "research_nonrigid_proposed_s8_allpairs_broadfail_probe_20260623" / compact_name),
        read_rows(
            root
            / "research_nonrigid_proposed_s8_allpairs_broadfail_probe_20260623"
            / "broadfail_probe_selection.csv"
        ),
    )
    if args.facescape_scale_dict:
        rows = attach_metric_distances(rows, load_facescape_scales(args.facescape_scale_dict))
    summary = write_outputs(rows, root / "research_allpairs_final_status_reproduced")
    if args.expect_paper_counts:
        expected = {
            "total_pairs": 380,
            "main_and_recovered_rigid_pass": 236,
            "anchor_only_s8_accept": 84,
            "broad_failure_s8_accept": 26,
            "final_accepted": 346,
            "residual_failures": 34,
        }
        observed = {key: summary[key] for key in expected}
        if observed != expected:
            raise SystemExit(f"Paper-count mismatch: observed={observed}, expected={expected}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
