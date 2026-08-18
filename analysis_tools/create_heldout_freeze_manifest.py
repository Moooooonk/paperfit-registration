#!/usr/bin/env python3
"""Create the fail-closed authorization record for one held-out execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REVISION = Path(__file__).resolve().parents[1]
SPLIT_PATH = REVISION / "identity_disjoint_split.json"
EXPECTED_DEVELOPMENT = (
    "001",
    "002",
    "007",
    "008",
    "009",
    "010",
    "011",
    "012",
    "015",
    "020",
)
EXPECTED_HELDOUT = (
    "003",
    "004",
    "005",
    "006",
    "013",
    "014",
    "016",
    "017",
    "018",
    "019",
)
EXPECTED_PIPELINE_STEPS = {
    "aggregate real blinded development ratings",
    "calibrate HRN development common QC",
    "HRN reference S8 development QC",
    "3DDFA-V2 reference S8 transfer QC",
    "HRN conventional Open3D development QC",
    "3DDFA-V2 conventional Open3D transfer QC",
    "HRN shared-cues Open3D development QC",
    "3DDFA-V2 shared-cues Open3D transfer QC",
    "ARAP development grid QC",
    "select one ARAP development setting",
    "all-case S8 component ablation QC",
    "S8 sensitivity stages4 development QC",
    "S8 sensitivity stages12 development QC",
    "S8 sensitivity step075 development QC",
    "S8 sensitivity step125 development QC",
    "S8 sensitivity gain080 development QC",
    "S8 sensitivity gain120 development QC",
    "common-QC threshold sensitivity",
}
HELDOUT_TRUE_FLAGS = {
    "heldout_inspected",
    "heldout_test_inspected",
    "heldout_used",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-rating-output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--freeze-note", default="")
    return parser.parse_args()


def inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_subjects(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(sorted(f"{int(value):03d}" for value in values))


def validate_split(split: dict[str, Any]) -> None:
    development = normalize_subjects(split.get("development_subjects", []))
    heldout = normalize_subjects(split.get("test_subjects", []))
    if development != tuple(sorted(EXPECTED_DEVELOPMENT)):
        raise ValueError(f"Development split changed: {development}")
    if heldout != tuple(sorted(EXPECTED_HELDOUT)):
        raise ValueError(f"Held-out split changed: {heldout}")
    if set(development) & set(heldout):
        raise ValueError("Development and held-out identities overlap")
    if split.get("metric_blind") is not True:
        raise ValueError("Frozen split is not marked metric-blind")


def heldout_markers(value: object, location: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if key in HELDOUT_TRUE_FLAGS and child is True:
                findings.append(f"{child_location}=true")
            if key == "subset" and str(child).strip().lower() == "heldout":
                findings.append(f"{child_location}=heldout")
            if key == "partition" and str(child).strip().lower() in {
                "heldout",
                "test",
            }:
                findings.append(f"{child_location}={child}")
            findings.extend(heldout_markers(child, child_location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(heldout_markers(child, f"{location}[{index}]"))
    return findings


def scan_json_for_heldout_evidence(
    root: Path, excluded_roots: Iterable[Path] = ()
) -> list[str]:
    excluded = [path.resolve() for path in excluded_roots]
    findings: list[str] = []
    for path in sorted(root.rglob("*.json")):
        resolved = path.resolve()
        if any(inside(resolved, parent) for parent in excluded):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        for marker in heldout_markers(payload):
            findings.append(f"{path.relative_to(root)}:{marker}")
    return findings


def verify_input_hashes(claims: dict[str, object]) -> dict[str, str]:
    if not claims:
        raise ValueError("Post-rating manifest has no input hashes")
    verified: dict[str, str] = {}
    for raw_path, expected in sorted(claims.items()):
        path = Path(raw_path).resolve(strict=True)
        if not inside(path, REVISION):
            raise ValueError(f"Hashed input escapes revision workspace: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Input changed after post-rating run: {path}")
        verified[str(path.relative_to(REVISION))] = actual
    return verified


def code_hashes() -> dict[str, str]:
    files: list[Path] = []
    files.extend((REVISION / "tools").glob("*.py"))
    for root in (
        REVISION / "03_code_revision" / "revision_tools",
        REVISION / "03_code_revision" / "scripts",
        REVISION / "03_code_revision" / "configs",
    ):
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file())
    for path in (
        REVISION / "03_code_revision" / "requirements.txt",
        REVISION / "03_code_revision" / "README.md",
    ):
        if path.exists():
            files.append(path)
    return {
        path.relative_to(REVISION).as_posix(): sha256(path)
        for path in sorted(set(files))
        if "__pycache__" not in path.parts
    }


def validate_rating_summary(
    summary: dict[str, Any], split_sha: str
) -> dict[str, object]:
    if (
        summary.get("subset") != "development"
        or summary.get("development_only") is not True
        or summary.get("heldout_inspected") is not False
    ):
        raise ValueError("Rating summary is not development-only and pre-heldout")
    if normalize_subjects(summary.get("partition_subjects", [])) != tuple(
        sorted(EXPECTED_DEVELOPMENT)
    ):
        raise ValueError("Rating subjects do not match the frozen development split")
    if summary.get("cases") != 190:
        raise ValueError("Expected exactly 190 development ratings")
    if summary.get("raters") not in (2, 3):
        raise ValueError("Exactly two or three real rater forms are required")
    if int(summary.get("primary_independent_raters", 0)) < 2:
        raise ValueError("At least two full independent raters are required")
    if summary.get("qualified_rater_metadata_complete") is not True:
        raise ValueError("Qualified-rater metadata is incomplete")
    if int(summary.get("unresolved_cases", -1)) != 0:
        raise ValueError("All development rating disagreements must be resolved")
    if summary.get("split_sha256") != split_sha:
        raise ValueError("Rating summary split hash mismatch")
    for rater in summary.get("rater_qualification_summary", []):
        involvement = str(
            rater.get("registration_method_development_involvement", "")
        ).strip().lower()
        if involvement not in {"no", "false", "0"}:
            raise ValueError("A rater was involved in registration-method development")
    return {
        "raters": summary["raters"],
        "primary_independent_raters": summary["primary_independent_raters"],
        "cases": summary["cases"],
        "unresolved_cases": summary["unresolved_cases"],
        "positive_consensus_cases": summary.get("positive_consensus_cases"),
        "rating_summary_sha256": None,
    }


def main() -> None:
    args = parse_args()
    post_dir = args.post_rating_output_dir.resolve(strict=True)
    output = args.output_json.resolve()
    if not inside(post_dir, REVISION) or not inside(output, REVISION):
        raise ValueError("All paths must remain inside the revision workspace")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite freeze manifest: {output}")

    split_path = SPLIT_PATH.resolve(strict=True)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    validate_split(split)
    split_sha = sha256(split_path)

    pipeline_path = post_dir / "pipeline_manifest.json"
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    if (
        pipeline.get("status") != "complete"
        or pipeline.get("development_only") is not True
        or pipeline.get("heldout_inspected") is not False
        or pipeline.get("common_qc_frozen") is not True
    ):
        raise ValueError("Post-rating pipeline is not complete, frozen, and pre-heldout")
    steps = pipeline.get("steps", [])
    completed_steps = {step.get("name") for step in steps if step.get("returncode") == 0}
    if completed_steps != EXPECTED_PIPELINE_STEPS:
        missing = sorted(EXPECTED_PIPELINE_STEPS - completed_steps)
        extra = sorted(completed_steps - EXPECTED_PIPELINE_STEPS)
        raise ValueError(f"Unexpected post-rating step set; missing={missing}, extra={extra}")
    verified_inputs = verify_input_hashes(pipeline.get("input_hashes", {}))

    rating_summary_path = post_dir / "01_rating_consensus" / "rating_summary.json"
    rating_summary = json.loads(rating_summary_path.read_text(encoding="utf-8"))
    rating_record = validate_rating_summary(rating_summary, split_sha)
    rating_record["rating_summary_sha256"] = sha256(rating_summary_path)

    calibration_path = post_dir / "02_common_qc_calibration" / "calibration_summary.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if (
        calibration.get("common_qc_frozen") is not True
        or calibration.get("final_target_reached") is not True
        or not calibration.get("final_frozen_rule")
        or calibration.get("development_only") is not True
        or calibration.get("heldout_inspected") is not False
        or calibration.get("heldout_test_inspected") is not False
    ):
        raise ValueError("Common QC did not satisfy the prespecified freeze policy")
    if normalize_subjects(calibration.get("subjects", [])) != tuple(
        sorted(EXPECTED_DEVELOPMENT)
    ) or calibration.get("cases") != 190:
        raise ValueError("Common-QC calibration denominator or subjects changed")
    if calibration.get("split_sha256") != split_sha:
        raise ValueError("Common-QC split hash mismatch")
    if calibration.get("rating_summary_sha256") != sha256(rating_summary_path):
        raise ValueError("Common-QC rating-summary hash mismatch")
    if pipeline.get("calibration_summary_sha256") != sha256(calibration_path):
        raise ValueError("Pipeline calibration hash mismatch")

    arap_path = post_dir / "10_arap_selection" / "arap_selected_setting.json"
    arap = json.loads(arap_path.read_text(encoding="utf-8"))
    if (
        arap.get("development_only") is not True
        or arap.get("heldout_inspected") is not False
        or arap.get("selected_setting_id") not in {"ARAP-A", "ARAP-B", "ARAP-C", "ARAP-D"}
    ):
        raise ValueError("ARAP setting selection is not a valid development-only artifact")
    if arap.get("split_sha256") != split_sha:
        raise ValueError("ARAP selection split hash mismatch")
    if arap.get("common_qc_calibration_sha256") != sha256(calibration_path):
        raise ValueError("ARAP selection calibration hash mismatch")
    if pipeline.get("arap_selection_sha256") != sha256(arap_path):
        raise ValueError("Pipeline ARAP-selection hash mismatch")

    heldout_findings = scan_json_for_heldout_evidence(
        REVISION,
        excluded_roots=(REVISION / "01_submitted_snapshot",),
    )
    if heldout_findings:
        raise ValueError(
            "Held-out evidence already exists; freeze is forbidden: "
            + "; ".join(heldout_findings[:10])
        )

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate_status": "FROZEN_FOR_ONE_TIME_HELDOUT_EXECUTION",
        "one_time_execution_consumed": False,
        "development_only_until_creation": True,
        "heldout_inspected": False,
        "freeze_note": args.freeze_note,
        "warning": (
            "Do not alter thresholds, routing, model settings, baseline settings, "
            "code, or this manifest after held-out execution begins."
        ),
        "split": {
            "development_subjects": list(EXPECTED_DEVELOPMENT),
            "heldout_subjects": list(EXPECTED_HELDOUT),
            "split_sha256": split_sha,
        },
        "rating_provenance": rating_record,
        "common_qc": {
            "final_frozen_rule": calibration["final_frozen_rule"],
            "final_development_fit": calibration["final_development_fit"],
            "calibration_summary_sha256": sha256(calibration_path),
        },
        "arap": {
            "selected_setting_id": arap["selected_setting_id"],
            "selected_summary": arap["selected_summary"],
            "selection_sha256": sha256(arap_path),
        },
        "post_rating_pipeline": {
            "path": str(post_dir.relative_to(REVISION)),
            "manifest_sha256": sha256(pipeline_path),
            "completed_steps": sorted(completed_steps),
            "verified_input_hashes": verified_inputs,
        },
        "frozen_code_hashes": code_hashes(),
        "pre_freeze_heldout_evidence_scan": {
            "json_files_with_positive_heldout_markers": 0,
            "submitted_snapshot_excluded": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
