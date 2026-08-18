#!/usr/bin/env python3
"""Aggregate held-out ratings and compare them with the frozen common QC."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REVISION = Path(__file__).resolve().parents[1]
TOOLS = REVISION / "tools"
SPLIT = REVISION / "identity_disjoint_split.json"
CALIBRATION = (
    REVISION
    / "85_post_rating_development_pipeline_v3"
    / "02_common_qc_calibration"
    / "calibration_summary.json"
)
PACKAGE = REVISION / "97_blinded_rating_hrn_heldout_v11_v3"
PRIVATE_KEY = PACKAGE / "PRIVATE_DO_NOT_SHARE" / "PRIVATE_case_key.csv"
QC_ROOT = REVISION / "98_heldout_qc_and_statistics_v3"
QC_DECISIONS = (
    QC_ROOT
    / "01_primary_fail_closed_qc"
    / "01_hrn"
    / "common_qc_cases.csv"
)
PRIMARY_METRICS = (
    QC_ROOT
    / "00_inputs"
    / "primary_hrn"
    / "primary_fail_closed_rows.csv"
)
ANALYSIS_SCRIPTS = (
    Path(__file__).resolve(),
    TOOLS / "subject_anonymization.py",
    TOOLS / "aggregate_blinded_ratings.py",
    TOOLS / "analyze_qc_against_blinded_ratings.py",
    TOOLS / "analyze_common_qc_threshold_sensitivity.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rater-csv", type=Path, action="append", required=True)
    parser.add_argument("--rater-metadata-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def run_step(name: str, argv: list[str], records: list[dict[str, Any]]) -> None:
    completed = subprocess.run(argv, text=True, capture_output=True)
    records.append(
        {
            "name": name,
            "argv": argv,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    )
    if completed.returncode:
        raise RuntimeError(
            f"Step {name} failed ({completed.returncode}):\n"
            f"{completed.stdout}\n{completed.stderr}"
        )


def main() -> None:
    args = parse_args()
    if len(args.rater_csv) not in (2, 3):
        raise ValueError("Exactly two full forms or two full forms plus R3 are required")

    output = args.output_dir.resolve()
    if not inside(output, REVISION):
        raise ValueError("Output must remain inside the revision workspace")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")

    inputs = [
        SPLIT,
        CALIBRATION,
        PRIVATE_KEY,
        QC_DECISIONS,
        PRIMARY_METRICS,
        args.rater_metadata_csv,
        *args.rater_csv,
        *ANALYSIS_SCRIPTS,
    ]
    resolved_inputs = [path.resolve(strict=True) for path in inputs]
    for path in resolved_inputs:
        if not inside(path, REVISION):
            raise ValueError(f"Input escapes revision workspace: {path}")

    output.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "pipeline": "held-out blinded perceptual validation",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "subset": "heldout",
        "ratings_generated_or_imputed": False,
        "automatic_evidence_available_to_raters": False,
        "input_hashes": {str(path): sha256(path) for path in resolved_inputs},
        "steps": records,
    }
    manifest_path = output / "pipeline_manifest.json"

    consensus = output / "01_consensus"
    qc_validation = output / "02_frozen_qc_vs_consensus"
    sensitivity = output / "03_threshold_sensitivity_with_consensus"
    try:
        aggregate = [
            sys.executable,
            str(TOOLS / "aggregate_blinded_ratings.py"),
            "--key-csv",
            str(PRIVATE_KEY),
        ]
        for path in args.rater_csv:
            aggregate.extend(("--rater-csv", str(path.resolve())))
        aggregate.extend(
            (
                "--rater-metadata-csv",
                str(args.rater_metadata_csv.resolve()),
                "--split-json",
                str(SPLIT),
                "--subset",
                "heldout",
                "--output-dir",
                str(consensus),
            )
        )
        run_step("aggregate held-out blinded ratings", aggregate, records)

        run_step(
            "compare frozen held-out QC with resolved perceptual consensus",
            [
                sys.executable,
                str(TOOLS / "analyze_qc_against_blinded_ratings.py"),
                "--decision-csv",
                str(QC_DECISIONS),
                "--ratings-csv",
                str(consensus / "consensus_ratings.csv"),
                "--rating-summary-json",
                str(consensus / "rating_summary.json"),
                "--rater-metadata-csv",
                str(args.rater_metadata_csv.resolve()),
                "--split-json",
                str(SPLIT),
                "--partition",
                "test",
                "--output-dir",
                str(qc_validation),
            ],
            records,
        )

        run_step(
            "held-out threshold sensitivity with perceptual consensus",
            [
                sys.executable,
                str(TOOLS / "analyze_common_qc_threshold_sensitivity.py"),
                "--metrics-csv",
                str(PRIMARY_METRICS),
                "--calibration-json",
                str(CALIBRATION),
                "--split-json",
                str(SPLIT),
                "--subset",
                "heldout",
                "--ratings-csv",
                str(consensus / "consensus_ratings.csv"),
                "--output-dir",
                str(sensitivity),
                "--analysis-label",
                "HRN held-out frozen common-QC threshold sensitivity with consensus",
            ],
            records,
        )
        manifest["status"] = "complete"
        manifest["consensus_sha256"] = sha256(consensus / "consensus_ratings.csv")
        manifest["qc_rating_summary_sha256"] = sha256(
            qc_validation / "qc_rating_summary.json"
        )
        manifest["threshold_sensitivity_sha256"] = sha256(
            sensitivity / "threshold_sensitivity_summary.json"
        )
    except Exception as error:
        manifest["status"] = "failed_closed"
        manifest["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
