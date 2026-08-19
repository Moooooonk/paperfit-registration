#!/usr/bin/env python3
"""Run full-resolution ARAP cases in isolated, single-process subprocesses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-case-script", type=Path, required=True)
    parser.add_argument("--tools-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scale-dict", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument(
        "--subset", choices=("development", "evaluation", "heldout"), required=True
    )
    parser.add_argument("--target-anchor-json", type=Path, required=True)
    parser.add_argument("--rigid-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--setting-id", default="ARAP-FR")
    parser.add_argument("--control-count", type=int, default=300)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-step-mm", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retain-final-vertices", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def failure_payload(case: str, error_type: str, message: str) -> dict[str, Any]:
    return {
        "case": case,
        "completed": 0,
        "error_type": error_type,
        "error_message": message[:4000],
    }


def flatten(case: str, payload: dict[str, Any], setting_id: str) -> dict[str, Any]:
    subject, expression = case.split("_", 1)
    base: dict[str, Any] = {
        "case": case,
        "subject": f"{int(subject):03d}",
        "expression": expression,
        "setting_id": setting_id,
        "completed": int(payload.get("completed", 1)),
        "error_type": payload.get("error_type", ""),
        "error_message": payload.get("error_message", ""),
    }
    if not base["completed"]:
        return base
    config = payload["configuration"]
    base.update(
        {
            "control_count": config["control_count"],
            "rounds": config["rounds"],
            "max_step_mm": config["max_step_mm"],
            "iterations": config["iterations"],
            "mesh_mode": config["mesh_mode"],
            "energy": config["energy"],
            "eye_constraints": config["eye_constraints"],
            "transfer": config["transfer"],
            "pre_full_median_mm": payload["pre_full_distance_mm"]["median"],
            "pre_full_p90_mm": payload["pre_full_distance_mm"]["p90"],
            "pre_nose_median_mm": payload["pre_nose_distance_mm"]["median"],
            "pre_nose_p90_mm": payload["pre_nose_distance_mm"]["p90"],
            "pre_anchor_mm": payload["pre_anchor_mm"],
            "post_orientation_pass": payload["post_orientation_pass"],
            "post_full_median_mm": payload["post_full_distance_mm"]["median"],
            "post_full_p90_mm": payload["post_full_distance_mm"]["p90"],
            "post_nose_median_mm": payload["post_nose_distance_mm"]["median"],
            "post_nose_p90_mm": payload["post_nose_distance_mm"]["p90"],
            "post_anchor_mm": payload["post_anchor_mm"],
            "edge_strain_p99": payload["full_edge_strain"]["all"]["p99"],
            "eye_boundary_strain_p99": payload["full_edge_strain"]["eye_boundary"]["p99"],
            "non_eye_strain_p99": payload["full_edge_strain"]["non_eye_non_eye"]["p99"],
            "displacement_p90_mm": payload["final_displacement_mm"]["p90"],
            "pre_restore_eye_displacement_p99_mm": payload[
                "pre_restore_eye_displacement_mm"
            ]["p99"],
            "eye_fixed_max_mm": payload["eye_fixed_max_mm"],
            "runtime_seconds": payload["runtime_seconds"],
        }
    )
    return base


def write_aggregate(
    output: Path,
    cases: list[str],
    setting_id: str,
    config: dict[str, Any],
) -> None:
    rows = []
    for case in cases:
        path = output / "case_details" / f"{case}.json"
        if path.exists():
            rows.append(flatten(case, json.loads(path.read_text(encoding="utf-8")), setting_id))
    fields = [
        "case",
        "subject",
        "expression",
        "setting_id",
        "completed",
        "error_type",
        "error_message",
        "control_count",
        "rounds",
        "max_step_mm",
        "iterations",
        "mesh_mode",
        "energy",
        "eye_constraints",
        "transfer",
        "pre_full_median_mm",
        "pre_full_p90_mm",
        "pre_nose_median_mm",
        "pre_nose_p90_mm",
        "pre_anchor_mm",
        "post_orientation_pass",
        "post_full_median_mm",
        "post_full_p90_mm",
        "post_nose_median_mm",
        "post_nose_p90_mm",
        "post_anchor_mm",
        "edge_strain_p99",
        "eye_boundary_strain_p99",
        "non_eye_strain_p99",
        "displacement_p90_mm",
        "pre_restore_eye_displacement_p99_mm",
        "eye_fixed_max_mm",
        "runtime_seconds",
    ]
    with (output / "arap_full_resolution_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    completed = [row for row in rows if int(row["completed"]) == 1]
    summary = {
        **config,
        "recorded_cases": len(rows),
        "completed_cases": len(completed),
        "failed_cases": len(rows) - len(completed),
        "pending_cases": len(cases) - len(rows),
        "total_runtime_seconds": sum(
            float(row.get("runtime_seconds") or 0.0) for row in completed
        ),
    }
    (output / "arap_full_resolution_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    script = args.single_case_script.resolve(strict=True)
    tools_dir = args.tools_dir.resolve(strict=True)
    root = args.root.resolve(strict=True)
    scale_dict = args.scale_dict.resolve(strict=True)
    split_path = args.split.resolve(strict=True)
    target_anchor = args.target_anchor_json.resolve(strict=True)
    rigid_output = args.rigid_output.resolve(strict=True)
    output = args.output_dir.resolve(strict=False)
    if output == root or root not in output.parents:
        raise ValueError("Output directory must remain inside the configured workspace root")
    if output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True, exist_ok=args.resume)
    (output / "case_details").mkdir(exist_ok=True)
    (output / "final_vertices_mm").mkdir(exist_ok=True)

    split = json.loads(split_path.read_text(encoding="utf-8"))
    split_key = (
        "development_subjects"
        if args.subset == "development"
        else "test_subjects"
    )
    expected_subjects = {f"{int(value):03d}" for value in split[split_key]}
    cases = sorted(path.stem for path in (rigid_output / "case_rows").glob("*.json"))
    observed_subjects = {f"{int(case.split('_', 1)[0]):03d}" for case in cases}
    if observed_subjects != expected_subjects or len(cases) != 19 * len(expected_subjects):
        raise ValueError("Rigid output does not match the requested identity partition")
    if args.case_limit is not None:
        if args.case_limit < 1:
            raise ValueError("case-limit must be positive")
        cases = cases[: args.case_limit]

    config: dict[str, Any] = {
        "setting_id": args.setting_id,
        "subset": args.subset,
        "attempted_cases": len(cases),
        "control_count": args.control_count,
        "rounds": args.rounds,
        "max_step_mm": args.max_step_mm,
        "iterations": args.iterations,
        "mesh_mode": "full",
        "energy": "spokes",
        "eye_constraints": "all",
        "transfer": "direct",
        "single_case_script_sha256": sha256(script),
        "split_sha256": sha256(split_path),
        "scale_dict_sha256": sha256(scale_dict),
        "target_anchor_sha256": sha256(target_anchor),
        "execution_policy": "one isolated subprocess at a time; one numerical thread",
        "retain_final_vertices": bool(args.retain_final_vertices),
    }
    (output / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    for index, case in enumerate(cases, start=1):
        detail_path = output / "case_details" / f"{case}.json"
        final_path = output / "final_vertices_mm" / f"{case}.npz"
        if args.resume and detail_path.exists():
            existing = json.loads(detail_path.read_text(encoding="utf-8"))
            if int(existing.get("completed", 1)) == 0:
                continue
            if not args.retain_final_vertices or final_path.exists():
                continue
        command = [
            sys.executable,
            str(script),
            "--tools-dir",
            str(tools_dir),
            "--root",
            str(root),
            "--scale-dict",
            str(scale_dict),
            "--target-anchor-json",
            str(target_anchor),
            "--rigid-output",
            str(rigid_output),
            "--case",
            case,
            "--control-count",
            str(args.control_count),
            "--rounds",
            str(args.rounds),
            "--max-step-mm",
            str(args.max_step_mm),
            "--iterations",
            str(args.iterations),
            "--output-json",
            str(detail_path),
            "--output-npz",
            str(final_path),
        ]
        started = time.time()
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=args.timeout_seconds,
                env=environment,
                check=False,
            )
            if result.returncode != 0 or not detail_path.exists() or not final_path.exists():
                payload = failure_payload(
                    case,
                    "subprocess_failure",
                    f"returncode={result.returncode}; stderr={result.stderr}",
                )
                payload["runtime_seconds"] = time.time() - started
                detail_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            elif not args.retain_final_vertices:
                final_path.unlink()
        except subprocess.TimeoutExpired as error:
            payload = failure_payload(case, "timeout", str(error))
            payload["runtime_seconds"] = time.time() - started
            detail_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_aggregate(output, cases, args.setting_id, config)
        print(json.dumps({"case": case, "index": index, "total": len(cases)}), flush=True)
    write_aggregate(output, cases, args.setting_id, config)


if __name__ == "__main__":
    main()
