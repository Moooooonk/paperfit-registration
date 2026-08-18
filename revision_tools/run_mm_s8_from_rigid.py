#!/usr/bin/env python3
"""Run scale-aware S8 from pair-wise rigid outputs without accepting cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

import run_anchor_aware_s8_pilot as s8
import run_pairwise_mm_rigid as registration
import frozen_nonrigid_nasal_solver as frozen_nonrigid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scale-dict", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--target-anchor-json", type=Path, required=True)
    parser.add_argument(
        "--subset", choices=("development", "heldout", "all"), required=True
    )
    parser.add_argument("--source-method", choices=("hrn", "3ddfa"), required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--topology-file", type=Path)
    parser.add_argument("--rigid-output", type=Path, required=True)
    parser.add_argument("--branch-assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contour-stages", type=int, choices=(4, 8, 12), default=8)
    parser.add_argument("--step-multiplier", type=float, default=1.0)
    parser.add_argument("--gain-multiplier", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--anchor-step-mm", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contour_schedule(stages: int) -> tuple[float, ...]:
    if stages == 8:
        return s8.S8_CONTOURS
    if stages == 4:
        return tuple(s8.S8_CONTOURS[index] for index in (0, 2, 4, 7))
    positions = np.linspace(0.0, len(s8.S8_CONTOURS) - 1, stages)
    return tuple(
        float(value)
        for value in np.interp(
            positions, np.arange(len(s8.S8_CONTOURS)), s8.S8_CONTOURS
        )
    )


def expected_scheduled_passes(contour_stages: int) -> int:
    return contour_stages + 5 + 3


def quantiles(distances: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    values = distances[np.asarray(mask, dtype=bool)]
    return float(np.median(values)), float(np.quantile(values, 0.90))


def target_for_case(
    case: str,
    root: Path,
    scale_dict: dict[str, Any],
    anchor_payload: dict[str, Any],
    anchor_json_path: Path,
) -> tuple[np.ndarray, np.ndarray, Path, Path, float]:
    manifest_path = root / "prepared_cohort" / "facescape_frontal_pairs_manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        by_pair = {row["pair_id"]: row for row in csv.DictReader(handle)}
    subject = f"{int(by_pair[case]['subject']):03d}"
    target_case = f"{subject}_18_eye_closed"
    target_path = registration.checked_existing(
        Path(by_pair[target_case]["mesh"]), root, "target mesh"
    )
    target_world, _ = registration.load_trimesh(target_path)
    mm_per_unit = float(scale_dict[str(int(subject))]["18"][0])
    target_full = registration.target_registration_frame(
        target_world, target_path.parent / "selected_camera.json"
    ) * mm_per_unit
    anchor_record = anchor_payload.get("anchors", {}).get(subject)
    if anchor_record is None:
        raise KeyError(f"No precomputed target anchor for subject {subject}")
    if str(anchor_record.get("target_pair")) != target_case:
        raise ValueError(f"Target-anchor pair mismatch for subject {subject}")
    target_anchor = np.asarray(anchor_record["anchor_mm"], dtype=np.float64)
    roi_relative = Path(str(anchor_record.get("face_roi_npz", "")))
    if not roi_relative.as_posix() or roi_relative.is_absolute():
        raise ValueError(f"Invalid target face-ROI path for subject {subject}")
    roi_path = registration.checked_existing(
        anchor_json_path.parent / roi_relative,
        root,
        f"target face ROI for subject {subject}",
    )
    with np.load(roi_path) as roi_payload:
        target_full_roi = np.asarray(
            roi_payload["target_face_roi_mm"], dtype=np.float64
        )
    target_sample = target_full_roi[
        registration.deterministic_indices(len(target_full_roi), 160000)
    ]
    return target_sample, target_anchor, target_path, roi_path, mm_per_unit


def process_case(
    case: str,
    root_string: str,
    scale_path_string: str,
    anchor_path_string: str,
    source_method: str,
    source_root_string: str | None,
    topology_file_string: str | None,
    rigid_output_string: str,
    output_string: str,
    stages: int,
    step_multiplier: float,
    gain_multiplier: float,
    anchor_weight: float,
    anchor_step_mm: float,
    pre_s8_branch: str,
    routing_failure_reasons: str,
) -> dict[str, Any]:
    root = Path(root_string)
    output = Path(output_string)
    row_path = output / "case_rows" / f"{case}.json"
    final_path = output / "final_vertices_mm" / f"{case}.npz"
    if row_path.exists() and final_path.exists():
        return json.loads(row_path.read_text(encoding="utf-8"))

    scale_dict = json.loads(Path(scale_path_string).read_text(encoding="utf-8"))
    anchor_json_path = Path(anchor_path_string)
    anchor_payload = json.loads(anchor_json_path.read_text(encoding="utf-8"))
    source_root = Path(source_root_string) if source_root_string else None
    topology_file = Path(topology_file_string) if topology_file_string else None
    source_path, _, faces, _, masks = registration.load_source(
        case, source_method, root, source_root, topology_file
    )
    rigid_path = Path(rigid_output_string) / "rigid_vertices_mm" / f"{case}.npz"
    rigid_payload = np.load(rigid_path)
    initial_mm = np.asarray(rigid_payload["vertices_mm"], dtype=np.float64)
    target_sample, target_anchor, target_path, target_roi_path, mm_per_unit = target_for_case(
        case, root, scale_dict, anchor_payload, anchor_json_path
    )

    variant = {
        "name": f"S{stages}_w{anchor_weight:g}_s{anchor_step_mm:g}",
        "anchor_weight": anchor_weight,
        "anchor_step_mm": anchor_step_mm,
        "contours": contour_schedule(stages),
        "step_multiplier": step_multiplier,
        "gain_multiplier": gain_multiplier,
    }
    started = time.time()
    final_mm, history = s8.run_variant(
        initial_mm,
        faces,
        target_sample,
        masks,
        target_anchor,
        frozen_nonrigid,
        variant,
    )
    expected_passes = expected_scheduled_passes(stages)
    if len(history) != expected_passes:
        raise RuntimeError(
            f"{case}: expected {expected_passes} scheduled S8 passes "
            f"({stages} contour + 5 local + 3 full), found {len(history)}"
        )
    executed_solves = sum(
        bool(item.get("executed_constrained_solve", True)) for item in history
    )
    skipped_passes = len(history) - executed_solves
    if not np.all(np.isfinite(final_mm)):
        raise FloatingPointError(f"{case}: non-finite S8 vertex coordinates")
    target_tree = cKDTree(target_sample)
    pre_distances, _ = target_tree.query(initial_mm, k=1, workers=1)
    post_distances, _ = target_tree.query(final_mm, k=1, workers=1)
    pre_full_median, pre_full_p90 = quantiles(
        pre_distances, masks["full_no_eye"]
    )
    post_full_median, post_full_p90 = quantiles(
        post_distances, masks["full_no_eye"]
    )
    pre_nose_median, pre_nose_p90 = quantiles(pre_distances, masks["nose"])
    post_nose_median, post_nose_p90 = quantiles(post_distances, masks["nose"])
    pre_anchor_surface = registration.anchor_consistency_distance(
        initial_mm, masks, target_anchor
    )
    post_anchor_surface = registration.anchor_consistency_distance(
        final_mm, masks, target_anchor
    )
    pre_anchor = float(
        np.linalg.norm(
            s8.source_anchor(initial_mm, masks["source_anchor"])
            - target_anchor
        )
    )
    post_anchor = float(
        np.linalg.norm(
            s8.source_anchor(final_mm, masks["source_anchor"])
            - target_anchor
        )
    )
    orientation = registration.orientation_metrics(final_mm, masks)
    eye_displacement = np.linalg.norm(
        final_mm[masks["eye_soft"]] - initial_mm[masks["eye_soft"]], axis=1
    )
    displacement = np.linalg.norm(final_mm - initial_mm, axis=1)
    strain = s8.edge_strain(initial_mm, final_mm, faces, frozen_nonrigid)
    audit_values = np.asarray(
        [
            pre_full_median,
            pre_full_p90,
            pre_nose_median,
            pre_nose_p90,
            pre_anchor,
            post_full_median,
            post_full_p90,
            post_nose_median,
            post_nose_p90,
            post_anchor,
            *strain.values(),
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(audit_values)) or np.any(audit_values < 0.0):
        raise FloatingPointError(f"{case}: invalid S8 metric evidence")
    np.savez_compressed(final_path, vertices_mm=final_mm.astype(np.float32))

    subject = f"{int(case.split('_', 1)[0]):03d}"
    expression = case.split("_", 1)[1]
    expression_index_text, expression_name = expression.split("_", 1)
    row: dict[str, Any] = {
        "case": case,
        "subject": subject,
        "expression": expression,
        "expression_index": int(expression_index_text),
        "expression_name": expression_name,
        "source_method": source_method,
        "pre_s8_branch": pre_s8_branch,
        "routing_failure_reasons": routing_failure_reasons,
        "completed": 1,
        "source_obj": str(source_path),
        "target_mesh": str(target_path),
        "target_face_roi_npz": str(target_roi_path),
        "mm_per_target_unit": mm_per_unit,
        "s8_name": "eight-stage nasal depth-contour schedule",
        "contour_stages": stages,
        "total_scheduled_passes": len(history),
        "expected_scheduled_passes": expected_passes,
        "executed_constrained_solves": executed_solves,
        "skipped_scheduled_passes": skipped_passes,
        "step_multiplier": step_multiplier,
        "gain_multiplier": gain_multiplier,
        "anchor_force_used": int(anchor_weight > 0.0 and anchor_step_mm > 0.0),
        "anchor_weight": anchor_weight,
        "anchor_step_mm": anchor_step_mm,
        "source_anchor_definition": registration.source_anchor_definition(source_method),
        "nose_anchor_metric_definition": (
            f"transformed {registration.source_anchor_definition(source_method)} "
            "to target nose-tip anchor"
        ),
        "pre_full_median_mm": pre_full_median,
        "pre_full_p90_mm": pre_full_p90,
        "pre_nose_median_mm": pre_nose_median,
        "pre_nose_p90_mm": pre_nose_p90,
        "pre_anchor_mm": pre_anchor,
        "pre_anchor_point_mm": pre_anchor,
        "pre_anchor_surface_mm": pre_anchor_surface,
        "post_full_median_mm": post_full_median,
        "post_full_p90_mm": post_full_p90,
        "post_nose_median_mm": post_nose_median,
        "post_nose_p90_mm": post_nose_p90,
        "post_anchor_mm": post_anchor,
        "post_anchor_point_mm": post_anchor,
        "post_anchor_surface_mm": post_anchor_surface,
        "post_orientation_pass": int(not bool(orientation["upside_down"])),
        "eye_fixed_max_mm": float(np.max(eye_displacement)) if len(eye_displacement) else 0.0,
        "displacement_p90_mm": float(np.quantile(displacement, 0.90)),
        "displacement_max_mm": float(np.max(displacement)),
        "runtime_seconds": float(time.time() - started),
        "rigid_vertices_npz": str(rigid_path),
        "final_vertices_npz": str(final_path),
        **strain,
    }
    detail = {
        "row": row,
        "history": history,
        "acceptance_threshold_applied": False,
        "fixed_eye_is_constraint_audit_not_alignment_evidence": True,
        "surface_anchor_distance_is_diagnostic_not_acceptance_evidence": True,
    }
    (output / "case_details" / f"{case}.json").write_text(
        json.dumps(detail, indent=2), encoding="utf-8"
    )
    row_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def read_cases(
    rigid_output: Path, explicit_cases: list[str]
) -> list[str]:
    rows = list((rigid_output / "case_rows").glob("*.json"))
    cases = sorted(path.stem for path in rows)
    if explicit_cases:
        requested = set(explicit_cases)
        missing = sorted(requested - set(cases))
        if missing:
            raise ValueError(f"Cases missing from rigid output: {missing}")
        cases = sorted(requested)
    return cases


def load_branch_assignments(
    path: Path, cases: list[str], source_method: str
) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"case", "subject", "source_method", "pre_s8_branch", "s8_eligible"}
    if not rows or not required.issubset(rows[0]):
        raise KeyError(f"Missing pre-S8 routing fields: {required - set(rows[0])}")
    assignments: dict[str, dict[str, str]] = {}
    valid_branches = {
        "rigid_pass",
        "anchor_only",
        "broad_failure",
        "residual_invalid_evidence",
    }
    for row in rows:
        case = str(row["case"])
        if case in assignments:
            raise ValueError(f"Duplicate pre-S8 routing case: {case}")
        subject = f"{int(row['subject']):03d}"
        if subject != f"{int(case.split('_', 1)[0]):03d}":
            raise ValueError(f"Pre-S8 routing case/subject mismatch: {case}")
        if str(row["source_method"]).strip().lower() != source_method:
            raise ValueError(f"Pre-S8 routing source-method mismatch: {case}")
        branch = str(row["pre_s8_branch"])
        if branch not in valid_branches:
            raise ValueError(f"Unknown pre-S8 branch for {case}: {branch}")
        eligible = int(float(row["s8_eligible"]))
        expected_eligible = int(branch != "residual_invalid_evidence")
        if eligible != expected_eligible:
            raise ValueError(f"Inconsistent S8 eligibility for {case}")
        assignments[case] = row
    if set(assignments) != set(cases):
        raise ValueError(
            "Pre-S8 routing cases do not match rigid cases: "
            f"missing={sorted(set(cases) - set(assignments))}, "
            f"extra={sorted(set(assignments) - set(cases))}"
        )
    return assignments


def ineligible_s8_row(
    case: str, source_method: str, assignment: dict[str, str]
) -> dict[str, Any]:
    expression = case.split("_", 1)[1]
    expression_index_text, expression_name = expression.split("_", 1)
    return {
        "case": case,
        "subject": f"{int(case.split('_', 1)[0]):03d}",
        "expression": expression,
        "expression_index": int(expression_index_text),
        "expression_name": expression_name,
        "source_method": source_method,
        "pre_s8_branch": assignment["pre_s8_branch"],
        "routing_failure_reasons": assignment.get("routing_failure_reasons", ""),
        "completed": 0,
        "execution_failure_reason": "invalid_pre_s8_evidence_not_processed",
    }


def execution_failure_s8_row(
    case: str,
    source_method: str,
    assignment: dict[str, str],
    error: Exception,
) -> dict[str, Any]:
    expression = case.split("_", 1)[1]
    expression_index_text, expression_name = expression.split("_", 1)
    return {
        "case": case,
        "subject": f"{int(case.split('_', 1)[0]):03d}",
        "expression": expression,
        "expression_index": int(expression_index_text),
        "expression_name": expression_name,
        "source_method": source_method,
        "pre_s8_branch": assignment["pre_s8_branch"],
        "routing_failure_reasons": assignment.get("routing_failure_reasons", ""),
        "completed": 0,
        "execution_failure_reason": "s8_execution_failure",
        "execution_exception_type": type(error).__name__,
        "execution_exception_message": str(error),
    }


def retain_ineligible_terminal_output(
    row: dict[str, Any], rigid_output: Path, output: Path
) -> dict[str, Any]:
    case = str(row["case"])
    rigid_row = json.loads(
        (rigid_output / "case_rows" / f"{case}.json").read_text(
            encoding="utf-8"
        )
    )
    rigid_vertices_path = rigid_output / "rigid_vertices_mm" / f"{case}.npz"
    if not rigid_vertices_path.is_file():
        raise FileNotFoundError(
            f"Missing terminal rigid mesh for invalid-evidence case: {case}"
        )
    for field in (
        "source_obj",
        "target_mesh",
        "target_face_roi_npz",
        "mm_per_target_unit",
        "source_anchor_definition",
        "nose_anchor_metric_definition",
    ):
        if field not in rigid_row:
            raise KeyError(f"Rigid row for {case} is missing {field}")
        row[field] = rigid_row[field]
    row["rigid_vertices_npz"] = str(rigid_vertices_path)
    row["final_vertices_npz"] = str(rigid_vertices_path)
    row["terminal_output_definition"] = (
        "rigid result retained as the terminal failed output because "
        "pre-S8 evidence was invalid"
    )
    (output / "case_rows" / f"{case}.json").write_text(
        json.dumps(row, indent=2), encoding="utf-8"
    )
    (output / "case_details" / f"{case}.json").write_text(
        json.dumps(
            {
                "row": row,
                "history": [],
                "acceptance_threshold_applied": False,
                "not_processed_due_to_invalid_pre_s8_evidence": (
                    row.get("execution_failure_reason")
                    == "invalid_pre_s8_evidence_not_processed"
                ),
                "s8_execution_failed": (
                    row.get("execution_failure_reason") == "s8_execution_failure"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return row


def write_summary(
    output: Path, rows: list[dict[str, Any]], config: dict[str, Any]
) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda row: str(row["case"]))
    fields = sorted({key for row in rows for key in row})
    with (output / "mm_s8_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    completed_rows = [
        row for row in rows if int(float(row.get("completed", 1))) == 1
    ]
    deltas = [
        float(row["post_nose_median_mm"]) - float(row["pre_nose_median_mm"])
        for row in completed_rows
    ]
    summary = {
        **config,
        "attempted_cases": len(rows),
        "completed_cases": len(completed_rows),
        "unprocessed_invalid_evidence_cases": sum(
            row.get("execution_failure_reason")
            == "invalid_pre_s8_evidence_not_processed"
            for row in rows
        ),
        "s8_execution_failure_cases": sum(
            row.get("execution_failure_reason") == "s8_execution_failure"
            for row in rows
        ),
        "cases_with_skipped_scheduled_passes": sum(
            int(float(row.get("skipped_scheduled_passes", 0))) > 0
            for row in completed_rows
        ),
        "skipped_scheduled_passes_total": sum(
            int(float(row.get("skipped_scheduled_passes", 0)))
            for row in completed_rows
        ),
        "completed_subjects": len(
            {str(row["subject"]) for row in completed_rows}
        ),
        "mean_nose_median_change_mm": (
            float(np.mean(deltas)) if deltas else None
        ),
        "orientation_failures": int(
            sum(not bool(row["post_orientation_pass"]) for row in completed_rows)
        ),
        "eye_constraint_failures": int(
            sum(
                float(row["eye_fixed_max_mm"]) > 1e-6
                for row in completed_rows
            )
        ),
        "note": "Evidence only; no branch-specific or common acceptance gate applied.",
    }
    (output / "mm_s8_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.step_multiplier <= 2.0:
        raise ValueError("--step-multiplier must be in (0, 2]")
    if not 0.0 < args.gain_multiplier <= 2.0:
        raise ValueError("--gain-multiplier must be in (0, 2]")
    if args.anchor_weight < 0.0 or args.anchor_step_mm < 0.0:
        raise ValueError("Anchor parameters must be nonnegative")
    if (args.anchor_weight == 0.0) != (args.anchor_step_mm == 0.0):
        raise ValueError(
            "--anchor-weight and --anchor-step-mm must both be zero or both positive"
        )
    root = args.root.expanduser().resolve(strict=True)
    revision = registration.checked_existing(root / "cmes_revision_20260816", root, "revision")
    scale = registration.checked_existing(args.scale_dict, revision, "scale dictionary")
    split = registration.checked_existing(args.split, revision, "identity split")
    anchor_path = registration.checked_existing(
        args.target_anchor_json, revision, "target-anchor JSON"
    )
    rigid_output = registration.checked_existing(args.rigid_output, revision, "rigid output")
    branch_assignments_path = registration.checked_existing(
        args.branch_assignments, revision, "pre-S8 branch assignments"
    )
    manifest = registration.checked_existing(
        root / "prepared_cohort" / "facescape_frontal_pairs_manifest.csv",
        root,
        "pair manifest",
    )
    source_root = (
        registration.checked_existing(args.source_root, revision, "source root")
        if args.source_root else None
    )
    topology = (
        registration.checked_existing(args.topology_file, revision, "topology")
        if args.topology_file else None
    )
    if args.source_method == "3ddfa" and (source_root is None or topology is None):
        raise ValueError("3DDFA requires --source-root and --topology-file")
    output = args.output_dir.expanduser().resolve(strict=False)
    if output == revision or not registration.is_relative_to(output, revision):
        raise ValueError(f"Output escapes revision root: {output}")
    if output.exists() and not args.resume and not args.dry_run:
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    cases = read_cases(rigid_output, args.case)
    expected_cases, _ = registration.read_selected_cases(
        manifest, split, args.subset, args.case
    )
    if cases != expected_cases:
        missing = sorted(set(expected_cases) - set(cases))
        extra = sorted(set(cases) - set(expected_cases))
        raise ValueError(
            f"Rigid cases do not match frozen {args.subset} split; "
            f"missing={missing}, extra={extra}"
        )
    branch_assignments = load_branch_assignments(
        branch_assignments_path, cases, args.source_method
    )
    eligible_cases = [
        case
        for case in cases
        if int(float(branch_assignments[case]["s8_eligible"])) == 1
    ]
    config = {
        "source_method": args.source_method,
        "subset": args.subset,
        "case_count": len(cases),
        "contour_stages": args.contour_stages,
        "contour_schedule": contour_schedule(args.contour_stages),
        "step_multiplier": args.step_multiplier,
        "gain_multiplier": args.gain_multiplier,
        "anchor_force_used": bool(args.anchor_weight > 0.0),
        "anchor_weight": args.anchor_weight,
        "anchor_step_mm": args.anchor_step_mm,
        "rigid_output": str(rigid_output),
        "branch_assignments": str(branch_assignments_path),
        "branch_assignments_sha256": sha256(branch_assignments_path),
        "s8_eligible_cases": len(eligible_cases),
        "identity_split": str(split),
        "identity_split_sha256": sha256(split),
        "pair_manifest": str(manifest),
        "pair_manifest_sha256": sha256(manifest),
        "target_anchor_json": str(anchor_path),
        "target_anchor_json_sha256": sha256(anchor_path),
        "scale_dictionary": str(scale),
        "scale_dictionary_sha256": sha256(scale),
    }
    if args.dry_run:
        print(json.dumps({**config, "first_case": cases[0], "last_case": cases[-1]}, indent=2))
        return

    output.mkdir(parents=True, exist_ok=args.resume)
    for child in ("case_rows", "case_details", "final_vertices_mm"):
        (output / child).mkdir(exist_ok=True)
    eligible_set = set(eligible_cases)
    rows: list[dict[str, Any]] = [
        ineligible_s8_row(case, args.source_method, branch_assignments[case])
        for case in cases
        if case not in eligible_set
    ]
    for row in rows:
        retain_ineligible_terminal_output(row, rigid_output, output)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_case,
                case,
                str(root),
                str(scale),
                str(anchor_path),
                args.source_method,
                str(source_root) if source_root else None,
                str(topology) if topology else None,
                str(rigid_output),
                str(output),
                args.contour_stages,
                args.step_multiplier,
                args.gain_multiplier,
                args.anchor_weight,
                args.anchor_step_mm,
                branch_assignments[case]["pre_s8_branch"],
                branch_assignments[case].get("routing_failure_reasons", ""),
            ): case
            for case in eligible_cases
        }
        for future in as_completed(futures):
            case = futures[future]
            try:
                row = future.result()
            except Exception as error:
                row = execution_failure_s8_row(
                    case,
                    args.source_method,
                    branch_assignments[case],
                    error,
                )
                retain_ineligible_terminal_output(row, rigid_output, output)
            rows.append(row)
            write_summary(output, rows, config)
            progress = {
                "completed": case,
                "count": len(rows),
                "total": len(cases),
                "success": int(float(row.get("completed", 0))) == 1,
            }
            if progress["success"]:
                progress["nose_median_change_mm"] = (
                    float(row["post_nose_median_mm"])
                    - float(row["pre_nose_median_mm"])
                )
            else:
                progress["execution_failure_reason"] = row[
                    "execution_failure_reason"
                ]
            print(json.dumps(progress), flush=True)
    write_summary(output, rows, config)
    if len(rows) != len(cases):
        raise RuntimeError(
            f"S8 denominator mismatch: recorded {len(rows)} of {len(cases)} cases"
        )


if __name__ == "__main__":
    main()
