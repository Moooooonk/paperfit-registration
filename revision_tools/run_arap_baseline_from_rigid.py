#!/usr/bin/env python3
"""Open3D ARAP non-rigid baseline initialized by the common rigid stage."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

import run_anchor_aware_s8_pilot as audits
import run_mm_s8_from_rigid as s8_runner
import run_pairwise_mm_rigid as registration
import frozen_nonrigid_nasal_solver as frozen_nonrigid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scale-dict", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument(
        "--subset", choices=("development", "heldout", "all"), required=True
    )
    parser.add_argument("--target-anchor-json", type=Path, required=True)
    parser.add_argument("--source-method", choices=("hrn", "3ddfa"), required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--topology-file", type=Path)
    parser.add_argument("--rigid-output", type=Path, required=True)
    parser.add_argument("--branch-assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--control-count", type=int, default=1200)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-step-mm", type=float, default=4.0)
    parser.add_argument("--arap-iterations", type=int, default=30)
    parser.add_argument("--decimated-triangles", type=int, default=12000)
    parser.add_argument(
        "--energy", choices=("spokes", "smoothed"), default="spokes"
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--isolate-native-cases", action="store_true")
    parser.add_argument("--case-timeout-seconds", type=int, default=1800)
    parser.add_argument("--native-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def target_for_case(
    case: str,
    root: Path,
    scales: dict[str, Any],
    anchor_payload: dict[str, Any],
    anchor_json_path: Path,
) -> tuple[np.ndarray, np.ndarray, Path, Path, float, dict[str, Any]]:
    manifest = root / "prepared_cohort" / "facescape_frontal_pairs_manifest.csv"
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        by_pair = {row["pair_id"]: row for row in csv.DictReader(handle)}
    subject = f"{int(by_pair[case]['subject']):03d}"
    target_path = registration.checked_existing(
        Path(by_pair[f"{subject}_18_eye_closed"]["mesh"]), root, "target mesh"
    )
    mm_per_unit = float(scales[str(int(subject))]["18"][0])
    anchor_record = anchor_payload.get("anchors", {}).get(subject)
    if anchor_record is None:
        raise KeyError(f"No precomputed target anchor for subject {subject}")
    if str(anchor_record.get("target_pair")) != f"{subject}_18_eye_closed":
        raise ValueError(f"Target-anchor pair mismatch for subject {subject}")
    anchor = np.asarray(anchor_record["anchor_mm"], dtype=np.float64)
    if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
        raise ValueError(f"Invalid target anchor for subject {subject}")
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
    if len(target_full_roi) < 3000:
        raise ValueError(f"Target face ROI is too small for subject {subject}")
    target_sample = target_full_roi[
        registration.deterministic_indices(len(target_full_roi), 160000)
    ]
    return target_sample, anchor, target_path, roi_path, mm_per_unit, anchor_record


def clipped_targets(
    current: np.ndarray,
    indices: np.ndarray,
    tree: cKDTree,
    target: np.ndarray,
    max_step_mm: float,
) -> np.ndarray:
    _, nearest = tree.query(current[indices], k=1, workers=1)
    delta = target[nearest] - current[indices]
    norm = np.linalg.norm(delta, axis=1)
    scale = np.minimum(1.0, max_step_mm / np.maximum(norm, 1e-8))
    return current[indices] + delta * scale[:, None]


def run_arap(
    initial: np.ndarray,
    faces: np.ndarray,
    target: np.ndarray,
    masks: dict[str, np.ndarray],
    control_count: int,
    rounds: int,
    max_step_mm: float,
    iterations: int,
    energy_name: str,
    decimated_triangles: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    full_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(initial),
        o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
    )
    simplified = full_mesh.simplify_quadric_decimation(
        target_number_of_triangles=decimated_triangles
    )
    simplified.remove_duplicated_vertices()
    simplified.remove_duplicated_triangles()
    simplified.remove_degenerate_triangles()
    simplified.remove_non_manifold_edges()
    simplified.remove_unreferenced_vertices()
    decimated_initial = np.asarray(simplified.vertices, dtype=np.float64).copy()
    decimated_faces = np.asarray(simplified.triangles, dtype=np.int32).copy()
    if len(decimated_initial) < 1000 or len(decimated_faces) < 1000:
        raise RuntimeError("ARAP decimation produced an unexpectedly small mesh")

    original_tree = cKDTree(initial)
    _, nearest_original = original_tree.query(decimated_initial, k=1, workers=1)
    decimated_eye = masks["eye_soft"][nearest_original]
    decimated_full = masks["full_no_eye"][nearest_original]
    current = decimated_initial.copy()
    fixed_indices = np.flatnonzero(decimated_eye)
    fixed_constraint_indices = fixed_indices[
        registration.deterministic_indices(len(fixed_indices), 100)
    ]
    active_indices = np.flatnonzero(decimated_full & ~decimated_eye)
    active_indices = active_indices[
        registration.deterministic_indices(len(active_indices), control_count)
    ]
    constraint_indices = np.concatenate(
        [fixed_constraint_indices, active_indices]
    ).astype(np.int32)
    if len(np.unique(constraint_indices)) != len(constraint_indices):
        raise ValueError("ARAP constraints contain duplicate vertices")
    tree = cKDTree(target)
    energy = (
        o3d.geometry.DeformAsRigidAsPossibleEnergy.Spokes
        if energy_name == "spokes"
        else o3d.geometry.DeformAsRigidAsPossibleEnergy.Smoothed
    )
    history = []
    for round_index in range(1, rounds + 1):
        active_targets = clipped_targets(
            current, active_indices, tree, target, max_step_mm
        )
        constraint_positions = np.vstack(
            [decimated_initial[fixed_constraint_indices], active_targets]
        ).astype(np.float64)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(current),
            o3d.utility.Vector3iVector(decimated_faces),
        )
        deformed = mesh.deform_as_rigid_as_possible(
            o3d.utility.IntVector(constraint_indices.tolist()),
            o3d.utility.Vector3dVector(constraint_positions),
            max_iter=iterations,
            energy=energy,
            smoothed_alpha=0.01,
        )
        current = np.asarray(deformed.vertices, dtype=np.float64).copy()
        current[fixed_indices] = decimated_initial[fixed_indices]
        distances, _ = tree.query(current[active_indices], k=1, workers=1)
        history.append(
            {
                "round": round_index,
                "control_vertices": int(len(active_indices)),
                "fixed_eye_constraints": int(len(fixed_constraint_indices)),
                "restored_eye_vertices": int(len(fixed_indices)),
                "decimated_vertices": int(len(decimated_initial)),
                "decimated_triangles": int(len(decimated_faces)),
                "control_median_nn_mm": float(np.median(distances)),
                "control_p90_nn_mm": float(np.quantile(distances, 0.90)),
            }
        )
    decimated_displacement = current - decimated_initial
    decimated_tree = cKDTree(decimated_initial)
    distances, nearest = decimated_tree.query(initial, k=4, workers=1)
    weights = 1.0 / np.maximum(distances, 1e-6)
    weights /= np.sum(weights, axis=1, keepdims=True)
    interpolated_displacement = np.sum(
        decimated_displacement[nearest] * weights[:, :, None], axis=1
    )
    final = initial + interpolated_displacement
    final[masks["eye_soft"]] = initial[masks["eye_soft"]]
    return final, history


def quantiles(distances: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    values = distances[mask]
    return float(np.median(values)), float(np.quantile(values, 0.90))


def process_case(
    case: str,
    root_string: str,
    scale_string: str,
    anchor_path_string: str,
    source_method: str,
    source_root_string: str | None,
    topology_string: str | None,
    rigid_output_string: str,
    output_string: str,
    control_count: int,
    rounds: int,
    max_step_mm: float,
    arap_iterations: int,
    energy: str,
    decimated_triangles: int,
    pre_s8_branch: str,
    routing_failure_reasons: str,
) -> dict[str, Any]:
    root = Path(root_string)
    output = Path(output_string)
    row_path = output / "case_rows" / f"{case}.json"
    final_path = output / "final_vertices_mm" / f"{case}.npz"
    if row_path.exists() and final_path.exists():
        return json.loads(row_path.read_text(encoding="utf-8"))
    source_root = Path(source_root_string) if source_root_string else None
    topology = Path(topology_string) if topology_string else None
    source_path, _, faces, _, masks = registration.load_source(
        case, source_method, root, source_root, topology
    )
    initial = np.asarray(
        np.load(Path(rigid_output_string) / "rigid_vertices_mm" / f"{case}.npz")[
            "vertices_mm"
        ],
        dtype=np.float64,
    )
    scales = json.loads(Path(scale_string).read_text(encoding="utf-8"))
    anchor_json_path = Path(anchor_path_string)
    anchor_payload = json.loads(anchor_json_path.read_text(encoding="utf-8"))
    (
        target,
        target_anchor,
        target_path,
        target_roi_path,
        mm_per_unit,
        anchor_record,
    ) = target_for_case(case, root, scales, anchor_payload, anchor_json_path)
    started = time.time()
    final, history = run_arap(
        initial,
        faces,
        target,
        masks,
        control_count,
        rounds,
        max_step_mm,
        arap_iterations,
        energy,
        decimated_triangles,
    )
    tree = cKDTree(target)
    pre_distances, _ = tree.query(initial, k=1, workers=1)
    post_distances, _ = tree.query(final, k=1, workers=1)
    pre_full_median, pre_full_p90 = quantiles(pre_distances, masks["full_no_eye"])
    post_full_median, post_full_p90 = quantiles(post_distances, masks["full_no_eye"])
    pre_nose_median, pre_nose_p90 = quantiles(pre_distances, masks["nose"])
    post_nose_median, post_nose_p90 = quantiles(post_distances, masks["nose"])
    pre_anchor_surface = registration.anchor_consistency_distance(
        initial, masks, target_anchor
    )
    post_anchor_surface = registration.anchor_consistency_distance(
        final, masks, target_anchor
    )
    pre_anchor = float(
        np.linalg.norm(registration.source_anchor(initial, masks) - target_anchor)
    )
    post_anchor = float(
        np.linalg.norm(registration.source_anchor(final, masks) - target_anchor)
    )
    eye_displacement = np.linalg.norm(
        final[masks["eye_soft"]] - initial[masks["eye_soft"]], axis=1
    )
    displacement = np.linalg.norm(final - initial, axis=1)
    strain = audits.edge_strain(initial, final, faces, frozen_nonrigid)
    orientation = registration.orientation_metrics(final, masks)
    np.savez_compressed(final_path, vertices_mm=final.astype(np.float32))
    expression = case.split("_", 1)[1]
    expression_index, expression_name = expression.split("_", 1)
    row: dict[str, Any] = {
        "case": case,
        "subject": f"{int(case.split('_', 1)[0]):03d}",
        "expression": expression,
        "expression_index": int(expression_index),
        "expression_name": expression_name,
        "source_method": source_method,
        "pre_s8_branch": pre_s8_branch,
        "routing_failure_reasons": routing_failure_reasons,
        "completed": 1,
        "baseline": "Open3D iterative eye-constrained ARAP",
        "source_obj": str(source_path),
        "target_mesh": str(target_path),
        "target_face_roi_npz": str(target_roi_path),
        "target_face_roi_definition": str(
            anchor_record.get("face_roi_definition", "precomputed")
        ),
        "mm_per_target_unit": mm_per_unit,
        "control_count": control_count,
        "rounds": rounds,
        "max_step_mm": max_step_mm,
        "arap_iterations": arap_iterations,
        "energy": energy,
        "decimated_triangles_requested": decimated_triangles,
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
        "source_anchor_definition": registration.source_anchor_definition(source_method),
        "nose_anchor_metric_definition": (
            f"transformed {registration.source_anchor_definition(source_method)} "
            "to target nose-tip anchor"
        ),
        "target_anchor_definition": str(
            anchor_record.get("method", "precomputed")
        ),
        "post_orientation_pass": int(not bool(orientation["upside_down"])),
        "eye_fixed_max_mm": float(np.max(eye_displacement)) if len(eye_displacement) else 0.0,
        "displacement_p90_mm": float(np.quantile(displacement, 0.90)),
        "displacement_max_mm": float(np.max(displacement)),
        "runtime_seconds": float(time.time() - started),
        "final_vertices_npz": str(final_path),
        **strain,
    }
    (output / "case_details" / f"{case}.json").write_text(
        json.dumps({"row": row, "history": history}, indent=2), encoding="utf-8"
    )
    row_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def failed_arap_row(
    case: str,
    source_method: str,
    assignment: dict[str, str],
    reason: str,
    error: Exception | None = None,
) -> dict[str, Any]:
    expression = case.split("_", 1)[1]
    expression_index, expression_name = expression.split("_", 1)
    row: dict[str, Any] = {
        "case": case,
        "subject": f"{int(case.split('_', 1)[0]):03d}",
        "expression": expression,
        "expression_index": int(expression_index),
        "expression_name": expression_name,
        "source_method": source_method,
        "pre_s8_branch": assignment["pre_s8_branch"],
        "routing_failure_reasons": assignment.get("routing_failure_reasons", ""),
        "completed": 0,
        "baseline": "Open3D iterative eye-constrained ARAP",
        "execution_failure_reason": reason,
    }
    if error is not None:
        row["execution_exception_type"] = type(error).__name__
        row["execution_exception_message"] = str(error)
    return row


def retain_failed_arap_output(
    row: dict[str, Any], rigid_output: Path, output: Path
) -> dict[str, Any]:
    case = str(row["case"])
    rigid_row = json.loads(
        (rigid_output / "case_rows" / f"{case}.json").read_text(
            encoding="utf-8"
        )
    )
    rigid_vertices = rigid_output / "rigid_vertices_mm" / f"{case}.npz"
    if not rigid_vertices.is_file():
        raise FileNotFoundError(f"Missing rigid terminal output for {case}")
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
    row["rigid_vertices_npz"] = str(rigid_vertices)
    row["final_vertices_npz"] = str(rigid_vertices)
    row["terminal_output_definition"] = (
        "rigid result retained as the terminal failed ARAP output"
    )
    (output / "case_rows" / f"{case}.json").write_text(
        json.dumps(row, indent=2), encoding="utf-8"
    )
    (output / "case_details" / f"{case}.json").write_text(
        json.dumps(
            {
                "row": row,
                "history": [],
                "not_processed_due_to_invalid_pre_s8_evidence": (
                    row.get("execution_failure_reason")
                    == "invalid_pre_s8_evidence_not_processed"
                ),
                "arap_execution_failed": (
                    row.get("execution_failure_reason")
                    == "arap_execution_failure"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return row


def recorded_rows(output: Path, cases: list[str]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        path = output / "case_rows" / f"{case}.json"
        if not path.is_file():
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if str(row.get("case", "")) != case:
            raise ValueError(f"Recorded ARAP row has the wrong case ID: {path}")
        rows.append(row)
    return rows


def clean_subprocess_message(value: str, limit: int = 3000) -> str:
    without_ansi = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", value)
    printable = "".join(
        character for character in without_ansi if character in "\n\t" or character.isprintable()
    )
    return printable[-limit:].strip()


def isolated_child_command(
    args: argparse.Namespace,
    case: str,
    root: Path,
    scale: Path,
    split: Path,
    anchor_path: Path,
    rigid: Path,
    branch_assignments_path: Path,
    output: Path,
    source_root: Path | None,
    topology: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--root",
        str(root),
        "--scale-dict",
        str(scale),
        "--split",
        str(split),
        "--subset",
        args.subset,
        "--target-anchor-json",
        str(anchor_path),
        "--source-method",
        args.source_method,
        "--rigid-output",
        str(rigid),
        "--branch-assignments",
        str(branch_assignments_path),
        "--output-dir",
        str(output),
        "--control-count",
        str(args.control_count),
        "--rounds",
        str(args.rounds),
        "--max-step-mm",
        str(args.max_step_mm),
        "--arap-iterations",
        str(args.arap_iterations),
        "--decimated-triangles",
        str(args.decimated_triangles),
        "--energy",
        args.energy,
        "--workers",
        "1",
        "--case",
        case,
        "--resume",
        "--native-child",
    ]
    if source_root is not None:
        command.extend(("--source-root", str(source_root)))
    if topology is not None:
        command.extend(("--topology-file", str(topology)))
    return command


def run_cases_in_isolated_subprocesses(
    args: argparse.Namespace,
    cases: list[str],
    eligible_set: set[str],
    branch_assignments: dict[str, dict[str, str]],
    root: Path,
    scale: Path,
    split: Path,
    anchor_path: Path,
    rigid: Path,
    branch_assignments_path: Path,
    output: Path,
    source_root: Path | None,
    topology: Path | None,
    config: dict[str, Any],
) -> None:
    for case in cases:
        row_path = output / "case_rows" / f"{case}.json"
        if row_path.is_file():
            if not args.resume:
                raise FileExistsError(f"Recorded ARAP case already exists: {row_path}")
            print(json.dumps({"resumed_existing": case}), flush=True)
            continue
        assignment = branch_assignments[case]
        if case not in eligible_set:
            row = failed_arap_row(
                case,
                args.source_method,
                assignment,
                "invalid_pre_s8_evidence_not_processed",
            )
            retain_failed_arap_output(row, rigid, output)
        else:
            command = isolated_child_command(
                args,
                case,
                root,
                scale,
                split,
                anchor_path,
                rigid,
                branch_assignments_path,
                output,
                source_root,
                topology,
            )
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=args.case_timeout_seconds,
                    check=False,
                )
                if result.returncode != 0:
                    message = clean_subprocess_message(
                        "\n".join(part for part in (result.stdout, result.stderr) if part)
                    )
                    raise RuntimeError(
                        f"isolated ARAP child exited {result.returncode}: {message}"
                    )
                if not row_path.is_file():
                    raise RuntimeError("isolated ARAP child returned without a case row")
            except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
                for stale in (
                    output / "case_details" / f"{case}.json",
                    output / "final_vertices_mm" / f"{case}.npz",
                    row_path,
                ):
                    if stale.exists():
                        stale.unlink()
                row = failed_arap_row(
                    case,
                    args.source_method,
                    assignment,
                    "arap_execution_failure",
                    error,
                )
                row["native_case_isolation"] = 1
                retain_failed_arap_output(row, rigid, output)
        rows = recorded_rows(output, cases)
        write_summary(output, rows, config)
        print(
            json.dumps({"completed": case, "count": len(rows), "total": len(cases)}),
            flush=True,
        )
    rows = recorded_rows(output, cases)
    write_summary(output, rows, config)
    if len(rows) != len(cases):
        raise RuntimeError(
            f"ARAP denominator mismatch: recorded {len(rows)} of {len(cases)} cases"
        )


def write_summary(output: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda row: str(row["case"]))
    fields = sorted({key for row in rows for key in row})
    with (output / "arap_baseline_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    completed_rows = [
        row for row in rows if int(float(row.get("completed", 1))) == 1
    ]
    changes = [
        float(row["post_nose_median_mm"])
        - float(row["pre_nose_median_mm"])
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
        "arap_execution_failure_cases": sum(
            row.get("execution_failure_reason") == "arap_execution_failure"
            for row in rows
        ),
        "completed_subjects": len(
            {str(row["subject"]) for row in completed_rows}
        ),
        "mean_nose_median_change_mm": float(np.mean(changes)) if changes else None,
        "note": "No acceptance threshold was applied during the ARAP baseline.",
    }
    (output / "arap_baseline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if (
        args.control_count < 50
        or args.rounds < 1
        or args.max_step_mm <= 0
        or args.decimated_triangles < 2000
        or args.case_timeout_seconds < 1
    ):
        raise ValueError("Invalid ARAP parameters")
    root = args.root.expanduser().resolve(strict=True)
    revision = registration.checked_existing(root / "cmes_revision_20260816", root, "revision")
    scale = registration.checked_existing(args.scale_dict, revision, "scale dictionary")
    split = registration.checked_existing(args.split, revision, "identity split")
    anchor_path = registration.checked_existing(
        args.target_anchor_json, revision, "target-anchor JSON"
    )
    rigid = registration.checked_existing(args.rigid_output, revision, "rigid output")
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
    all_cases = sorted(path.stem for path in (rigid / "case_rows").glob("*.json"))
    expected_all_cases, _ = registration.read_selected_cases(
        manifest, split, args.subset, []
    )
    if all_cases != expected_all_cases:
        missing = sorted(set(expected_all_cases) - set(all_cases))
        extra = sorted(set(all_cases) - set(expected_all_cases))
        raise ValueError(
            f"Rigid cases do not match frozen {args.subset} split; "
            f"missing={missing}, extra={extra}"
        )
    cases = all_cases
    if args.case:
        requested = set(args.case)
        missing = sorted(requested - set(all_cases))
        if missing:
            raise ValueError(f"Cases missing from rigid output: {missing}")
        cases = sorted(requested)
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
    all_branch_assignments = s8_runner.load_branch_assignments(
        branch_assignments_path, all_cases, args.source_method
    )
    branch_assignments = {case: all_branch_assignments[case] for case in cases}
    eligible_cases = [
        case
        for case in cases
        if int(float(branch_assignments[case]["s8_eligible"])) == 1
    ]
    config = {
        "source_method": args.source_method,
        "subset": args.subset,
        "case_count": len(cases),
        "control_count": args.control_count,
        "rounds": args.rounds,
        "max_step_mm": args.max_step_mm,
        "arap_iterations": args.arap_iterations,
        "decimated_triangles": args.decimated_triangles,
        "energy": args.energy,
        "open3d_version": o3d.__version__,
        "rigid_output": str(rigid),
        "branch_assignments": str(branch_assignments_path),
        "branch_assignments_sha256": registration.sha256(branch_assignments_path),
        "eligible_cases": len(eligible_cases),
        "target_anchor_json": str(anchor_path),
        "target_anchor_json_sha256": registration.sha256(anchor_path),
        "identity_split": str(split),
        "identity_split_sha256": registration.sha256(split),
        "pair_manifest": str(manifest),
        "pair_manifest_sha256": registration.sha256(manifest),
        "common_target_face_roi": True,
        "acceptance_threshold_applied": False,
        "native_case_isolation": bool(args.isolate_native_cases and not args.native_child),
        "case_timeout_seconds": args.case_timeout_seconds,
        "post_decimation_mesh_cleanup": (
            "remove duplicated vertices/triangles, degenerate triangles, "
            "non-manifold edges, and unreferenced vertices"
        ),
    }
    if args.dry_run:
        print(json.dumps({**config, "first_case": cases[0], "last_case": cases[-1]}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=args.resume)
    for child in ("case_rows", "case_details", "final_vertices_mm"):
        (output / child).mkdir(exist_ok=True)
    eligible_set = set(eligible_cases)
    if args.isolate_native_cases and not args.native_child:
        run_cases_in_isolated_subprocesses(
            args,
            cases,
            eligible_set,
            branch_assignments,
            root,
            scale,
            split,
            anchor_path,
            rigid,
            branch_assignments_path,
            output,
            source_root,
            topology,
            config,
        )
        return
    rows = [
        failed_arap_row(
            case,
            args.source_method,
            branch_assignments[case],
            "invalid_pre_s8_evidence_not_processed",
        )
        for case in cases
        if case not in eligible_set
    ]
    for row in rows:
        retain_failed_arap_output(row, rigid, output)

    def arguments(case: str) -> tuple[Any, ...]:
        return (
            case,
            str(root),
            str(scale),
            str(anchor_path),
            args.source_method,
            str(source_root) if source_root else None,
            str(topology) if topology else None,
            str(rigid),
            str(output),
            args.control_count,
            args.rounds,
            args.max_step_mm,
            args.arap_iterations,
            args.energy,
            args.decimated_triangles,
            branch_assignments[case]["pre_s8_branch"],
            branch_assignments[case].get("routing_failure_reasons", ""),
        )

    if args.workers == 1:
        for case in eligible_cases:
            try:
                row = process_case(*arguments(case))
            except Exception as error:
                row = failed_arap_row(
                    case,
                    args.source_method,
                    branch_assignments[case],
                    "arap_execution_failure",
                    error,
                )
                retain_failed_arap_output(row, rigid, output)
            rows.append(row)
            write_summary(output, rows, config)
            print(
                json.dumps({"completed": case, "count": len(rows), "total": len(cases)}),
                flush=True,
            )
        write_summary(output, rows, config)
        if len(rows) != len(cases):
            raise RuntimeError(
                f"ARAP denominator mismatch: recorded {len(rows)} of {len(cases)} cases"
            )
        return

    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {
            pool.submit(process_case, *arguments(case)): case
            for case in eligible_cases
        }
        for future in as_completed(futures):
            case = futures[future]
            try:
                row = future.result()
            except Exception as error:
                row = failed_arap_row(
                    case,
                    args.source_method,
                    branch_assignments[case],
                    "arap_execution_failure",
                    error,
                )
                retain_failed_arap_output(row, rigid, output)
            rows.append(row)
            write_summary(output, rows, config)
            print(json.dumps({"completed": case, "count": len(rows), "total": len(cases)}), flush=True)
    write_summary(output, rows, config)
    if len(rows) != len(cases):
        raise RuntimeError(
            f"ARAP denominator mismatch: recorded {len(rows)} of {len(cases)} cases"
        )


if __name__ == "__main__":
    main()
