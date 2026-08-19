#!/usr/bin/env python3
"""Run one isolated full-resolution, eye-constrained ARAP case."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tools-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scale-dict", type=Path, required=True)
    parser.add_argument("--target-anchor-json", type=Path, required=True)
    parser.add_argument("--rigid-output", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--control-count", type=int, default=300)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-step-mm", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    return parser.parse_args()


def checked_output(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"{label} must remain inside the configured workspace root")
    return resolved


def edge_table(faces: np.ndarray) -> np.ndarray:
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def quantiles(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0, "median": None, "p90": None, "p99": None, "max": None}
    return {
        "count": int(len(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.90)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
    }


def strain_regions(
    initial: np.ndarray,
    final: np.ndarray,
    faces: np.ndarray,
    eye: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    edges = edge_table(faces)
    before = np.linalg.norm(initial[edges[:, 0]] - initial[edges[:, 1]], axis=1)
    after = np.linalg.norm(final[edges[:, 0]] - final[edges[:, 1]], axis=1)
    strain = np.abs(after - before) / np.maximum(before, 1e-8)
    left = eye[edges[:, 0]]
    right = eye[edges[:, 1]]
    regions = {
        "all": np.ones(len(edges), dtype=bool),
        "eye_eye": left & right,
        "eye_boundary": left ^ right,
        "non_eye_non_eye": ~left & ~right,
    }
    return {name: quantiles(strain[mask]) for name, mask in regions.items()}


def masked_distance_quantiles(
    distances: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    values = distances[np.asarray(mask, dtype=bool)]
    return {
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
    }


def target_for_case(
    case: str,
    root: Path,
    scales: dict[str, Any],
    anchor_payload: dict[str, Any],
    anchor_json_path: Path,
    registration: Any,
) -> tuple[np.ndarray, np.ndarray, Path, float]:
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
        target_full_roi = np.asarray(roi_payload["target_face_roi_mm"], dtype=np.float64)
    if len(target_full_roi) < 3000:
        raise ValueError(f"Target face ROI is too small for subject {subject}")
    target_sample = target_full_roi[
        registration.deterministic_indices(len(target_full_roi), 160000)
    ]
    return target_sample, anchor, roi_path, mm_per_unit


def main() -> None:
    args = parse_args()
    tools_dir = args.tools_dir.resolve(strict=True)
    sys.path.insert(0, str(tools_dir))
    import run_pairwise_mm_rigid as registration

    root = args.root.resolve(strict=True)
    output_json = checked_output(args.output_json, root, "JSON output")
    output_npz = checked_output(args.output_npz, root, "NPZ output")
    rigid_output = args.rigid_output.resolve(strict=True)
    source_path, _, faces, _, masks = registration.load_source(
        args.case, "hrn", root, None, None
    )
    initial_path = rigid_output / "rigid_vertices_mm" / f"{args.case}.npz"
    initial = np.asarray(np.load(initial_path)["vertices_mm"], dtype=np.float64)
    scale_dict = json.loads(args.scale_dict.resolve(strict=True).read_text(encoding="utf-8"))
    anchor_path = args.target_anchor_json.resolve(strict=True)
    anchor_payload = json.loads(anchor_path.read_text(encoding="utf-8"))
    target, target_anchor, target_roi_path, mm_per_unit = target_for_case(
        args.case, root, scale_dict, anchor_payload, anchor_path, registration
    )

    started = time.time()
    faces = np.asarray(faces, dtype=np.int32)
    eye_mask = np.asarray(masks["eye_soft"], dtype=bool)
    active_mask = np.asarray(masks["full_no_eye"], dtype=bool) & ~eye_mask
    fixed_indices = np.flatnonzero(eye_mask)
    active_indices = np.flatnonzero(active_mask)
    active_indices = active_indices[
        registration.deterministic_indices(len(active_indices), args.control_count)
    ]
    constraint_indices = np.concatenate((fixed_indices, active_indices)).astype(np.int32)
    if len(np.unique(constraint_indices)) != len(constraint_indices):
        raise ValueError("ARAP constraints contain duplicate vertices")

    target_tree = cKDTree(target)
    current = initial.copy()
    history: list[dict[str, Any]] = []
    energy = o3d.geometry.DeformAsRigidAsPossibleEnergy.Spokes
    for round_index in range(1, args.rounds + 1):
        _, nearest = target_tree.query(current[active_indices], k=1, workers=1)
        delta = target[nearest] - current[active_indices]
        norm = np.linalg.norm(delta, axis=1)
        scale = np.minimum(1.0, args.max_step_mm / np.maximum(norm, 1e-8))
        active_targets = current[active_indices] + delta * scale[:, None]
        constraint_positions = np.vstack(
            (initial[fixed_indices], active_targets)
        ).astype(np.float64)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(current),
            o3d.utility.Vector3iVector(faces),
        )
        deformed = mesh.deform_as_rigid_as_possible(
            o3d.utility.IntVector(constraint_indices.tolist()),
            o3d.utility.Vector3dVector(constraint_positions),
            max_iter=args.iterations,
            energy=energy,
            smoothed_alpha=0.01,
        )
        current = np.asarray(deformed.vertices, dtype=np.float64).copy()
        control_distances, _ = target_tree.query(
            current[active_indices], k=1, workers=1
        )
        history.append(
            {
                "round": round_index,
                "control_distance_mm": quantiles(control_distances),
                "full_resolution_strain": strain_regions(
                    initial, current, faces, eye_mask
                ),
            }
        )

    final = current
    pre_restore_eye_displacement = np.linalg.norm(
        final[eye_mask] - initial[eye_mask], axis=1
    )
    eye_displacement = np.linalg.norm(final[eye_mask] - initial[eye_mask], axis=1)

    pre_distances, _ = target_tree.query(initial, k=1, workers=1)
    post_distances, _ = target_tree.query(final, k=1, workers=1)
    orientation = registration.orientation_metrics(final, masks)
    full_strain = strain_regions(initial, final, np.asarray(faces), eye_mask)
    final_displacement = np.linalg.norm(final - initial, axis=1)
    result: dict[str, Any] = {
        "case": args.case,
        "source_obj": str(source_path),
        "target_face_roi_npz": str(target_roi_path),
        "mm_per_target_unit": mm_per_unit,
        "configuration": {
            "mesh_mode": "full",
            "control_count": args.control_count,
            "rounds": args.rounds,
            "max_step_mm": args.max_step_mm,
            "iterations": args.iterations,
            "energy": "spokes",
            "eye_constraints": "all",
            "transfer": "direct",
        },
        "full_resolution_vertices": int(len(initial)),
        "full_resolution_triangles": int(len(faces)),
        "fixed_eye_constraints": int(len(fixed_indices)),
        "active_constraints": int(len(active_indices)),
        "history": history,
        "pre_restore_eye_displacement_mm": quantiles(pre_restore_eye_displacement),
        "eye_displacement_mm": quantiles(eye_displacement),
        "eye_fixed_max_mm": float(np.max(eye_displacement)) if len(eye_displacement) else 0.0,
        "final_displacement_mm": quantiles(final_displacement),
        "full_edge_strain": full_strain,
        "pre_full_distance_mm": masked_distance_quantiles(
            pre_distances, masks["full_no_eye"]
        ),
        "post_full_distance_mm": masked_distance_quantiles(
            post_distances, masks["full_no_eye"]
        ),
        "pre_nose_distance_mm": masked_distance_quantiles(pre_distances, masks["nose"]),
        "post_nose_distance_mm": masked_distance_quantiles(post_distances, masks["nose"]),
        "pre_anchor_mm": float(
            np.linalg.norm(registration.source_anchor(initial, masks) - target_anchor)
        ),
        "post_anchor_mm": float(
            np.linalg.norm(registration.source_anchor(final, masks) - target_anchor)
        ),
        "post_orientation_pass": int(not bool(orientation["upside_down"])),
        "runtime_seconds": float(time.time() - started),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(output_npz, vertices_mm=final.astype(np.float32))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
