#!/usr/bin/env python3
"""Audit whether source and target nose anchors denote the same anatomy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

import run_pairwise_mm_rigid as registration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scale-dict", type=Path, required=True)
    parser.add_argument("--rigid-output", type=Path, required=True)
    parser.add_argument("--source-method", choices=("hrn", "3ddfa"), required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--topology-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve(strict=True)
    revision = registration.checked_existing(
        root / "cmes_revision_20260816", root, "revision root"
    )
    scale_path = registration.checked_existing(
        args.scale_dict, revision, "scale dictionary"
    )
    rigid_output = registration.checked_existing(
        args.rigid_output, revision, "rigid output"
    )
    source_root = (
        registration.checked_existing(args.source_root, revision, "source root")
        if args.source_root
        else None
    )
    topology_file = (
        registration.checked_existing(args.topology_file, revision, "topology file")
        if args.topology_file
        else None
    )
    if args.source_method == "3ddfa" and (source_root is None or topology_file is None):
        raise ValueError("3DDFA requires --source-root and --topology-file")
    output = args.output_dir.expanduser().resolve(strict=False)
    if output == revision or not registration.is_relative_to(output, revision):
        raise ValueError(f"Output escapes revision root: {output}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    output.mkdir(parents=True)

    scales = json.loads(scale_path.read_text(encoding="utf-8"))
    manifest_path = registration.checked_existing(
        root / "prepared_cohort" / "facescape_frontal_pairs_manifest.csv",
        root,
        "FaceScape pair manifest",
    )
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest = {row["pair_id"]: row for row in csv.DictReader(handle)}

    rigid_rows = sorted((rigid_output / "case_rows").glob("*.json"))
    if len(rigid_rows) != 190:
        raise ValueError(f"Expected 190 development rows, found {len(rigid_rows)}")

    rows: list[dict[str, Any]] = []
    target_cache: dict[str, tuple[np.ndarray, np.ndarray, cKDTree]] = {}
    for index, rigid_row_path in enumerate(rigid_rows, start=1):
        rigid_row = json.loads(rigid_row_path.read_text(encoding="utf-8"))
        case = str(rigid_row["case"])
        subject = f"{int(rigid_row['subject']):03d}"
        _, _, _, _, masks = registration.load_source(
            case, args.source_method, root, source_root, topology_file
        )
        rigid_vertices = np.asarray(
            np.load(rigid_output / "rigid_vertices_mm" / f"{case}.npz")["vertices_mm"],
            dtype=np.float64,
        )
        if subject not in target_cache:
            target_case = f"{subject}_18_eye_closed"
            target_path = registration.checked_existing(
                Path(manifest[target_case]["mesh"]), root, f"target mesh for {subject}"
            )
            target_world, _ = registration.load_trimesh(target_path)
            mm_per_unit = float(scales[str(int(subject))]["18"][0])
            target_vertices = registration.target_registration_frame(
                target_world, target_path.parent / "selected_camera.json"
            ) * mm_per_unit
            target_cache[subject] = (
                target_vertices,
                registration.target_nose_anchor(target_vertices),
                cKDTree(target_vertices),
            )
        target_vertices, target_anchor, target_tree = target_cache[subject]
        source_anchor = registration.source_anchor(rigid_vertices, masks)
        anchor_delta = source_anchor - target_anchor
        source_nose_tree = cKDTree(rigid_vertices[masks["nose"]])
        source_anchor_surface_mm = float(target_tree.query(source_anchor, k=1)[0])
        target_anchor_nose_surface_mm = float(source_nose_tree.query(target_anchor, k=1)[0])
        nose_distances = target_tree.query(rigid_vertices[masks["nose"]], k=1)[0]
        row = {
            "case": case,
            "subject": subject,
            "expression": str(rigid_row["expression"]),
            "source_method": args.source_method,
            "anchor_distance_mm": float(np.linalg.norm(anchor_delta)),
            "anchor_delta_x_mm": float(anchor_delta[0]),
            "anchor_delta_front_mm": float(anchor_delta[1]),
            "anchor_delta_vertical_mm": float(anchor_delta[2]),
            "source_anchor_x_mm": float(source_anchor[0]),
            "source_anchor_front_mm": float(source_anchor[1]),
            "source_anchor_vertical_mm": float(source_anchor[2]),
            "target_anchor_x_mm": float(target_anchor[0]),
            "target_anchor_front_mm": float(target_anchor[1]),
            "target_anchor_vertical_mm": float(target_anchor[2]),
            "source_anchor_to_target_surface_mm": source_anchor_surface_mm,
            "target_anchor_to_source_nose_surface_mm": target_anchor_nose_surface_mm,
            "nose_surface_median_mm": float(np.median(nose_distances)),
            "nose_surface_p90_mm": float(np.quantile(nose_distances, 0.90)),
        }
        rows.append(row)
        print(json.dumps({"completed": case, "count": index}), flush=True)

    write_csv(output / "anchor_coordinate_rows.csv", rows)
    subject_rows: list[dict[str, Any]] = []
    for subject in sorted({str(row["subject"]) for row in rows}):
        subset = [row for row in rows if row["subject"] == subject]
        subject_rows.append(
            {
                "subject": subject,
                "pairs": len(subset),
                "mean_anchor_distance_mm": float(
                    np.mean([float(row["anchor_distance_mm"]) for row in subset])
                ),
                "mean_abs_delta_x_mm": float(
                    np.mean([abs(float(row["anchor_delta_x_mm"])) for row in subset])
                ),
                "mean_abs_delta_front_mm": float(
                    np.mean([abs(float(row["anchor_delta_front_mm"])) for row in subset])
                ),
                "mean_abs_delta_vertical_mm": float(
                    np.mean([abs(float(row["anchor_delta_vertical_mm"])) for row in subset])
                ),
                "mean_source_anchor_to_target_surface_mm": float(
                    np.mean(
                        [
                            float(row["source_anchor_to_target_surface_mm"])
                            for row in subset
                        ]
                    )
                ),
                "mean_target_anchor_to_source_nose_surface_mm": float(
                    np.mean(
                        [
                            float(row["target_anchor_to_source_nose_surface_mm"])
                            for row in subset
                        ]
                    )
                ),
            }
        )
    write_csv(output / "anchor_coordinate_subject_summary.csv", subject_rows)
    summary = {
        "development_only": True,
        "source_method": args.source_method,
        "pairs": len(rows),
        "subjects": len(subject_rows),
        "mean_anchor_distance_mm": float(
            np.mean([float(row["anchor_distance_mm"]) for row in rows])
        ),
        "median_anchor_distance_mm": float(
            np.median([float(row["anchor_distance_mm"]) for row in rows])
        ),
        "mean_source_anchor_to_target_surface_mm": float(
            np.mean(
                [float(row["source_anchor_to_target_surface_mm"]) for row in rows]
            )
        ),
        "mean_target_anchor_to_source_nose_surface_mm": float(
            np.mean(
                [float(row["target_anchor_to_source_nose_surface_mm"]) for row in rows]
            )
        ),
        "interpretation": (
            "If Euclidean anchor distance is much larger than both surface distances, "
            "the two anchor estimators do not denote homologous anatomy and must not be "
            "used as a final acceptance criterion without revision."
        ),
    }
    (output / "anchor_coordinate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
