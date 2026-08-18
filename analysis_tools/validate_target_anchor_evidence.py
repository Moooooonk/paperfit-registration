#!/usr/bin/env python3
"""Validate target anchors, anatomical scale evidence, and ROI provenance."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DIRECT_RAY_METHOD = "perspective-correct triangle-ray intersection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-json", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument(
        "--subset", choices=("development", "heldout"), required=True
    )
    parser.add_argument(
        "--roi-root",
        type=Path,
        help="If supplied, require every referenced target-face ROI file.",
    )
    parser.add_argument(
        "--allow-visible-point-fallback",
        action="store_true",
        help="Permit recorded target-anchor or eye-corner fallback evidence.",
    )
    return parser.parse_args()


def finite_vector(value: object, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} contains a non-finite value")
    return result


def validate_payload(
    payload: dict[str, Any],
    split: dict[str, Any],
    subset: str,
    *,
    roi_root: Path | None = None,
    require_direct_ray_intersections: bool = True,
) -> dict[str, Any]:
    if payload.get("subset") != subset:
        raise ValueError(
            f"Target-anchor subset mismatch: {payload.get('subset')!r} != {subset!r}"
        )
    split_key = "development_subjects" if subset == "development" else "test_subjects"
    expected = {f"{int(subject):03d}" for subject in split[split_key]}
    declared = {f"{int(subject):03d}" for subject in payload.get("subjects", [])}
    anchors = payload.get("anchors")
    if not isinstance(anchors, dict):
        raise ValueError("Target-anchor payload lacks an anchors object")
    observed = {f"{int(subject):03d}" for subject in anchors}
    if expected != declared or expected != observed:
        raise ValueError(
            "Target-anchor subjects do not match the frozen split: "
            f"expected={sorted(expected)}, declared={sorted(declared)}, "
            f"observed={sorted(observed)}"
        )

    normalized_roi_root = roi_root.resolve(strict=True) if roi_root else None
    direct_anchor_count = 0
    direct_eye_count = 0
    interocular_values = []
    roi_vertex_counts = []
    for subject in sorted(expected):
        record = anchors[subject]
        if str(record.get("target_pair")) != f"{subject}_18_eye_closed":
            raise ValueError(f"Unexpected target pair for subject {subject}")
        finite_vector(record.get("anchor_mm"), 3, f"{subject} anchor_mm")
        interocular = float(record.get("interocular_distance_mm"))
        if not math.isfinite(interocular) or interocular <= 0.0:
            raise ValueError(f"Invalid interocular distance for subject {subject}")
        interocular_values.append(interocular)

        eye_points = record.get("eye_corner_points_mm")
        if not isinstance(eye_points, dict):
            raise ValueError(f"Missing eye-corner evidence for subject {subject}")
        for side in ("left", "right"):
            points = eye_points.get(side)
            if not isinstance(points, list) or len(points) != 2:
                raise ValueError(f"Expected two {side} canthus points for {subject}")
            for index, point in enumerate(points):
                finite_vector(point, 3, f"{subject} {side} canthus {index}")

        diagnostics = record.get("diagnostics")
        if not isinstance(diagnostics, dict):
            raise ValueError(f"Missing diagnostics for subject {subject}")
        anchor_method = str(diagnostics.get("anchor_surface_method", ""))
        eye_fallbacks = int(diagnostics.get("eye_corner_ray_fallbacks", -1))
        if anchor_method == DIRECT_RAY_METHOD:
            direct_anchor_count += 1
        if eye_fallbacks == 0:
            direct_eye_count += 1
        if require_direct_ray_intersections:
            if anchor_method != DIRECT_RAY_METHOD:
                raise ValueError(
                    f"Target nose-anchor fallback detected for subject {subject}: "
                    f"{anchor_method or 'missing method'}"
                )
            if eye_fallbacks != 0:
                raise ValueError(
                    f"Eye-corner ray fallback detected for subject {subject}: "
                    f"{eye_fallbacks}"
                )
        ray_count = int(diagnostics.get("ray_intersection_count", 0))
        if anchor_method == DIRECT_RAY_METHOD and ray_count < 1:
            raise ValueError(f"Missing direct nose-ray intersection for {subject}")
        roi_vertices = int(diagnostics.get("visible_face_roi_vertices", 0))
        if roi_vertices < 3000:
            raise ValueError(
                f"Visible target face ROI is too small for {subject}: {roi_vertices}"
            )
        roi_vertex_counts.append(roi_vertices)

        if normalized_roi_root is not None:
            relative = Path(str(record.get("face_roi_npz", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe ROI path for subject {subject}: {relative}")
            roi_path = (normalized_roi_root / relative).resolve(strict=True)
            if normalized_roi_root != roi_path and normalized_roi_root not in roi_path.parents:
                raise ValueError(f"ROI path escapes its evidence directory: {roi_path}")

    return {
        "subset": subset,
        "subjects": sorted(expected),
        "subject_count": len(expected),
        "direct_target_anchor_subjects": direct_anchor_count,
        "zero_eye_fallback_subjects": direct_eye_count,
        "direct_ray_intersections_required": require_direct_ray_intersections,
        "interocular_distance_mm_min": min(interocular_values),
        "interocular_distance_mm_max": max(interocular_values),
        "visible_face_roi_vertices_min": min(roi_vertex_counts),
        "visible_face_roi_vertices_max": max(roi_vertex_counts),
    }


def validate_files(
    anchor_json: Path,
    split_json: Path,
    subset: str,
    *,
    roi_root: Path | None = None,
    require_direct_ray_intersections: bool = True,
) -> dict[str, Any]:
    payload = json.loads(anchor_json.resolve(strict=True).read_text(encoding="utf-8"))
    split = json.loads(split_json.resolve(strict=True).read_text(encoding="utf-8"))
    return validate_payload(
        payload,
        split,
        subset,
        roi_root=roi_root,
        require_direct_ray_intersections=require_direct_ray_intersections,
    )


def main() -> None:
    args = parse_args()
    summary = validate_files(
        args.anchor_json,
        args.split_json,
        args.subset,
        roi_root=args.roi_root,
        require_direct_ray_intersections=not args.allow_visible_point_fallback,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
