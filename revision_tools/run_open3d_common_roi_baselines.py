#!/usr/bin/env python3
"""Rigid Open3D baselines using the exact common target face ROI and QC evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

import run_mm_s8_from_rigid as s8_runner
import run_pairwise_mm_rigid as registration


METHODS = (
    "prescale_only",
    "prescale_icp",
    "fpfh_fgr",
    "fpfh_fgr_icp",
    "fpfh_ransac",
    "fpfh_ransac_icp",
)


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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prescale-mode",
        choices=("bbox", "anatomical_interocular"),
        default="bbox",
    )
    parser.add_argument("--voxel-divisor", type=float, default=42.0)
    parser.add_argument("--voxel-floor-mm", type=float, default=3.0)
    parser.add_argument("--ransac-iterations", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def to_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return cloud


def weighted_sample(
    points: np.ndarray, weights: np.ndarray, maximum: int, seed: int
) -> np.ndarray:
    indices = np.flatnonzero(weights > 0.08)
    if len(indices) > maximum:
        probabilities = np.maximum(weights[indices], 0.0).astype(np.float64)
        probabilities /= probabilities.sum()
        rng = np.random.default_rng(seed)
        indices = rng.choice(indices, maximum, replace=False, p=probabilities)
    return points[indices]


def preprocess(
    points: np.ndarray, voxel_mm: float
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    cloud = to_point_cloud(points).voxel_down_sample(voxel_mm)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=2.0 * voxel_mm, max_nn=30)
    )
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=5.0 * voxel_mm, max_nn=100),
    )
    return cloud, feature


def bbox_prescale(
    source: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    source_center = np.median(source, axis=0)
    target_center = np.median(target, axis=0)
    source_extent = np.quantile(source, 0.95, axis=0) - np.quantile(
        source, 0.05, axis=0
    )
    target_extent = np.quantile(target, 0.95, axis=0) - np.quantile(
        target, 0.05, axis=0
    )
    scale = float(np.median(target_extent / np.maximum(source_extent, 1e-8)))
    rotation = np.eye(3, dtype=np.float64)
    translation = target_center - scale * source_center
    return scale, rotation, translation


def matrix_to_similarity(
    matrix: np.ndarray,
    prescale: float,
    prerotation: np.ndarray,
    pretranslation: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    linear = np.asarray(matrix[:3, :3], dtype=np.float64)
    delta_scale = float(np.cbrt(abs(np.linalg.det(linear))))
    if not np.isfinite(delta_scale) or delta_scale < 1e-8:
        raise ValueError("Open3D returned a degenerate similarity matrix")
    delta_rotation = linear / delta_scale
    u, _, vt = np.linalg.svd(delta_rotation)
    delta_rotation = u @ vt
    if np.linalg.det(delta_rotation) < 0.0:
        u[:, -1] *= -1.0
        delta_rotation = u @ vt
    scale = delta_scale * prescale
    rotation = delta_rotation @ prerotation
    translation = (
        delta_scale * (pretranslation @ delta_rotation.T) + matrix[:3, 3]
    )
    return float(scale), rotation, translation


def refine_icp(
    source_prescaled: np.ndarray,
    target: np.ndarray,
    initial_matrix: np.ndarray,
    voxel_mm: float,
) -> np.ndarray:
    source_cloud = to_point_cloud(source_prescaled)
    target_cloud = to_point_cloud(target)
    result = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        2.5 * voxel_mm,
        initial_matrix,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
    )
    return np.asarray(result.transformation, dtype=np.float64)


def evaluate(
    case: str,
    source_method: str,
    method: str,
    source: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    masks: dict[str, np.ndarray],
    target: np.ndarray,
    target_anchor: np.ndarray,
    runtime_seconds: float,
    voxel_mm: float,
    error: str = "",
) -> dict[str, Any]:
    expression = case.split("_", 1)[1]
    expression_index, expression_name = expression.split("_", 1)
    common = {
        "case": case,
        "subject": f"{int(case.split('_', 1)[0]):03d}",
        "expression": expression,
        "expression_index": int(expression_index),
        "expression_name": expression_name,
        "source_method": source_method,
        "baseline_method": method,
        "source_anchor_definition": registration.source_anchor_definition(
            source_method
        ),
        "nose_anchor_metric_definition": (
            f"transformed {registration.source_anchor_definition(source_method)} "
            "to target nose-tip anchor"
        ),
        "attempted": 1,
        "runtime_seconds": runtime_seconds,
        "voxel_mm": voxel_mm,
        "error": error,
    }
    if error:
        return {
            **common,
            "completed": 0,
            "post_full_median_mm": None,
            "post_full_p90_mm": None,
            "post_nose_median_mm": None,
            "post_nose_p90_mm": None,
            "post_anchor_mm": None,
            "post_anchor_point_mm": None,
            "post_anchor_surface_mm": None,
            "post_orientation_pass": 0,
            "eye_fixed_max_mm": None,
            "edge_strain_p99": None,
        }
    aligned = registration.transform(source, scale, rotation, translation)
    tree = cKDTree(target)
    distances, _ = tree.query(aligned, k=1, workers=1)
    full_median, full_p90 = s8_runner.quantiles(distances, masks["full_no_eye"])
    nose_median, nose_p90 = s8_runner.quantiles(distances, masks["nose"])
    orientation = registration.orientation_metrics(aligned, masks)
    return {
        **common,
        "completed": 1,
        "scale_source_to_mm": scale,
        "rotation_det": float(np.linalg.det(rotation)),
        "post_full_median_mm": full_median,
        "post_full_p90_mm": full_p90,
        "post_nose_median_mm": nose_median,
        "post_nose_p90_mm": nose_p90,
        "post_anchor_mm": float(
            np.linalg.norm(registration.source_anchor(aligned, masks) - target_anchor)
        ),
        "post_anchor_point_mm": float(
            np.linalg.norm(registration.source_anchor(aligned, masks) - target_anchor)
        ),
        "post_anchor_surface_mm": registration.anchor_consistency_distance(
            aligned, masks, target_anchor
        ),
        "post_orientation_pass": int(not bool(orientation["upside_down"])),
        "eye_fixed_max_mm": 0.0,
        "edge_strain_p99": 0.0,
    }


def process_case(
    case: str,
    root_string: str,
    scale_path_string: str,
    anchor_path_string: str,
    source_method: str,
    source_root_string: str | None,
    topology_string: str | None,
    output_string: str,
    voxel_divisor: float,
    voxel_floor_mm: float,
    ransac_iterations: int,
    prescale_mode: str,
) -> list[dict[str, Any]]:
    root = Path(root_string)
    output = Path(output_string)
    row_path = output / "case_rows" / f"{case}.json"
    if row_path.exists():
        return json.loads(row_path.read_text(encoding="utf-8"))["rows"]
    scales = json.loads(Path(scale_path_string).read_text(encoding="utf-8"))
    anchor_path = Path(anchor_path_string)
    anchors = json.loads(anchor_path.read_text(encoding="utf-8"))
    source_root = Path(source_root_string) if source_root_string else None
    topology = Path(topology_string) if topology_string else None
    _, source, _, fit_weight, masks = registration.load_source(
        case, source_method, root, source_root, topology
    )
    target, target_anchor, _, _, _ = s8_runner.target_for_case(
        case, root, scales, anchors, anchor_path
    )
    subject = f"{int(case.split('_', 1)[0]):03d}"
    anchor_record = anchors.get("anchors", {}).get(subject)
    if anchor_record is None:
        raise KeyError(f"Missing target anatomical record for {subject}")
    target_interocular_mm = float(anchor_record["interocular_distance_mm"])
    source_interocular = registration.source_interocular_distance(source, masks)
    seed = int(hashlib.sha256(case.encode("utf-8")).hexdigest()[:8], 16) % 2147483647
    o3d.utility.random.seed(seed)
    source_fit = weighted_sample(source, fit_weight, 28000, seed)
    target_fit = target[
        registration.deterministic_indices(len(target), 90000)
    ]
    if prescale_mode == "anatomical_interocular":
        prescale = registration.anatomical_scale_from_interocular(
            source_interocular, target_interocular_mm
        )
        prerotation = np.eye(3, dtype=np.float64)
        pretranslation = np.median(target_fit, axis=0) - prescale * np.median(
            source_fit, axis=0
        )
    else:
        prescale, prerotation, pretranslation = bbox_prescale(source_fit, target_fit)
    source_prescaled = registration.transform(
        source_fit, prescale, prerotation, pretranslation
    )
    target_extent = float(
        np.linalg.norm(
            np.quantile(target_fit, 0.95, axis=0)
            - np.quantile(target_fit, 0.05, axis=0)
        )
    )
    voxel_mm = max(target_extent / voxel_divisor, voxel_floor_mm)
    source_down, source_feature = preprocess(source_prescaled, voxel_mm)
    target_down, target_feature = preprocess(target_fit, voxel_mm)
    correspondence_distance = 2.5 * voxel_mm
    identity = np.eye(4, dtype=np.float64)
    transforms: dict[str, np.ndarray | str] = {"prescale_only": identity}
    started = time.time()
    try:
        transforms["prescale_icp"] = refine_icp(
            source_prescaled, target_fit, identity, voxel_mm
        )
    except Exception as exc:  # pragma: no cover - external library failure path
        transforms["prescale_icp"] = f"{type(exc).__name__}: {exc}"
    try:
        fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down,
            target_down,
            source_feature,
            target_feature,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=correspondence_distance
            ),
        )
        transforms["fpfh_fgr"] = np.asarray(fgr.transformation, dtype=np.float64)
        transforms["fpfh_fgr_icp"] = refine_icp(
            source_prescaled,
            target_fit,
            transforms["fpfh_fgr"],
            voxel_mm,
        )
    except Exception as exc:  # pragma: no cover - external library failure path
        message = f"{type(exc).__name__}: {exc}"
        transforms.setdefault("fpfh_fgr", message)
        transforms.setdefault("fpfh_fgr_icp", message)
    try:
        ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_feature,
            target_feature,
            True,
            correspondence_distance,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(
                prescale_mode == "bbox"
            ),
            4,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9
                ),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    correspondence_distance
                ),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(
                ransac_iterations, 0.999
            ),
        )
        transforms["fpfh_ransac"] = np.asarray(
            ransac.transformation, dtype=np.float64
        )
        transforms["fpfh_ransac_icp"] = refine_icp(
            source_prescaled,
            target_fit,
            transforms["fpfh_ransac"],
            voxel_mm,
        )
    except Exception as exc:  # pragma: no cover - external library failure path
        message = f"{type(exc).__name__}: {exc}"
        transforms.setdefault("fpfh_ransac", message)
        transforms.setdefault("fpfh_ransac_icp", message)

    runtime = float(time.time() - started)
    rows = []
    details = {}
    for method in METHODS:
        value = transforms.get(method, "missing transform")
        if isinstance(value, str):
            row = evaluate(
                case,
                source_method,
                method,
                source,
                prescale,
                prerotation,
                pretranslation,
                masks,
                target,
                target_anchor,
                runtime,
                voxel_mm,
                value,
            )
            details[method] = {"error": value}
        else:
            try:
                scale, rotation, translation = matrix_to_similarity(
                    value, prescale, prerotation, pretranslation
                )
                row = evaluate(
                    case,
                    source_method,
                    method,
                    source,
                    scale,
                    rotation,
                    translation,
                    masks,
                    target,
                    target_anchor,
                    runtime,
                    voxel_mm,
                )
                details[method] = {
                    "scale": scale,
                    "rotation": rotation.tolist(),
                    "translation_mm": translation.tolist(),
                }
            except Exception as exc:  # pragma: no cover - Open3D result audit path
                message = f"{type(exc).__name__}: {exc}"
                row = evaluate(
                    case,
                    source_method,
                    method,
                    source,
                    prescale,
                    prerotation,
                    pretranslation,
                    masks,
                    target,
                    target_anchor,
                    runtime,
                    voxel_mm,
                    message,
                )
                details[method] = {"error": message}
        completed = int(row["completed"])
        row.update(
            {
                "prescale_mode": prescale_mode,
                "target_interocular_mm": target_interocular_mm,
                "source_interocular_before_scaling": source_interocular,
                "anatomical_scale_residual_mm": (
                    abs(
                        float(row["scale_source_to_mm"])
                        * source_interocular
                        - target_interocular_mm
                    )
                    if completed
                    else None
                ),
                "ransac_similarity_scale_enabled": int(prescale_mode == "bbox"),
            }
        )
        rows.append(row)
    row_path.write_text(
        json.dumps(
            {
                "rows": rows,
                "transforms": details,
                "source_down_points": len(source_down.points),
                "target_down_points": len(target_down.points),
                "target_policy": "common camera-visible MediaPipe face-oval ROI",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows


def failed_case_rows(
    case: str,
    source_method: str,
    prescale_mode: str,
    error: Exception,
) -> list[dict[str, Any]]:
    message = f"{type(error).__name__}: {error}"
    rows = []
    for method in METHODS:
        row = evaluate(
            case,
            source_method,
            method,
            np.empty((0, 3), dtype=np.float64),
            1.0,
            np.eye(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            {},
            np.empty((0, 3), dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            0.0,
            0.0,
            message,
        )
        row.update(
            {
                "prescale_mode": prescale_mode,
                "target_interocular_mm": None,
                "source_interocular_before_scaling": None,
                "anatomical_scale_residual_mm": None,
                "ransac_similarity_scale_enabled": int(prescale_mode == "bbox"),
            }
        )
        rows.append(row)
    return rows


def write_summary(
    output: Path, rows: list[dict[str, Any]], config: dict[str, Any]
) -> None:
    if not rows:
        return
    ordered = sorted(rows, key=lambda row: (str(row["case"]), str(row["baseline_method"])))
    fields = sorted({key for row in ordered for key in row})
    with (output / "open3d_common_roi_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)
    by_method = {}
    for method in METHODS:
        subset = [row for row in ordered if row["baseline_method"] == method]
        by_method[method] = {
            "attempted": len(subset),
            "completed": sum(int(row["completed"]) for row in subset),
            "errors": sum(not int(row["completed"]) for row in subset),
        }
    successful_rows = sum(int(row["completed"]) for row in ordered)
    (output / "open3d_common_roi_summary.json").write_text(
        json.dumps(
            {
                **config,
                "recorded_rows": len(ordered),
                "completed_rows": successful_rows,
                "error_rows": len(ordered) - successful_rows,
                "methods": by_method,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if (
        args.voxel_divisor <= 0.0
        or args.voxel_floor_mm <= 0.0
        or args.ransac_iterations < 1000
        or args.workers < 1
    ):
        raise ValueError("Invalid Open3D baseline parameters")
    root = args.root.expanduser().resolve(strict=True)
    revision = registration.checked_existing(
        root / "cmes_revision_20260816", root, "revision"
    )
    scale = registration.checked_existing(args.scale_dict, revision, "scale dictionary")
    split = registration.checked_existing(args.split, revision, "identity split")
    anchor = registration.checked_existing(
        args.target_anchor_json, revision, "target-anchor JSON"
    )
    source_root = (
        registration.checked_existing(args.source_root, revision, "source root")
        if args.source_root
        else None
    )
    topology = (
        registration.checked_existing(args.topology_file, revision, "topology")
        if args.topology_file
        else None
    )
    if args.source_method == "3ddfa" and (source_root is None or topology is None):
        raise ValueError("3DDFA requires --source-root and --topology-file")
    output = args.output_dir.expanduser().resolve(strict=False)
    if output == revision or not registration.is_relative_to(output, revision):
        raise ValueError(f"Output escapes revision root: {output}")
    if output.exists() and not args.resume and not args.dry_run:
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    manifest = registration.checked_existing(
        root / "prepared_cohort" / "facescape_frontal_pairs_manifest.csv",
        root,
        "manifest",
    )
    cases, _ = registration.read_selected_cases(
        manifest, split, args.subset, args.case
    )
    config = {
        "source_method": args.source_method,
        "subset": args.subset,
        "case_count": len(cases),
        "methods": METHODS,
        "voxel_divisor": args.voxel_divisor,
        "voxel_floor_mm": args.voxel_floor_mm,
        "ransac_iterations": args.ransac_iterations,
        "prescale_mode": args.prescale_mode,
        "ransac_similarity_scale_enabled": args.prescale_mode == "bbox",
        "open3d_version": o3d.__version__,
        "target_anchor_json": str(anchor),
        "target_anchor_json_sha256": registration.sha256(anchor),
        "common_target_face_roi": True,
        "acceptance_threshold_applied": False,
    }
    if args.dry_run:
        print(json.dumps({**config, "first_case": cases[0], "last_case": cases[-1]}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=args.resume)
    (output / "case_rows").mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []

    def arguments(case: str) -> tuple[Any, ...]:
        return (
            case,
            str(root),
            str(scale),
            str(anchor),
            args.source_method,
            str(source_root) if source_root else None,
            str(topology) if topology else None,
            str(output),
            args.voxel_divisor,
            args.voxel_floor_mm,
            args.ransac_iterations,
            args.prescale_mode,
        )

    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {pool.submit(process_case, *arguments(case)): case for case in cases}
        for future in as_completed(futures):
            case = futures[future]
            try:
                case_rows = future.result()
            except Exception as error:
                case_rows = failed_case_rows(
                    case, args.source_method, args.prescale_mode, error
                )
                (output / "case_rows" / f"{case}.json").write_text(
                    json.dumps(
                        {
                            "rows": case_rows,
                            "transforms": {},
                            "case_execution_failed": True,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            rows.extend(case_rows)
            write_summary(output, rows, config)
            print(
                json.dumps({"completed": case, "cases": len(rows) // len(METHODS), "total": len(cases)}),
                flush=True,
            )
    write_summary(output, rows, config)
    expected_rows = len(cases) * len(METHODS)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Open3D baseline denominator mismatch: {len(rows)} of {expected_rows}"
        )


if __name__ == "__main__":
    main()
