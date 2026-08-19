#!/usr/bin/env python3
"""Identity-disjoint, pair-wise rigid registration in physical millimeters."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree


REVISION_TOOLS = Path(__file__).resolve().parent
if str(REVISION_TOOLS) not in sys.path:
    sys.path.insert(0, str(REVISION_TOOLS))
import frozen_nonrigid_nasal_solver as frozen_nonrigid  # noqa: E402


BASE_ROTATION = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
# Scale is fixed by the 3D interocular-distance ratio. Surface overlap is used
# only to rank orientation and translation candidates, so it cannot shrink or
# enlarge a partial source mesh to game a one-way distance objective.
SCALE_RATIOS = np.asarray([1.0], dtype=np.float64)
# Physical versions of the submitted 0.010 and 0.055 target-unit guards.
# The conversion uses only the frozen development identities (median official
# FaceScape expression-18 scale), never an evaluation identity.
DEVELOPMENT_REFERENCE_MM_PER_UNIT = 262.58104988598365
TRANSLATION_STEP_CAP_MM = 0.010 * DEVELOPMENT_REFERENCE_MM_PER_UNIT
MOUTH_TO_NOSE_GUARD_MM = 0.055 * DEVELOPMENT_REFERENCE_MM_PER_UNIT
COARSE_TRANSLATION_MODES = ("surface", "nose_anchor")
TARGET_COVERAGE_COARSE_POINTS = 8_000
TARGET_COVERAGE_REFINED_POINTS = 30_000
DEFAULT_FINAL_ANCHOR_CAP_MM = 10.0


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
    parser.add_argument("--keep", type=int, default=8)
    parser.add_argument("--rigid-iterations", type=int, default=20)
    parser.add_argument(
        "--final-anchor-cap-mm", type=float, default=DEFAULT_FINAL_ANCHOR_CAP_MM
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def checked_existing(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not is_relative_to(resolved, root):
        raise ValueError(f"{label} escapes project root: {resolved}")
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_indices(count: int, maximum: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, maximum, dtype=np.int64)


def transform(
    points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    return scale * (points @ rotation.T) + translation


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    if len(values) == 0:
        return float("nan")
    if float(weights.sum()) <= 1e-12:
        return float(np.quantile(values, quantile))
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order]) / float(weights.sum())
    return float(np.interp(quantile, cumulative, sorted_values))


def proper_axis_rotations() -> list[np.ndarray]:
    rotations = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                rotation = np.diag([sx, sy, sz]) @ BASE_ROTATION
                if np.linalg.det(rotation) > 0.0:
                    rotations.append(rotation)
    if len(rotations) != 4:
        raise RuntimeError(f"Expected four proper rotations, found {len(rotations)}")
    return rotations


def load_trimesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = tuple(loaded.geometry.values())
        if len(geometries) != 1:
            raise ValueError(f"Expected one mesh in {path}, found {len(geometries)}")
        loaded = geometries[0]
    return (
        np.asarray(loaded.vertices, dtype=np.float64),
        np.asarray(loaded.faces, dtype=np.int32),
    )


def target_registration_frame(vertices: np.ndarray, camera_json: Path) -> np.ndarray:
    camera = json.loads(camera_json.read_text(encoding="utf-8"))
    rt = np.asarray(camera["Rt"], dtype=np.float64)
    camera_vertices = vertices @ rt[:, :3].T + rt[:, 3]
    return np.column_stack(
        [camera_vertices[:, 0], -camera_vertices[:, 2], -camera_vertices[:, 1]]
    )


def canonical_coordinates(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    extent = np.maximum(maximum - minimum, 1e-8)
    x = (vertices[:, 0] - 0.5 * (minimum[0] + maximum[0])) / extent[0]
    front = (vertices[:, 1] - minimum[1]) / extent[1]
    vertical = (vertices[:, 2] - minimum[2]) / extent[2]
    return x, front, vertical


def target_nose_anchor(vertices: np.ndarray) -> np.ndarray:
    x, front, vertical = canonical_coordinates(vertices)
    central = (np.abs(x) < 0.18) & (vertical > 0.34) & (vertical < 0.70)
    if int(central.sum()) < 500:
        central = (np.abs(x) < 0.26) & (vertical > 0.28) & (vertical < 0.76)
    candidates = vertices[central]
    if len(candidates) < 100:
        candidates = vertices
    band = candidates[candidates[:, 1] >= np.quantile(candidates[:, 1], 0.985)]
    if len(band) < 30:
        band = candidates[candidates[:, 1] >= np.quantile(candidates[:, 1], 0.965)]
    return np.median(band, axis=0)


def distances_to_seeds(vertices: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    tree = cKDTree(np.asarray(seeds, dtype=np.float64))
    distances, _ = tree.query(vertices, k=1, workers=1)
    return distances


def fixed_source_anchor_mask(
    vertices: np.ndarray, masks: dict[str, np.ndarray]
) -> np.ndarray:
    """Freeze the HRN UV nose-tip/alar front in its canonical source frame."""
    candidates = np.asarray(masks["nose_tip"] | masks["alar"], dtype=bool)
    indices = np.flatnonzero(candidates)
    if len(indices) < 20:
        raise ValueError("Source nose-tip/alar region is too small")
    depth = np.asarray(vertices[indices, 2], dtype=np.float64)
    threshold = float(np.quantile(depth, 0.80))
    selected = indices[depth >= threshold]
    if len(selected) < 4:
        order = np.argsort(depth)
        selected = indices[order[-min(4, len(indices)):]]
    result = np.zeros(len(vertices), dtype=bool)
    result[selected] = True
    return result


def fixed_3ddfa_source_anchor_mask(
    vertex_count: int, landmark_vertex_ids: np.ndarray
) -> np.ndarray:
    """Freeze the official 68-landmark nose-tip vertex (zero-based index 30)."""
    landmark_vertex_ids = np.asarray(landmark_vertex_ids, dtype=np.int64)
    if len(landmark_vertex_ids) != 68:
        raise ValueError("3DDFA source anchor requires 68 landmark vertices")
    nose_tip_vertex = int(landmark_vertex_ids[30])
    if not 0 <= nose_tip_vertex < vertex_count:
        raise ValueError("3DDFA nose-tip landmark is outside the source topology")
    result = np.zeros(vertex_count, dtype=bool)
    result[nose_tip_vertex] = True
    return result


def build_3ddfa_masks(
    vertices: np.ndarray, landmark_vertex_ids: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    landmark_vertex_ids = np.asarray(landmark_vertex_ids, dtype=np.int64)
    if len(landmark_vertex_ids) != 68 or int(landmark_vertex_ids.max()) >= len(vertices):
        raise ValueError("Invalid 3DDFA landmark vertex indices")
    landmarks = vertices[landmark_vertex_ids]
    left_eye_center = np.median(landmarks[36:42], axis=0)
    right_eye_center = np.median(landmarks[42:48], axis=0)
    interocular = float(np.linalg.norm(left_eye_center - right_eye_center))
    if interocular <= 1e-8:
        raise ValueError("Degenerate interocular distance")

    eye_distance = distances_to_seeds(vertices, landmarks[36:48])
    bridge_distance = distances_to_seeds(vertices, landmarks[27:30])
    dorsum_distance = distances_to_seeds(vertices, landmarks[28:31])
    tip_distance = distances_to_seeds(vertices, landmarks[[30, 33]])
    alar_distance = distances_to_seeds(vertices, landmarks[[31, 32, 34, 35]])
    mouth_distance = distances_to_seeds(vertices, landmarks[48:68])
    nasal_distance = distances_to_seeds(vertices, landmarks[27:36])

    eye_hard = eye_distance <= 0.13 * interocular
    eye_soft = eye_distance <= 0.24 * interocular
    nasal_bridge = bridge_distance <= 0.12 * interocular
    nasal_dorsum = dorsum_distance <= 0.13 * interocular
    nose_tip = tip_distance <= 0.12 * interocular
    alar = alar_distance <= 0.12 * interocular
    nose = nasal_bridge | nasal_dorsum | nose_tip | alar
    mouth = mouth_distance <= 0.13 * interocular
    midface = (nasal_distance <= 0.48 * interocular) & ~eye_soft & ~mouth

    nose_base = np.median(landmarks[31:36], axis=0)
    upper_lip = np.median(landmarks[[50, 51, 52, 61, 62, 63]], axis=0)
    subnasal_center = 0.72 * nose_base + 0.28 * upper_lip
    philtrum_center = 0.42 * nose_base + 0.58 * upper_lip
    subnasal = np.linalg.norm(vertices - subnasal_center, axis=1) <= 0.11 * interocular
    philtrum = np.linalg.norm(vertices - philtrum_center, axis=1) <= 0.10 * interocular

    fit_weight = np.ones(len(vertices), dtype=np.float64)
    fit_weight[eye_soft] = 0.0
    fit_weight[midface] *= 1.20
    fit_weight[nasal_bridge | nasal_dorsum] *= 2.05
    fit_weight[nose_tip | alar] *= 2.35
    fit_weight[subnasal | philtrum] *= 1.04
    fit_weight[mouth] *= 0.42
    fit_weight = np.clip(fit_weight, 0.0, 3.0)
    full_no_eye = fit_weight > 0.08
    masks = {
        "eye": eye_hard,
        "eye_soft": eye_soft,
        "full_no_eye": full_no_eye,
        "midface": midface & full_no_eye,
        "nose": nose & full_no_eye,
        "nasal_bridge": nasal_bridge & full_no_eye,
        "nasal_dorsum": nasal_dorsum & full_no_eye,
        "nose_tip": nose_tip & full_no_eye,
        "alar": alar & full_no_eye,
        "subnasal": subnasal & full_no_eye,
        "philtrum": philtrum & full_no_eye,
        "mouth": mouth & full_no_eye,
        "mouth_downweighted": mouth & full_no_eye,
        "nose_bridge": nasal_bridge & full_no_eye,
        "nose_dorsum": nasal_dorsum & full_no_eye,
    }
    masks["source_anchor"] = fixed_3ddfa_source_anchor_mask(
        len(vertices), landmark_vertex_ids
    )
    minimum_counts = {
        "eye_soft": 50,
        "nose": 80,
        "nose_tip": 20,
        "alar": 20,
        "mouth": 50,
    }
    for name, minimum in minimum_counts.items():
        if int(masks[name].sum()) < minimum:
            raise ValueError(f"3DDFA {name} mask is too small: {int(masks[name].sum())}")
    return fit_weight, masks


def load_source(
    case: str,
    source_method: str,
    root: Path,
    source_root: Path | None,
    topology_file: Path | None,
) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if source_method == "hrn":
        source_path = checked_existing(
            root / "hrn_outputs" / case / f"{case}_0_hrn_mid_mesh.obj",
            root,
            f"HRN source for {case}",
        )
        vertices, _, faces, _, _, _ = frozen_nonrigid.parse_obj_with_uv(source_path)
        fit_weight, masks, _ = frozen_nonrigid.build_masks(source_path, vertices)
        masks = {
            **masks,
            "mouth_downweighted": masks["mouth"],
            "nose_bridge": masks["nasal_bridge"],
            "nose_dorsum": masks["nasal_dorsum"],
        }
        masks["source_anchor"] = fixed_source_anchor_mask(vertices, masks)
        return source_path, vertices, faces, fit_weight, masks

    if source_root is None or topology_file is None:
        raise ValueError("3DDFA requires --source-root and --topology-file")
    source_path = checked_existing(source_root / "objects" / f"{case}.obj", root, f"3DDFA source for {case}")
    vertices, faces = load_trimesh(source_path)
    topology = np.load(topology_file)
    landmark_vertex_ids = np.asarray(topology["landmark_vertex_ids"], dtype=np.int64)
    fit_weight, masks = build_3ddfa_masks(vertices, landmark_vertex_ids)
    return source_path, vertices, faces, fit_weight, masks


def source_anchor(vertices: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    """Return the centroid of the source-frame-frozen semantic anchor set."""
    anchor_mask = np.asarray(masks["source_anchor"], dtype=bool)
    candidates = vertices[anchor_mask]
    if len(candidates) < 1:
        raise ValueError("Frozen source-anchor mask is empty")
    # The arithmetic centroid commutes exactly with every similarity transform,
    # so this quantity is the transformed native-frame anchor g(a_s).
    return np.mean(candidates, axis=0)


def source_anchor_definition(source_method: str) -> str:
    if source_method == "hrn":
        return (
            "centroid of the anterior 20% of the HRN UV nose-tip/alar region "
            "along canonical source z"
        )
    if source_method == "3ddfa":
        return "official 3DDFA-V2 68-landmark nose-tip vertex (index 30)"
    raise ValueError(f"Unsupported source method: {source_method}")


def anchor_consistency_distance(
    vertices: np.ndarray, masks: dict[str, np.ndarray], target_anchor: np.ndarray
) -> float:
    """Distance from the target nose-tip anchor to the transformed source nasal surface."""
    source_nose = vertices[np.asarray(masks["nose"], dtype=bool)]
    if len(source_nose) < 20:
        raise ValueError("Source nasal mask is too small for anchor consistency")
    return float(np.min(np.linalg.norm(source_nose - target_anchor[None, :], axis=1)))


def source_interocular_distance(
    vertices: np.ndarray, masks: dict[str, np.ndarray]
) -> float:
    eye_vertices = vertices[np.asarray(masks["eye_soft"], dtype=bool)]
    if len(eye_vertices) < 20:
        raise ValueError("Source eye/orbit mask is too small")
    middle = float(np.median(eye_vertices[:, 0]))
    left = eye_vertices[eye_vertices[:, 0] < middle]
    right = eye_vertices[eye_vertices[:, 0] >= middle]
    if len(left) < 5 or len(right) < 5:
        raise ValueError("Could not split source eye/orbit mask into two sides")
    return float(np.linalg.norm(np.median(left, axis=0) - np.median(right, axis=0)))


def anatomical_scale_from_interocular(
    source_interocular: float, target_interocular_mm: float
) -> float:
    if not np.isfinite(source_interocular) or source_interocular <= 0.0:
        raise ValueError("Source interocular distance must be positive and finite")
    if not np.isfinite(target_interocular_mm) or target_interocular_mm <= 0.0:
        raise ValueError("Target interocular distance must be positive and finite")
    return float(target_interocular_mm / source_interocular)


def orientation_metrics(
    vertices: np.ndarray, masks: dict[str, np.ndarray]
) -> dict[str, float | int]:
    eyes = np.median(vertices[masks["eye_soft"]], axis=0)
    nose = np.median(vertices[masks["nose"]], axis=0)
    mouth = np.median(vertices[masks["mouth_downweighted"]], axis=0)
    extent = np.maximum(
        np.quantile(vertices, 0.95, axis=0) - np.quantile(vertices, 0.05, axis=0),
        1e-8,
    )
    height = float(extent[2])
    eye_over_mouth = float(eyes[2] - mouth[2])
    nose_over_mouth = float(nose[2] - mouth[2])
    vertical = eyes - mouth
    vertical_z = float(vertical[2] / max(float(np.linalg.norm(vertical)), 1e-12))
    eye_norm = eye_over_mouth / height
    nose_norm = nose_over_mouth / height
    upside_down = eye_norm < 0.10 or nose_norm < 0.02 or vertical_z < 0.35
    penalty = (
        max(0.0, 0.28 - eye_norm)
        + max(0.0, 0.08 - nose_norm)
        + max(0.0, 0.55 - vertical_z)
    )
    return {
        "eye_over_mouth_norm": float(eye_norm),
        "nose_over_mouth_norm": float(nose_norm),
        "vertical_z": vertical_z,
        "upside_down": int(upside_down),
        "upright_penalty": float(penalty),
    }


def geometric_metrics(
    vertices: np.ndarray,
    target_tree: cKDTree,
    target_points: np.ndarray,
    fit_weight: np.ndarray,
    masks: dict[str, np.ndarray],
    target_anchor: np.ndarray,
) -> dict[str, float]:
    fit_indices = np.flatnonzero(fit_weight > 0.08)
    fit_indices = fit_indices[deterministic_indices(len(fit_indices), 30000)]
    distances, _ = target_tree.query(vertices[fit_indices], k=1, workers=1)
    weights = fit_weight[fit_indices]
    nose_indices = np.flatnonzero(masks["nose"] & (fit_weight > 0.08))
    nose_indices = nose_indices[deterministic_indices(len(nose_indices), 9000)]
    nose_distances, _ = target_tree.query(vertices[nose_indices], k=1, workers=1)
    source_coverage = vertices[fit_indices]
    source_coverage = source_coverage[
        deterministic_indices(len(source_coverage), TARGET_COVERAGE_REFINED_POINTS)
    ]
    target_coverage = target_points[
        deterministic_indices(len(target_points), TARGET_COVERAGE_REFINED_POINTS)
    ]
    target_to_source, _ = cKDTree(source_coverage).query(
        target_coverage, k=1, workers=1
    )
    anchor_surface_distance = anchor_consistency_distance(
        vertices, masks, target_anchor
    )
    source_anchor_point = source_anchor(vertices, masks)
    anchor_point_distance = float(np.linalg.norm(source_anchor_point - target_anchor))
    mouth_center = np.median(vertices[masks["mouth_downweighted"]], axis=0)
    mouth_anchor_distance = float(np.linalg.norm(mouth_center - target_anchor))
    mouth_guard = max(0.0, MOUTH_TO_NOSE_GUARD_MM - mouth_anchor_distance)
    result = {
        "full_mean_mm": float(np.average(distances, weights=np.maximum(weights, 1e-8))),
        "full_median_mm": weighted_quantile(distances, weights, 0.50),
        "full_p90_mm": weighted_quantile(distances, weights, 0.90),
        "full_p95_mm": weighted_quantile(distances, weights, 0.95),
        "target_coverage_median_mm": float(np.median(target_to_source)),
        "target_coverage_p90_mm": float(np.quantile(target_to_source, 0.90)),
        "nose_median_mm": float(np.median(nose_distances)),
        "nose_p90_mm": float(np.quantile(nose_distances, 0.90)),
        "nose_anchor_mm": anchor_point_distance,
        "nose_anchor_surface_mm": anchor_surface_distance,
        "nose_anchor_point_mm": anchor_point_distance,
        "mouth_to_anchor_mm": mouth_anchor_distance,
        "mouth_guard_mm": mouth_guard,
    }
    result["symmetric_full_median_mm"] = 0.5 * (
        result["full_median_mm"] + result["target_coverage_median_mm"]
    )
    result["symmetric_full_p90_mm"] = 0.5 * (
        result["full_p90_mm"] + result["target_coverage_p90_mm"]
    )
    result["selection_score_mm"] = float(
        result["symmetric_full_median_mm"]
        + 0.22 * result["symmetric_full_p90_mm"]
        + 0.32 * result["nose_median_mm"]
        + 0.08 * result["nose_p90_mm"]
        + 0.35 * result["mouth_guard_mm"]
    )
    return result


def combined_score(metrics: dict[str, float], orientation: dict[str, Any]) -> float:
    return float(
        metrics["selection_score_mm"]
        + (1000.0 if bool(orientation["upside_down"]) else 0.0)
        + 100.0 * float(orientation["upright_penalty"])
    )


def umeyama_rigid(
    source: np.ndarray, destination: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    normalized_weights = weights / max(float(weights.sum()), 1e-12)
    source_center = np.sum(source * normalized_weights[:, None], axis=0)
    destination_center = np.sum(destination * normalized_weights[:, None], axis=0)
    source_zero = source - source_center
    destination_zero = destination - destination_center
    covariance = source_zero.T @ (destination_zero * normalized_weights[:, None])
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = destination_center - rotation @ source_center
    return rotation, translation


def global_candidates(
    source: np.ndarray,
    target: np.ndarray,
    fit_weight: np.ndarray,
    masks: dict[str, np.ndarray],
    target_anchor: np.ndarray,
    target_interocular_mm: float,
    keep: int,
) -> list[dict[str, Any]]:
    fit_mask = fit_weight > 0.35
    source_fit = source[fit_mask]
    source_weights = fit_weight[fit_mask]
    evaluation_indices = deterministic_indices(len(source_fit), 5000)
    target_evaluation = target[deterministic_indices(len(target), 60000)]
    target_coverage = target[
        deterministic_indices(len(target), TARGET_COVERAGE_COARSE_POINTS)
    ]
    target_tree = cKDTree(target_evaluation)
    target_front_coordinate = target[:, 1]
    target_front = target[target_front_coordinate >= np.quantile(target_front_coordinate, 0.78)]
    if len(target_front) < 5000:
        target_front = target
    target_center = np.median(target_front, axis=0)
    target_front_y = float(np.quantile(target_front_coordinate, 0.94))
    source_nose_anchor = source_anchor(source, masks)
    source_interocular = source_interocular_distance(source, masks)
    base_scale = anatomical_scale_from_interocular(
        source_interocular, target_interocular_mm
    )

    candidates = []
    for rotation in proper_axis_rotations():
        for ratio in SCALE_RATIOS:
            scale = base_scale * float(ratio)
            moved_all = transform(source, scale, rotation, np.zeros(3))
            surface_translation = target_center - np.median(
                moved_all[fit_mask], axis=0
            )
            surface_translation[1] = target_front_y - np.max(
                moved_all[fit_mask, 1]
            )
            anchor_translation = target_anchor - (
                scale * (source_nose_anchor @ rotation.T)
            )
            for translation_mode, translation in (
                ("surface", surface_translation),
                ("nose_anchor", anchor_translation),
            ):
                aligned_evaluation = (
                    transform(
                        source_fit[evaluation_indices],
                        scale,
                        rotation,
                        np.zeros(3),
                    )
                    + translation
                )
                distances, _ = target_tree.query(
                    aligned_evaluation, k=1, workers=1
                )
                weights = source_weights[evaluation_indices]
                aligned_all = moved_all + translation
                aligned_coverage = aligned_all[fit_mask]
                aligned_coverage = aligned_coverage[
                    deterministic_indices(len(aligned_coverage), 12_000)
                ]
                coverage_distances, _ = cKDTree(aligned_coverage).query(
                    target_coverage, k=1, workers=1
                )
                anchor = float(
                    np.linalg.norm(source_anchor(aligned_all, masks) - target_anchor)
                )
                orientation = orientation_metrics(aligned_all, masks)
                mouth = np.median(
                    aligned_all[masks["mouth_downweighted"]], axis=0
                )
                mouth_guard = max(
                    0.0,
                    MOUTH_TO_NOSE_GUARD_MM
                    - float(np.linalg.norm(mouth - target_anchor)),
                )
                source_median = weighted_quantile(distances, weights, 0.50)
                source_p90 = weighted_quantile(distances, weights, 0.90)
                target_median = float(np.median(coverage_distances))
                target_p90 = float(np.quantile(coverage_distances, 0.90))
                score = float(
                    0.5 * (source_median + target_median)
                    + 0.10 * (source_p90 + target_p90)
                    + 0.08 * weighted_quantile(distances, weights, 0.98)
                    + 0.45 * mouth_guard
                    + 100.0 * float(orientation["upright_penalty"])
                    + (1000.0 if bool(orientation["upside_down"]) else 0.0)
                )
                candidates.append(
                    {
                        "coarse_candidate_id": int(len(candidates) + 1),
                        "coarse_score_mm": score,
                        "scale": scale,
                        "rotation": rotation.copy(),
                        "translation": translation.copy(),
                        "translation_mode": translation_mode,
                        "scale_ratio": float(ratio),
                        "scale_definition": "target_3d_interocular_mm/source_3d_interocular",
                        "anchor_mm": anchor,
                        "coarse_source_median_mm": source_median,
                        "coarse_source_p90_mm": source_p90,
                        "coarse_target_coverage_median_mm": target_median,
                        "coarse_target_coverage_p90_mm": target_p90,
                        "upside_down": int(orientation["upside_down"]),
                    }
                )
    surface_order = sorted(candidates, key=lambda item: float(item["coarse_score_mm"]))
    anchor_order = sorted(
        candidates,
        key=lambda item: (
            int(item["upside_down"]),
            float(item["anchor_mm"]),
            float(item["coarse_score_mm"]),
        ),
    )
    for rank, item in enumerate(surface_order, start=1):
        item["coarse_surface_rank"] = int(rank)
        item["coarse_overlap_rank"] = int(rank)
        item["retention_reasons"] = []
    for rank, item in enumerate(anchor_order, start=1):
        item["coarse_anchor_rank"] = int(rank)

    if keep == 1:
        surface_order[0]["retention_reasons"].append("symmetric_overlap")
        return surface_order[:1]

    surface_quota = (keep + 1) // 2
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()

    def retain(item: dict[str, Any], reason: str) -> None:
        candidate_id = int(item["coarse_candidate_id"])
        if candidate_id not in selected_ids:
            selected.append(item)
            selected_ids.add(candidate_id)
        if reason not in item["retention_reasons"]:
            item["retention_reasons"].append(reason)

    for item in surface_order[:surface_quota]:
        retain(item, "symmetric_overlap")
    for item in anchor_order:
        if len(selected) >= keep:
            break
        retain(item, "semantic_anchor")
    for item in surface_order:
        if len(selected) >= keep:
            break
        retain(item, "symmetric_overlap_fill")
    return selected


def refine_fixed_scale(
    source: np.ndarray,
    target_tree: cKDTree,
    target_points: np.ndarray,
    fit_weight: np.ndarray,
    masks: dict[str, np.ndarray],
    target_anchor: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    iterations: int,
) -> tuple[float, np.ndarray, np.ndarray, list[float]]:
    fit_indices = np.flatnonzero(fit_weight > 0.08)
    fit_indices = fit_indices[deterministic_indices(len(fit_indices), 18000)]
    history: list[float] = []
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    trim_schedule = np.linspace(0.68, 0.90, iterations)
    for iteration, trim in enumerate(trim_schedule):
        moved = transform(source[fit_indices], scale, rotation, translation)
        distances, nearest = target_tree.query(moved, k=1, workers=1)
        keep = distances <= np.quantile(distances, float(trim))
        if int(keep.sum()) < 200:
            keep = distances <= np.quantile(distances, 0.92)
        weights = fit_weight[fit_indices][keep].copy()
        robust_scale = max(float(np.quantile(distances, 0.94)), 1e-8)
        weights *= np.clip(1.0 - distances[keep] / robust_scale, 0.03, 1.0)
        delta_rotation, delta_translation = umeyama_rigid(
            moved[keep], target_points[nearest[keep]], weights
        )
        rotation = delta_rotation @ rotation
        translation = delta_rotation @ translation + delta_translation
        moved_after = transform(source[fit_indices], scale, rotation, translation)
        distances_after, _ = target_tree.query(moved_after, k=1, workers=1)
        score_weights = fit_weight[fit_indices]
        aligned = transform(source, scale, rotation, translation)
        anchor = anchor_consistency_distance(aligned, masks, target_anchor)
        orientation = orientation_metrics(aligned, masks)
        score = float(
            weighted_quantile(distances_after, score_weights, 0.50)
            + 0.30 * weighted_quantile(distances_after, score_weights, 0.90)
            + (1000.0 if bool(orientation["upside_down"]) else 0.0)
            + 100.0 * float(orientation["upright_penalty"])
        )
        history.append(score)
        if best is None or score < best[0]:
            best = (score, rotation.copy(), translation.copy())
        if iteration >= 6 and len(history) >= 5:
            recent = np.asarray(history[-5:])
            if float(recent.max() - recent.min()) < 1e-4:
                break
    if best is None:
        raise RuntimeError("Rigid refinement produced no state")
    return scale, best[1], best[2], history


def translation_polish(
    source: np.ndarray,
    target_tree: cKDTree,
    target_points: np.ndarray,
    fit_weight: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    iterations: int = 12,
) -> np.ndarray:
    fit_indices = np.flatnonzero(fit_weight > 0.35)
    fit_indices = fit_indices[deterministic_indices(len(fit_indices), 20000)]
    for _ in range(iterations):
        moved = transform(source[fit_indices], scale, rotation, translation)
        distances, nearest = target_tree.query(moved, k=1, workers=1)
        keep = distances <= np.quantile(distances, 0.72)
        if int(keep.sum()) < 200:
            break
        weights = fit_weight[fit_indices][keep]
        step = np.average(
            target_points[nearest[keep]] - moved[keep], axis=0, weights=weights
        )
        norm = float(np.linalg.norm(step))
        if norm < 1e-5:
            break
        if norm > TRANSLATION_STEP_CAP_MM:
            step *= TRANSLATION_STEP_CAP_MM / norm
        translation = translation + step
    return translation


def process_case(
    case: str,
    manifest_row: dict[str, str],
    root_string: str,
    scale_path_string: str,
    anchor_path_string: str,
    source_method: str,
    source_root_string: str | None,
    topology_file_string: str | None,
    output_string: str,
    keep: int,
    rigid_iterations: int,
    final_anchor_cap_mm: float,
) -> dict[str, Any]:
    root = Path(root_string)
    output = Path(output_string)
    row_path = output / "case_rows" / f"{case}.json"
    vertices_path = output / "rigid_vertices_mm" / f"{case}.npz"
    if row_path.exists() and vertices_path.exists():
        return json.loads(row_path.read_text(encoding="utf-8"))
    scales = json.loads(Path(scale_path_string).read_text(encoding="utf-8"))
    anchor_payload = json.loads(Path(anchor_path_string).read_text(encoding="utf-8"))
    subject = f"{int(manifest_row['subject']):03d}"
    mm_per_unit = float(scales[str(int(subject))]["18"][0])
    source_root = Path(source_root_string) if source_root_string else None
    topology_file = Path(topology_file_string) if topology_file_string else None
    source_path, source, faces, fit_weight, masks = load_source(
        case, source_method, root, source_root, topology_file
    )
    source_iod = source_interocular_distance(source, masks)

    target_pair = f"{subject}_18_eye_closed"
    manifest_path = root / "prepared_cohort" / "facescape_frontal_pairs_manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        by_pair = {item["pair_id"]: item for item in csv.DictReader(handle)}
    target_mesh = checked_existing(Path(by_pair[target_pair]["mesh"]), root, "target mesh")
    target_world, _ = load_trimesh(target_mesh)
    camera_json = checked_existing(
        target_mesh.parent / "selected_camera.json", root, "target camera"
    )
    target_mm_full = target_registration_frame(target_world, camera_json) * mm_per_unit
    anchor_record = anchor_payload.get("anchors", {}).get(subject)
    if anchor_record is None:
        raise KeyError(f"No precomputed target anchor for subject {subject}")
    if str(anchor_record.get("target_pair")) != target_pair:
        raise ValueError(
            f"Target-anchor pair mismatch for {subject}: {anchor_record.get('target_pair')}"
        )
    target_anchor = np.asarray(anchor_record["anchor_mm"], dtype=np.float64)
    if target_anchor.shape != (3,) or not np.all(np.isfinite(target_anchor)):
        raise ValueError(f"Invalid target anchor for subject {subject}")
    target_interocular_mm = float(anchor_record.get("interocular_distance_mm", np.nan))
    if not np.isfinite(target_interocular_mm) or target_interocular_mm <= 0.0:
        raise ValueError(f"Missing or invalid target 3D interocular distance for {subject}")
    face_roi_relative = Path(str(anchor_record.get("face_roi_npz", "")))
    if not face_roi_relative.as_posix() or face_roi_relative.is_absolute():
        raise ValueError(f"Invalid target face-ROI path for subject {subject}")
    target_face_roi_path = checked_existing(
        Path(anchor_path_string).parent / face_roi_relative,
        root,
        f"target face ROI for subject {subject}",
    )
    with np.load(target_face_roi_path) as roi_payload:
        target_face_roi_full = np.asarray(
            roi_payload["target_face_roi_mm"], dtype=np.float64
        )
    if len(target_face_roi_full) < 3000:
        raise ValueError(f"Target face ROI is too small for subject {subject}")
    target_mm = target_face_roi_full[
        deterministic_indices(len(target_face_roi_full), 160000)
    ]
    target_tree = cKDTree(target_mm)

    started = time.time()
    initial = global_candidates(
        source,
        target_mm,
        fit_weight,
        masks,
        target_anchor,
        target_interocular_mm,
        keep=keep,
    )
    tried = []
    states: list[dict[str, Any]] = []
    for index, candidate in enumerate(initial, start=1):
        scale, rotation, translation, history = refine_fixed_scale(
            source,
            target_tree,
            target_mm,
            fit_weight,
            masks,
            target_anchor,
            float(candidate["scale"]),
            np.asarray(candidate["rotation"], dtype=np.float64),
            np.asarray(candidate["translation"], dtype=np.float64),
            rigid_iterations,
        )
        translation = translation_polish(
            source,
            target_tree,
            target_mm,
            fit_weight,
            scale,
            rotation,
            translation,
        )
        aligned = transform(source, scale, rotation, translation)
        metrics = geometric_metrics(
            aligned,
            target_tree,
            target_mm,
            fit_weight,
            masks,
            target_anchor,
        )
        orientation = orientation_metrics(aligned, masks)
        score = combined_score(metrics, orientation)
        tried.append(
            {
                "candidate": index,
                "coarse_candidate_id": int(candidate["coarse_candidate_id"]),
                "coarse_score_mm": float(candidate["coarse_score_mm"]),
                "coarse_anchor_mm": float(candidate["anchor_mm"]),
                "coarse_surface_rank": int(candidate["coarse_surface_rank"]),
                "coarse_overlap_rank": int(candidate["coarse_overlap_rank"]),
                "coarse_anchor_rank": int(candidate["coarse_anchor_rank"]),
                "coarse_retention_reasons": list(candidate["retention_reasons"]),
                "coarse_scale_ratio": float(candidate["scale_ratio"]),
                "coarse_translation_mode": str(candidate["translation_mode"]),
                "coarse_target_coverage_median_mm": float(
                    candidate["coarse_target_coverage_median_mm"]
                ),
                "coarse_target_coverage_p90_mm": float(
                    candidate["coarse_target_coverage_p90_mm"]
                ),
                "refined_score_mm": score,
                "source_interocular_mm": float(scale * source_iod),
                "nose_anchor_to_interocular_ratio": float(
                    metrics["nose_anchor_mm"] / max(scale * source_iod, 1e-8)
                ),
                "refinement_iterations": len(history),
                **metrics,
                **orientation,
            }
        )
        states.append(
            {
                "candidate_index": index,
                "coarse_candidate": candidate,
                "score": score,
                "scale": scale,
                "rotation": rotation.copy(),
                "translation": translation.copy(),
                "aligned": aligned.copy(),
                "metrics": metrics,
                "orientation": orientation,
            }
        )
    if not states:
        raise RuntimeError(f"No rigid candidate completed for {case}")
    upright_states = [
        state for state in states if not bool(state["orientation"]["upside_down"])
    ]
    ranking_pool = upright_states if upright_states else states
    anchor_eligible = [
        state
        for state in ranking_pool
        if float(state["metrics"]["nose_anchor_point_mm"])
        <= final_anchor_cap_mm
    ]
    anchor_cap_fallback = not bool(anchor_eligible)
    if anchor_eligible:
        ranking_pool = anchor_eligible
    selected = min(
        ranking_pool,
        key=lambda state: (
            float(state["score"]),
            float(state["metrics"]["nose_anchor_point_mm"]),
            int(state["candidate_index"]),
        ),
    )
    score = float(selected["score"])
    scale = float(selected["scale"])
    rotation = np.asarray(selected["rotation"], dtype=np.float64)
    translation = np.asarray(selected["translation"], dtype=np.float64)
    aligned = np.asarray(selected["aligned"], dtype=np.float64)
    metrics = dict(selected["metrics"])
    orientation = dict(selected["orientation"])
    selected_coarse = dict(selected["coarse_candidate"])
    expected_source_anchor = transform(
        source_anchor(source, masks)[None, :], scale, rotation, translation
    )[0]
    measured_source_anchor = source_anchor(aligned, masks)
    source_anchor_transform_residual_mm = float(
        np.linalg.norm(measured_source_anchor - expected_source_anchor)
    )
    if source_anchor_transform_residual_mm > 1e-8:
        raise RuntimeError(
            f"{case}: source anchor is not similarity-transform equivariant "
            f"({source_anchor_transform_residual_mm:.3e} mm)"
        )

    np.savez_compressed(
        vertices_path,
        vertices_mm=aligned.astype(np.float32),
        scale=np.asarray(scale, dtype=np.float64),
        rotation=rotation.astype(np.float64),
        translation_mm=translation.astype(np.float64),
    )
    expression = case.split("_", 1)[1]
    expression_index_text, expression_name = expression.split("_", 1)
    row: dict[str, Any] = {
        "case": case,
        "subject": subject,
        "expression": expression,
        "expression_index": int(expression_index_text),
        "expression_name": expression_name,
        "source_method": source_method,
        "source_obj": str(source_path),
        "target_mesh": str(target_mesh),
        "target_face_roi_npz": str(target_face_roi_path),
        "target_face_roi_vertices": int(len(target_face_roi_full)),
        "target_face_roi_definition": str(
            anchor_record.get("face_roi_definition", "precomputed")
        ),
        "mm_per_target_unit": mm_per_unit,
        "coarse_candidate_count": int(
            len(proper_axis_rotations())
            * len(SCALE_RATIOS)
            * len(COARSE_TRANSLATION_MODES)
        ),
        "refined_candidate_count": len(initial),
        "rigid_iterations_requested": rigid_iterations,
        "selected_score_mm": score,
        "selected_candidate_index": int(selected["candidate_index"]),
        "selected_coarse_candidate_id": int(
            selected_coarse["coarse_candidate_id"]
        ),
        "selected_coarse_scale_ratio": float(selected_coarse["scale_ratio"]),
        "selected_coarse_translation_mode": str(
            selected_coarse["translation_mode"]
        ),
        "selected_scale_source_to_mm": scale,
        "source_interocular_before_scaling": source_iod,
        "target_interocular_mm": target_interocular_mm,
        "anatomical_scale_definition": (
            "target 3D interocular distance divided by source 3D interocular distance"
        ),
        "anatomical_scale_residual_mm": float(
            abs(scale * source_iod - target_interocular_mm)
        ),
        "runtime_seconds": float(time.time() - started),
        "rigid_vertices_npz": str(vertices_path),
        "eye_mask_vertices": int(masks["eye_soft"].sum()),
        "nose_mask_vertices": int(masks["nose"].sum()),
        "nose_anchor_metric_definition": (
            f"transformed {source_anchor_definition(source_method)} to target "
            "nose-tip anchor"
        ),
        "nose_anchor_surface_metric_definition": (
            "target nose-tip anchor to transformed source nasal surface"
        ),
        "nose_anchor_point_metric_definition": (
            f"transformed {source_anchor_definition(source_method)} to target "
            "nose-tip anchor"
        ),
        "source_anchor_definition": source_anchor_definition(source_method),
        "source_anchor_transform_residual_mm": source_anchor_transform_residual_mm,
        "target_anchor_definition": str(anchor_record.get("method", "precomputed")),
        "target_anchor_json": str(anchor_path_string),
        "source_interocular_mm": float(scale * source_iod),
        "nose_anchor_to_interocular_ratio": float(
            metrics["nose_anchor_mm"] / max(scale * source_iod, 1e-8)
        ),
        "nose_anchor_used_for_candidate_retention": 1,
        "nose_anchor_used_for_final_ranking": 1,
        "final_anchor_ranking_cap_mm": final_anchor_cap_mm,
        "final_anchor_eligible_candidates": len(anchor_eligible),
        "final_anchor_cap_fallback": int(anchor_cap_fallback),
        "bidirectional_overlap_used_for_candidate_ranking": 1,
        **metrics,
        **orientation,
    }
    detail = {
        "row": row,
        "selected_rotation": rotation.tolist(),
        "selected_translation_mm": translation.tolist(),
        "tried_candidates": tried,
        "target_sampling": (
            "deterministic 160000-point maximum from the camera-visible "
            "MediaPipe face-oval target scan ROI"
        ),
        "uses_same_subject_prior": False,
        "uses_acceptance_threshold_during_selection": False,
        "nose_anchor_used_for_candidate_retention": True,
        "nose_anchor_used_for_final_ranking": True,
        "final_anchor_ranking_cap_mm": final_anchor_cap_mm,
        "final_anchor_eligible_candidates": len(anchor_eligible),
        "final_anchor_cap_fallback": anchor_cap_fallback,
        "bidirectional_overlap_used_for_candidate_ranking": True,
        "nose_anchor_metric_definition": (
            "Euclidean distance from the transformed "
            f"{source_anchor_definition(source_method)} to the target nose-tip anchor"
        ),
        "target_anchor_definition": str(anchor_record.get("method", "precomputed")),
        "candidate_retention_policy": (
            "union of the best coarse bidirectional-overlap candidates and the "
            "best upright semantic point-anchor candidates"
        ),
        "final_candidate_policy": (
            "minimum bidirectional-overlap score among upright candidates with "
            f"semantic point-anchor distance <= {final_anchor_cap_mm:g} mm; "
            "overlap-ranked fallback only when no candidate satisfies the cap"
        ),
    }
    (output / "case_details" / f"{case}.json").write_text(
        json.dumps(detail, indent=2), encoding="utf-8"
    )
    row_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def read_selected_cases(
    manifest: Path, split: Path, subset: str, explicit_cases: list[str]
) -> tuple[list[str], dict[str, dict[str, str]]]:
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_pair = {str(row["pair_id"]): row for row in rows}
    frozen_split = json.loads(split.read_text(encoding="utf-8"))
    if subset == "all":
        subjects = set(frozen_split["development_subjects"]) | set(
            frozen_split["test_subjects"]
        )
    elif subset == "heldout":
        subjects = set(frozen_split["test_subjects"])
    else:
        subjects = set(frozen_split["development_subjects"])
    cases = sorted(
        case
        for case, row in by_pair.items()
        if f"{int(row['subject']):03d}" in subjects
        and not case.endswith("_18_eye_closed")
    )
    if explicit_cases:
        requested = set(explicit_cases)
        missing = sorted(requested - set(cases))
        if missing:
            raise ValueError(f"Requested cases are outside {subset}: {missing}")
        cases = sorted(requested)
    else:
        expected = 380 if subset == "all" else 190
        if len(cases) != expected:
            raise ValueError(f"Expected {expected} {subset} cases, found {len(cases)}")
    return cases, by_pair


def write_summary(
    output: Path, rows: list[dict[str, Any]], config: dict[str, Any]
) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda row: str(row["case"]))
    fields = sorted({key for row in rows for key in row})
    with (output / "pairwise_mm_rigid_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    subject_means: dict[str, list[float]] = {}
    for row in rows:
        subject_means.setdefault(str(row["subject"]), []).append(
            float(row["full_median_mm"])
        )
    summary = {
        **config,
        "completed_cases": len(rows),
        "completed_subjects": len(subject_means),
        "orientation_failures": int(sum(bool(row["upside_down"]) for row in rows)),
        "mean_pair_full_median_mm": float(
            np.mean([float(row["full_median_mm"]) for row in rows])
        ),
        "mean_subject_full_median_mm": float(
            np.mean([np.mean(values) for values in subject_means.values()])
        ),
        "note": "No acceptance threshold or same-subject prior was used.",
    }
    (output / "pairwise_mm_rigid_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    maximum_candidates = (
        len(proper_axis_rotations())
        * len(SCALE_RATIOS)
        * len(COARSE_TRANSLATION_MODES)
    )
    if args.keep < 1 or args.keep > maximum_candidates:
        raise ValueError(f"--keep must be in [1, {maximum_candidates}]")
    if args.rigid_iterations < 1:
        raise ValueError("--rigid-iterations must be positive")
    if not np.isfinite(args.final_anchor_cap_mm) or args.final_anchor_cap_mm <= 0.0:
        raise ValueError("--final-anchor-cap-mm must be a positive finite value")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    root = args.root.expanduser().resolve(strict=True)
    revision_root = checked_existing(root / "cmes_revision_20260816", root, "revision root")
    scale_path = checked_existing(args.scale_dict, revision_root, "scale dictionary")
    split_path = checked_existing(args.split, revision_root, "identity split")
    target_anchor_path = checked_existing(
        args.target_anchor_json, revision_root, "target-anchor JSON"
    )
    manifest = checked_existing(
        root / "prepared_cohort" / "facescape_frontal_pairs_manifest.csv",
        root,
        "manifest",
    )
    source_root = (
        checked_existing(args.source_root, revision_root, "source root")
        if args.source_root
        else None
    )
    topology_file = (
        checked_existing(args.topology_file, revision_root, "topology file")
        if args.topology_file
        else None
    )
    if args.source_method == "3ddfa" and (source_root is None or topology_file is None):
        raise ValueError("3DDFA requires --source-root and --topology-file")
    output = args.output_dir.expanduser().resolve(strict=False)
    if output == revision_root or not is_relative_to(output, revision_root):
        raise ValueError(f"Output escapes revision root: {output}")
    if output.exists() and not args.resume and not args.dry_run:
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    cases, by_pair = read_selected_cases(
        manifest, split_path, args.subset, args.case
    )
    config = {
        "source_method": args.source_method,
        "subset": args.subset,
        "case_count": len(cases),
        "coarse_axis_conventions": 4,
        "coarse_scale_ratios": SCALE_RATIOS.tolist(),
        "coarse_translation_modes": list(COARSE_TRANSLATION_MODES),
        "coarse_candidate_count": maximum_candidates,
        "refined_candidate_count": args.keep,
        "rigid_iterations": args.rigid_iterations,
        "candidate_ranking_metric": (
            "bidirectional overlap between the source evaluation mask and the "
            "camera-visible target face ROI"
        ),
        "scale_normalization": (
            "fixed target/source 3D interocular-distance ratio; surface overlap "
            "does not optimize scale"
        ),
        "target_coverage_coarse_points": TARGET_COVERAGE_COARSE_POINTS,
        "target_coverage_refined_points": TARGET_COVERAGE_REFINED_POINTS,
        "final_semantic_anchor_candidate_cap_mm": args.final_anchor_cap_mm,
        "final_semantic_anchor_cap_selected_on": "frozen development identities only",
        "translation_step_cap_mm": TRANSLATION_STEP_CAP_MM,
        "mouth_to_nose_guard_mm": MOUTH_TO_NOSE_GUARD_MM,
        "development_reference_mm_per_unit": (
            DEVELOPMENT_REFERENCE_MM_PER_UNIT
        ),
        "physical_parameter_reference": (
            "median official expression-18 scale of the frozen development "
            "identities only"
        ),
        "uses_same_subject_prior": False,
        "uses_acceptance_threshold_during_selection": False,
        "identity_split": str(split_path),
        "identity_split_sha256": sha256(split_path),
        "target_anchor_json": str(target_anchor_path),
        "target_anchor_json_sha256": sha256(target_anchor_path),
        "scale_dictionary": str(scale_path),
        "scale_dictionary_sha256": sha256(scale_path),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
    }
    if args.dry_run:
        print(json.dumps({**config, "first_case": cases[0], "last_case": cases[-1]}, indent=2))
        return

    output.mkdir(parents=True, exist_ok=args.resume)
    for child in ("case_rows", "case_details", "rigid_vertices_mm"):
        (output / child).mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_case,
                case,
                by_pair[case],
                str(root),
                str(scale_path),
                str(target_anchor_path),
                args.source_method,
                str(source_root) if source_root else None,
                str(topology_file) if topology_file else None,
                str(output),
                args.keep,
                args.rigid_iterations,
                args.final_anchor_cap_mm,
            ): case
            for case in cases
        }
        for future in as_completed(futures):
            case = futures[future]
            row = future.result()
            rows.append(row)
            write_summary(output, rows, config)
            print(
                json.dumps(
                    {
                        "completed": case,
                        "count": len(rows),
                        "total": len(cases),
                        "full_median_mm": row["full_median_mm"],
                        "nose_anchor_mm": row["nose_anchor_mm"],
                        "upside_down": row["upside_down"],
                    }
                ),
                flush=True,
            )
    write_summary(output, rows, config)


if __name__ == "__main__":
    main()
