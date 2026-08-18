#!/usr/bin/env python3
"""Build target nose-tip anchors by projecting a 2D landmark onto visible scan surface."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
from google.protobuf import message_factory, symbol_database


if not hasattr(message_factory.MessageFactory, "GetPrototype"):
    message_factory.MessageFactory.GetPrototype = (  # type: ignore[attr-defined]
        lambda self, descriptor: message_factory.GetMessageClass(descriptor)
    )
if not hasattr(symbol_database.SymbolDatabase, "GetPrototype"):
    symbol_database.SymbolDatabase.GetPrototype = (  # type: ignore[attr-defined]
        lambda self, descriptor: message_factory.GetMessageClass(descriptor)
    )

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from scipy.spatial import cKDTree

import run_pairwise_mm_rigid as registration


MEDIAPIPE_NOSE_TIP_INDEX = 1
MEDIAPIPE_LEFT_EYE_CORNERS = (33, 133)
MEDIAPIPE_RIGHT_EYE_CORNERS = (362, 263)
INITIAL_PIXEL_RADIUS = 2.0
MAXIMUM_PIXEL_RADIUS = 12.0
MINIMUM_SURFACE_CANDIDATES = 8
FRONT_DEPTH_QUANTILE = 0.10
FRONT_DEPTH_MARGIN_MM = 1.0
ANCHOR_AVERAGE_POINTS = 5
FACE_ROI_BIN_SIZE_PX = 3
FACE_ROI_DEPTH_MARGIN_MM = 1.0
FACE_OVAL_INDICES = sorted(
    {
        int(index)
        for connection in mp.solutions.face_mesh.FACEMESH_FACE_OVAL
        for index in connection
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scale-dict", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--subset", choices=("development", "heldout"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def project_world_to_pixels(
    vertices_world: np.ndarray, camera_json: Path
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    camera = json.loads(camera_json.read_text(encoding="utf-8"))
    rt = np.asarray(camera["Rt"], dtype=np.float64)
    intrinsic = np.asarray(camera["K"], dtype=np.float64)
    camera_vertices = vertices_world @ rt[:, :3].T + rt[:, 3]
    depth = camera_vertices[:, 2]
    safe_depth = np.maximum(depth, 1e-8)
    u = intrinsic[0, 0] * camera_vertices[:, 0] / safe_depth + intrinsic[0, 2]
    v = intrinsic[1, 1] * camera_vertices[:, 1] / safe_depth + intrinsic[1, 2]
    return np.column_stack([u, v]), camera_vertices, camera


def face_landmark_pixels(
    face_mesh: Any, image_path: Path
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read target image: {image_path}")
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        raise RuntimeError(f"MediaPipe did not detect a face: {image_path}")
    landmarks = np.asarray(
        [
            [landmark.x * width, landmark.y * height]
            for landmark in result.multi_face_landmarks[0].landmark
        ],
        dtype=np.float64,
    )
    return landmarks[MEDIAPIPE_NOSE_TIP_INDEX], landmarks, (width, height)


def visible_face_roi(
    vertices_world: np.ndarray,
    vertices_reg_mm: np.ndarray,
    camera_json: Path,
    landmarks_pixel: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    projected, camera_vertices, camera = project_world_to_pixels(
        vertices_world, camera_json
    )
    width = int(camera["width"])
    height = int(camera["height"])
    oval = cv2.convexHull(
        landmarks_pixel[np.asarray(FACE_OVAL_INDICES, dtype=np.int64)].astype(
            np.float32
        )
    )
    oval_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(oval_mask, np.rint(oval[:, 0]).astype(np.int32), 1)

    rounded = np.rint(projected).astype(np.int64)
    valid = (
        (camera_vertices[:, 2] > 0)
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    valid_indices = np.flatnonzero(valid)
    inside = oval_mask[
        rounded[valid_indices, 1], rounded[valid_indices, 0]
    ].astype(bool)
    local_indices = valid_indices[inside]
    if len(local_indices) < 5000:
        raise RuntimeError(
            f"Face-oval projection retained too few target vertices: {len(local_indices)}"
        )

    bins_x = rounded[local_indices, 0] // FACE_ROI_BIN_SIZE_PX
    bins_y = rounded[local_indices, 1] // FACE_ROI_BIN_SIZE_PX
    bin_width = (width + FACE_ROI_BIN_SIZE_PX - 1) // FACE_ROI_BIN_SIZE_PX
    keys = bins_y * bin_width + bins_x
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    local_depth_mm = -vertices_reg_mm[local_indices, 1]
    minimum_depth = np.full(len(unique_keys), np.inf, dtype=np.float64)
    np.minimum.at(minimum_depth, inverse, local_depth_mm)
    visible = local_depth_mm <= minimum_depth[inverse] + FACE_ROI_DEPTH_MARGIN_MM
    roi_indices = local_indices[visible]
    if len(roi_indices) < 3000:
        raise RuntimeError(
            f"Visible face ROI retained too few target vertices: {len(roi_indices)}"
        )
    diagnostics = {
        "face_oval_landmark_count": int(len(FACE_OVAL_INDICES)),
        "face_oval_projected_vertices": int(len(local_indices)),
        "visible_face_roi_vertices": int(len(roi_indices)),
        "face_roi_bin_size_px": FACE_ROI_BIN_SIZE_PX,
        "face_roi_depth_margin_mm": FACE_ROI_DEPTH_MARGIN_MM,
    }
    return vertices_reg_mm[roi_indices], diagnostics


def visible_surface_anchor(
    vertices_world: np.ndarray,
    faces: np.ndarray,
    vertices_reg_mm: np.ndarray,
    camera_json: Path,
    landmark_pixel: np.ndarray,
    mm_per_unit: float,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    projected, camera_vertices, camera = project_world_to_pixels(
        vertices_world, camera_json
    )
    width = int(camera["width"])
    height = int(camera["height"])
    valid = (
        (camera_vertices[:, 2] > 0)
        & (projected[:, 0] >= 0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < height)
    )
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < 1000:
        raise RuntimeError("Too few target vertices project into the selected camera")
    projected_valid = projected[valid_indices]
    pixel_distances = np.linalg.norm(
        projected_valid - landmark_pixel[None, :], axis=1
    )

    # Intersect the landmark ray with projected mesh triangles. Screen-space
    # barycentric weights are perspective-corrected through reciprocal depth.
    triangle_pixels = projected[faces]
    triangle_depths = camera_vertices[faces, 2]
    triangle_valid = np.all(triangle_depths > 1e-8, axis=1)
    px, py = float(landmark_pixel[0]), float(landmark_pixel[1])
    triangle_valid &= (
        (np.min(triangle_pixels[:, :, 0], axis=1) <= px)
        & (np.max(triangle_pixels[:, :, 0], axis=1) >= px)
        & (np.min(triangle_pixels[:, :, 1], axis=1) <= py)
        & (np.max(triangle_pixels[:, :, 1], axis=1) >= py)
    )
    candidate_faces = np.flatnonzero(triangle_valid)
    ray_depths: list[float] = []
    hit_faces: list[int] = []
    tolerance = 1e-7
    for face_index in candidate_faces:
        a, b, c = triangle_pixels[face_index]
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (
            a[1] - c[1]
        )
        if abs(float(denominator)) <= 1e-12:
            continue
        weight_a = (
            (b[1] - c[1]) * (px - c[0])
            + (c[0] - b[0]) * (py - c[1])
        ) / denominator
        weight_b = (
            (c[1] - a[1]) * (px - c[0])
            + (a[0] - c[0]) * (py - c[1])
        ) / denominator
        weight_c = 1.0 - weight_a - weight_b
        if min(weight_a, weight_b, weight_c) < -tolerance:
            continue
        depths = triangle_depths[face_index]
        inverse_depth = (
            weight_a / depths[0]
            + weight_b / depths[1]
            + weight_c / depths[2]
        )
        if inverse_depth <= 0:
            continue
        ray_depths.append(float(1.0 / inverse_depth))
        hit_faces.append(int(face_index))

    if ray_depths:
        order = np.argsort(np.asarray(ray_depths))
        depth = float(ray_depths[int(order[0])])
        camera = json.loads(camera_json.read_text(encoding="utf-8"))
        intrinsic = np.asarray(camera["K"], dtype=np.float64)
        camera_x = (px - intrinsic[0, 2]) * depth / intrinsic[0, 0]
        camera_y = (py - intrinsic[1, 2]) * depth / intrinsic[1, 1]
        anchor = (
            np.asarray([camera_x, -depth, -camera_y], dtype=np.float64)
            * mm_per_unit
        )
        sorted_depths = np.asarray(ray_depths, dtype=np.float64)[order]
        second_depth_gap_mm = (
            float((sorted_depths[1] - sorted_depths[0]) * mm_per_unit)
            if len(sorted_depths) > 1
            else None
        )
        diagnostics = {
            "anchor_surface_method": "perspective-correct triangle-ray intersection",
            "ray_intersection_count": int(len(ray_depths)),
            "nearest_mesh_projection_residual_px": float(np.min(pixel_distances)),
            "visible_anchor_projection_residual_px": 0.0,
            "nearest_ray_depth_mm": float(depth * mm_per_unit),
            "second_ray_depth_gap_mm": second_depth_gap_mm,
            "pixel_radius_used": 0.0,
            "local_surface_candidates": 0,
            "visible_surface_candidates": 1,
            "front_depth_quantile": None,
            "front_depth_margin_mm": None,
            "anchor_average_points": 0,
        }
        return (
            anchor,
            diagnostics,
            np.asarray([hit_faces[int(order[0])]], dtype=np.int64),
            landmark_pixel.copy(),
        )

    # A sparse or locally incomplete mesh may not intersect the exact ray.
    # Fall back to a small visible-point neighborhood and record that event.
    radius = INITIAL_PIXEL_RADIUS
    local = pixel_distances <= radius
    while int(local.sum()) < MINIMUM_SURFACE_CANDIDATES and radius < MAXIMUM_PIXEL_RADIUS:
        radius = min(MAXIMUM_PIXEL_RADIUS, radius * 2.0)
        local = pixel_distances <= radius
    if int(local.sum()) < MINIMUM_SURFACE_CANDIDATES:
        nearest = np.argsort(pixel_distances)[:MINIMUM_SURFACE_CANDIDATES]
        local_indices = valid_indices[nearest]
    else:
        local_indices = valid_indices[np.flatnonzero(local)]
    local_depth_mm = camera_vertices[local_indices, 2] * mm_per_unit
    front_reference_mm = float(
        np.quantile(local_depth_mm, FRONT_DEPTH_QUANTILE)
    )
    visible_indices = local_indices[
        local_depth_mm <= front_reference_mm + FRONT_DEPTH_MARGIN_MM
    ]
    if len(visible_indices) < ANCHOR_AVERAGE_POINTS:
        visible_indices = local_indices[
            np.argsort(local_depth_mm)[:ANCHOR_AVERAGE_POINTS]
        ]

    visible_pixel_distances = np.linalg.norm(
        projected[visible_indices] - landmark_pixel[None, :], axis=1
    )
    selected_indices = visible_indices[
        np.argsort(visible_pixel_distances)[:ANCHOR_AVERAGE_POINTS]
    ]
    anchor = np.mean(vertices_reg_mm[selected_indices], axis=0)
    anchor_world = np.mean(vertices_world[selected_indices], axis=0, keepdims=True)
    anchor_projected, _, _ = project_world_to_pixels(anchor_world, camera_json)
    anchor_pixel = anchor_projected[0]
    nearest_projected_distance = float(np.min(pixel_distances))
    anchor_projected_distance = float(np.linalg.norm(anchor_pixel - landmark_pixel))
    diagnostics = {
        "anchor_surface_method": "visible-point fallback",
        "ray_intersection_count": 0,
        "pixel_radius_used": float(radius),
        "local_surface_candidates": int(len(local_indices)),
        "visible_surface_candidates": int(len(visible_indices)),
        "nearest_mesh_projection_residual_px": nearest_projected_distance,
        "visible_anchor_projection_residual_px": anchor_projected_distance,
        "front_depth_quantile": FRONT_DEPTH_QUANTILE,
        "front_depth_margin_mm": FRONT_DEPTH_MARGIN_MM,
        "anchor_average_points": int(len(selected_indices)),
    }
    return anchor, diagnostics, selected_indices, anchor_pixel


def render_image_overlay(
    path: Path,
    image_path: Path,
    subject: str,
    landmark_pixel: np.ndarray,
    anchor_pixel: np.ndarray,
    diagnostics: dict[str, Any],
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read target image: {image_path}")
    display = image.copy()
    landmark = tuple(np.rint(landmark_pixel).astype(int))
    projected = tuple(np.rint(anchor_pixel).astype(int))
    cv2.drawMarker(
        display,
        landmark,
        (70, 220, 70),
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
        thickness=3,
    )
    cv2.circle(display, projected, 8, (0, 210, 255), thickness=3)
    cv2.line(display, landmark, projected, (255, 180, 0), thickness=2)
    label = (
        f"subject {subject} | reprojection residual "
        f"{diagnostics['visible_anchor_projection_residual_px']:.2f} px"
    )
    cv2.rectangle(display, (10, 10), (min(display.shape[1] - 10, 780), 84), (0, 0, 0), -1)
    cv2.putText(
        display,
        "INTERNAL AUDIT ONLY - DO NOT PUBLISH",
        (22, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        label,
        (22, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), display)


def render_anatomical_scale_overlay(
    path: Path,
    image_path: Path,
    subject: str,
    landmarks_pixel: np.ndarray,
    surface_pixels: dict[str, list[np.ndarray]],
    interocular_distance_mm: float,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read target image: {image_path}")
    display = image.copy()
    eye_centers: dict[str, tuple[int, int]] = {}
    eye_groups = {
        "left": MEDIAPIPE_LEFT_EYE_CORNERS,
        "right": MEDIAPIPE_RIGHT_EYE_CORNERS,
    }
    colors = {"left": (70, 220, 70), "right": (255, 180, 0)}
    for name, indices in eye_groups.items():
        projected = np.asarray(surface_pixels[name], dtype=np.float64)
        eye_centers[name] = tuple(np.rint(np.mean(projected, axis=0)).astype(int))
        for landmark_index, surface_pixel in zip(indices, projected, strict=True):
            detected = tuple(np.rint(landmarks_pixel[landmark_index]).astype(int))
            on_surface = tuple(np.rint(surface_pixel).astype(int))
            cv2.drawMarker(
                display,
                detected,
                colors[name],
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=3,
            )
            cv2.circle(display, on_surface, 7, (0, 210, 255), thickness=3)
            cv2.line(display, detected, on_surface, colors[name], thickness=2)
    cv2.line(display, eye_centers["left"], eye_centers["right"], (255, 255, 255), 3)
    cv2.rectangle(display, (10, 10), (min(display.shape[1] - 10, 860), 84), (0, 0, 0), -1)
    cv2.putText(
        display,
        "INTERNAL AUDIT ONLY - DO NOT PUBLISH",
        (22, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        f"subject {subject} | target 3D interocular distance {interocular_distance_mm:.2f} mm",
        (22, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), display)


def render_diagnostic(
    path: Path,
    subject: str,
    target_mm: np.ndarray,
    old_anchor: np.ndarray,
    new_anchor: np.ndarray,
) -> None:
    sample = target_mm[registration.deterministic_indices(len(target_mm), 70000)]
    figure, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=170)
    for axis, coordinates, title in (
        (axes[0], (0, 2), "Frontal view"),
        (axes[1], (1, 2), "Profile view"),
    ):
        x_index, y_index = coordinates
        axis.scatter(
            sample[:, x_index],
            sample[:, y_index],
            s=0.12,
            color="#a9afb5",
            alpha=0.28,
            linewidths=0,
        )
        axis.scatter(
            [old_anchor[x_index]],
            [old_anchor[y_index]],
            marker="x",
            s=70,
            color="#d44b4b",
            linewidths=1.8,
            label="extent-based anchor",
        )
        axis.scatter(
            [new_anchor[x_index]],
            [new_anchor[y_index]],
            marker="o",
            s=55,
            color="#ffd23f",
            edgecolors="#111111",
            linewidths=0.7,
            label="image-projected anchor",
        )
        axis.set_aspect("equal")
        axis.set_title(title)
        axis.set_xlabel(("x", "y", "z")[x_index] + " (mm)")
        axis.set_ylabel(("x", "y", "z")[y_index] + " (mm)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.suptitle(f"Internal target-anchor audit | subject {subject}")
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve(strict=True)
    revision = registration.checked_existing(
        root / "cmes_revision_20260816", root, "revision root"
    )
    scale_path = registration.checked_existing(
        args.scale_dict, revision, "scale dictionary"
    )
    split_path = registration.checked_existing(args.split, revision, "identity split")
    output = args.output_dir.expanduser().resolve(strict=False)
    if output == revision or not registration.is_relative_to(output, revision):
        raise ValueError(f"Output escapes revision root: {output}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    diagnostics_dir = output / "diagnostics"
    diagnostics_dir.mkdir()
    roi_dir = output / "target_face_roi_mm"
    roi_dir.mkdir()

    split = json.loads(split_path.read_text(encoding="utf-8"))
    subject_key = (
        "development_subjects" if args.subset == "development" else "test_subjects"
    )
    subjects = [f"{int(value):03d}" for value in split[subject_key]]
    scales = json.loads(scale_path.read_text(encoding="utf-8"))
    manifest_path = registration.checked_existing(
        root / "prepared_cohort" / "facescape_frontal_pairs_manifest.csv",
        root,
        "FaceScape pair manifest",
    )
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest = {row["pair_id"]: row for row in csv.DictReader(handle)}

    rows: list[dict[str, Any]] = []
    anchors: dict[str, Any] = {}
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        for subject in subjects:
            pair = f"{subject}_18_eye_closed"
            target_path = registration.checked_existing(
                Path(manifest[pair]["mesh"]), root, f"target mesh for {subject}"
            )
            image_path = registration.checked_existing(
                Path(manifest[pair]["image"]), root, f"target image for {subject}"
            )
            camera_json = registration.checked_existing(
                target_path.parent / "selected_camera.json", root, "selected camera"
            )
            target_world, target_faces = registration.load_trimesh(target_path)
            mm_per_unit = float(scales[str(int(subject))]["18"][0])
            target_mm = registration.target_registration_frame(
                target_world, camera_json
            ) * mm_per_unit
            landmark_pixel, landmarks_pixel, image_size = face_landmark_pixels(
                face_mesh, image_path
            )
            target_face_roi, roi_diagnostic = visible_face_roi(
                target_world,
                target_mm,
                camera_json,
                landmarks_pixel,
            )
            roi_relative = Path("target_face_roi_mm") / f"{subject}.npz"
            np.savez_compressed(
                output / roi_relative,
                target_face_roi_mm=target_face_roi.astype(np.float32),
            )
            new_anchor, diagnostic, _, anchor_pixel = visible_surface_anchor(
                target_world,
                target_faces,
                target_mm,
                camera_json,
                landmark_pixel,
                mm_per_unit,
            )
            eye_corner_points: dict[str, list[np.ndarray]] = {}
            eye_surface_pixels: dict[str, list[np.ndarray]] = {}
            eye_diagnostics: dict[str, list[dict[str, Any]]] = {}
            for eye_name, corner_indices in {
                "left": MEDIAPIPE_LEFT_EYE_CORNERS,
                "right": MEDIAPIPE_RIGHT_EYE_CORNERS,
            }.items():
                eye_corner_points[eye_name] = []
                eye_surface_pixels[eye_name] = []
                eye_diagnostics[eye_name] = []
                for landmark_index in corner_indices:
                    point, point_diagnostic, _, surface_pixel = visible_surface_anchor(
                        target_world,
                        target_faces,
                        target_mm,
                        camera_json,
                        landmarks_pixel[landmark_index],
                        mm_per_unit,
                    )
                    eye_corner_points[eye_name].append(point)
                    eye_surface_pixels[eye_name].append(surface_pixel)
                    eye_diagnostics[eye_name].append(point_diagnostic)
            eye_centers = {
                name: np.mean(np.asarray(points, dtype=np.float64), axis=0)
                for name, points in eye_corner_points.items()
            }
            interocular_distance_mm = float(
                np.linalg.norm(eye_centers["left"] - eye_centers["right"])
            )
            if not 30.0 <= interocular_distance_mm <= 100.0:
                raise ValueError(
                    f"Implausible target interocular distance for {subject}: "
                    f"{interocular_distance_mm:.3f} mm"
                )
            old_anchor = registration.target_nose_anchor(target_mm)
            row = {
                "subject": subject,
                "target_pair": pair,
                "image_width": int(image_size[0]),
                "image_height": int(image_size[1]),
                "landmark_x_px": float(landmark_pixel[0]),
                "landmark_y_px": float(landmark_pixel[1]),
                "old_to_new_anchor_mm": float(np.linalg.norm(old_anchor - new_anchor)),
                "old_to_new_delta_x_mm": float(old_anchor[0] - new_anchor[0]),
                "old_to_new_delta_front_mm": float(old_anchor[1] - new_anchor[1]),
                "old_to_new_delta_vertical_mm": float(old_anchor[2] - new_anchor[2]),
                "target_interocular_distance_mm": interocular_distance_mm,
                "eye_corner_ray_fallbacks": int(
                    sum(
                        diagnostic_item["anchor_surface_method"]
                        != "perspective-correct triangle-ray intersection"
                        for diagnostic_list in eye_diagnostics.values()
                        for diagnostic_item in diagnostic_list
                    )
                ),
                **diagnostic,
                **roi_diagnostic,
            }
            rows.append(row)
            anchors[subject] = {
                "target_pair": pair,
                "anchor_mm": [float(value) for value in new_anchor],
                "method": "MediaPipe FaceMesh landmark 1 projected to visible target scan surface",
                "interocular_distance_mm": interocular_distance_mm,
                "interocular_method": (
                    "Euclidean distance between 3D eye centers; each center is the "
                    "mean of the two MediaPipe canthus rays intersected with the "
                    "visible target scan surface"
                ),
                "eye_corner_landmark_indices": {
                    "left": list(MEDIAPIPE_LEFT_EYE_CORNERS),
                    "right": list(MEDIAPIPE_RIGHT_EYE_CORNERS),
                },
                "eye_corner_points_mm": {
                    name: [[float(value) for value in point] for point in points]
                    for name, points in eye_corner_points.items()
                },
                "eye_centers_mm": {
                    name: [float(value) for value in center]
                    for name, center in eye_centers.items()
                },
                "face_roi_npz": roi_relative.as_posix(),
                "face_roi_definition": (
                    "camera-visible target scan vertices inside the MediaPipe "
                    "FaceMesh face-oval convex hull"
                ),
                "diagnostics": row,
            }
            render_diagnostic(
                diagnostics_dir / f"target_anchor_{subject}.png",
                subject,
                target_mm,
                old_anchor,
                new_anchor,
            )
            render_image_overlay(
                diagnostics_dir / f"target_anchor_image_{subject}.png",
                image_path,
                subject,
                landmark_pixel,
                anchor_pixel,
                diagnostic,
            )
            render_anatomical_scale_overlay(
                diagnostics_dir / f"target_anatomical_scale_{subject}.png",
                image_path,
                subject,
                landmarks_pixel,
                eye_surface_pixels,
                interocular_distance_mm,
            )
            print(json.dumps({"completed_subject": subject, **diagnostic}), flush=True)

    fields = sorted({key for row in rows for key in row})
    with (output / "target_anchor_audit_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "subset": args.subset,
        "subjects": subjects,
        "development_only_during_parameter_selection": args.subset == "development",
        "mediapipe_nose_tip_index": MEDIAPIPE_NOSE_TIP_INDEX,
        "mediapipe_left_eye_corners": list(MEDIAPIPE_LEFT_EYE_CORNERS),
        "mediapipe_right_eye_corners": list(MEDIAPIPE_RIGHT_EYE_CORNERS),
        "initial_pixel_radius": INITIAL_PIXEL_RADIUS,
        "maximum_pixel_radius": MAXIMUM_PIXEL_RADIUS,
        "minimum_surface_candidates": MINIMUM_SURFACE_CANDIDATES,
        "front_depth_quantile": FRONT_DEPTH_QUANTILE,
        "front_depth_margin_mm": FRONT_DEPTH_MARGIN_MM,
        "anchor_average_points": ANCHOR_AVERAGE_POINTS,
        "face_roi_bin_size_px": FACE_ROI_BIN_SIZE_PX,
        "face_roi_depth_margin_mm": FACE_ROI_DEPTH_MARGIN_MM,
        "anchors": anchors,
    }
    (output / "target_anchors_mm.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
