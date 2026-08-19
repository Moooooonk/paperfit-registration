#!/usr/bin/env python3
"""Development-only pilot for a millimeter-scaled anchor-aware S8 update."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


REPORT_PATHS: tuple[str, ...] = ()
PILOT_CASES: tuple[str, ...] = ()
VARIANTS = (
    {"name": "A0_mm_no_anchor", "anchor_weight": 0.0, "anchor_step_mm": 0.0},
    {"name": "A1_w25_s4", "anchor_weight": 25.0, "anchor_step_mm": 4.0},
    {"name": "A2_w60_s6", "anchor_weight": 60.0, "anchor_step_mm": 6.0},
    {"name": "A3_w120_s10", "anchor_weight": 120.0, "anchor_step_mm": 10.0},
)
S8_CONTOURS = (0.00, 0.22, 0.40, 0.55, 0.68, 0.78, 0.87, 0.94)
# Median official expression-18 scale of the frozen development identities.
# Evaluation-identity scales are not used to define any S8 physical parameter.
REFERENCE_MM_PER_UNIT = 262.58104988598365


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def checked_existing(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not is_relative_to(resolved, root):
        raise ValueError(f"{label} escapes root: {resolved}")
    return resolved


def checked_output(path: Path, root: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if resolved == root or not is_relative_to(resolved, root):
        raise ValueError(f"Output escapes root: {resolved}")
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite output: {resolved}")
    return resolved


def load_reports(root: Path) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    branches = ("rigid_pass", "anchor_only", "broad_failure")
    for branch, relative in zip(branches, REPORT_PATHS):
        path = checked_existing(root / relative, root, "S8 report")
        for report in json.loads(path.read_text(encoding="utf-8")):
            case = str(report["case"])
            reports[case] = {**report, "_input_branch": branch}
            counts[case] += 1
    repeated = sorted(case for case, count in counts.items() if count != 1)
    if repeated or len(reports) != 380:
        raise ValueError(f"Invalid report partition: n={len(reports)}, repeated={repeated}")
    return reports


def source_anchor(vertices: np.ndarray, mask: np.ndarray) -> np.ndarray:
    candidates = vertices[np.asarray(mask, dtype=bool)]
    if len(candidates) < 1:
        raise ValueError("Frozen source-anchor mask is empty")
    return np.mean(candidates, axis=0)


def clipped_vector(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm <= 1e-12:
        return vector
    return vector * (max_norm / norm)


def blend_anchor_targets(
    current: np.ndarray,
    target_idx: np.ndarray,
    target_pos: np.ndarray,
    target_weights: np.ndarray,
    anchor_mask: np.ndarray,
    target_anchor: np.ndarray,
    anchor_weight: float,
    anchor_step_mm: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    before = float(np.linalg.norm(source_anchor(current, anchor_mask) - target_anchor))
    if anchor_weight <= 0.0 or anchor_step_mm <= 0.0:
        return target_pos, target_weights, before, before
    shift = clipped_vector(
        target_anchor - source_anchor(current, anchor_mask), anchor_step_mm
    )
    active_anchor = anchor_mask[target_idx]
    if not np.any(active_anchor):
        return target_pos, target_weights, before, before
    anchor_targets = current[target_idx[active_anchor]] + shift
    old_weights = target_weights[active_anchor]
    target_pos[active_anchor] = (
        old_weights[:, None] * target_pos[active_anchor]
        + anchor_weight * anchor_targets
    ) / (old_weights + anchor_weight)[:, None]
    target_weights[active_anchor] = old_weights + anchor_weight
    predicted = current.copy()
    predicted[target_idx[active_anchor]] = target_pos[active_anchor]
    after_target = float(np.linalg.norm(source_anchor(predicted, anchor_mask) - target_anchor))
    return target_pos, target_weights, before, after_target


def run_variant(
    initial_mm: np.ndarray,
    faces: np.ndarray,
    target_sample_mm: np.ndarray,
    masks: dict[str, np.ndarray],
    target_anchor_mm: np.ndarray,
    nonrigid: Any,
    variant: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    current = initial_mm.copy()
    start = initial_mm.copy()
    edges = nonrigid.build_edges(faces)
    tree = cKDTree(target_sample_mm)
    disable_eye_exclusion = bool(variant.get("disable_eye_exclusion", False))
    disable_contours = bool(variant.get("disable_contours", False))
    disable_region_weights = bool(variant.get("disable_region_weights", False))
    if disable_eye_exclusion:
        fixed_idx = np.empty(0, dtype=np.int64)
        excluded_eye = np.zeros(len(current), dtype=bool)
        fitting_full = masks["full_no_eye"] | masks["eye_soft"]
    else:
        fixed_idx = np.flatnonzero(masks["eye_soft"])
        if len(fixed_idx) < 50:
            fixed_idx = np.flatnonzero(masks["eye"])
        excluded_eye = masks["eye_soft"]
        fitting_full = masks["full_no_eye"]
    anchor_mask = masks["source_anchor"]
    history: list[dict[str, Any]] = []

    contours = tuple(float(value) for value in variant.get("contours", S8_CONTOURS))
    step_multiplier = float(variant.get("step_multiplier", 1.0))
    gain_multiplier = float(variant.get("gain_multiplier", 1.0))
    contour_sets = []
    if not disable_contours:
        contour_sets, _ = nonrigid.depth_contours(current, masks, contours)
    local_sets = contour_sets + [
        ("bridge", masks["nasal_bridge"]),
        ("dorsum", masks["nasal_dorsum"]),
        ("tip_alar", masks["nose_tip"] | masks["alar"]),
        ("subnasal", masks["subnasal"]),
        ("philtrum", masks["philtrum"]),
    ]
    local_weight_slope = 85.0 / REFERENCE_MM_PER_UNIT
    local_weight_clip_mm = 0.20 * REFERENCE_MM_PER_UNIT
    for pass_i, (name, active_mask) in enumerate(local_sets, 1):
        target_idx = np.flatnonzero(
            active_mask & fitting_full & ~excluded_eye
        )
        before_anchor = float(
            np.linalg.norm(source_anchor(current, anchor_mask) - target_anchor_mm)
        )
        if len(target_idx) < 20:
            history.append(
                {
                    "phase": "local",
                    "pass": pass_i,
                    "name": str(name),
                    "active_vertices": int(len(target_idx)),
                    "executed_constrained_solve": False,
                    "skipped_reason": "fewer_than_20_active_vertices",
                    "anchor_before_mm": before_anchor,
                    "anchor_blended_target_mm": before_anchor,
                    "anchor_after_mm": before_anchor,
                }
            )
            continue
        d, nearest = tree.query(current[target_idx], k=1, workers=1)
        delta = target_sample_mm[nearest] - current[target_idx]
        norm = np.linalg.norm(delta, axis=1)
        max_step = (
            0.018 * REFERENCE_MM_PER_UNIT
            if pass_i <= len(contour_sets)
            else 0.014 * REFERENCE_MM_PER_UNIT
        ) * step_multiplier
        step = np.minimum(1.0, max_step / np.maximum(norm, 1e-8))
        target_pos = current[target_idx] + delta * step[:, None]
        weights = 18.0 + np.clip(d, 0.0, local_weight_clip_mm) * local_weight_slope
        if not disable_region_weights and (
            name == "tip_alar" or (isinstance(name, float) and name >= 0.75)
        ):
            weights *= 1.25
        if name == "tip_alar":
            target_pos, weights, before_anchor, target_anchor_goal = blend_anchor_targets(
                current,
                target_idx,
                target_pos,
                weights,
                anchor_mask,
                target_anchor_mm,
                float(variant["anchor_weight"]),
                float(variant["anchor_step_mm"]),
            )
        else:
            target_anchor_goal = before_anchor
        solved = nonrigid.solve(
            current,
            edges,
            target_idx,
            target_pos,
            weights,
            fixed_idx,
            start[fixed_idx],
            edge_w=190.0,
            fixed_w=32000.0,
        )
        local_gain = min(1.0, 0.43 * gain_multiplier)
        current = local_gain * solved + (1.0 - local_gain) * current
        current[fixed_idx] = start[fixed_idx]
        history.append(
            {
                "phase": "local",
                "pass": pass_i,
                "name": str(name),
                "active_vertices": int(len(target_idx)),
                "executed_constrained_solve": True,
                "skipped_reason": "",
                "anchor_before_mm": before_anchor,
                "anchor_blended_target_mm": target_anchor_goal,
                "anchor_after_mm": float(
                    np.linalg.norm(source_anchor(current, anchor_mask) - target_anchor_mm)
                ),
            }
        )

    full_schedule = (
        (210.0, 9.0, 0.12, 0.018 * REFERENCE_MM_PER_UNIT),
        (185.0, 11.0, 0.14, 0.016 * REFERENCE_MM_PER_UNIT),
        (165.0, 12.0, 0.13, 0.014 * REFERENCE_MM_PER_UNIT),
    )
    full_weight_slope = 65.0 / REFERENCE_MM_PER_UNIT
    full_weight_clip_mm = 0.25 * REFERENCE_MM_PER_UNIT
    target_idx = np.flatnonzero(fitting_full & ~excluded_eye)
    for pass_i, (edge_w, base_w, gain, max_step) in enumerate(full_schedule, 1):
        max_step *= step_multiplier
        gain = min(1.0, gain * gain_multiplier)
        d, nearest = tree.query(current[target_idx], k=1, workers=1)
        delta = target_sample_mm[nearest] - current[target_idx]
        norm = np.linalg.norm(delta, axis=1)
        step = np.minimum(1.0, max_step / np.maximum(norm, 1e-8))
        target_pos = current[target_idx] + delta * step[:, None]
        weights = np.full(len(target_idx), base_w, dtype=np.float64)
        weights += np.clip(d, 0.0, full_weight_clip_mm) * full_weight_slope
        if not disable_region_weights:
            for key, factor in (
                ("nasal_bridge", 1.30),
                ("nasal_dorsum", 1.35),
                ("nose_tip", 1.42),
                ("alar", 1.35),
                ("subnasal", 1.18),
                ("philtrum", 1.12),
            ):
                weights[masks[key][target_idx]] *= factor
        target_pos, weights, before_anchor, target_anchor_goal = blend_anchor_targets(
            current,
            target_idx,
            target_pos,
            weights,
            anchor_mask,
            target_anchor_mm,
            float(variant["anchor_weight"]),
            float(variant["anchor_step_mm"]),
        )
        solved = nonrigid.solve(
            current,
            edges,
            target_idx,
            target_pos,
            weights,
            fixed_idx,
            start[fixed_idx],
            edge_w=edge_w,
            fixed_w=32000.0,
        )
        current = (1.0 - gain) * current + gain * solved
        current[fixed_idx] = start[fixed_idx]
        history.append(
            {
                "phase": "full",
                "pass": pass_i,
                "active_vertices": int(len(target_idx)),
                "executed_constrained_solve": True,
                "skipped_reason": "",
                "anchor_before_mm": before_anchor,
                "anchor_blended_target_mm": target_anchor_goal,
                "anchor_after_mm": float(
                    np.linalg.norm(source_anchor(current, anchor_mask) - target_anchor_mm)
                ),
            }
        )
    return current, history


def quantiles(distances: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    values = distances[mask]
    return float(np.median(values)), float(np.quantile(values, 0.90))


def edge_strain(
    initial: np.ndarray, final: np.ndarray, faces: np.ndarray, nonrigid: Any
) -> dict[str, float]:
    edges = nonrigid.build_edges(faces)
    initial_length = np.linalg.norm(initial[edges[:, 0]] - initial[edges[:, 1]], axis=1)
    final_length = np.linalg.norm(final[edges[:, 0]] - final[edges[:, 1]], axis=1)
    strain = np.abs(final_length / np.maximum(initial_length, 1e-8) - 1.0)
    return {
        "edge_strain_median": float(np.median(strain)),
        "edge_strain_p90": float(np.quantile(strain, 0.90)),
        "edge_strain_p99": float(np.quantile(strain, 0.99)),
        "edge_strain_max": float(np.max(strain)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("PAPERFIT_ROOT", ".")),
    )
    parser.add_argument("--scale-dict", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--development-split",
        type=Path,
        help="Frozen split JSON used with --all-development-anchor-only.",
    )
    parser.add_argument("--all-development-anchor-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve(strict=True)
    output = checked_output(args.output_dir, root)
    scale_path = checked_existing(args.scale_dict, root, "scale dictionary")
    scales = json.loads(scale_path.read_text(encoding="utf-8"))
    reports = load_reports(root)
    cases = list(PILOT_CASES)
    case_selection = (
        "10th/50th/90th percentiles of development anchor-only post-S8 anchor distance"
    )
    if args.all_development_anchor_only:
        if args.development_split is None:
            raise ValueError("--development-split is required")
        split_path = checked_existing(args.development_split, root, "development split")
        split = json.loads(split_path.read_text(encoding="utf-8"))
        development = set(split["development_subjects"])
        cases = sorted(
            case
            for case, report in reports.items()
            if report["_input_branch"] == "anchor_only"
            and f"{int(case.split('_', 1)[0]):03d}" in development
        )
        if len(cases) != 37:
            raise ValueError(f"Expected 37 development anchor-only cases, found {len(cases)}")
        case_selection = "all anchor-only cases from frozen development identities"
    for case in cases:
        report = reports[case]
        source = checked_existing(Path(report["source_rigid_obj"]), root, "source mesh")
        checked_existing(source.with_suffix(".jpg"), root, "source texture")
        target = checked_existing(Path(report["target_mesh"]), root, "target mesh")
        checked_existing(target.parent / "selected_camera.json", root, "target camera")
    if args.dry_run:
        print(
            json.dumps(
                {"status": "validated", "cases": cases, "variants": VARIANTS},
                indent=2,
            )
        )
        return

    tools = checked_existing(root / "research_tools", root, "research tools")
    sys.path.insert(0, str(tools))
    import run_nonrigid_nasal_depth_ablation_3case as nonrigid  # noqa: PLC0415
    import run_rigid_upright_hardgate_3case_20260619 as orient_module  # noqa: PLC0415
    import run_scratch_surface_registration_3case_final_attempt as rigid  # noqa: PLC0415

    rows = []
    details = []
    for case in cases:
        report = reports[case]
        subject = f"{int(case.split('_', 1)[0]):03d}"
        mm_per_unit = float(scales[str(int(subject))]["18"][0])
        source_obj = Path(report["source_rigid_obj"])
        initial_raw, _, faces, _, _, _ = nonrigid.parse_obj_with_uv(source_obj)
        _, masks, _ = nonrigid.build_masks(source_obj, initial_raw)
        rigid_weight, eye_hard, _ = rigid.texture_eye_weight(source_obj)
        _, rigid_masks = rigid.apply_nose_midface_weight(
            source_obj, initial_raw, rigid_weight
        )
        target_path = Path(report["target_mesh"])
        _, target_world, _ = nonrigid.load_mesh(target_path)
        target_raw = nonrigid.target_registration_frame(
            target_world, target_path.parent / "selected_camera.json"
        )
        target_sample_mm = nonrigid.deterministic_sample(target_raw, 140000) * mm_per_unit
        target_anchor_mm = rigid.target_nose_anchor(target_raw) * mm_per_unit
        initial_mm = initial_raw * mm_per_unit
        target_tree = cKDTree(target_sample_mm)
        native_source = root / "hrn_outputs" / case / f"{case}_0_hrn_mid_mesh.obj"
        native_source = checked_existing(native_source, root, "native source mesh")
        native_vertices, _, _, _, _, _ = nonrigid.parse_obj_with_uv(native_source)
        if len(native_vertices) != len(initial_raw):
            raise ValueError(f"{case}: native and rigid source topologies differ")
        import run_pairwise_mm_rigid as registration  # noqa: PLC0415

        masks["source_anchor"] = registration.fixed_source_anchor_mask(
            native_vertices, masks
        )
        anchor_mask = masks["source_anchor"]
        fixed_mask = masks["eye_soft"]

        for variant in VARIANTS:
            started = time.time()
            final_mm, history = run_variant(
                initial_mm,
                faces,
                target_sample_mm,
                masks,
                target_anchor_mm,
                nonrigid,
                variant,
            )
            distances, _ = target_tree.query(final_mm, k=1, workers=-1)
            full_median, full_p90 = quantiles(distances, masks["full_no_eye"])
            nose_median, nose_p90 = quantiles(distances, masks["nose"])
            anchor_distance = float(
                np.linalg.norm(source_anchor(final_mm, anchor_mask) - target_anchor_mm)
            )
            eye_disp = np.linalg.norm(
                final_mm[fixed_mask] - initial_mm[fixed_mask], axis=1
            )
            orientation = orient_module.orientation_metrics(
                final_mm, rigid_masks, eye_hard
            )
            displacement = np.linalg.norm(final_mm - initial_mm, axis=1)
            row = {
                "case": case,
                "subject": subject,
                "variant": variant["name"],
                "anchor_weight": variant["anchor_weight"],
                "anchor_step_mm": variant["anchor_step_mm"],
                "full_median_mm": full_median,
                "full_p90_mm": full_p90,
                "nose_median_mm": nose_median,
                "nose_p90_mm": nose_p90,
                "uv_tip_alar_front_anchor_mm": anchor_distance,
                "orientation_pass": int(not bool(orientation["upside_down"])),
                "eye_fixed_max_mm": float(np.max(eye_disp)) if len(eye_disp) else 0.0,
                "displacement_p90_mm": float(np.quantile(displacement, 0.90)),
                "displacement_max_mm": float(np.max(displacement)),
                "runtime_seconds": time.time() - started,
                **edge_strain(initial_mm, final_mm, faces, nonrigid),
            }
            rows.append(row)
            details.append({"row": row, "history": history})
            print(json.dumps(row), flush=True)

    output.mkdir(parents=True, exist_ok=False)
    with (output / "anchor_aware_s8_pilot_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output / "anchor_aware_s8_pilot_details.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )
    (output / "anchor_aware_s8_pilot_config.json").write_text(
        json.dumps(
            {
                "development_only": True,
                "case_selection": case_selection,
                "cases": cases,
                "variants": VARIANTS,
                "reference_mm_per_unit": REFERENCE_MM_PER_UNIT,
                "contours": S8_CONTOURS,
                "stored_meshes": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
