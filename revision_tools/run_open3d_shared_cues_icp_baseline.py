#!/usr/bin/env python3
"""Open3D ICP baseline given the proposed method's anatomical input cues."""

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
from scipy.spatial import cKDTree

import frozen_nonrigid_nasal_solver as frozen_nonrigid
import run_mm_s8_from_rigid as s8_runner
import run_open3d_common_roi_baselines as open3d_baseline
import run_pairwise_mm_rigid as registration
from shared_cues_candidate_policy import select_candidate


METHOD = "open3d_shared_cues_multistart_icp"


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
    parser.add_argument("--anchor-cap-mm", type=float, required=True)
    parser.add_argument("--voxel-divisor", type=float, default=42.0)
    parser.add_argument("--voxel-floor-mm", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=2)
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


def process_case(
    case: str,
    root_string: str,
    scale_path_string: str,
    anchor_path_string: str,
    source_method: str,
    source_root_string: str | None,
    topology_string: str | None,
    output_string: str,
    anchor_cap_mm: float,
    voxel_divisor: float,
    voxel_floor_mm: float,
) -> dict[str, Any]:
    root = Path(root_string)
    output = Path(output_string)
    row_path = output / "case_rows" / f"{case}.json"
    detail_path = output / "case_details" / f"{case}.json"
    if row_path.exists() and detail_path.exists():
        return json.loads(row_path.read_text(encoding="utf-8"))

    scale_dict = json.loads(Path(scale_path_string).read_text(encoding="utf-8"))
    anchor_path = Path(anchor_path_string)
    anchor_payload = json.loads(anchor_path.read_text(encoding="utf-8"))
    source_root = Path(source_root_string) if source_root_string else None
    topology = Path(topology_string) if topology_string else None
    _, source, _, fit_weight, masks = registration.load_source(
        case, source_method, root, source_root, topology
    )
    target, target_anchor, _, _, _ = s8_runner.target_for_case(
        case, root, scale_dict, anchor_payload, anchor_path
    )
    subject = f"{int(case.split('_', 1)[0]):03d}"
    target_record = anchor_payload.get("anchors", {}).get(subject)
    if target_record is None:
        raise KeyError(f"Missing target anatomical record for {subject}")
    target_interocular_mm = float(target_record["interocular_distance_mm"])
    source_interocular = registration.source_interocular_distance(source, masks)
    fixed_scale = registration.anatomical_scale_from_interocular(
        source_interocular, target_interocular_mm
    )

    seed = int(hashlib.sha256(case.encode("utf-8")).hexdigest()[:8], 16) % 2147483647
    source_fit = open3d_baseline.weighted_sample(source, fit_weight, 28000, seed)
    target_fit = target[registration.deterministic_indices(len(target), 90000)]
    target_extent = float(
        np.linalg.norm(
            np.quantile(target_fit, 0.95, axis=0)
            - np.quantile(target_fit, 0.05, axis=0)
        )
    )
    voxel_mm = max(target_extent / voxel_divisor, voxel_floor_mm)
    target_front_coordinate = target_fit[:, 1]
    target_front = target_fit[
        target_front_coordinate >= np.quantile(target_front_coordinate, 0.78)
    ]
    if len(target_front) < 5000:
        target_front = target_fit
    target_center = np.median(target_front, axis=0)
    target_front_y = float(np.quantile(target_front_coordinate, 0.94))

    states = []
    started = time.time()
    identity = np.eye(4, dtype=np.float64)
    for rotation_index, rotation in enumerate(registration.proper_axis_rotations(), 1):
        moved_all = registration.transform(
            source, fixed_scale, rotation, np.zeros(3, dtype=np.float64)
        )
        moved_fit = registration.transform(
            source_fit, fixed_scale, rotation, np.zeros(3, dtype=np.float64)
        )
        surface_translation = target_center - np.median(moved_fit, axis=0)
        surface_translation[1] = target_front_y - np.max(moved_fit[:, 1])
        anchor_translation = target_anchor - registration.source_anchor(
            moved_all, masks
        )
        for translation_mode, initial_translation in (
            ("surface", surface_translation),
            ("nose_anchor", anchor_translation),
        ):
            candidate_index = len(states) + 1
            initial_fit = moved_fit + initial_translation
            try:
                delta = open3d_baseline.refine_icp(
                    initial_fit, target_fit, identity, voxel_mm
                )
                scale, final_rotation, final_translation = (
                    open3d_baseline.matrix_to_similarity(
                        delta, fixed_scale, rotation, initial_translation
                    )
                )
                aligned = registration.transform(
                    source, scale, final_rotation, final_translation
                )
                metrics = registration.geometric_metrics(
                    aligned,
                    cKDTree(target),
                    target,
                    fit_weight,
                    masks,
                    target_anchor,
                )
                orientation = registration.orientation_metrics(aligned, masks)
                state = {
                    "candidate_index": candidate_index,
                    "rotation_index": rotation_index,
                    "translation_mode": translation_mode,
                    "scale": float(scale),
                    "rotation": final_rotation,
                    "translation": final_translation,
                    "selection_score_mm": registration.combined_score(
                        metrics, orientation
                    ),
                    "nose_anchor_point_mm": float(metrics["nose_anchor_point_mm"]),
                    "upside_down": int(orientation["upside_down"]),
                    "metrics": metrics,
                    "orientation": orientation,
                    "error": "",
                }
            except Exception as exc:  # pragma: no cover - Open3D failure path
                state = {
                    "candidate_index": candidate_index,
                    "rotation_index": rotation_index,
                    "translation_mode": translation_mode,
                    "selection_score_mm": None,
                    "nose_anchor_point_mm": None,
                    "upside_down": 1,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            states.append(state)

    valid_states = [state for state in states if not state["error"]]
    runtime = float(time.time() - started)
    if not valid_states:
        row = open3d_baseline.evaluate(
            case,
            source_method,
            METHOD,
            source,
            1.0,
            np.eye(3),
            np.zeros(3),
            masks,
            target,
            target_anchor,
            runtime,
            voxel_mm,
            error="all eight Open3D ICP candidates failed",
        )
        selected_index = None
        eligible_count = 0
        cap_fallback = True
    else:
        selected, eligible_count, cap_fallback = select_candidate(
            valid_states, anchor_cap_mm
        )
        selected_index = int(selected["candidate_index"])
        row = open3d_baseline.evaluate(
            case,
            source_method,
            METHOD,
            source,
            float(selected["scale"]),
            np.asarray(selected["rotation"], dtype=np.float64),
            np.asarray(selected["translation"], dtype=np.float64),
            masks,
            target,
            target_anchor,
            runtime,
            voxel_mm,
        )
    row.update(
        {
            "coarse_candidate_count": 8,
            "selected_candidate_index": selected_index,
            "anchor_cap_mm": anchor_cap_mm,
            "anchor_eligible_candidates": eligible_count,
            "anchor_cap_fallback": int(cap_fallback),
            "uses_common_target_face_roi": 1,
            "uses_anatomical_interocular_scale": 1,
            "uses_shared_surface_and_anchor_initializations": 1,
            "uses_shared_orientation_and_anchor_candidate_policy": 1,
            "anatomical_scale_residual_mm": (
                float(
                    abs(
                        float(row.get("scale_source_to_mm", fixed_scale))
                        * source_interocular
                        - target_interocular_mm
                    )
                )
                if int(row["completed"])
                else None
            ),
            "acceptance_threshold_applied": 0,
        }
    )
    serializable_states = []
    for state in states:
        serializable = {
            key: value
            for key, value in state.items()
            if key not in {"rotation", "translation", "metrics", "orientation"}
        }
        if not state["error"]:
            serializable["rotation"] = np.asarray(state["rotation"]).tolist()
            serializable["translation_mm"] = np.asarray(state["translation"]).tolist()
            serializable["metrics"] = state["metrics"]
            serializable["orientation"] = state["orientation"]
        serializable_states.append(serializable)
    detail_path.write_text(
        json.dumps(
            {
                "row": row,
                "candidates": serializable_states,
                "selection_policy": (
                    "same upright and semantic-anchor-cap candidate policy as the "
                    "revised proposed rigid stage; standard Open3D point-to-point "
                    "ICP supplies each candidate"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    row_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def failed_case_row(
    case: str,
    source_method: str,
    anchor_cap_mm: float,
    error: Exception,
) -> dict[str, Any]:
    row = open3d_baseline.evaluate(
        case,
        source_method,
        METHOD,
        np.empty((0, 3), dtype=np.float64),
        1.0,
        np.eye(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        {},
        np.empty((0, 3), dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        0.0,
        0.0,
        error=f"{type(error).__name__}: {error}",
    )
    row.update(
        {
            "coarse_candidate_count": 8,
            "selected_candidate_index": None,
            "anchor_cap_mm": anchor_cap_mm,
            "anchor_eligible_candidates": 0,
            "anchor_cap_fallback": 1,
            "uses_common_target_face_roi": 1,
            "uses_anatomical_interocular_scale": 1,
            "uses_shared_surface_and_anchor_initializations": 1,
            "uses_shared_orientation_and_anchor_candidate_policy": 1,
            "anatomical_scale_residual_mm": None,
            "acceptance_threshold_applied": 0,
        }
    )
    return row


def write_summary(output: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    ordered = sorted(rows, key=lambda row: str(row["case"]))
    if not ordered:
        return
    fields = sorted({key for row in ordered for key in row})
    with (output / "open3d_shared_cues_icp_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)
    summary = {
        **config,
        "attempted_cases": len(ordered),
        "completed_cases": sum(int(row["completed"]) for row in ordered),
        "successful_executions": sum(int(row["completed"]) for row in ordered),
        "execution_failures": sum(not int(row["completed"]) for row in ordered),
        "anchor_cap_fallback_cases": sum(
            int(row["anchor_cap_fallback"]) for row in ordered
        ),
    }
    (output / "open3d_shared_cues_icp_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.voxel_divisor <= 0.0 or args.voxel_floor_mm <= 0.0:
        raise ValueError("Invalid Open3D voxel parameters")
    if not np.isfinite(args.anchor_cap_mm) or args.anchor_cap_mm <= 0.0:
        raise ValueError("--anchor-cap-mm must be positive and finite")
    root = args.root.expanduser().resolve(strict=True)
    revision = registration.checked_existing(
        root / "cmes_revision_20260816", root, "revision root"
    )
    scale = registration.checked_existing(args.scale_dict, revision, "scale dictionary")
    split = registration.checked_existing(args.split, revision, "identity split")
    anchor = registration.checked_existing(
        args.target_anchor_json, revision, "target-anchor JSON"
    )
    manifest = registration.checked_existing(
        root / "prepared_cohort" / "facescape_frontal_pairs_manifest.csv",
        root,
        "manifest",
    )
    source_root = (
        registration.checked_existing(args.source_root, revision, "source root")
        if args.source_root
        else None
    )
    topology = (
        registration.checked_existing(args.topology_file, revision, "topology file")
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
    cases, _ = registration.read_selected_cases(
        manifest, split, args.subset, args.case
    )
    config = {
        "baseline_method": METHOD,
        "source_method": args.source_method,
        "subset": args.subset,
        "case_count": len(cases),
        "anchor_cap_mm": args.anchor_cap_mm,
        "coarse_candidate_count": 8,
        "open3d_version": open3d_baseline.o3d.__version__,
        "voxel_divisor": args.voxel_divisor,
        "voxel_floor_mm": args.voxel_floor_mm,
        "common_target_face_roi": True,
        "anatomical_interocular_scale": True,
        "shared_anatomical_initializations": True,
        "acceptance_threshold_applied": False,
        "split_sha256": sha256(split),
        "target_anchor_sha256": sha256(anchor),
    }
    if args.dry_run:
        print(json.dumps({**config, "first_case": cases[0], "last_case": cases[-1]}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=args.resume)
    (output / "case_rows").mkdir(exist_ok=True)
    (output / "case_details").mkdir(exist_ok=True)

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
            args.anchor_cap_mm,
            args.voxel_divisor,
            args.voxel_floor_mm,
        )

    rows = []
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {pool.submit(process_case, *arguments(case)): case for case in cases}
        for future in as_completed(futures):
            case = futures[future]
            try:
                row = future.result()
            except Exception as error:
                row = failed_case_row(
                    case, args.source_method, args.anchor_cap_mm, error
                )
                (output / "case_rows" / f"{case}.json").write_text(
                    json.dumps(row, indent=2), encoding="utf-8"
                )
                (output / "case_details" / f"{case}.json").write_text(
                    json.dumps(
                        {
                            "row": row,
                            "candidates": [],
                            "case_execution_failed": True,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            rows.append(row)
            write_summary(output, rows, config)
            print(
                json.dumps(
                    {"completed": row["case"], "count": len(rows), "total": len(cases)}
                ),
                flush=True,
            )
    write_summary(output, rows, config)
    if len(rows) != len(cases):
        raise RuntimeError(
            f"Shared-cues denominator mismatch: {len(rows)} of {len(cases)}"
        )


if __name__ == "__main__":
    main()
