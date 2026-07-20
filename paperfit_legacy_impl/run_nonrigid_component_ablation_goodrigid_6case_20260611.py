#!/usr/bin/env python3
import csv
import os
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("PAPERFIT_ROOT", "/path/to/prepared/facescape_pipeline"))
TOOLS = ROOT / "research_tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_nonrigid_nasal_depth_ablation_3case as nonrigid  # noqa: E402


RIGID_ROOT = ROOT / "research_scratch_surface_registration_10case_final_attempt_20260603"
OUT = ROOT / "research_nonrigid_component_ablation_goodrigid_6case_20260611"

CASES = [
    "001_1_neutral",
    "001_10_dimpler",
    "002_1_neutral",
    "002_10_dimpler",
    "003_1_neutral",
    "003_10_dimpler",
]

BEST_THRESHOLDS = [0.00, 0.22, 0.40, 0.55, 0.68, 0.78, 0.87, 0.94]

CONDITIONS = {
    "proposed_S8": {
        "description": "Eye/orbit exclusion, nasal depth-contour staging, and nasal/midface weights.",
        "disable_eye_exclusion": False,
        "disable_contours": False,
        "disable_region_weights": False,
    },
    "no_eye_exclusion": {
        "description": "Uses the same source-target fitting but allows the eyes-open source eye/orbit vertices to participate.",
        "disable_eye_exclusion": True,
        "disable_contours": False,
        "disable_region_weights": False,
    },
    "no_nasal_depth_contours": {
        "description": "Keeps eye/orbit exclusion and local anatomical regions, but removes depth-ordered nasal contour staging.",
        "disable_eye_exclusion": False,
        "disable_contours": True,
        "disable_region_weights": False,
    },
    "no_region_weights": {
        "description": "Keeps eye/orbit exclusion and depth contours, but removes additional nasal/midface correspondence weights.",
        "disable_eye_exclusion": False,
        "disable_contours": False,
        "disable_region_weights": True,
    },
}


def copy_masks(masks):
    return {k: v.copy() for k, v in masks.items()}


def mutate_masks_for_condition(base_masks, condition):
    masks = copy_masks(base_masks)
    if condition["disable_eye_exclusion"]:
        eye_area = masks["eye_soft"].copy()
        masks["eye"][:] = False
        masks["eye_soft"][:] = False
        masks["full_no_eye"] = masks["full_no_eye"] | eye_area
    return masks


def run_nonrigid_variant(init_vertices, faces, target_sample, masks, thresholds, condition):
    current = init_vertices.copy()
    start = init_vertices.copy()
    edges = nonrigid.build_edges(faces)
    tree = nonrigid.cKDTree(target_sample)
    fixed_idx = np.flatnonzero(masks["eye_soft"])
    if len(fixed_idx) < 50:
        fixed_idx = np.flatnonzero(masks["eye"])

    history = []
    protrusion = np.zeros(len(init_vertices), dtype=np.float64)
    local_sets = []
    if not condition["disable_contours"]:
        contour_sets, protrusion = nonrigid.depth_contours(current, masks, thresholds)
        local_sets.extend(contour_sets)

    local_sets.extend([
        ("bridge", masks["nasal_bridge"]),
        ("dorsum", masks["nasal_dorsum"]),
        ("tip_alar", masks["nose_tip"] | masks["alar"]),
        ("subnasal", masks["subnasal"]),
        ("philtrum", masks["philtrum"]),
    ])

    for pass_i, (name, active_mask) in enumerate(local_sets, 1):
        target_idx = np.flatnonzero(active_mask & masks["full_no_eye"] & ~masks["eye_soft"])
        if len(target_idx) < 20:
            continue
        d, nn = tree.query(current[target_idx], k=1, workers=-1)
        delta = target_sample[nn] - current[target_idx]
        norm = np.linalg.norm(delta, axis=1)
        max_step = 0.018 if not isinstance(name, str) else 0.014
        step = np.minimum(1.0, max_step / np.maximum(norm, 1e-8))
        target = current[target_idx] + delta * step[:, None]
        weights = 18.0 + np.clip(d, 0.0, 0.20) * 85.0
        if not condition["disable_region_weights"]:
            if name == "tip_alar" or (not isinstance(name, str) and float(name) >= 0.75):
                weights *= 1.25
        solved = nonrigid.solve(
            current,
            edges,
            target_idx,
            target,
            weights,
            fixed_idx,
            start[fixed_idx],
            edge_w=190.0,
            fixed_w=32000.0,
        )
        current = 0.43 * solved + 0.57 * current
        current[fixed_idx] = start[fixed_idx]
        history.append({
            "phase": "nasal_depth" if not isinstance(name, str) else "anatomical_local",
            "pass": pass_i,
            "name": str(name),
            "active_vertices": int(len(target_idx)),
            "median_nn": float(np.median(d)),
            "p90_nn": float(np.quantile(d, 0.90)),
        })

    local_refined = current.copy()
    target_idx = np.flatnonzero(masks["full_no_eye"] & ~masks["eye_soft"])
    schedule = [
        (210.0, 9.0, 0.12, 0.018),
        (185.0, 11.0, 0.14, 0.016),
        (165.0, 12.0, 0.13, 0.014),
    ]
    for pass_i, (edge_w, base_w, gain, max_step) in enumerate(schedule, 1):
        d, nn = tree.query(current[target_idx], k=1, workers=-1)
        delta = target_sample[nn] - current[target_idx]
        norm = np.linalg.norm(delta, axis=1)
        step = np.minimum(1.0, max_step / np.maximum(norm, 1e-8))
        target = current[target_idx] + delta * step[:, None]
        weights = np.full(len(target_idx), base_w, dtype=np.float64) + np.clip(d, 0.0, 0.25) * 65.0
        if not condition["disable_region_weights"]:
            for key, factor in [
                ("nasal_bridge", 1.30),
                ("nasal_dorsum", 1.35),
                ("nose_tip", 1.42),
                ("alar", 1.35),
                ("subnasal", 1.18),
                ("philtrum", 1.12),
            ]:
                weights[masks[key][target_idx]] *= factor
        solved = nonrigid.solve(
            current,
            edges,
            target_idx,
            target,
            weights,
            fixed_idx,
            start[fixed_idx],
            edge_w=edge_w,
            fixed_w=32000.0,
        )
        current = (1.0 - gain) * current + gain * solved
        current[fixed_idx] = start[fixed_idx]
        history.append({
            "phase": "full_no_eye",
            "pass": pass_i,
            "active_vertices": int(len(target_idx)),
            "median_nn": float(np.median(d)),
            "p90_nn": float(np.quantile(d, 0.90)),
        })

    eye_disp = np.linalg.norm(current[fixed_idx] - start[fixed_idx], axis=1) if len(fixed_idx) else np.zeros(0)
    return local_refined, current, history, protrusion, {
        "fixed_eye_vertices": int(len(fixed_idx)),
        "eye_fixed_max": float(eye_disp.max()) if len(eye_disp) else 0.0,
        "eye_fixed_mean": float(eye_disp.mean()) if len(eye_disp) else 0.0,
    }


def process_case(case, manifest):
    target_id = f"{int(case[:3]):03d}_18_eye_closed"
    target_mesh = Path(manifest[target_id]["mesh"])
    _, target_world, _ = nonrigid.load_mesh(target_mesh)
    target_reg = nonrigid.target_registration_frame(target_world, target_mesh.parent / "selected_camera.json")
    target_sample = nonrigid.deterministic_sample(target_reg, 140000)

    src_obj = RIGID_ROOT / case / f"{case}_similarity_icp_regframe.obj"
    init_v, _, faces, _, _, _ = nonrigid.parse_obj_with_uv(src_obj)
    fit_weight, eval_masks, ellipses = nonrigid.build_masks(src_obj, init_v)

    case_dir = OUT / case
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / f"{case}_eye_soft_mask.npy", eval_masks["eye_soft"])
    np.save(case_dir / f"{case}_nose_mask.npy", eval_masks["nose"])

    baseline_metrics = nonrigid.all_metrics(init_v, target_sample, eval_masks)
    reports = []
    for condition_name, condition in CONDITIONS.items():
        fit_masks = mutate_masks_for_condition(eval_masks, condition)
        local_v, final_v, history, protrusion, eye_report = run_nonrigid_variant(
            init_v,
            faces,
            target_sample,
            fit_masks,
            BEST_THRESHOLDS,
            condition,
        )
        schedule_dir = case_dir / condition_name
        local_obj = schedule_dir / f"{case}_{condition_name}_nasal_local_regframe.obj"
        final_obj = schedule_dir / f"{case}_{condition_name}_nonrigid_final_regframe.obj"
        nonrigid.write_obj_like(src_obj, local_obj, local_v)
        nonrigid.write_obj_like(src_obj, final_obj, final_v)
        local_metrics = nonrigid.all_metrics(local_v, target_sample, eval_masks)
        final_metrics = nonrigid.all_metrics(final_v, target_sample, eval_masks)
        disp = np.linalg.norm(final_v - init_v, axis=1)
        report = {
            "case": case,
            "condition": condition_name,
            "description": condition["description"],
            "thresholds": BEST_THRESHOLDS,
            "source_rigid_obj": str(src_obj),
            "target_mesh": str(target_mesh),
            "target_policy": "full eye-closed target mesh converted to registration frame, no target ROI crop",
            "eval_policy": "All metrics use the original proposed source masks for fair cross-condition comparison.",
            "nasal_local_obj": str(local_obj),
            "nonrigid_final_obj": str(final_obj),
            "fit_vertices": int(fit_masks["full_no_eye"].sum()),
            "mask_vertices_for_fit": {k: int(v.sum()) for k, v in fit_masks.items() if v.dtype == bool},
            "mask_vertices_for_eval": {k: int(v.sum()) for k, v in eval_masks.items() if v.dtype == bool},
            "eye_ellipses": ellipses,
            **eye_report,
            "displacement_mean": float(np.mean(disp)),
            "displacement_p90": float(np.quantile(disp, 0.90)),
            "displacement_max": float(np.max(disp)),
            "baseline_metrics": baseline_metrics,
            "nasal_local_metrics": local_metrics,
            "nonrigid_final_metrics": final_metrics,
            "history": history,
        }
        review = nonrigid.make_review(schedule_dir, case, condition_name, src_obj, target_reg, init_v, local_v, final_v)
        report["review_png"] = str(review)
        (schedule_dir / f"{case}_{condition_name}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        reports.append(report)
        b = baseline_metrics["nose"]
        f = final_metrics["nose"]
        print(
            f"{case} {condition_name} nose median {b['median']:.6f}->{f['median']:.6f} "
            f"p90 {b['p90']:.6f}->{f['p90']:.6f} eye_max={eye_report['eye_fixed_max']:.8f}",
            flush=True,
        )
    return reports


def summarize(reports):
    rows = []
    for report in reports:
        base = report["baseline_metrics"]
        final = report["nonrigid_final_metrics"]
        local = report["nasal_local_metrics"]
        rows.append({
            "case": report["case"],
            "condition": report["condition"],
            "full_median_before": base["full_no_eye"]["median"],
            "full_median_after": final["full_no_eye"]["median"],
            "midface_median_before": base["midface"]["median"],
            "midface_median_after": final["midface"]["median"],
            "nose_median_before": base["nose"]["median"],
            "nose_median_local": local["nose"]["median"],
            "nose_median_after": final["nose"]["median"],
            "nose_p90_before": base["nose"]["p90"],
            "nose_p90_after": final["nose"]["p90"],
            "nose_tip_median_before": base["nose_tip"]["median"],
            "nose_tip_median_after": final["nose_tip"]["median"],
            "eye_fixed_max": report["eye_fixed_max"],
            "fit_vertices": report["fit_vertices"],
            "displacement_p90": report["displacement_p90"],
            "review_png": report["review_png"],
        })

    fields = list(rows[0].keys())
    with (OUT / "component_ablation_compact_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {}
    for condition in CONDITIONS:
        subset = [r for r in rows if r["condition"] == condition]
        aggregate[condition] = {
            "cases": len(subset),
            "description": CONDITIONS[condition]["description"],
            "mean_nose_median_after": float(np.mean([r["nose_median_after"] for r in subset])),
            "mean_nose_p90_after": float(np.mean([r["nose_p90_after"] for r in subset])),
            "mean_full_median_after": float(np.mean([r["full_median_after"] for r in subset])),
            "mean_displacement_p90": float(np.mean([r["displacement_p90"] for r in subset])),
        }
    (OUT / "component_ablation_aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({"aggregate": aggregate, "reports": reports}, indent=2), encoding="utf-8")
    return rows, aggregate


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = nonrigid.read_manifest()
    nonrigid.RIGID_ROOT = RIGID_ROOT
    nonrigid.OUT = OUT
    reports = []
    for case in CASES:
        reports.extend(process_case(case, manifest))
    rows, aggregate = summarize(reports)
    summary = {
        "cases": CASES,
        "rigid_root": str(RIGID_ROOT),
        "output_root": str(OUT),
        "thresholds": BEST_THRESHOLDS,
        "conditions": CONDITIONS,
        "aggregate": aggregate,
        "purpose": "IEEE Access component ablation: isolate eye/orbit exclusion, nasal depth-contour staging, and nasal/midface weighting.",
    }
    (OUT / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()


