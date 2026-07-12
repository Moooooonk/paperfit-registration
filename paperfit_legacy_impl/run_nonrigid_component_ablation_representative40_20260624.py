#!/usr/bin/env python3
import csv
import os
import json
import os
import sys
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import os


ROOT = Path(os.environ.get("PAPERFIT_ROOT", "/path/to/prepared/facescape_pipeline"))
TOOLS = ROOT / "research_tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_nonrigid_component_ablation_goodrigid_6case_20260611 as comp  # noqa: E402
import os
import run_nonrigid_nasal_depth_ablation_3case as nonrigid  # noqa: E402
import os


RIGID_ROOT = ROOT / "research_rigid_upright_hardgate_allpairs_mergedpass_20260622"
RIGID_AGG = RIGID_ROOT / "allpairs_mergedpass_aggregate.json"
OUT = ROOT / "research_nonrigid_component_ablation_representative40_20260624"
THRESHOLDS = [0.00, 0.22, 0.40, 0.55, 0.68, 0.78, 0.87, 0.94]


CONDITIONS = {
    "proposed_S8": {
        "description": "Eye/orbit exclusion, nasal depth-contour staging, and nasal/midface weights.",
        "disable_eye_exclusion": False,
        "disable_contours": False,
        "disable_region_weights": False,
    },
    "no_eye_exclusion": {
        "description": "Allows eyes-open source eye/orbit vertices to participate in fitting.",
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


def select_cases(max_per_subject=2):
    aggregate = json.loads(RIGID_AGG.read_text(encoding="utf-8"))
    pass_cases = sorted(aggregate["pass_cases"])
    by_subject = defaultdict(list)
    for case in pass_cases:
        by_subject[case[:3]].append(case)
    selected = []
    for subject in sorted(by_subject):
        subject_cases = by_subject[subject]
        neutral = [c for c in subject_cases if c.endswith("_1_neutral")]
        dimpler = [c for c in subject_cases if c.endswith("_10_dimpler")]
        priority = neutral[:1] + dimpler[:1]
        for case in subject_cases:
            if len(priority) >= max_per_subject:
                break
            if case not in priority:
                priority.append(case)
        selected.extend(priority[:max_per_subject])
    return selected[:40]


def target_sample_for_case(case, manifest):
    target_id = f"{int(case[:3]):03d}_18_eye_closed"
    target_mesh = Path(manifest[target_id]["mesh"])
    _, target_world, _ = nonrigid.load_mesh(target_mesh)
    target_reg = nonrigid.target_registration_frame(target_world, target_mesh.parent / "selected_camera.json")
    return target_mesh, nonrigid.deterministic_sample(target_reg, 110000)


def process_case(case, manifest):
    target_mesh, target_sample = target_sample_for_case(case, manifest)
    src_obj = RIGID_ROOT / case / f"{case}_similarity_icp_regframe.obj"
    init_v, _, faces, _, _, _ = nonrigid.parse_obj_with_uv(src_obj)
    _, eval_masks, ellipses = nonrigid.build_masks(src_obj, init_v)
    baseline_metrics = nonrigid.all_metrics(init_v, target_sample, eval_masks)
    rows = []
    for condition_name, condition in CONDITIONS.items():
        fit_masks = comp.mutate_masks_for_condition(eval_masks, condition)
        local_v, final_v, history, _, eye_report = comp.run_nonrigid_variant(
            init_v,
            faces,
            target_sample,
            fit_masks,
            THRESHOLDS,
            condition,
        )
        local_metrics = nonrigid.all_metrics(local_v, target_sample, eval_masks)
        final_metrics = nonrigid.all_metrics(final_v, target_sample, eval_masks)
        disp = np.linalg.norm(final_v - init_v, axis=1)
        row = {
            "case": case,
            "subject": case[:3],
            "condition": condition_name,
            "description": condition["description"],
            "source_rigid_obj": str(src_obj),
            "target_mesh": str(target_mesh),
            "fit_vertices": int(fit_masks["full_no_eye"].sum()),
            "eval_full_no_eye_vertices": int(eval_masks["full_no_eye"].sum()),
            "eval_eye_soft_vertices": int(eval_masks["eye_soft"].sum()),
            "fixed_eye_vertices": eye_report["fixed_eye_vertices"],
            "eye_fixed_max": eye_report["eye_fixed_max"],
            "eye_fixed_mean": eye_report["eye_fixed_mean"],
            "displacement_mean": float(np.mean(disp)),
            "displacement_p90": float(np.quantile(disp, 0.90)),
            "displacement_max": float(np.max(disp)),
            "full_median_before": baseline_metrics["full_no_eye"]["median"],
            "full_median_local": local_metrics["full_no_eye"]["median"],
            "full_median_after": final_metrics["full_no_eye"]["median"],
            "full_p90_before": baseline_metrics["full_no_eye"]["p90"],
            "full_p90_after": final_metrics["full_no_eye"]["p90"],
            "nose_median_before": baseline_metrics["nose"]["median"],
            "nose_median_local": local_metrics["nose"]["median"],
            "nose_median_after": final_metrics["nose"]["median"],
            "nose_p90_before": baseline_metrics["nose"]["p90"],
            "nose_p90_after": final_metrics["nose"]["p90"],
            "midface_median_before": baseline_metrics["midface"]["median"],
            "midface_median_after": final_metrics["midface"]["median"],
            "history_passes": len(history),
            "eye_ellipse_count": len(ellipses),
        }
        rows.append(row)
        print(
            f"{case} {condition_name} full {row['full_median_before']:.5f}->{row['full_median_after']:.5f} "
            f"nose {row['nose_median_before']:.5f}->{row['nose_median_after']:.5f} "
            f"eye={row['eye_fixed_max']:.6f}",
            flush=True,
        )
    return rows


def summarize(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with (OUT / "representative40_component_ablation_rows.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {}
    for condition_name, condition in CONDITIONS.items():
        subset = [r for r in rows if r["condition"] == condition_name]
        aggregate[condition_name] = {
            "description": condition["description"],
            "cases": len(subset),
            "mean_full_median_before": float(np.mean([r["full_median_before"] for r in subset])),
            "mean_full_median_after": float(np.mean([r["full_median_after"] for r in subset])),
            "mean_full_reduction_pct": float(
                100.0
                * (np.mean([r["full_median_before"] for r in subset]) - np.mean([r["full_median_after"] for r in subset]))
                / max(np.mean([r["full_median_before"] for r in subset]), 1e-12)
            ),
            "mean_nose_median_before": float(np.mean([r["nose_median_before"] for r in subset])),
            "mean_nose_median_after": float(np.mean([r["nose_median_after"] for r in subset])),
            "mean_nose_reduction_pct": float(
                100.0
                * (np.mean([r["nose_median_before"] for r in subset]) - np.mean([r["nose_median_after"] for r in subset]))
                / max(np.mean([r["nose_median_before"] for r in subset]), 1e-12)
            ),
            "mean_nose_p90_after": float(np.mean([r["nose_p90_after"] for r in subset])),
            "mean_displacement_p90": float(np.mean([r["displacement_p90"] for r in subset])),
            "max_eye_fixed": float(np.max([r["eye_fixed_max"] for r in subset])),
            "full_improved_cases": int(sum(r["full_median_after"] < r["full_median_before"] for r in subset)),
            "nose_improved_cases": int(sum(r["nose_median_after"] < r["nose_median_before"] for r in subset)),
        }
    payload = {
        "cases": sorted({r["case"] for r in rows}),
        "case_count": len(set(r["case"] for r in rows)),
        "rigid_root": str(RIGID_ROOT),
        "output_root": str(OUT),
        "thresholds": THRESHOLDS,
        "conditions": CONDITIONS,
        "aggregate": aggregate,
        "purpose": "Representative 40-case metrics-only component ablation for manuscript robustness review.",
    }
    (OUT / "representative40_component_ablation_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = nonrigid.read_manifest()
    cases = select_cases(max_per_subject=2)
    all_rows = []
    for case in cases:
        all_rows.extend(process_case(case, manifest))
    summarize(all_rows)


if __name__ == "__main__":
    main()


