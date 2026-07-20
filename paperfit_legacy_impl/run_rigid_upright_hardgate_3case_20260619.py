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

import run_scratch_surface_registration_3case_final_attempt as rigid  # noqa: E402
import run_rigid_global_upright_failed11_20260617 as global_rigid  # noqa: E402


MANIFEST = ROOT / "prepared_001_020_cv" / "facescape_frontal_pairs_manifest.csv"
OUT = ROOT / "research_rigid_upright_hardgate_3case_20260619"
CASES = ["004_1_neutral", "011_10_dimpler", "012_10_dimpler"]


def read_manifest():
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return {r["pair_id"]: r for r in csv.DictReader(f)}


def orientation_metrics(aligned, region_masks, eye_mask):
    nose_mask = (
        region_masks["nose_bridge"]
        | region_masks["nose_dorsum"]
        | region_masks["nose_tip"]
        | region_masks["alar"]
    )
    mouth_mask = region_masks["mouth_downweighted"]
    eyes = np.median(aligned[eye_mask], axis=0)
    nose = np.median(aligned[nose_mask], axis=0)
    mouth = np.median(aligned[mouth_mask], axis=0)
    ext = np.maximum(np.quantile(aligned, 0.95, axis=0) - np.quantile(aligned, 0.05, axis=0), 1e-8)
    height = float(ext[2])
    eye_over_mouth = float(eyes[2] - mouth[2])
    nose_over_mouth = float(nose[2] - mouth[2])
    vertical = eyes - mouth
    vertical_z = float(vertical[2] / max(float(np.linalg.norm(vertical)), 1e-12))
    eye_over_mouth_norm = eye_over_mouth / height
    nose_over_mouth_norm = nose_over_mouth / height
    upside_down = (
        eye_over_mouth_norm < 0.10
        or nose_over_mouth_norm < 0.02
        or vertical_z < 0.35
    )
    upright_penalty = 0.0
    upright_penalty += max(0.0, 0.28 - eye_over_mouth_norm)
    upright_penalty += max(0.0, 0.08 - nose_over_mouth_norm)
    upright_penalty += max(0.0, 0.55 - vertical_z)
    return {
        "eye_over_mouth": eye_over_mouth,
        "nose_over_mouth": nose_over_mouth,
        "eye_over_mouth_norm": float(eye_over_mouth_norm),
        "nose_over_mouth_norm": float(nose_over_mouth_norm),
        "vertical_z": vertical_z,
        "upside_down": int(upside_down),
        "upright_penalty": float(upright_penalty),
    }


def combined_score(metric, orient):
    hard = 10.0 if orient["upside_down"] else 0.0
    return float(metric["selection_score"] + hard + 3.5 * orient["upright_penalty"])


def qc_row(report):
    m = report["similarity_metrics"]
    o = report["similarity_orientation_metrics"]
    med = float(m["median"])
    p90 = float(m["p90"])
    nose = float(m.get("nose_weighted_median", med))
    anchor = float(m.get("source_nose_anchor_distance", 999.0))
    qc_pass = (
        not bool(o["upside_down"])
        and med <= 0.023
        and p90 <= 0.070
        and nose <= 0.040
        and anchor <= 0.120
    )
    return {
        "case": report["case"],
        "qc_pass": int(qc_pass),
        "upside_down": int(o["upside_down"]),
        "eye_over_mouth_norm": float(o["eye_over_mouth_norm"]),
        "nose_over_mouth_norm": float(o["nose_over_mouth_norm"]),
        "vertical_z": float(o["vertical_z"]),
        "similarity_median": med,
        "similarity_p90": p90,
        "nose_weighted_median": nose,
        "source_nose_anchor_distance": anchor,
        "rigid_scale": float(report["rigid_scale"]),
        "final_similarity_scale": float(report["final_similarity_scale"]),
        "diagnostic_png": report["diagnostic_png"],
        "report_json": str(OUT / report["case"] / f"{report['case']}_scratch_surface_registration_report.json"),
    }


def process(case, by_pair):
    print(f"{case} start", flush=True)
    row = by_pair[case]
    target_row = by_pair["%03d_18_eye_closed" % int(row["subject"])]
    src_obj = rigid.hrn_obj(case)
    _, src_v, _ = rigid.load_mesh(src_obj)
    target_mesh = Path(target_row["mesh"])
    _, target_world, _ = rigid.load_mesh(target_mesh)
    cam_json = target_mesh.parent / "selected_camera.json"
    target_reg = rigid.target_registration_frame(target_world, cam_json)
    nose_anchor = rigid.target_nose_anchor(target_reg)
    fit_weight_base, eye_mask, eye_ellipses = rigid.texture_eye_weight(src_obj)
    fit_weight, region_masks = rigid.apply_nose_midface_weight(src_obj, src_v, fit_weight_base)

    best = None
    tried = []
    initial = global_rigid.global_initial_candidates(
        src_v, target_reg, fit_weight, region_masks, eye_mask, nose_anchor, keep=16
    )
    for ci, cand in enumerate(initial, start=1):
        scale, rot, trans, hist = rigid.refine_rigid_fixed_scale(
            src_v,
            target_reg,
            fit_weight,
            cand["scale"],
            cand["rot"],
            cand["trans"],
            iterations=30,
            region_masks=region_masks,
            nose_anchor=nose_anchor,
        )
        scale, rot, trans = rigid.translation_only_polish(src_v, target_reg, fit_weight, scale, rot, trans)
        aligned = rigid.transform(src_v, scale, rot, trans)
        metric = rigid.metrics(aligned, target_reg, fit_weight, region_masks=region_masks, nose_anchor=nose_anchor)
        orient = orientation_metrics(aligned, region_masks, eye_mask)
        score = combined_score(metric, orient)
        tried.append({"candidate": ci, "score": score, "scale": float(scale), **metric, **orient})
        print(
            f"{case} cand={ci}/{len(initial)} score={score:.6f} med={metric['median']:.6f} "
            f"p90={metric['p90']:.6f} anchor={metric['source_nose_anchor_distance']:.6f} "
            f"up={orient['upside_down']} eye_norm={orient['eye_over_mouth_norm']:.3f} "
            f"vertical_z={orient['vertical_z']:.3f} scale={scale:.6f}",
            flush=True,
        )
        if best is None or score < best[0]:
            best = (score, scale, rot, trans, aligned, metric, orient, hist)

    _, rigid_scale, rigid_rot, rigid_trans, rigid_aligned, rigid_m, rigid_o, rigid_hist = best

    scale_only_scale, scale_only_rot, scale_only_trans, scale_only_hist = rigid.adaptive_scale_prefit_only(
        src_v, target_reg, fit_weight, rigid_scale, rigid_rot, rigid_trans,
        region_masks=region_masks, nose_anchor=nose_anchor
    )
    scale_only_aligned = rigid.transform(src_v, scale_only_scale, scale_only_rot, scale_only_trans)
    scale_only_m = rigid.metrics(scale_only_aligned, target_reg, fit_weight, region_masks=region_masks, nose_anchor=nose_anchor)
    scale_only_o = orientation_metrics(scale_only_aligned, region_masks, eye_mask)

    sim_scale, sim_rot, sim_trans, sim_hist = rigid.scale_first_then_rigid_icp(
        src_v, target_reg, fit_weight, rigid_scale, rigid_rot, rigid_trans,
        region_masks=region_masks, nose_anchor=nose_anchor
    )
    sim_aligned = rigid.transform(src_v, sim_scale, sim_rot, sim_trans)
    final_scale, final_rot, final_trans, final_pose_hist = rigid.final_small_pose_search(
        src_v, target_reg, fit_weight, sim_scale, sim_rot, sim_trans,
        region_masks=region_masks, nose_anchor=nose_anchor
    )
    sim_aligned = rigid.transform(src_v, final_scale, final_rot, final_trans)
    sim_m = rigid.metrics(sim_aligned, target_reg, fit_weight, region_masks=region_masks, nose_anchor=nose_anchor)
    sim_o = orientation_metrics(sim_aligned, region_masks, eye_mask)

    stages = [
        ("rigid", rigid_scale, rigid_rot, rigid_trans, rigid_aligned, rigid_m, rigid_o),
        ("scale_only", scale_only_scale, scale_only_rot, scale_only_trans, scale_only_aligned, scale_only_m, scale_only_o),
        ("similarity", final_scale, final_rot, final_trans, sim_aligned, sim_m, sim_o),
    ]
    scored = sorted(
        [(combined_score(m, o), name, scale, rot, trans, aligned, m, o) for name, scale, rot, trans, aligned, m, o in stages],
        key=lambda item: item[0],
    )
    _, selected_stage, final_scale, final_rot, final_trans, final_aligned, final_m, final_o = scored[0]

    case_dir = OUT / case
    case_dir.mkdir(parents=True, exist_ok=True)
    rigid.write_obj_like(src_obj, case_dir / f"{case}_rigid_icp_regframe.obj", rigid_aligned)
    rigid.write_obj_like(src_obj, case_dir / f"{case}_adaptive_scale_only_regframe.obj", scale_only_aligned)
    sim_reg = case_dir / f"{case}_similarity_icp_regframe.obj"
    rigid.write_obj_like(src_obj, sim_reg, final_aligned)
    fig = rigid.plot_review(
        case_dir,
        case,
        src_v,
        target_reg,
        rigid_aligned,
        scale_only_aligned,
        final_aligned,
        fit_weight,
        eye_mask,
        region_masks,
        rigid_m,
        scale_only_m,
        final_m,
    )
    report = {
        "case": case,
        "method": "upright hard-gated rerun for upside-down rigid failures",
        "selected_stage": selected_stage,
        "source_obj": str(src_obj),
        "target_mesh": str(target_mesh),
        "similarity_regframe_obj": str(sim_reg),
        "diagnostic_png": str(fig),
        "rigid_scale": float(rigid_scale),
        "adaptive_scale_only_scale": float(scale_only_scale),
        "similarity_scale": float(sim_scale),
        "final_similarity_scale": float(final_scale),
        "tried_initializations": tried,
        "stage_scores": [
            {"stage": name, "score": float(score), **m, **o}
            for score, name, _, _, _, _, m, o in scored
        ],
        "rigid_metrics": rigid_m,
        "rigid_orientation_metrics": rigid_o,
        "adaptive_scale_only_metrics": scale_only_m,
        "adaptive_scale_only_orientation_metrics": scale_only_o,
        "similarity_metrics": final_m,
        "similarity_orientation_metrics": final_o,
        "scale_only_history": scale_only_hist,
        "similarity_history": sim_hist,
        "final_small_pose_history": final_pose_hist,
    }
    (case_dir / f"{case}_scratch_surface_registration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"{case} selected={selected_stage} med={final_m['median']:.6f}/{final_m['p90']:.6f} "
        f"anchor={final_m['source_nose_anchor_distance']:.6f} up={final_o['upside_down']} "
        f"eye_norm={final_o['eye_over_mouth_norm']:.3f} fig={fig}",
        flush=True,
    )
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    by_pair = read_manifest()
    reports = [process(case, by_pair) for case in CASES]
    rows = [qc_row(report) for report in reports]
    with (OUT / "upright_hardgate_3case_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {
        "cases": CASES,
        "qc_pass": int(sum(row["qc_pass"] for row in rows)),
        "upside_down_remaining": [row["case"] for row in rows if row["upside_down"]],
        "pass_cases": [row["case"] for row in rows if row["qc_pass"]],
        "fail_cases": [row["case"] for row in rows if not row["qc_pass"]],
    }
    (OUT / "upright_hardgate_3case_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    (OUT / "upright_hardgate_3case_aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(OUT), **aggregate}, indent=2), flush=True)


if __name__ == "__main__":
    main()


