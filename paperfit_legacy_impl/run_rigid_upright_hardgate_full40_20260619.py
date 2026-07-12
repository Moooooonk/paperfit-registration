#!/usr/bin/env python3
import csv
import os
import json
import os
import sys
import os
from pathlib import Path

import numpy as np
import os


ROOT = Path(os.environ.get("PAPERFIT_ROOT", "/path/to/prepared/facescape_pipeline"))
TOOLS = ROOT / "research_tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_scratch_surface_registration_3case_final_attempt as rigid  # noqa: E402
import os
import run_rigid_global_upright_failed11_20260617 as global_rigid  # noqa: E402
import os
import run_rigid_upright_hardgate_3case_20260619 as hard3  # noqa: E402
import os


MANIFEST = ROOT / "prepared_001_020_cv" / "facescape_frontal_pairs_manifest.csv"
BASELINE_QC = ROOT / "research_rigid_expanded_qc_40case_20260611" / "rigid_expanded_qc_compact_summary.csv"
OUT = ROOT / "research_rigid_upright_hardgate_full40_20260619"


def read_manifest():
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return {r["pair_id"]: r for r in csv.DictReader(f)}


def read_cases():
    with BASELINE_QC.open("r", encoding="utf-8", newline="") as f:
        return [r["case"] for r in csv.DictReader(f)]


def process(case, by_pair, keep=8, rigid_iters=20):
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
        src_v, target_reg, fit_weight, region_masks, eye_mask, nose_anchor, keep=keep
    )
    for ci, cand in enumerate(initial, start=1):
        scale, rot, trans, hist = rigid.refine_rigid_fixed_scale(
            src_v,
            target_reg,
            fit_weight,
            cand["scale"],
            cand["rot"],
            cand["trans"],
            iterations=rigid_iters,
            region_masks=region_masks,
            nose_anchor=nose_anchor,
        )
        scale, rot, trans = rigid.translation_only_polish(src_v, target_reg, fit_weight, scale, rot, trans)
        aligned = rigid.transform(src_v, scale, rot, trans)
        metric = rigid.metrics(aligned, target_reg, fit_weight, region_masks=region_masks, nose_anchor=nose_anchor)
        orient = hard3.orientation_metrics(aligned, region_masks, eye_mask)
        score = hard3.combined_score(metric, orient)
        tried.append({"candidate": ci, "score": score, "scale": float(scale), **metric, **orient})
        print(
            f"{case} cand={ci}/{len(initial)} score={score:.6f} med={metric['median']:.6f} "
            f"p90={metric['p90']:.6f} anchor={metric['source_nose_anchor_distance']:.6f} "
            f"up={orient['upside_down']} eye_norm={orient['eye_over_mouth_norm']:.3f} scale={scale:.6f}",
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
    scale_only_o = hard3.orientation_metrics(scale_only_aligned, region_masks, eye_mask)

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
    sim_o = hard3.orientation_metrics(sim_aligned, region_masks, eye_mask)

    stages = [
        ("rigid", rigid_scale, rigid_rot, rigid_trans, rigid_aligned, rigid_m, rigid_o),
        ("scale_only", scale_only_scale, scale_only_rot, scale_only_trans, scale_only_aligned, scale_only_m, scale_only_o),
        ("similarity", final_scale, final_rot, final_trans, sim_aligned, sim_m, sim_o),
    ]
    scored = sorted(
        [(hard3.combined_score(m, o), name, scale, rot, trans, aligned, m, o) for name, scale, rot, trans, aligned, m, o in stages],
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
        "method": "upright hard-gated rigid full40 probe",
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
        f"anchor={final_m['source_nose_anchor_distance']:.6f} up={final_o['upside_down']} fig={fig}",
        flush=True,
    )
    return report


def qc_row(report):
    row = hard3.qc_row(report)
    row["report_json"] = str(OUT / report["case"] / f"{report['case']}_scratch_surface_registration_report.json")
    return row


def write_summary(reports):
    rows = [qc_row(report) for report in reports]
    if rows:
        with (OUT / "upright_hardgate_full40_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    aggregate = {
        "attempted": len(rows),
        "qc_pass": int(sum(row["qc_pass"] for row in rows)),
        "qc_fail": int(len(rows) - sum(row["qc_pass"] for row in rows)),
        "upside_down_remaining": [row["case"] for row in rows if row["upside_down"]],
        "pass_cases": [row["case"] for row in rows if row["qc_pass"]],
        "fail_cases": [row["case"] for row in rows if not row["qc_pass"]],
    }
    (OUT / "upright_hardgate_full40_aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return rows, aggregate


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    by_pair = read_manifest()
    reports = []
    for case in read_cases():
        report_path = OUT / case / f"{case}_scratch_surface_registration_report.json"
        if report_path.exists():
            print(f"{case} skip existing", flush=True)
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = process(case, by_pair)
        reports.append(report)
        write_summary(reports)
    _, aggregate = write_summary(reports)
    (OUT / "upright_hardgate_full40_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(OUT), **aggregate}, indent=2), flush=True)


if __name__ == "__main__":
    main()


