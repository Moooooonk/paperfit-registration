#!/usr/bin/env python3
import csv
import os
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(os.environ.get("PAPERFIT_ROOT", "/path/to/prepared/facescape_pipeline"))
TOOLS = ROOT / "research_tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_scratch_surface_registration_3case_final_attempt as rigid  # noqa: E402


MANIFEST = ROOT / "prepared_001_020_cv" / "facescape_frontal_pairs_manifest.csv"
OUT = ROOT / "research_rigid_global_upright_failed11_20260617"

CASES = [
    "004_10_dimpler",
    "006_1_neutral",
    "006_10_dimpler",
    "007_1_neutral",
    "007_10_dimpler",
    "013_1_neutral",
    "013_10_dimpler",
    "014_1_neutral",
    "016_1_neutral",
    "016_10_dimpler",
    "020_10_dimpler",
]


def read_manifest():
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return {r["pair_id"]: r for r in csv.DictReader(f)}


def rotation_grid(base):
    """Broad deterministic pose grid around each legal axis-convention candidate."""
    angles = np.deg2rad([0])
    small = np.deg2rad([0])
    seen = set()
    for yaw in angles:
        for pitch in angles:
            for roll in small:
                rot = rigid.axis_angle([0, 0, 1], roll) @ rigid.axis_angle([1, 0, 0], pitch) @ rigid.axis_angle([0, 1, 0], yaw) @ base
                if np.linalg.det(rot) <= 0.0:
                    continue
                key = tuple(np.round(rot.reshape(-1), 5))
                if key in seen:
                    continue
                seen.add(key)
                yield rot


def anatomical_upright_penalty(aligned, region_masks, eye_mask):
    nose_mask = (
        region_masks.get("nose_tip", False)
        | region_masks.get("alar", False)
        | region_masks.get("nose_dorsum", False)
        | region_masks.get("nose_bridge", False)
    )
    mouth_mask = region_masks.get("mouth_downweighted", False)
    if isinstance(nose_mask, bool) or isinstance(mouth_mask, bool):
        return 0.0
    if int(nose_mask.sum()) < 80 or int(mouth_mask.sum()) < 80 or int(eye_mask.sum()) < 80:
        return 0.0

    nose = np.median(aligned[nose_mask], axis=0)
    mouth = np.median(aligned[mouth_mask], axis=0)
    eyes = np.median(aligned[eye_mask], axis=0)
    ext = np.maximum(np.quantile(aligned, 0.95, axis=0) - np.quantile(aligned, 0.05, axis=0), 1e-8)
    height = float(ext[2])

    # In the registration frame z is the vertical image axis. For an upright
    # face, eyes should be above mouth and nose should be above mouth.
    eye_over_mouth = eyes[2] - mouth[2]
    nose_over_mouth = nose[2] - mouth[2]
    nose_front_over_mouth = nose[1] - mouth[1]

    penalty = 0.0
    penalty += max(0.0, 0.28 * height - eye_over_mouth)
    penalty += max(0.0, 0.08 * height - nose_over_mouth)
    penalty += max(0.0, 0.00 - nose_front_over_mouth)
    return float(penalty)


def global_initial_candidates(src, target_full, fit_weight, region_masks, eye_mask, nose_anchor, keep=4):
    fit_mask = fit_weight > 0.35
    src_fit = src[fit_mask]
    src_fit_weights = fit_weight[fit_mask]
    eval_idx = rigid.sample_idx(len(src_fit), 4000, 141)
    target_eval = target_full[rigid.sample_idx(len(target_full), 50000, 142)]
    tree = cKDTree(target_eval)

    base = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    base_rots = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                rot = np.diag([sx, sy, sz]) @ base
                if np.linalg.det(rot) > 0.0:
                    base_rots.append(rot)

    y = target_full[:, 1]
    front = target_full[y >= np.quantile(y, 0.78)]
    if len(front) < 5000:
        front = target_full
    src_ext = np.maximum(np.quantile(src_fit, 0.95, axis=0) - np.quantile(src_fit, 0.05, axis=0), 1e-8)
    front_ext = np.maximum(np.quantile(front, 0.92, axis=0) - np.quantile(front, 0.08, axis=0), 1e-8)
    target_center = np.median(front, axis=0)
    target_front_y = float(np.quantile(y, 0.94))

    candidates = []
    for base_rot in base_rots:
        for rot in rotation_grid(base_rot):
            mapped_ext = np.maximum(np.abs(rot) @ src_ext, 1e-8)
            scale0 = float(np.median([front_ext[0] / mapped_ext[0], front_ext[2] / mapped_ext[2]]))
            for scale in scale0 * np.linspace(0.58, 1.38, 17):
                moved_all = rigid.transform(src, float(scale), rot, np.zeros(3))
                moved_fit = rigid.transform(src_fit[eval_idx], float(scale), rot, np.zeros(3))
                trans = target_center - np.median(moved_all[fit_mask], axis=0)
                trans[1] = target_front_y - np.max(moved_all[fit_mask, 1])
                aligned_eval = moved_fit + trans
                d, _ = tree.query(aligned_eval, k=1, workers=-1)
                ew = src_fit_weights[eval_idx]
                aligned_all = moved_all + trans
                anchor = rigid.source_nose_anchor_distance(aligned_all, region_masks, nose_anchor)
                mouth_guard = rigid.source_mouth_to_nose_guard(aligned_all, region_masks, nose_anchor)
                upright = anatomical_upright_penalty(aligned_all, region_masks, eye_mask)
                coarse = float(
                    rigid.weighted_quantile(d, ew, 0.50)
                    + 0.20 * rigid.weighted_quantile(d, ew, 0.90)
                    + 0.08 * rigid.weighted_quantile(d, ew, 0.98)
                    + 0.65 * anchor
                    + 0.45 * mouth_guard
                    + 3.0 * upright
                )
                candidates.append({
                    "score": coarse,
                    "scale": float(scale),
                    "rot": rot.copy(),
                    "trans": trans.copy(),
                    "upright_penalty": upright,
                    "anchor": float(anchor),
                    "mouth_guard": float(mouth_guard),
                })

    candidates.sort(key=lambda r: r["score"])
    return candidates[:keep]


def qc_row(report):
    sim = report["similarity_metrics"]
    scale_ratio = float(report.get("scale_ratio_similarity_over_rigid", 0.0))
    med = float(sim["median"])
    p90 = float(sim["p90"])
    nose_med = float(sim.get("nose_weighted_median", med))
    anchor = float(sim.get("source_nose_anchor_distance", 999.0))
    mouth_guard = float(sim.get("source_mouth_to_nose_guard", 999.0))
    qc_pass = (
        med <= 0.020
        and p90 <= 0.055
        and nose_med <= 0.035
        and anchor <= 0.090
        and mouth_guard <= 0.010
        and 0.60 <= scale_ratio <= 1.18
    )
    return {
        "case": report["case"],
        "qc_pass": int(qc_pass),
        "similarity_median": med,
        "similarity_p90": p90,
        "similarity_nose_weighted_median": nose_med,
        "similarity_nose_weighted_p90": float(sim.get("nose_weighted_p90", p90)),
        "source_nose_anchor_distance": anchor,
        "source_mouth_to_nose_guard": mouth_guard,
        "rigid_median": float(report["rigid_metrics"]["median"]),
        "adaptive_scale_only_median": float(report["adaptive_scale_only_metrics"]["median"]),
        "rigid_scale": float(report["rigid_scale"]),
        "similarity_scale": float(report["similarity_scale"]),
        "final_similarity_scale": float(report["final_similarity_scale"]),
        "scale_ratio_similarity_over_rigid": scale_ratio,
        "diagnostic_png": report["diagnostic_png"],
        "report_json": str(OUT / report["case"] / f"{report['case']}_scratch_surface_registration_report.json"),
    }


def process(pair_id, by_pair):
    row = by_pair[pair_id]
    target_row = by_pair[f"{int(row['subject']):03d}_18_eye_closed"]
    src_obj = rigid.hrn_obj(pair_id)
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
    initial = global_initial_candidates(src_v, target_reg, fit_weight, region_masks, eye_mask, nose_anchor)
    for cand in initial:
        scale, rot, trans, hist = rigid.refine_rigid_fixed_scale(
            src_v,
            target_reg,
            fit_weight,
            cand["scale"],
            cand["rot"],
            cand["trans"],
            iterations=24,
            region_masks=region_masks,
            nose_anchor=nose_anchor,
        )
        scale, rot, trans = rigid.translation_only_polish(src_v, target_reg, fit_weight, scale, rot, trans)
        aligned_reg = rigid.transform(src_v, scale, rot, trans)
        m = rigid.metrics(aligned_reg, target_reg, fit_weight, region_masks=region_masks, nose_anchor=nose_anchor)
        upright = anatomical_upright_penalty(aligned_reg, region_masks, eye_mask)
        score = float(m["selection_score"] + 3.0 * upright)
        tried.append({
            "initial_score": float(cand["score"]),
            "initial_upright_penalty": float(cand["upright_penalty"]),
            "initial_anchor": float(cand["anchor"]),
            "initial_mouth_guard": float(cand["mouth_guard"]),
            "scale0": float(cand["scale"]),
            "scale": float(scale),
            "upright_penalty": float(upright),
            "guarded_selection_score": score,
            **m,
        })
        if best is None or score < best[0]:
            best = (score, scale, rot, trans, aligned_reg, m, hist)

    _, rigid_scale, rigid_rot, rigid_trans, rigid_aligned_reg, rigid_m, rigid_hist = best
    scale_only_scale, scale_only_rot, scale_only_trans, scale_only_hist = rigid.adaptive_scale_prefit_only(
        src_v,
        target_reg,
        fit_weight,
        rigid_scale,
        rigid_rot,
        rigid_trans,
        region_masks=region_masks,
        nose_anchor=nose_anchor,
    )
    scale_only_aligned_reg = rigid.transform(src_v, scale_only_scale, scale_only_rot, scale_only_trans)
    scale_only_m = rigid.metrics(scale_only_aligned_reg, target_reg, fit_weight, region_masks=region_masks, nose_anchor=nose_anchor)

    sim_scale, sim_rot, sim_trans, sim_hist = rigid.scale_first_then_rigid_icp(
        src_v,
        target_reg,
        fit_weight,
        rigid_scale,
        rigid_rot,
        rigid_trans,
        region_masks=region_masks,
        nose_anchor=nose_anchor,
    )
    similarity_aligned_reg = rigid.transform(src_v, sim_scale, sim_rot, sim_trans)
    similarity_pre_polish_m = rigid.metrics(similarity_aligned_reg, target_reg, fit_weight, region_masks=region_masks, nose_anchor=nose_anchor)
    final_scale, final_rot, final_trans, final_pose_hist = rigid.final_small_pose_search(
        src_v,
        target_reg,
        fit_weight,
        sim_scale,
        sim_rot,
        sim_trans,
        region_masks=region_masks,
        nose_anchor=nose_anchor,
    )
    similarity_aligned_reg = rigid.transform(src_v, final_scale, final_rot, final_trans)
    similarity_m = rigid.metrics(similarity_aligned_reg, target_reg, fit_weight, region_masks=region_masks, nose_anchor=nose_anchor)

    rigid_region_m = rigid.region_metrics(rigid_aligned_reg, target_reg, region_masks, fit_weight)
    scale_region_m = rigid.region_metrics(scale_only_aligned_reg, target_reg, region_masks, fit_weight)
    similarity_region_m = rigid.region_metrics(similarity_aligned_reg, target_reg, region_masks, fit_weight)

    rigid_aligned_world = rigid.inv_target_registration_frame(rigid_aligned_reg, cam_json)
    scale_only_aligned_world = rigid.inv_target_registration_frame(scale_only_aligned_reg, cam_json)
    similarity_aligned_world = rigid.inv_target_registration_frame(similarity_aligned_reg, cam_json)
    case_dir = OUT / pair_id
    rigid_out_reg = case_dir / f"{pair_id}_rigid_icp_regframe.obj"
    rigid_out_world = case_dir / f"{pair_id}_rigid_icp_world.obj"
    scale_out_reg = case_dir / f"{pair_id}_adaptive_scale_only_regframe.obj"
    scale_out_world = case_dir / f"{pair_id}_adaptive_scale_only_world.obj"
    sim_out_reg = case_dir / f"{pair_id}_similarity_icp_regframe.obj"
    sim_out_world = case_dir / f"{pair_id}_similarity_icp_world.obj"
    rigid.write_obj_like(src_obj, rigid_out_reg, rigid_aligned_reg)
    rigid.write_obj_like(src_obj, rigid_out_world, rigid_aligned_world)
    rigid.write_obj_like(src_obj, scale_out_reg, scale_only_aligned_reg)
    rigid.write_obj_like(src_obj, scale_out_world, scale_only_aligned_world)
    rigid.write_obj_like(src_obj, sim_out_reg, similarity_aligned_reg)
    rigid.write_obj_like(src_obj, sim_out_world, similarity_aligned_world)
    np.save(case_dir / f"{pair_id}_source_eye_orbit_mask.npy", eye_mask)
    np.save(case_dir / f"{pair_id}_source_fit_weight_with_eye_and_midface.npy", fit_weight)
    fig = rigid.plot_review(
        case_dir,
        pair_id,
        src_v,
        target_reg,
        rigid_aligned_reg,
        scale_only_aligned_reg,
        similarity_aligned_reg,
        fit_weight,
        eye_mask,
        region_masks,
        rigid_m,
        scale_only_m,
        similarity_m,
    )
    curve_fig = rigid.plot_scale_history(case_dir, pair_id, scale_only_hist, sim_hist)
    report = {
        "case": pair_id,
        "method": "global multi-start upright guarded rigid initialization: broad proper-rotation grid, anatomical eye/nose/mouth upright penalty, then adaptive scale and fixed-scale rigid refinement",
        "source_obj": str(src_obj),
        "target_mesh": str(target_mesh),
        "target_policy": "full eye-closed target mesh, no ROI crop",
        "target_nose_anchor_regframe": [float(v) for v in nose_anchor],
        "rigid_regframe_obj": str(rigid_out_reg),
        "rigid_world_obj": str(rigid_out_world),
        "adaptive_scale_only_regframe_obj": str(scale_out_reg),
        "adaptive_scale_only_world_obj": str(scale_out_world),
        "similarity_regframe_obj": str(sim_out_reg),
        "similarity_world_obj": str(sim_out_world),
        "diagnostic_png": str(fig),
        "scale_search_curves_png": str(curve_fig),
        "rigid_scale": float(rigid_scale),
        "adaptive_scale_only_scale": float(scale_only_scale),
        "similarity_scale": float(sim_scale),
        "final_similarity_scale": float(final_scale),
        "scale_ratio_adaptive_over_rigid": float(scale_only_scale / max(rigid_scale, 1e-12)),
        "scale_ratio_similarity_over_rigid": float(sim_scale / max(rigid_scale, 1e-12)),
        "fit_vertices_weighted": int((fit_weight > 0.08).sum()),
        "fit_vertices_full_weight": int((fit_weight > 0.60).sum()),
        "eye_excluded_vertices": int(eye_mask.sum()),
        "eye_mask_source": "HRN texture/UV detected eye-orbit mask",
        "eye_ellipses": eye_ellipses,
        "region_weighting": "source-side nose and midface weighted; outer face downweighted; source nose anchor scored against automatically detected target nose from full target mesh",
        "region_vertices": {k: int(v.sum()) for k, v in region_masks.items()},
        "rigid_history_last": float(rigid_hist[-1]) if rigid_hist else None,
        "adaptive_scale_only_history": scale_only_hist,
        "adaptive_scale_only_selected_by_round": rigid.selected_by_round(scale_only_hist, "scale"),
        "similarity_history": sim_hist,
        "similarity_selected_by_round": rigid.selected_by_round(sim_hist, "result_scale"),
        "final_small_pose_history": final_pose_hist,
        "tried_initializations": tried,
        "rigid_metrics": rigid_m,
        "adaptive_scale_only_metrics": scale_only_m,
        "similarity_pre_polish_metrics": similarity_pre_polish_m,
        "similarity_metrics": similarity_m,
        "rigid_region_metrics": rigid_region_m,
        "adaptive_scale_only_region_metrics": scale_region_m,
        "similarity_region_metrics": similarity_region_m,
    }
    (case_dir / f"{pair_id}_scratch_surface_registration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"{pair_id} rigid={rigid_m['median']:.6f}/{rigid_m['p90']:.6f} "
        f"scale_only={scale_only_m['median']:.6f}/{scale_only_m['p90']:.6f} "
        f"similarity={similarity_m['median']:.6f}/{similarity_m['p90']:.6f} "
        f"scale={rigid_scale:.6f}->{scale_only_scale:.6f}->{sim_scale:.6f}->{final_scale:.6f} fig={fig}",
        flush=True,
    )
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    by_pair = read_manifest()
    reports = []
    rows = []
    for case in CASES:
        report_path = OUT / case / f"{case}_scratch_surface_registration_report.json"
        if report_path.exists():
            print(f"{case} skip existing", flush=True)
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = process(case, by_pair)
        reports.append(report)
        rows.append(qc_row(report))
        fields = list(rows[0].keys())
        with (OUT / "failed11_global_upright_qc_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    aggregate = {
        "cases": CASES,
        "attempted": len(rows),
        "qc_pass": int(sum(r["qc_pass"] for r in rows)),
        "qc_fail": int(len(rows) - sum(r["qc_pass"] for r in rows)),
        "pass_cases": [r["case"] for r in rows if r["qc_pass"]],
        "fail_cases": [r["case"] for r in rows if not r["qc_pass"]],
    }
    (OUT / "summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    (OUT / "failed11_global_upright_qc_aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(OUT), **aggregate}, indent=2), flush=True)


if __name__ == "__main__":
    main()


