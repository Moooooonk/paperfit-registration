#!/usr/bin/env python3
import csv
import os
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d


ROOT = Path(os.environ.get("PAPERFIT_ROOT", "/path/to/prepared/facescape_pipeline"))
TOOLS = ROOT / "research_tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_scratch_surface_registration_3case_final_attempt as rigid  # noqa: E402
import run_rigid_upright_hardgate_3case_20260619 as hard3  # noqa: E402


MANIFEST = ROOT / "prepared_001_020_cv" / "facescape_frontal_pairs_manifest.csv"
BASELINE_QC = ROOT / "research_rigid_expanded_qc_40case_20260611" / "rigid_expanded_qc_compact_summary.csv"
OUT = ROOT / "research_rigid_open3d_baselines_40_20260622"


def read_manifest():
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return {r["pair_id"]: r for r in csv.DictReader(f)}


def read_cases():
    with BASELINE_QC.open("r", encoding="utf-8", newline="") as f:
        return [r["case"] for r in csv.DictReader(f)]


def to_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pcd


def sample_weighted(points, weights, max_n, seed):
    idx = np.flatnonzero(weights > 0.08)
    if len(idx) > max_n:
        w = weights[idx].astype(np.float64)
        w = w / np.sum(w)
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, max_n, replace=False, p=w)
    return points[idx]


def preprocess(points, voxel):
    pcd = to_pcd(points)
    pcd = pcd.voxel_down_sample(voxel)
    radius_normal = voxel * 2.0
    radius_feature = voxel * 5.0
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd, fpfh


def bbox_prescale(src, target):
    src_center = np.median(src, axis=0)
    tgt_center = np.median(target, axis=0)
    src_extent = np.quantile(src, 0.95, axis=0) - np.quantile(src, 0.05, axis=0)
    tgt_extent = np.quantile(target, 0.95, axis=0) - np.quantile(target, 0.05, axis=0)
    scale = float(np.median(tgt_extent / np.maximum(src_extent, 1e-8)))
    trans = tgt_center - scale * src_center
    return scale, np.eye(3), trans


def apply_matrix(src, mat):
    homog = np.ones((len(src), 4), dtype=np.float64)
    homog[:, :3] = src
    return (homog @ mat.T)[:, :3]


def matrix_to_similarity(mat, pre_scale, pre_rot, pre_trans):
    a = mat[:3, :3]
    s_delta = float(np.cbrt(abs(np.linalg.det(a))))
    if not np.isfinite(s_delta) or s_delta < 1e-8:
        s_delta = 1.0
    r_delta = a / s_delta
    if np.linalg.det(r_delta) < 0:
        r_delta[:, -1] *= -1.0
    scale = float(s_delta * pre_scale)
    rot = r_delta @ pre_rot
    trans = s_delta * (pre_trans @ r_delta.T) + mat[:3, 3]
    return scale, rot, trans


def refine_icp(src_pre, target, init_mat, voxel, point_to_plane=False):
    source = to_pcd(src_pre)
    target_pcd = to_pcd(target)
    target_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))
    estimation = (
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
        if point_to_plane
        else o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
    )
    return o3d.pipelines.registration.registration_icp(
        source, target_pcd, voxel * 2.5, init_mat, estimation,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
    )


def run_open3d_methods(src_v, target_reg, fit_weight):
    src_fit = sample_weighted(src_v, fit_weight, 28000, 20260622)
    target_fit = target_reg
    if len(target_fit) > 90000:
        rng = np.random.default_rng(20260623)
        target_fit = target_fit[rng.choice(len(target_fit), 90000, replace=False)]

    pre_scale, pre_rot, pre_trans = bbox_prescale(src_fit, target_fit)
    src_pre = rigid.transform(src_fit, pre_scale, pre_rot, pre_trans)
    full_pre = rigid.transform(src_v, pre_scale, pre_rot, pre_trans)

    extent = np.linalg.norm(np.quantile(target_fit, 0.95, axis=0) - np.quantile(target_fit, 0.05, axis=0))
    voxel = max(float(extent / 42.0), 0.012)
    src_down, src_fpfh = preprocess(src_pre, voxel)
    tgt_down, tgt_fpfh = preprocess(target_fit, voxel)
    dist = voxel * 2.5
    methods = {}

    try:
        ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_down,
            tgt_down,
            src_fpfh,
            tgt_fpfh,
            True,
            dist,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
            4,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
        )
        methods["fpfh_ransac"] = ransac.transformation
        methods["fpfh_ransac_icp"] = refine_icp(src_pre, target_fit, ransac.transformation, voxel).transformation
    except Exception as exc:
        methods["fpfh_ransac_error"] = str(exc)

    try:
        fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            src_down,
            tgt_down,
            src_fpfh,
            tgt_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=dist),
        )
        methods["fpfh_fgr"] = fgr.transformation
        methods["fpfh_fgr_icp"] = refine_icp(src_pre, target_fit, fgr.transformation, voxel).transformation
    except Exception as exc:
        methods["fpfh_fgr_error"] = str(exc)

    out = []
    for name, mat in methods.items():
        if isinstance(mat, str):
            out.append({"method": name, "error": mat})
            continue
        scale, rot, trans = matrix_to_similarity(mat, pre_scale, pre_rot, pre_trans)
        aligned = rigid.transform(src_v, scale, rot, trans)
        out.append({
            "method": name,
            "scale": float(scale),
            "rot": rot,
            "trans": trans,
            "aligned": aligned,
            "matrix": mat,
            "pre_scale": float(pre_scale),
            "voxel": float(voxel),
            "down_source": int(np.asarray(src_down.points).shape[0]),
            "down_target": int(np.asarray(tgt_down.points).shape[0]),
        })
    out.append({
        "method": "bbox_prescale_only",
        "scale": float(pre_scale),
        "rot": pre_rot,
        "trans": pre_trans,
        "aligned": full_pre,
        "matrix": np.eye(4),
        "pre_scale": float(pre_scale),
        "voxel": float(voxel),
        "down_source": int(np.asarray(src_down.points).shape[0]),
        "down_target": int(np.asarray(tgt_down.points).shape[0]),
    })
    return out


def process(case, by_pair):
    t0 = time.time()
    row = by_pair[case]
    target_row = by_pair["%03d_18_eye_closed" % int(row["subject"])]
    src_obj = rigid.hrn_obj(case)
    _, src_v, _ = rigid.load_mesh(src_obj)
    target_mesh = Path(target_row["mesh"])
    _, target_world, _ = rigid.load_mesh(target_mesh)
    target_reg = rigid.target_registration_frame(target_world, target_mesh.parent / "selected_camera.json")
    nose_anchor = rigid.target_nose_anchor(target_reg)
    fit_weight_base, eye_mask, _ = rigid.texture_eye_weight(src_obj)
    fit_weight, region_masks = rigid.apply_nose_midface_weight(src_obj, src_v, fit_weight_base)

    case_dir = OUT / case
    case_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    reports = []
    for item in run_open3d_methods(src_v, target_reg, fit_weight):
        name = item["method"]
        if "error" in item:
            rows.append({"case": case, "method": name, "error": item["error"], "qc_pass": 0})
            continue
        aligned = item["aligned"]
        m = rigid.metrics(aligned, target_reg, fit_weight, region_masks=region_masks, nose_anchor=nose_anchor)
        o = hard3.orientation_metrics(aligned, region_masks, eye_mask)
        report = {
            "case": case,
            "method": name,
            "selected_stage": name,
            "source_obj": str(src_obj),
            "target_mesh": str(target_mesh),
            "diagnostic_png": "",
            "rigid_scale": float(item["scale"]),
            "final_similarity_scale": float(item["scale"]),
            "similarity_metrics": m,
            "similarity_orientation_metrics": o,
            "pre_scale": item["pre_scale"],
            "voxel": item["voxel"],
            "down_source": item["down_source"],
            "down_target": item["down_target"],
        }
        qc = hard3.qc_row(report)
        obj_path = case_dir / f"{case}_{name}_regframe.obj"
        rigid.write_obj_like(src_obj, obj_path, aligned)
        report["similarity_regframe_obj"] = str(obj_path)
        report["diagnostic_png"] = ""
        row_out = {
            **qc,
            "method": name,
            "error": "",
            "runtime_sec": float(time.time() - t0),
            "pre_scale": item["pre_scale"],
            "voxel": item["voxel"],
            "down_source": item["down_source"],
            "down_target": item["down_target"],
            "regframe_obj": str(obj_path),
        }
        rows.append(row_out)
        reports.append(report)
    (case_dir / f"{case}_open3d_baseline_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps({"case": case, "rows": len(rows), "elapsed": time.time() - t0}), flush=True)
    return rows


def write_summary(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "open3d_baselines_40_summary.csv"
    fields = sorted({k for row in rows for k in row.keys()})
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    aggregate = {}
    for method in sorted({r["method"] for r in rows}):
        mr = [r for r in rows if r["method"] == method and not r.get("error")]
        if not mr:
            aggregate[method] = {"attempted": 0, "errors": len([r for r in rows if r["method"] == method])}
            continue
        aggregate[method] = {
            "attempted": len(mr),
            "qc_pass": int(sum(int(r.get("qc_pass", 0)) for r in mr)),
            "qc_fail": int(len(mr) - sum(int(r.get("qc_pass", 0)) for r in mr)),
            "upside_down": int(sum(int(r.get("upside_down", 0)) for r in mr)),
            "mean_median": float(np.mean([float(r["similarity_median"]) for r in mr])),
            "mean_p90": float(np.mean([float(r["similarity_p90"]) for r in mr])),
            "mean_anchor": float(np.mean([float(r["source_nose_anchor_distance"]) for r in mr])),
            "pass_cases": [r["case"] for r in mr if int(r.get("qc_pass", 0))],
            "fail_cases": [r["case"] for r in mr if not int(r.get("qc_pass", 0))],
        }
    (OUT / "open3d_baselines_40_aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return aggregate


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    by_pair = read_manifest()
    all_rows = []
    for case in read_cases():
        case_report = OUT / case / f"{case}_open3d_baseline_reports.json"
        if case_report.exists():
            print(f"{case} existing report ignored for summary refresh", flush=True)
        all_rows.extend(process(case, by_pair))
        write_summary(all_rows)
    aggregate = write_summary(all_rows)
    print(json.dumps({"output_root": str(OUT), "aggregate": aggregate}, indent=2), flush=True)


if __name__ == "__main__":
    main()


