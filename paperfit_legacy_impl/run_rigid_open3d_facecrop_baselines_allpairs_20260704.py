#!/usr/bin/env python3
import csv
import json
import time
from pathlib import Path

import numpy as np

import run_rigid_open3d_baselines_40_20260622 as base


ROOT = base.ROOT
OUT = ROOT / "research_rigid_open3d_facecrop_baselines_allpairs_20260704"
MANIFEST = ROOT / "prepared_001_020_cv" / "facescape_frontal_pairs_manifest.csv"


def read_cases():
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return sorted(row["pair_id"] for row in rows if not row["pair_id"].endswith("_18_eye_closed"))


def target_face_roi(target_reg):
    x, y_front, z = base.rigid.canonical_coords(target_reg)
    return (
        (np.abs(x) < 0.34)
        & (z > 0.25)
        & (z < 0.82)
        & (y_front > np.quantile(y_front, 0.66))
    )


def run_open3d_methods_facecrop(src_v, target_reg, fit_weight):
    src_fit = base.sample_weighted(src_v, fit_weight, 28000, 20260704)
    roi_mask = target_face_roi(target_reg)
    target_fit = target_reg[roi_mask]
    if len(target_fit) > 90000:
        rng = np.random.default_rng(20260704)
        target_fit = target_fit[rng.choice(len(target_fit), 90000, replace=False)]

    pre_scale, pre_rot, pre_trans = base.bbox_prescale(src_fit, target_fit)
    src_pre = base.rigid.transform(src_fit, pre_scale, pre_rot, pre_trans)
    full_pre = base.rigid.transform(src_v, pre_scale, pre_rot, pre_trans)

    extent = np.linalg.norm(np.quantile(target_fit, 0.95, axis=0) - np.quantile(target_fit, 0.05, axis=0))
    voxel = max(float(extent / 42.0), 0.012)
    src_down, src_fpfh = base.preprocess(src_pre, voxel)
    tgt_down, tgt_fpfh = base.preprocess(target_fit, voxel)
    dist = voxel * 2.5
    methods = {}

    try:
        ransac = base.o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_down,
            tgt_down,
            src_fpfh,
            tgt_fpfh,
            True,
            dist,
            base.o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
            4,
            [
                base.o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                base.o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist),
            ],
            base.o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
        )
        methods["facecrop_fpfh_ransac"] = ransac.transformation
        methods["facecrop_fpfh_ransac_icp"] = base.refine_icp(
            src_pre, target_fit, ransac.transformation, voxel
        ).transformation
    except Exception as exc:
        methods["facecrop_fpfh_ransac_error"] = str(exc)

    try:
        fgr = base.o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            src_down,
            tgt_down,
            src_fpfh,
            tgt_fpfh,
            base.o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=dist),
        )
        methods["facecrop_fpfh_fgr"] = fgr.transformation
        methods["facecrop_fpfh_fgr_icp"] = base.refine_icp(
            src_pre, target_fit, fgr.transformation, voxel
        ).transformation
    except Exception as exc:
        methods["facecrop_fpfh_fgr_error"] = str(exc)

    out = []
    for name, mat in methods.items():
        if isinstance(mat, str):
            out.append({"method": name, "error": mat})
            continue
        scale, rot, trans = base.matrix_to_similarity(mat, pre_scale, pre_rot, pre_trans)
        aligned = base.rigid.transform(src_v, scale, rot, trans)
        out.append(
            {
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
                "target_roi_vertices": int(roi_mask.sum()),
                "target_fit_vertices": int(len(target_fit)),
                "target_policy": "frontal face ROI crop for Open3D fitting; full target used for final QC metrics",
            }
        )
    out.append(
        {
            "method": "facecrop_bbox_prescale_only",
            "scale": float(pre_scale),
            "rot": pre_rot,
            "trans": pre_trans,
            "aligned": full_pre,
            "matrix": np.eye(4),
            "pre_scale": float(pre_scale),
            "voxel": float(voxel),
            "down_source": int(np.asarray(src_down.points).shape[0]),
            "down_target": int(np.asarray(tgt_down.points).shape[0]),
            "target_roi_vertices": int(roi_mask.sum()),
            "target_fit_vertices": int(len(target_fit)),
            "target_policy": "frontal face ROI crop for Open3D fitting; full target used for final QC metrics",
        }
    )
    return out


def process(case, by_pair):
    t0 = time.time()
    row = by_pair[case]
    target_row = by_pair["%03d_18_eye_closed" % int(row["subject"])]
    src_obj = base.rigid.hrn_obj(case)
    _, src_v, _ = base.rigid.load_mesh(src_obj)
    target_mesh = Path(target_row["mesh"])
    _, target_world, _ = base.rigid.load_mesh(target_mesh)
    target_reg = base.rigid.target_registration_frame(target_world, target_mesh.parent / "selected_camera.json")
    nose_anchor = base.rigid.target_nose_anchor(target_reg)
    fit_weight_base, eye_mask, _ = base.rigid.texture_eye_weight(src_obj)
    fit_weight, region_masks = base.rigid.apply_nose_midface_weight(src_obj, src_v, fit_weight_base)

    case_dir = OUT / case
    case_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    reports = []
    for item in run_open3d_methods_facecrop(src_v, target_reg, fit_weight):
        name = item["method"]
        if "error" in item:
            rows.append({"case": case, "method": name, "error": item["error"], "qc_pass": 0})
            continue
        aligned = item["aligned"]
        metrics = base.rigid.metrics(aligned, target_reg, fit_weight, region_masks=region_masks, nose_anchor=nose_anchor)
        orient = base.hard3.orientation_metrics(aligned, region_masks, eye_mask)
        report = {
            "case": case,
            "method": name,
            "selected_stage": name,
            "source_obj": str(src_obj),
            "target_mesh": str(target_mesh),
            "target_policy": item["target_policy"],
            "diagnostic_png": "",
            "rigid_scale": float(item["scale"]),
            "final_similarity_scale": float(item["scale"]),
            "similarity_metrics": metrics,
            "similarity_orientation_metrics": orient,
            "pre_scale": item["pre_scale"],
            "voxel": item["voxel"],
            "down_source": item["down_source"],
            "down_target": item["down_target"],
            "target_roi_vertices": item["target_roi_vertices"],
            "target_fit_vertices": item["target_fit_vertices"],
        }
        qc = base.hard3.qc_row(report)
        report["similarity_regframe_obj"] = ""
        rows.append(
            {
                **qc,
                "method": name,
                "error": "",
                "runtime_sec": float(time.time() - t0),
                "pre_scale": item["pre_scale"],
                "voxel": item["voxel"],
                "down_source": item["down_source"],
                "down_target": item["down_target"],
                "target_roi_vertices": item["target_roi_vertices"],
                "target_fit_vertices": item["target_fit_vertices"],
            }
        )
        reports.append(report)

    (case_dir / f"{case}_open3d_facecrop_baseline_reports.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )
    print(json.dumps({"case": case, "rows": len(rows), "elapsed": time.time() - t0}), flush=True)
    return rows


def write_summary(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "open3d_facecrop_baselines_allpairs_summary.csv"
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
        passed = [r for r in mr if int(r.get("qc_pass", 0))]
        aggregate[method] = {
            "attempted": len(mr),
            "qc_pass": int(len(passed)),
            "qc_fail": int(len(mr) - len(passed)),
            "upside_down": int(sum(int(r.get("upside_down", 0)) for r in mr)),
            "mean_median": float(np.mean([float(r["similarity_median"]) for r in mr])),
            "mean_p90": float(np.mean([float(r["similarity_p90"]) for r in mr])),
            "mean_anchor": float(np.mean([float(r["source_nose_anchor_distance"]) for r in mr])),
            "pass_cases": [r["case"] for r in passed],
            "fail_cases": [r["case"] for r in mr if not int(r.get("qc_pass", 0))],
        }
    (OUT / "open3d_facecrop_baselines_allpairs_aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    return aggregate


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    by_pair = base.read_manifest()
    all_rows = []
    for case in read_cases():
        all_rows.extend(process(case, by_pair))
        write_summary(all_rows)
    aggregate = write_summary(all_rows)
    print(json.dumps({"output_root": str(OUT), "aggregate": aggregate}, indent=2), flush=True)


if __name__ == "__main__":
    main()


