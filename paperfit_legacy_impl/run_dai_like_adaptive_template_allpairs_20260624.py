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
from scipy.spatial import cKDTree


ROOT = Path(os.environ.get("PAPERFIT_ROOT", "/path/to/prepared/facescape_pipeline"))
TOOLS = ROOT / "research_tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_rigid_fail6_nose_tip_gradient_candidate_only_20260622 as cand40  # noqa: E402
import os
import run_rigid_upright_hardgate_3case_20260619 as hard3  # noqa: E402
import os
import run_scratch_surface_registration_3case_final_attempt as rigid  # noqa: E402
import os
import run_nonrigid_nasal_depth_ablation_3case as nonrigid  # noqa: E402
import os


MANIFEST = ROOT / "prepared_001_020_cv" / "facescape_frontal_pairs_manifest.csv"
OUT = ROOT / "research_dai_like_adaptive_template_allpairs_20260624"


def read_manifest():
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return {r["pair_id"]: r for r in csv.DictReader(f)}


def read_cases():
    by_pair = read_manifest()
    return sorted(pid for pid in by_pair if not pid.endswith("_18_eye_closed"))


def target_for(case, by_pair):
    target_id = f"{int(case[:3]):03d}_18_eye_closed"
    target_row = by_pair[target_id]
    target_mesh = Path(target_row["mesh"])
    _, target_world, _ = rigid.load_mesh(target_mesh)
    target_reg = rigid.target_registration_frame(target_world, target_mesh.parent / "selected_camera.json")
    return target_row, target_mesh, target_world, target_reg


def qc_from_vertices(vertices, target_reg, fit_weight, region_masks, eye_mask, nose_anchor):
    metric = rigid.metrics(vertices, target_reg, fit_weight, region_masks=region_masks, nose_anchor=nose_anchor)
    orient = hard3.orientation_metrics(vertices, region_masks, eye_mask)
    med = float(metric["median"])
    p90 = float(metric["p90"])
    nose = float(metric.get("nose_weighted_median", med))
    anchor = float(metric.get("source_nose_anchor_distance", 999.0))
    qc_pass = (
        not bool(orient["upside_down"])
        and med <= 0.023
        and p90 <= 0.070
        and nose <= 0.040
        and anchor <= 0.120
    )
    return {
        "qc_pass": int(qc_pass),
        "upside_down": int(orient["upside_down"]),
        "median": med,
        "p90": p90,
        "nose_weighted_median": nose,
        "source_nose_anchor_distance": anchor,
        "eye_over_mouth_norm": float(orient["eye_over_mouth_norm"]),
        "nose_over_mouth_norm": float(orient["nose_over_mouth_norm"]),
        "vertical_z": float(orient["vertical_z"]),
    }


def umeyama_similarity(src, dst):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    xs = src - mu_s
    xd = dst - mu_d
    cov = (xd.T @ xs) / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1] = -1
    rot = u @ np.diag(d) @ vt
    var = np.mean(np.sum(xs * xs, axis=1))
    scale = float(np.sum(s * d) / max(var, 1e-12))
    trans = mu_d - scale * (mu_s @ rot.T)
    return scale, rot, trans


def landmark_indices(vertices, landmarks):
    tree = cKDTree(vertices)
    out = {}
    for key, point in landmarks.items():
        _, idx = tree.query(point, k=1)
        out[key] = int(idx)
    return out


def landmark_error(vertices, lm_idx, target_landmarks, keys):
    vals = {}
    total = 0.0
    weights = {"nose": 1.35, "left_eye": 0.75, "right_eye": 0.75, "mouth": 0.55, "chin": 0.30}
    wsum = 0.0
    for key in keys:
        d = float(np.linalg.norm(vertices[lm_idx[key]] - target_landmarks[key]))
        vals[f"landmark_{key}_distance"] = d
        total += weights.get(key, 0.5) * d
        wsum += weights.get(key, 0.5)
    vals["landmark_penalty"] = float(total / max(wsum, 1e-8))
    return vals


def median_point(vertices, mask):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        raise ValueError("empty landmark mask")
    return np.median(vertices[idx], axis=0)


def source_geometry_landmarks(src_v, region_masks, eye_mask):
    x, y, _ = rigid.canonical_coords(src_v)
    eyes = np.flatnonzero(eye_mask)
    if len(eyes) < 20:
        raise RuntimeError("source eye mask too small")
    eye_x = src_v[eyes, 0]
    left_eye = eyes[eye_x <= np.median(eye_x)]
    right_eye = eyes[eye_x > np.median(eye_x)]
    central_lower = (np.abs(x) < 0.18) & (y < 0.20)
    mouth_mask = region_masks.get("mouth_downweighted", np.zeros(len(src_v), dtype=bool))
    return {
        "nose": median_point(src_v, region_masks["nose_tip"]),
        "left_eye": median_point(src_v, np.isin(np.arange(len(src_v)), left_eye)),
        "right_eye": median_point(src_v, np.isin(np.arange(len(src_v)), right_eye)),
        "mouth": median_point(src_v, mouth_mask),
        "chin": median_point(src_v, central_lower),
    }


def target_point_near(target_reg, ideal, radius_x, radius_z, prefer_front=True):
    dx = np.abs(target_reg[:, 0] - ideal[0])
    dz = np.abs(target_reg[:, 2] - ideal[2])
    mask = (dx <= radius_x) & (dz <= radius_z)
    pts = target_reg[mask]
    if len(pts) < 20:
        tree = cKDTree(target_reg[:, [0, 2]])
        _, idx = tree.query(ideal[[0, 2]], k=min(80, len(target_reg)))
        pts = target_reg[np.atleast_1d(idx)]
    if prefer_front:
        front_cut = np.quantile(pts[:, 1], 0.18)
        front = pts[pts[:, 1] <= front_cut]
        if len(front) >= 5:
            pts = front
    d = np.linalg.norm(pts[:, [0, 2]] - ideal[[0, 2]], axis=1)
    return pts[int(np.argmin(d))]


def target_geometry_landmarks(target_reg, nose_anchor):
    lo = np.quantile(target_reg, 0.05, axis=0)
    hi = np.quantile(target_reg, 0.95, axis=0)
    center = np.median(target_reg, axis=0)
    width = float(max(hi[0] - lo[0], 1e-8))
    height = float(max(hi[2] - lo[2], 1e-8))
    z0 = float(lo[2])
    eye_z = z0 + 0.66 * height
    mouth_z = z0 + 0.38 * height
    chin_z = z0 + 0.12 * height
    return {
        "nose": np.asarray(nose_anchor, dtype=np.float64),
        "left_eye": target_point_near(target_reg, np.array([center[0] - 0.18 * width, center[1], eye_z]), 0.11 * width, 0.09 * height),
        "right_eye": target_point_near(target_reg, np.array([center[0] + 0.18 * width, center[1], eye_z]), 0.11 * width, 0.09 * height),
        "mouth": target_point_near(target_reg, np.array([center[0], center[1], mouth_z]), 0.16 * width, 0.08 * height, prefer_front=False),
        "chin": target_point_near(target_reg, np.array([center[0], center[1], chin_z]), 0.14 * width, 0.07 * height, prefer_front=False),
    }


def dai_lb_adapt(aligned, faces, lm_idx, tgt_lm, keys):
    edges = nonrigid.build_edges(faces)
    target_idx = np.asarray([lm_idx[k] for k in keys], dtype=np.int64)
    target_pos = np.asarray([tgt_lm[k] for k in keys], dtype=np.float64)
    weights = np.asarray(
        [{"nose": 95000.0, "left_eye": 52000.0, "right_eye": 52000.0, "mouth": 36000.0, "chin": 22000.0}.get(k, 30000.0) for k in keys],
        dtype=np.float64,
    )
    return nonrigid.solve(
        aligned,
        edges,
        target_idx,
        target_pos,
        weights,
        np.asarray([], dtype=np.int64),
        np.zeros((0, 3), dtype=np.float64),
        edge_w=165.0,
        fixed_w=0.0,
    )


def process_case(case, by_pair):
    _, _, _, target_reg = target_for(case, by_pair)
    src_obj = rigid.hrn_obj(case)
    _, src_v, faces = rigid.load_mesh(src_obj)
    fit_weight_base, eye_mask, eye_ellipses = rigid.texture_eye_weight(src_obj)
    fit_weight, region_masks = cand40.apply_nose_tip_gradient_weight(src_obj, src_v, fit_weight_base)
    nose_anchor = rigid.target_nose_anchor(target_reg)

    src_lm = source_geometry_landmarks(src_v, region_masks, eye_mask)
    tgt_lm = target_geometry_landmarks(target_reg, nose_anchor)
    common = [k for k in ("nose", "left_eye", "right_eye", "mouth", "chin") if k in src_lm and k in tgt_lm]
    if len(common) < 3:
        return {
            "case": case,
            "status": "skipped",
            "reason": f"not enough landmarks: source={sorted(src_lm)} target={sorted(tgt_lm)}",
        }

    scale, rot, trans = umeyama_similarity([src_lm[k] for k in common], [tgt_lm[k] for k in common])
    aligned = rigid.transform(src_v, scale, rot, trans)
    lm_idx = landmark_indices(src_v, src_lm)
    adapted = dai_lb_adapt(aligned, faces, lm_idx, tgt_lm, common)

    rows = []
    for stage, vertices in [
        ("dai_landmark_similarity", aligned),
        ("dai_lb_adaptive_template", adapted),
    ]:
        q = qc_from_vertices(vertices, target_reg, fit_weight, region_masks, eye_mask, nose_anchor)
        lm = landmark_error(vertices, lm_idx, tgt_lm, common)
        rows.append(
            {
                "case": case,
                "subject": case[:3],
                "stage": stage,
                "status": "ok",
                "landmark_keys": ",".join(common),
                "similarity_scale": float(scale),
                **q,
                **lm,
            }
        )
    print(json.dumps({"case": case, "rows": rows}), flush=True)
    return {"case": case, "status": "ok", "rows": rows}


def write_outputs(results):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dai_like_allpairs_reports.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    rows = []
    skipped = []
    for result in results:
        if result.get("status") == "ok":
            rows.extend(result["rows"])
        else:
            skipped.append(result)
    if rows:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with (OUT / "dai_like_allpairs_rows.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    aggregate = {
        "attempted_cases": len(results),
        "processed_cases": len({r["case"] for r in rows}),
        "skipped_cases": skipped,
        "by_stage": {},
    }
    for stage in sorted({r["stage"] for r in rows}):
        subset = [r for r in rows if r["stage"] == stage]
        aggregate["by_stage"][stage] = {
            "cases": len(subset),
            "qc_pass": int(sum(r["qc_pass"] for r in subset)),
            "qc_pass_rate_pct": float(100.0 * sum(r["qc_pass"] for r in subset) / max(len(subset), 1)),
            "upside_down_count": int(sum(r["upside_down"] for r in subset)),
            "mean_median": float(np.mean([r["median"] for r in subset])),
            "mean_p90": float(np.mean([r["p90"] for r in subset])),
            "mean_nose_weighted_median": float(np.mean([r["nose_weighted_median"] for r in subset])),
            "mean_anchor": float(np.mean([r["source_nose_anchor_distance"] for r in subset])),
            "mean_landmark_penalty": float(np.mean([r["landmark_penalty"] for r in subset])),
        }
    (OUT / "dai_like_allpairs_aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    by_pair = read_manifest()
    results = []
    for case in read_cases():
        report_path = OUT / case / f"{case}_dai_like_report.json"
        if report_path.exists():
            result = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            result = process_case(case, by_pair)
            case_dir = OUT / case
            case_dir.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)
        write_outputs(results)
    write_outputs(results)


if __name__ == "__main__":
    main()


