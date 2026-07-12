#!/usr/bin/env python3
import csv
import os
import json
import os
import shutil
import os
from pathlib import Path

import matplotlib
import os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np
import os
import trimesh
import os
from scipy.spatial import cKDTree
from PIL import Image


ROOT = Path(os.environ.get("PAPERFIT_ROOT", "/path/to/prepared/facescape_pipeline"))
MANIFEST = ROOT / "prepared_001_020_cv" / "facescape_frontal_pairs_manifest.csv"
HRN_ROOT = ROOT / "hrn_outputs_001_020"
OUT = ROOT / "research_scratch_surface_registration_3case_final_attempt_20260531"
CASES = ["001_1_neutral", "001_10_dimpler", "002_1_neutral"]


def load_mesh(path):
    mesh = trimesh.load(str(path), process=False)
    return mesh, np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def read_manifest():
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def hrn_obj(pair_id):
    return HRN_ROOT / pair_id / f"{pair_id}_0_hrn_mid_mesh.obj"


def target_registration_frame(vertices, camera_json):
    """FaceScape world mesh -> frontal registration frame, without cropping target vertices."""
    data = json.loads(Path(camera_json).read_text(encoding="utf-8"))
    rt = np.asarray(data["Rt"], dtype=np.float64)
    cam = vertices @ rt[:, :3].T + rt[:, 3]
    return np.column_stack([cam[:, 0], -cam[:, 2], -cam[:, 1]])


def inv_target_registration_frame(vertices_reg, camera_json):
    data = json.loads(Path(camera_json).read_text(encoding="utf-8"))
    rt = np.asarray(data["Rt"], dtype=np.float64)
    cam = np.column_stack([vertices_reg[:, 0], -vertices_reg[:, 2], -vertices_reg[:, 1]])
    return (cam - rt[:, 3]) @ rt[:, :3]


def canonical_coords(vertices):
    mn = vertices.min(axis=0)
    mx = vertices.max(axis=0)
    ext = np.maximum(mx - mn, 1e-8)
    x = (vertices[:, 0] - (mn[0] + mx[0]) * 0.5) / ext[0]
    y = (vertices[:, 1] - mn[1]) / ext[1]
    z = (vertices[:, 2] - mn[2]) / ext[2]
    return x, y, z


def target_nose_anchor(vertices):
    x, front, z = canonical_coords(vertices)
    central = (np.abs(x) < 0.18) & (z > 0.34) & (z < 0.70)
    if int(central.sum()) < 500:
        central = (np.abs(x) < 0.26) & (z > 0.28) & (z < 0.76)
    candidates = vertices[central]
    if len(candidates) < 100:
        candidates = vertices
    y = candidates[:, 1]
    front_band = candidates[y >= np.quantile(y, 0.985)]
    if len(front_band) < 30:
        front_band = candidates[y >= np.quantile(y, 0.965)]
    return np.median(front_band, axis=0)


def parse_vertex_uv(obj_path):
    verts = []
    uvs = []
    pairs = []
    for line in Path(obj_path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            uvs.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("f "):
            for token in line.split()[1:]:
                parts = token.split("/")
                if len(parts) >= 2 and parts[0] and parts[1]:
                    pairs.append((int(parts[0]) - 1, int(parts[1]) - 1))
    n = len(verts)
    uvs = np.asarray(uvs, dtype=np.float64)
    vertex_uv = np.full((n, 2), np.nan, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    for vi, ti in pairs:
        if 0 <= vi < n and 0 <= ti < len(uvs):
            if np.isnan(vertex_uv[vi, 0]):
                vertex_uv[vi] = 0.0
            vertex_uv[vi] += uvs[ti]
            counts[vi] += 1.0
    valid = counts > 0
    vertex_uv[valid] /= counts[valid, None]
    return vertex_uv, valid


def detect_eye_ellipses(texture_path):
    image = Image.open(texture_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.float64)
    h, w = rgb.shape[:2]
    lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    ellipses = []
    for name, x0, x1 in [("left", 0.24, 0.50), ("right", 0.50, 0.76)]:
        xs = np.arange(int(w * x0), int(w * x1))
        ys = np.arange(int(h * 0.24), int(h * 0.52))
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        crop = lum[yy, xx]
        threshold = min(95.0, float(np.quantile(crop, 0.24)))
        dark = crop <= threshold
        if int(dark.sum()) < 20:
            threshold = min(115.0, float(np.quantile(crop, 0.34)))
            dark = crop <= threshold
        px = xx[dark].astype(np.float64)
        py = yy[dark].astype(np.float64)
        if len(px) == 0:
            cx = w * (0.37 if name == "left" else 0.63)
            cy = h * 0.38
            rx = w * 0.095
            ry = h * 0.065
        else:
            cx = float(np.median(px))
            cy = float(np.median(py))
            rx_raw = float(np.quantile(np.abs(px - cx), 0.80)) + w * 0.030
            ry_raw = float(np.quantile(np.abs(py - cy), 0.80)) + h * 0.026
            rx = float(np.clip(rx_raw, w * 0.070, w * 0.120))
            ry = float(np.clip(ry_raw, h * 0.045, h * 0.085))
        ellipses.append({"name": name, "cx": cx, "cy": cy, "rx": rx, "ry": ry, "w": w, "h": h})
    return ellipses, (w, h)


def texture_eye_weight(obj_path):
    obj_path = Path(obj_path)
    texture_path = obj_path.with_suffix(".jpg")
    vertex_uv, valid = parse_vertex_uv(obj_path)
    ellipses, (w, h) = detect_eye_ellipses(texture_path)
    px = vertex_uv[:, 0] * (w - 1)
    py = (1.0 - vertex_uv[:, 1]) * (h - 1)
    soft = np.zeros(len(vertex_uv), dtype=np.float64)
    hard = np.zeros(len(vertex_uv), dtype=bool)
    for e in ellipses:
        d = ((px - e["cx"]) / e["rx"]) ** 2 + ((py - e["cy"]) / e["ry"]) ** 2
        hard |= valid & (d <= 1.0)
        t = np.clip((d - 1.0) / 1.15, 0.0, 1.0)
        soft = np.maximum(soft, 1.0 - (t * t * (3.0 - 2.0 * t)))
    soft[~valid] = 0.0
    fit_weight = np.clip(1.0 - soft, 0.0, 1.0)
    return fit_weight, hard, ellipses


def uv_ellipse(px, py, cx, cy, rx, ry):
    return ((px - cx) / max(rx, 1e-8)) ** 2 + ((py - cy) / max(ry, 1e-8)) ** 2 <= 1.0


def apply_nose_midface_weight(obj_path, vertices, fit_weight):
    x, y, _ = canonical_coords(vertices)
    vertex_uv, valid_uv = parse_vertex_uv(obj_path)
    texture_path = Path(obj_path).with_suffix(".jpg")
    image = Image.open(texture_path)
    w, h = image.size
    px = vertex_uv[:, 0] * (w - 1)
    py = (1.0 - vertex_uv[:, 1]) * (h - 1)

    # Source-side UV regions. This follows the same stable texture domain as
    # the eye/orbit mask and avoids case-dependent 3D depth boxes pulling the
    # nose weight toward cheeks, mouth, or one side of a yawed reconstruction.
    nose_bridge = valid_uv & uv_ellipse(px, py, 0.500 * w, 0.435 * h, 0.048 * w, 0.105 * h)
    nose_dorsum = valid_uv & uv_ellipse(px, py, 0.500 * w, 0.505 * h, 0.066 * w, 0.125 * h)
    nose_tip = valid_uv & uv_ellipse(px, py, 0.500 * w, 0.585 * h, 0.082 * w, 0.078 * h)
    alar = valid_uv & uv_ellipse(px, py, 0.500 * w, 0.595 * h, 0.105 * w, 0.058 * h)
    philtrum = valid_uv & uv_ellipse(px, py, 0.500 * w, 0.665 * h, 0.075 * w, 0.035 * h)
    mouth = valid_uv & uv_ellipse(px, py, 0.500 * w, 0.725 * h, 0.205 * w, 0.070 * h)
    lower_lip = valid_uv & uv_ellipse(px, py, 0.500 * w, 0.760 * h, 0.170 * w, 0.052 * h)
    midface = valid_uv & uv_ellipse(px, py, 0.500 * w, 0.555 * h, 0.215 * w, 0.155 * h) & (py < 0.655 * h)
    outer = (np.abs(x) > 0.430) | (y > 0.775) | (y < 0.160)

    region_weight = np.ones(len(vertices), dtype=np.float64)
    region_weight[midface] *= 1.20
    region_weight[nose_bridge | nose_dorsum] *= 2.05
    region_weight[nose_tip | alar] *= 2.35
    region_weight[philtrum] *= 1.04
    region_weight[mouth | lower_lip] *= 0.42
    region_weight[outer] *= 0.62

    weighted = fit_weight * region_weight
    weighted = np.clip(weighted, 0.0, 3.0)
    masks = {
        "midface": midface,
        "nose_bridge": nose_bridge,
        "nose_dorsum": nose_dorsum,
        "nose_tip": nose_tip,
        "alar": alar,
        "philtrum": philtrum,
        "mouth_downweighted": mouth | lower_lip,
        "outer_downweighted": outer,
    }
    return weighted, masks


def source_nose_anchor_distance(src_aligned, region_masks, target_anchor):
    if region_masks is None or target_anchor is None:
        return 0.0
    nose_mask = (
        region_masks.get("nose_tip", False)
        | region_masks.get("alar", False)
        | region_masks.get("nose_dorsum", False)
        | region_masks.get("nose_bridge", False)
    )
    if isinstance(nose_mask, bool) or int(nose_mask.sum()) < 80:
        return 0.0
    nose = src_aligned[nose_mask]
    nose_center = np.median(nose, axis=0)
    return float(np.linalg.norm(nose_center - target_anchor))


def source_mouth_to_nose_guard(src_aligned, region_masks, target_anchor):
    if region_masks is None or target_anchor is None:
        return 0.0
    mouth_mask = region_masks.get("mouth_downweighted", False)
    if isinstance(mouth_mask, bool) or int(mouth_mask.sum()) < 80:
        return 0.0
    mouth = src_aligned[mouth_mask]
    mouth_center = np.median(mouth, axis=0)
    d = float(np.linalg.norm(mouth_center - target_anchor))
    return max(0.0, 0.055 - d)


def anchor_penalty_from_transform(src, scale, rot, trans, region_masks, target_anchor, weight=0.45):
    if region_masks is None or target_anchor is None:
        return 0.0
    nose_mask = (
        region_masks.get("nose_tip", False)
        | region_masks.get("alar", False)
        | region_masks.get("nose_dorsum", False)
        | region_masks.get("nose_bridge", False)
    )
    if isinstance(nose_mask, bool) or int(nose_mask.sum()) < 80:
        return 0.0
    nose = transform(src[nose_mask], scale, rot, trans)
    mouth_guard = 0.0
    mouth_mask = region_masks.get("mouth_downweighted", False)
    if not isinstance(mouth_mask, bool) and int(mouth_mask.sum()) >= 80:
        mouth = transform(src[mouth_mask], scale, rot, trans)
        mouth_guard = source_mouth_to_nose_guard(mouth, {"mouth_downweighted": np.ones(len(mouth), dtype=bool)}, target_anchor)
    return float(weight * np.linalg.norm(np.median(nose, axis=0) - target_anchor) + 0.35 * mouth_guard)


def sample_idx(n, max_n, seed):
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, max_n, replace=False)


def transform(points, scale, rot, trans):
    return scale * (points @ rot.T) + trans


def axis_angle(axis, angle_rad):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = axis
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    k = 1.0 - c
    return np.array([
        [c + x * x * k, x * y * k - z * s, x * z * k + y * s],
        [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
        [z * x * k - y * s, z * y * k + x * s, c + z * z * k],
    ], dtype=np.float64)


def umeyama_rigid(src_moved, dst, weights=None):
    a = src_moved
    b = dst
    if weights is None:
        weights = np.ones(len(a), dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1e-12)
    ac = np.sum(a * weights[:, None], axis=0)
    bc = np.sum(b * weights[:, None], axis=0)
    h = (a - ac).T @ ((b - bc) * weights[:, None])
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    t = bc - r @ ac
    return r, t


def umeyama_similarity(src_moved, dst, weights=None):
    a = src_moved
    b = dst
    if weights is None:
        weights = np.ones(len(a), dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1e-12)
    ac = np.sum(a * weights[:, None], axis=0)
    bc = np.sum(b * weights[:, None], axis=0)
    aa = a - ac
    bb = b - bc
    h = aa.T @ (bb * weights[:, None])
    u, s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        s[-1] *= -1
        r = vt.T @ u.T
    var_a = float(np.sum(weights * np.sum(aa * aa, axis=1)))
    ds = float(np.sum(s) / max(var_a, 1e-12))
    ds = float(np.clip(ds, 0.985, 1.015))
    t = bc - ds * (r @ ac)
    return ds, r, t


def weighted_quantile(values, weights, q):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(values) == 0:
        return float("nan")
    weights = np.maximum(weights, 0.0)
    if float(weights.sum()) <= 1e-12:
        return float(np.quantile(values, q))
    order = np.argsort(values)
    sv = values[order]
    sw = weights[order]
    cdf = np.cumsum(sw) / float(np.sum(sw))
    return float(np.interp(q, cdf, sv))


def refine_rigid_fixed_scale(
    src,
    target_full,
    fit_weight,
    scale,
    rot,
    trans,
    iterations=70,
    region_masks=None,
    nose_anchor=None,
):
    fit_idx = np.flatnonzero(fit_weight > 0.08)
    fit_idx = fit_idx[sample_idx(len(fit_idx), 24000, 31)]
    tree = cKDTree(target_full)
    history = []
    best = (float("inf"), scale, rot.copy(), trans.copy())
    trim_schedule = np.linspace(0.68, 0.90, iterations)
    for it in range(iterations):
        moved = transform(src[fit_idx], scale, rot, trans)
        d, nn = tree.query(moved, k=1, workers=-1)
        keep = d <= np.quantile(d, float(trim_schedule[it]))
        if int(keep.sum()) < 200:
            keep = d <= np.quantile(d, 0.92)
        w = fit_weight[fit_idx][keep].copy()
        w *= np.clip(1.0 - d[keep] / max(float(np.quantile(d, 0.94)), 1e-8), 0.03, 1.0)
        r_delta, t_delta = umeyama_rigid(moved[keep], target_full[nn[keep]], w)
        rot = r_delta @ rot
        trans = r_delta @ trans + t_delta
        moved2 = transform(src[fit_idx], scale, rot, trans)
        d2, _ = tree.query(moved2, k=1, workers=-1)
        score_w = fit_weight[fit_idx]
        score = float(
            weighted_quantile(d2, score_w, 0.50)
            + 0.30 * weighted_quantile(d2, score_w, 0.90)
            + anchor_penalty_from_transform(src, scale, rot, trans, region_masks, nose_anchor, weight=0.45)
        )
        history.append(score)
        if score < best[0]:
            best = (score, scale, rot.copy(), trans.copy())
    return best[1], best[2], best[3], history


def refine_similarity_adaptive_scale(src, target_full, fit_weight, scale, rot, trans, iterations=55):
    fit_idx = np.flatnonzero(fit_weight > 0.08)
    fit_idx = fit_idx[sample_idx(len(fit_idx), 26000, 81)]
    tree = cKDTree(target_full)
    scale0 = float(scale)
    lower = scale0 * 0.78
    upper = scale0 * 1.22
    history = []
    best = (float("inf"), scale, rot.copy(), trans.copy())
    trim_schedule = np.linspace(0.66, 0.88, iterations)
    for it in range(iterations):
        moved = transform(src[fit_idx], scale, rot, trans)
        d, nn = tree.query(moved, k=1, workers=-1)
        keep = d <= np.quantile(d, float(trim_schedule[it]))
        if int(keep.sum()) < 250:
            keep = d <= np.quantile(d, 0.92)
        w = fit_weight[fit_idx][keep].copy()
        w *= np.clip(1.0 - d[keep] / max(float(np.quantile(d, 0.94)), 1e-8), 0.04, 1.0)
        ds, r_delta, t_delta = umeyama_similarity(moved[keep], target_full[nn[keep]], w)
        new_scale = float(np.clip(ds * scale, lower, upper))
        applied_ds = new_scale / max(scale, 1e-12)
        rot = r_delta @ rot
        trans = applied_ds * (r_delta @ trans) + t_delta
        scale = new_scale
        moved2 = transform(src[fit_idx], scale, rot, trans)
        d2, _ = tree.query(moved2, k=1, workers=-1)
        scale_penalty = 0.010 * abs(np.log(scale / max(scale0, 1e-12)))
        score_w = fit_weight[fit_idx]
        score = float(
            weighted_quantile(d2, score_w, 0.50)
            + 0.30 * weighted_quantile(d2, score_w, 0.90)
            + scale_penalty
        )
        history.append({
            "score": score,
            "scale": float(scale),
            "median": weighted_quantile(d2, score_w, 0.50),
            "p90": weighted_quantile(d2, score_w, 0.90),
        })
        if score < best[0]:
            best = (score, scale, rot.copy(), trans.copy())
    return best[1], best[2], best[3], history


def translation_only_polish(src, target_full, fit_weight, scale, rot, trans, iterations=20):
    fit_idx = np.flatnonzero(fit_weight > 0.35)
    fit_idx = fit_idx[sample_idx(len(fit_idx), 26000, 71)]
    tree = cKDTree(target_full)
    for _ in range(iterations):
        moved = transform(src[fit_idx], scale, rot, trans)
        d, nn = tree.query(moved, k=1, workers=-1)
        keep = d <= np.quantile(d, 0.72)
        if int(keep.sum()) < 200:
            break
        w = fit_weight[fit_idx][keep]
        delta = target_full[nn[keep]] - moved[keep]
        step = np.average(delta, axis=0, weights=w)
        n = float(np.linalg.norm(step))
        if n < 1e-6:
            break
        if n > 0.010:
            step *= 0.010 / n
        trans = trans + step
    return scale, rot, trans


def scale_first_then_rigid_icp(
    src,
    target_full,
    fit_weight,
    scale,
    rot,
    trans,
    region_masks=None,
    nose_anchor=None,
    rounds=7,
):
    """Try scale first, then run fixed-scale rigid ICP for each candidate.

    This avoids resizing an already registered mesh. Each outer round proposes
    scale values, re-centers the source with that scale, runs rigid ICP with
    the scale fixed, then chooses the best rigid result.
    """
    fit_idx = np.flatnonzero(fit_weight > 0.35)
    fit_idx = fit_idx[sample_idx(len(fit_idx), 26000, 91)]
    tree = cKDTree(target_full)
    current_scale = float(scale)
    current_rot = rot.copy()
    current_trans = trans.copy()
    history = []
    span = 0.14
    original_scale = float(scale)
    lower = original_scale * 0.60
    upper = original_scale * 1.18

    for round_i in range(1, rounds + 1):
        current = transform(src[fit_idx], current_scale, current_rot, current_trans)
        d, nn = tree.query(current, k=1, workers=-1)
        keep = d <= np.quantile(d, 0.72)
        if int(keep.sum()) < 300:
            keep = d <= np.quantile(d, 0.88)
        w = fit_weight[fit_idx][keep]
        dst_center = np.average(target_full[nn[keep]], axis=0, weights=w)
        src_center_rot = np.average(src[fit_idx][keep] @ current_rot.T, axis=0, weights=w)

        best = None
        scale_values = np.unique(np.clip(current_scale * (1.0 + np.linspace(-span, span, 11)), lower, upper).round(9))
        for candidate_scale in scale_values:
            candidate_trans = dst_center - float(candidate_scale) * src_center_rot
            s, r, t, rigid_hist = refine_rigid_fixed_scale(
                src,
                target_full,
                fit_weight,
                float(candidate_scale),
                current_rot,
                candidate_trans,
                iterations=45,
                region_masks=region_masks,
                nose_anchor=nose_anchor,
            )
            aligned = transform(src, s, r, t)
            m = metrics(aligned, target_full, fit_weight, region_masks=region_masks, nose_anchor=nose_anchor)
            score = m["selection_score"]
            row = {
                "round": round_i,
                "candidate_scale": float(candidate_scale),
                "result_scale": float(s),
                "score": float(score),
                "scale_penalty": 0.0,
                **m,
            }
            history.append(row)
            if best is None or score < best[0]:
                best = (score, s, r, t, m, rigid_hist)
        _, current_scale, current_rot, current_trans, best_metric, _ = best
        best_pos = int(np.where(scale_values == np.round(current_scale, 9))[0][0]) if np.round(current_scale, 9) in scale_values else -1
        edge = best_pos in (0, len(scale_values) - 1)
        if edge:
            span *= 0.72
        else:
            span *= 0.45
        span = max(span, 0.012)

    return current_scale, current_rot, current_trans, history


def adaptive_scale_prefit_only(src, target_full, fit_weight, scale, rot, trans, region_masks=None, nose_anchor=None, rounds=6):
    """Coarse-to-fine scale search; translation recenter only, no ICP update."""
    fit_idx = np.flatnonzero(fit_weight > 0.35)
    fit_idx = fit_idx[sample_idx(len(fit_idx), 26000, 101)]
    tree = cKDTree(target_full)
    moved = transform(src[fit_idx], scale, rot, trans)
    d, nn = tree.query(moved, k=1, workers=-1)
    keep = d <= np.quantile(d, 0.72)
    if int(keep.sum()) < 300:
        keep = d <= np.quantile(d, 0.88)
    w = fit_weight[fit_idx][keep]
    dst_center = np.average(target_full[nn[keep]], axis=0, weights=w)
    src_center_rot = np.average(src[fit_idx][keep] @ rot.T, axis=0, weights=w)

    original_scale = float(scale)
    current_scale = float(scale)
    current_trans = trans.copy()
    history = []
    span = 0.16
    lower = original_scale * 0.60
    upper = original_scale * 1.14
    best_global = None
    for round_i in range(1, rounds + 1):
        round_best = None
        scale_values = np.unique(np.clip(current_scale * (1.0 + np.linspace(-span, span, 11)), lower, upper).round(9))
        for candidate_scale in scale_values:
            candidate_trans = dst_center - float(candidate_scale) * src_center_rot
            aligned = transform(src, float(candidate_scale), rot, candidate_trans)
            m = metrics(aligned, target_full, fit_weight, region_masks=region_masks, nose_anchor=nose_anchor)
            score = m["selection_score"]
            row = {
                "round": round_i,
                "scale": float(candidate_scale),
                "score": float(score),
                "scale_penalty": 0.0,
                **m,
            }
            history.append(row)
            if round_best is None or score < round_best[0]:
                round_best = (score, float(candidate_scale), rot.copy(), candidate_trans, m, row)
            if best_global is None or score < best_global[0]:
                best_global = (score, float(candidate_scale), rot.copy(), candidate_trans, m, row)
        current_scale = round_best[1]
        current_trans = round_best[3]
        best_pos = int(np.where(scale_values == np.round(current_scale, 9))[0][0]) if np.round(current_scale, 9) in scale_values else -1
        edge = best_pos in (0, len(scale_values) - 1)
        if edge:
            span *= 0.72
        else:
            span *= 0.42
        span = max(span, 0.010)
    return best_global[1], best_global[2], best_global[3], history


def final_small_pose_search(src, target_full, fit_weight, scale, rot, trans, region_masks=None, nose_anchor=None):
    """Local pose/scale polish around the current similarity solution.

    This is intentionally narrow: it searches scale, rotation, and translation
    near the already-good solution, then runs short fixed-scale ICP only for the
    best local candidate. It is meant for the rigid stage before non-rigid
    registration, not for large recovery from a bad initialization.
    """
    fit_idx = np.flatnonzero(fit_weight > 0.35)
    fit_idx = fit_idx[sample_idx(len(fit_idx), 26000, 131)]
    src_center = np.average(src[fit_idx], axis=0, weights=fit_weight[fit_idx])

    current_scale = float(scale)
    current_rot = rot.copy()
    current_trans = trans.copy()
    best_aligned = transform(src, current_scale, current_rot, current_trans)
    best_metric = metrics(best_aligned, target_full, fit_weight, region_masks=region_masks, nose_anchor=nose_anchor)
    history = [{
        "round": 0,
        "move": "initial_similarity",
        "scale": float(current_scale),
        "score": float(best_metric["selection_score"]),
        **best_metric,
    }]

    axes = [
        ("rx", np.array([1.0, 0.0, 0.0])),
        ("ry", np.array([0.0, 1.0, 0.0])),
        ("rz", np.array([0.0, 0.0, 1.0])),
    ]
    spans = [
        (0.026, np.deg2rad(3.0), 0.012),
        (0.014, np.deg2rad(1.8), 0.007),
        (0.008, np.deg2rad(1.0), 0.004),
        (0.004, np.deg2rad(0.55), 0.002),
    ]

    for round_i, (scale_span, angle_span, trans_span) in enumerate(spans, start=1):
        target_center = np.average(transform(src[fit_idx], current_scale, current_rot, current_trans), axis=0, weights=fit_weight[fit_idx])
        candidates = [("keep", current_scale, current_rot.copy(), current_trans.copy())]
        for f in (-scale_span, -0.5 * scale_span, 0.5 * scale_span, scale_span):
            s = current_scale * (1.0 + f)
            t = target_center - s * (src_center @ current_rot.T)
            candidates.append((f"scale_{f:+.4f}", s, current_rot.copy(), t))
        for name, axis in axes:
            for a in (-angle_span, -0.5 * angle_span, 0.5 * angle_span, angle_span):
                r = axis_angle(axis, a) @ current_rot
                t = target_center - current_scale * (src_center @ r.T)
                candidates.append((f"{name}_{np.rad2deg(a):+.2f}", current_scale, r, t))
        for axis_name, delta in [
            ("tx", np.array([trans_span, 0.0, 0.0])),
            ("ty", np.array([0.0, trans_span, 0.0])),
            ("tz", np.array([0.0, 0.0, trans_span])),
        ]:
            candidates.append((f"{axis_name}_plus", current_scale, current_rot.copy(), current_trans + delta))
            candidates.append((f"{axis_name}_minus", current_scale, current_rot.copy(), current_trans - delta))

        round_best = None
        for move, cand_scale, cand_rot, cand_trans in candidates:
            s, r, t, rigid_hist = refine_rigid_fixed_scale(
                src,
                target_full,
                fit_weight,
                float(cand_scale),
                cand_rot,
                cand_trans,
                iterations=18,
                region_masks=region_masks,
                nose_anchor=nose_anchor,
            )
            aligned = transform(src, s, r, t)
            m = metrics(aligned, target_full, fit_weight, region_masks=region_masks, nose_anchor=nose_anchor)
            row = {
                "round": round_i,
                "move": move,
                "scale": float(s),
                "score": float(m["selection_score"]),
                "rigid_last": float(rigid_hist[-1]) if rigid_hist else None,
                **m,
            }
            history.append(row)
            if round_best is None or row["score"] < round_best[0]:
                round_best = (row["score"], s, r, t, m)
        if round_best is not None and round_best[0] < best_metric["selection_score"]:
            _, current_scale, current_rot, current_trans, best_metric = round_best
        else:
            history.append({
                "round": round_i,
                "move": "no_accept",
                "scale": float(current_scale),
                "score": float(best_metric["selection_score"]),
                **best_metric,
            })

    return current_scale, current_rot, current_trans, history


def initial_candidates(src, target_full, fit_weight):
    fit_mask = fit_weight > 0.35
    src_fit = src[fit_mask]
    # HRN native source -> FaceScape registration frame. This is the same
    # axis convention used by the old CT registration adapter, but no target
    # vertices are cropped here.
    base_rots = []
    base = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                rot = np.diag([sx, sy, sz]) @ base
                # Reflections can give deceptively low one-way NN distances on
                # roughly symmetric faces while leaving the recovered face
                # anatomically flipped. Keep only proper rotations.
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
    eval_idx = sample_idx(len(src_fit), 12000, 41)
    src_fit_weights = fit_weight[fit_mask]
    target_eval = target_full[sample_idx(len(target_full), 180000, 42)]
    tree = cKDTree(target_eval)
    for rot in base_rots:
        mapped_ext = np.maximum(np.abs(rot) @ src_ext, 1e-8)
        scale0 = float(np.median([front_ext[0] / mapped_ext[0], front_ext[2] / mapped_ext[2]]))
        for scale in scale0 * np.linspace(0.70, 1.28, 15):
            moved_all = transform(src, float(scale), rot, np.zeros(3))
            moved_fit = transform(src_fit[eval_idx], float(scale), rot, np.zeros(3))
            trans = target_center - np.median(moved_all[fit_mask], axis=0)
            trans[1] = target_front_y - np.max(moved_all[fit_mask, 1])
            d, _ = tree.query(moved_fit + trans, k=1, workers=-1)
            ew = src_fit_weights[eval_idx]
            score = float(
                weighted_quantile(d, ew, 0.50)
                + 0.20 * weighted_quantile(d, ew, 0.90)
                + 0.08 * weighted_quantile(d, ew, 0.98)
            )
            candidates.append((score, float(scale), rot.copy(), trans.copy()))
    return sorted(candidates, key=lambda x: x[0])[:8]


def metrics(src_aligned, target_full, fit_weight, region_masks=None, nose_anchor=None):
    idx = np.flatnonzero(fit_weight > 0.08)
    idx = idx[sample_idx(len(idx), 30000, 51)]
    tree = cKDTree(target_full)
    d, _ = tree.query(src_aligned[idx], k=1, workers=-1)
    w = fit_weight[idx]
    out = {
        "mean": float(np.average(d, weights=np.maximum(w, 1e-8))),
        "median": weighted_quantile(d, w, 0.50),
        "p90": weighted_quantile(d, w, 0.90),
        "p95": weighted_quantile(d, w, 0.95),
        "unweighted_median": float(np.median(d)),
        "unweighted_p90": float(np.quantile(d, 0.90)),
    }
    nose_idx = idx[fit_weight[idx] > 1.75]
    if len(nose_idx) >= 80:
        nose_idx = nose_idx[sample_idx(len(nose_idx), 7000, 52)]
        nd, _ = tree.query(src_aligned[nose_idx], k=1, workers=-1)
        nw = fit_weight[nose_idx]
        out["nose_weighted_median"] = weighted_quantile(nd, nw, 0.50)
        out["nose_weighted_p90"] = weighted_quantile(nd, nw, 0.90)
    else:
        out["nose_weighted_median"] = out["median"]
        out["nose_weighted_p90"] = out["p90"]
    out["source_nose_anchor_distance"] = source_nose_anchor_distance(src_aligned, region_masks, nose_anchor)
    out["source_mouth_to_nose_guard"] = source_mouth_to_nose_guard(src_aligned, region_masks, nose_anchor)
    out["selection_score"] = float(
        out["median"]
        + 0.22 * out["p90"]
        + 0.32 * out["nose_weighted_median"]
        + 0.08 * out["nose_weighted_p90"]
        + 0.45 * out["source_nose_anchor_distance"]
        + 0.35 * out["source_mouth_to_nose_guard"]
    )
    return out


def region_metrics(src_aligned, target_full, masks, fit_weight):
    out = {}
    for name, mask in masks.items():
        local_weight = fit_weight * mask.astype(np.float64)
        if int((local_weight > 0.08).sum()) < 80:
            continue
        out[name] = metrics(src_aligned, target_full, local_weight)
    return out


def write_obj_like(src_obj, out_obj, vertices):
    out_obj.parent.mkdir(parents=True, exist_ok=True)
    src_obj = Path(src_obj)
    lines = []
    vi = 0
    for line in src_obj.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("v "):
            v = vertices[vi]
            lines.append(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}")
            vi += 1
        else:
            lines.append(line)
    out_obj.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for suffix in [".mtl", ".jpg", ".png", ".jpeg"]:
        side = src_obj.with_suffix(suffix)
        if side.exists():
            shutil.copy2(side, out_obj.with_suffix(suffix))


def equal_3d(ax, pts):
    c = np.median(pts, axis=0)
    r = np.percentile(np.linalg.norm(pts - c, axis=1), 96)
    r = max(float(r), 1e-8)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def plot_review(case_dir, case, source, target, rigid_aligned, scale_aligned, similarity_aligned, fit_weight, eye_mask, region_masks, rigid_metric, scale_metric, similarity_metric):
    src_s = source[sample_idx(len(source), 35000, 61)]
    tgt_s = target[sample_idx(len(target), 140000, 62)]
    rig_s = rigid_aligned[sample_idx(len(rigid_aligned), 35000, 63)]
    scl_s = scale_aligned[sample_idx(len(scale_aligned), 35000, 67)]
    sim_s = similarity_aligned[sample_idx(len(similarity_aligned), 35000, 66)]
    eye_s = source[eye_mask]
    eye_s = eye_s[sample_idx(len(eye_s), 5000, 65)] if len(eye_s) else eye_s
    nose_mask = region_masks["nose_bridge"] | region_masks["nose_dorsum"] | region_masks["nose_tip"] | region_masks["alar"] | region_masks["philtrum"]
    nose_s = source[nose_mask & (fit_weight > 0.08)]
    nose_s = nose_s[sample_idx(len(nose_s), 6500, 68)] if len(nose_s) else nose_s
    fig = plt.figure(figsize=(22, 16), dpi=170)
    front = (8, -88)
    side = (8, -8)
    oblique = (18, -48)
    panels = [
        ("source HRN + source eye/orbit mask", src_s, "#2878b5", eye_s, "#00a651", 75, -88),
        ("source nose/midface weighted vertices", src_s, "#f3a6a0", nose_s, "#f0c419", 75, -88),
        ("target full only, front", tgt_s, "#111111", None, None, *front),
        ("target full only, side", tgt_s, "#111111", None, None, *side),
        ("rigid ICP overlay, front", tgt_s, "#111111", rig_s, "#f08c00", *front),
        ("adaptive scale overlay, front", tgt_s, "#111111", scl_s, "#7b2cbf", *front),
        ("similarity ICP overlay, front", tgt_s, "#111111", sim_s, "#e43d30", *front),
        ("similarity ICP overlay, side", tgt_s, "#111111", sim_s, "#e43d30", *side),
        ("rigid ICP overlay, oblique", tgt_s, "#111111", rig_s, "#f08c00", *oblique),
        ("adaptive scale overlay, oblique", tgt_s, "#111111", scl_s, "#7b2cbf", *oblique),
        ("similarity ICP overlay, oblique", tgt_s, "#111111", sim_s, "#e43d30", *oblique),
        ("similarity source only, oblique", sim_s, "#e43d30", None, None, *oblique),
    ]
    for i, (title, pts, color, overlay, overlay_color, elev, azim) in enumerate(panels, 1):
        ax = fig.add_subplot(3, 4, i, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.0, c=color, alpha=0.16 if overlay is not None else 0.62, linewidths=0)
        all_pts = pts
        if overlay is not None and len(overlay):
            ax.scatter(overlay[:, 0], overlay[:, 1], overlay[:, 2], s=1.7, c=overlay_color, alpha=0.80, linewidths=0)
            all_pts = np.vstack([pts, overlay])
        equal_3d(ax, all_pts)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle(
        f"{case} adaptive similarity ICP | rigid median={rigid_metric['median']:.4f}, "
        f"scale-only median={scale_metric['median']:.4f}, similarity median={similarity_metric['median']:.4f}",
        fontsize=12,
    )
    fig.tight_layout()
    out = case_dir / f"{case}_adaptive_similarity_registration_review.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_scale_history(case_dir, case, adaptive_hist, similarity_hist):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=170)
    specs = [
        ("adaptive scale-only", adaptive_hist, "scale", "#7b2cbf"),
        ("similarity scale-first ICP", similarity_hist, "result_scale", "#e43d30"),
    ]
    for row_i, (name, hist, scale_key, color) in enumerate(specs):
        ax_score, ax_err, ax_prog = axes[row_i]
        if not hist:
            for ax in (ax_score, ax_err, ax_prog):
                ax.text(0.5, 0.5, "no history", ha="center", va="center")
                ax.set_axis_off()
            continue
        rounds = sorted({int(h.get("round", 1)) for h in hist})
        for r in rounds:
            rows = [h for h in hist if int(h.get("round", 1)) == r]
            rows = sorted(rows, key=lambda h: float(h.get(scale_key, h.get("scale", 0.0))))
            xs = np.asarray([float(h.get(scale_key, h.get("scale", 0.0))) for h in rows])
            score = np.asarray([float(h.get("score", np.nan)) for h in rows])
            median = np.asarray([float(h.get("median", np.nan)) for h in rows])
            p90 = np.asarray([float(h.get("p90", np.nan)) for h in rows])
            nose = np.asarray([float(h.get("nose_weighted_median", np.nan)) for h in rows])
            alpha = 0.45 + 0.12 * r
            ax_score.plot(xs, score, marker="o", ms=3.2, lw=1.2, alpha=min(alpha, 0.95), label=f"round {r}")
            ax_err.plot(xs, median, marker="o", ms=2.8, lw=1.0, alpha=min(alpha, 0.95), label=f"median r{r}")
            ax_err.plot(xs, p90, marker=".", ms=2.5, lw=0.9, alpha=min(alpha, 0.55), linestyle="--", label=f"p90 r{r}")
            if np.isfinite(nose).any():
                ax_err.plot(xs, nose, marker="x", ms=3.0, lw=0.9, alpha=min(alpha, 0.80), linestyle=":", label=f"nose r{r}")
        best_rows = []
        for r in rounds:
            rows = [h for h in hist if int(h.get("round", 1)) == r]
            best_rows.append(min(rows, key=lambda h: float(h.get("score", np.inf))))
        bx = [int(h.get("round", 1)) for h in best_rows]
        bs = [float(h.get(scale_key, h.get("scale", 0.0))) for h in best_rows]
        bscore = [float(h.get("score", np.nan)) for h in best_rows]
        ax_prog.plot(bx, bs, marker="o", lw=1.8, color=color)
        ax_prog.set_xticks(bx)
        ax_prog.set_ylabel("selected scale")
        ax_prog2 = ax_prog.twinx()
        ax_prog2.plot(bx, bscore, marker="s", lw=1.2, color="0.25", alpha=0.65)
        ax_prog2.set_ylabel("selected score")
        ax_score.set_title(f"{name}: scale vs selection score")
        ax_err.set_title(f"{name}: scale vs errors")
        ax_prog.set_title(f"{name}: selected scale by round")
        ax_score.set_xlabel("scale")
        ax_score.set_ylabel("selection score")
        ax_err.set_xlabel("scale")
        ax_err.set_ylabel("distance")
        ax_score.grid(True, alpha=0.25)
        ax_err.grid(True, alpha=0.25)
        ax_prog.grid(True, alpha=0.25)
        ax_score.legend(fontsize=7)
        ax_err.legend(fontsize=6, ncol=2)
    fig.suptitle(f"{case} scale search curves", fontsize=13)
    fig.tight_layout()
    out = case_dir / f"{case}_scale_search_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def selected_by_round(history, scale_key):
    selected = []
    for round_i in sorted({int(row["round"]) for row in history}):
        rows = [row for row in history if int(row["round"]) == round_i]
        best = min(rows, key=lambda row: float(row["selection_score"]))
        selected.append({
            "round": round_i,
            "scale": float(best[scale_key]),
            "selection_score": float(best["selection_score"]),
            "median": float(best["median"]),
            "p90": float(best["p90"]),
            "source_nose_anchor_distance": float(best.get("source_nose_anchor_distance", 0.0)),
            "source_mouth_to_nose_guard": float(best.get("source_mouth_to_nose_guard", 0.0)),
        })
    return selected


def process(pair_id, by_pair):
    row = by_pair[pair_id]
    target_row = by_pair[f"{int(row['subject']):03d}_18_eye_closed"]
    src_obj = hrn_obj(pair_id)
    _, src_v, _ = load_mesh(src_obj)
    target_mesh = Path(target_row["mesh"])
    _, target_world, _ = load_mesh(target_mesh)
    cam_json = target_mesh.parent / "selected_camera.json"
    target_reg = target_registration_frame(target_world, cam_json)
    nose_anchor = target_nose_anchor(target_reg)
    fit_weight_base, eye_mask, eye_ellipses = texture_eye_weight(src_obj)
    fit_weight, region_masks = apply_nose_midface_weight(src_obj, src_v, fit_weight_base)

    best = None
    tried = []
    for score0, scale0, rot0, trans0 in initial_candidates(src_v, target_reg, fit_weight):
        scale, rot, trans, hist = refine_rigid_fixed_scale(
            src_v,
            target_reg,
            fit_weight,
            scale0,
            rot0,
            trans0,
            region_masks=region_masks,
            nose_anchor=nose_anchor,
        )
        scale, rot, trans = translation_only_polish(src_v, target_reg, fit_weight, scale, rot, trans)
        aligned_reg = transform(src_v, scale, rot, trans)
        m = metrics(aligned_reg, target_reg, fit_weight, region_masks=region_masks, nose_anchor=nose_anchor)
        score = m["selection_score"]
        tried.append({"initial_score": float(score0), "scale0": float(scale0), "scale": float(scale), **m})
        if best is None or score < best[0]:
            best = (score, scale, rot, trans, aligned_reg, m, hist)

    _, rigid_scale, rigid_rot, rigid_trans, rigid_aligned_reg, rigid_m, rigid_hist = best
    scale_only_scale, scale_only_rot, scale_only_trans, scale_only_hist = adaptive_scale_prefit_only(
        src_v,
        target_reg,
        fit_weight,
        rigid_scale,
        rigid_rot,
        rigid_trans,
        region_masks=region_masks,
        nose_anchor=nose_anchor,
    )
    scale_only_aligned_reg = transform(src_v, scale_only_scale, scale_only_rot, scale_only_trans)
    scale_only_m = metrics(scale_only_aligned_reg, target_reg, fit_weight, region_masks=region_masks, nose_anchor=nose_anchor)

    sim_scale, sim_rot, sim_trans, sim_hist = scale_first_then_rigid_icp(
        src_v,
        target_reg,
        fit_weight,
        rigid_scale,
        rigid_rot,
        rigid_trans,
        region_masks=region_masks,
        nose_anchor=nose_anchor,
    )
    similarity_aligned_reg = transform(src_v, sim_scale, sim_rot, sim_trans)
    similarity_pre_polish_m = metrics(similarity_aligned_reg, target_reg, fit_weight, region_masks=region_masks, nose_anchor=nose_anchor)
    final_scale, final_rot, final_trans, final_pose_hist = final_small_pose_search(
        src_v,
        target_reg,
        fit_weight,
        sim_scale,
        sim_rot,
        sim_trans,
        region_masks=region_masks,
        nose_anchor=nose_anchor,
    )
    similarity_aligned_reg = transform(src_v, final_scale, final_rot, final_trans)
    similarity_m = metrics(similarity_aligned_reg, target_reg, fit_weight, region_masks=region_masks, nose_anchor=nose_anchor)

    rigid_region_m = region_metrics(rigid_aligned_reg, target_reg, region_masks, fit_weight)
    scale_region_m = region_metrics(scale_only_aligned_reg, target_reg, region_masks, fit_weight)
    similarity_region_m = region_metrics(similarity_aligned_reg, target_reg, region_masks, fit_weight)

    rigid_aligned_world = inv_target_registration_frame(rigid_aligned_reg, cam_json)
    scale_only_aligned_world = inv_target_registration_frame(scale_only_aligned_reg, cam_json)
    similarity_aligned_world = inv_target_registration_frame(similarity_aligned_reg, cam_json)
    case_dir = OUT / pair_id
    rigid_out_reg = case_dir / f"{pair_id}_rigid_icp_regframe.obj"
    rigid_out_world = case_dir / f"{pair_id}_rigid_icp_world.obj"
    scale_out_reg = case_dir / f"{pair_id}_adaptive_scale_only_regframe.obj"
    scale_out_world = case_dir / f"{pair_id}_adaptive_scale_only_world.obj"
    sim_out_reg = case_dir / f"{pair_id}_similarity_icp_regframe.obj"
    sim_out_world = case_dir / f"{pair_id}_similarity_icp_world.obj"
    write_obj_like(src_obj, rigid_out_reg, rigid_aligned_reg)
    write_obj_like(src_obj, rigid_out_world, rigid_aligned_world)
    write_obj_like(src_obj, scale_out_reg, scale_only_aligned_reg)
    write_obj_like(src_obj, scale_out_world, scale_only_aligned_world)
    write_obj_like(src_obj, sim_out_reg, similarity_aligned_reg)
    write_obj_like(src_obj, sim_out_world, similarity_aligned_world)
    np.save(case_dir / f"{pair_id}_source_eye_orbit_mask.npy", eye_mask)
    np.save(case_dir / f"{pair_id}_source_fit_weight_with_eye_and_midface.npy", fit_weight)
    fig = plot_review(
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
    curve_fig = plot_scale_history(case_dir, pair_id, scale_only_hist, sim_hist)
    adaptive_selected = selected_by_round(scale_only_hist, "scale")
    similarity_selected = selected_by_round(sim_hist, "result_scale")
    report = {
        "case": pair_id,
        "method": "scale-first adaptive registration from rigid baseline: scale candidate first, fixed-scale rigid ICP second, source-to-target one-way, source texture/UV eye mask only, no target ROI",
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
        "adaptive_scale_only_selected_by_round": adaptive_selected,
        "similarity_history": sim_hist,
        "similarity_selected_by_round": similarity_selected,
        "final_small_pose_history": final_pose_hist,
        "similarity_history_last_raw_candidate": sim_hist[-1] if sim_hist else None,
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
    rows = read_manifest()
    by_pair = {r["pair_id"]: r for r in rows}
    reports = [process(case, by_pair) for case in CASES]
    (OUT / "summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(OUT, flush=True)


if __name__ == "__main__":
    main()


