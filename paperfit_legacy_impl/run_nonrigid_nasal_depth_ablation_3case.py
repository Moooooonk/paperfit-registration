#!/usr/bin/env python3
import csv
import os
import json
import os
import shutil
import os
from pathlib import Path

import cv2
import os
import matplotlib
import os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import os
import trimesh
import os
from PIL import Image, ImageDraw
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree


ROOT = Path(os.environ.get("PAPERFIT_ROOT", "/path/to/prepared/facescape_pipeline"))
MANIFEST = ROOT / "prepared_001_020_cv" / "facescape_frontal_pairs_manifest.csv"
RIGID_ROOT = ROOT / "research_scratch_surface_registration_3case_final_attempt_20260531"
OUT = ROOT / "research_nonrigid_nasal_depth_ablation_3case_20260602"
CASES = ["001_1_neutral", "001_10_dimpler", "002_1_neutral"]

SCHEDULES = {
    "S4_linear": [0.00, 0.33, 0.60, 0.82],
    "S6_current": [0.00, 0.22, 0.42, 0.60, 0.75, 0.86],
    "S6_tip_dense": [0.00, 0.30, 0.52, 0.68, 0.80, 0.90],
    "S7_tip_dense": [0.00, 0.25, 0.45, 0.60, 0.72, 0.83, 0.92],
}


def read_manifest():
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return {r["pair_id"]: r for r in csv.DictReader(f)}


def load_mesh(path):
    mesh = trimesh.load(str(path), process=False)
    return mesh, np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def target_registration_frame(vertices, camera_json):
    data = json.loads(Path(camera_json).read_text(encoding="utf-8"))
    rt = np.asarray(data["Rt"], dtype=np.float64)
    cam = vertices @ rt[:, :3].T + rt[:, 3]
    return np.column_stack([cam[:, 0], -cam[:, 2], -cam[:, 1]])


def canonical_coords(vertices):
    mn = vertices.min(axis=0)
    mx = vertices.max(axis=0)
    ext = np.maximum(mx - mn, 1e-8)
    x = (vertices[:, 0] - (mn[0] + mx[0]) * 0.5) / ext[0]
    y = (vertices[:, 1] - mn[1]) / ext[1]
    z = (vertices[:, 2] - mn[2]) / ext[2]
    return x, y, z


def parse_obj_with_uv(path):
    vertices, uvs, faces, face_uvs = [], [], [], []
    pairs = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("vt "):
                uvs.append([float(x) for x in line.split()[1:3]])
            elif line.startswith("f "):
                v_idx, vt_idx = [], []
                for part in line.split()[1:]:
                    bits = part.split("/")
                    vi = int(bits[0]) - 1
                    ti = int(bits[1]) - 1 if len(bits) > 1 and bits[1] else 0
                    v_idx.append(vi)
                    vt_idx.append(ti)
                    pairs.append((vi, ti))
                if len(v_idx) == 3:
                    faces.append(v_idx)
                    face_uvs.append(vt_idx)
                elif len(v_idx) > 3:
                    for i in range(1, len(v_idx) - 1):
                        faces.append([v_idx[0], v_idx[i], v_idx[i + 1]])
                        face_uvs.append([vt_idx[0], vt_idx[i], vt_idx[i + 1]])
    vertices = np.asarray(vertices, dtype=np.float64)
    uvs = np.asarray(uvs, dtype=np.float64)
    vertex_uv = np.full((len(vertices), 2), np.nan, dtype=np.float64)
    counts = np.zeros(len(vertices), dtype=np.float64)
    for vi, ti in pairs:
        if 0 <= vi < len(vertices) and 0 <= ti < len(uvs):
            if np.isnan(vertex_uv[vi, 0]):
                vertex_uv[vi] = 0.0
            vertex_uv[vi] += uvs[ti]
            counts[vi] += 1.0
    valid = counts > 0
    vertex_uv[valid] /= counts[valid, None]
    return (
        vertices,
        uvs,
        np.asarray(faces, dtype=np.int32),
        np.asarray(face_uvs, dtype=np.int32),
        vertex_uv,
        valid,
    )


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


def uv_ellipse(px, py, cx, cy, rx, ry):
    return ((px - cx) / max(rx, 1e-8)) ** 2 + ((py - cy) / max(ry, 1e-8)) ** 2 <= 1.0


def build_masks(obj_path, vertices):
    _, _, _, _, vertex_uv, valid_uv = parse_obj_with_uv(obj_path)
    texture_path = Path(obj_path).with_suffix(".jpg")
    image = Image.open(texture_path)
    w, h = image.size
    px = vertex_uv[:, 0] * (w - 1)
    py = (1.0 - vertex_uv[:, 1]) * (h - 1)

    ellipses, _ = detect_eye_ellipses(texture_path)
    eye_soft = np.zeros(len(vertices), dtype=np.float64)
    eye_hard = np.zeros(len(vertices), dtype=bool)
    for e in ellipses:
        d = ((px - e["cx"]) / e["rx"]) ** 2 + ((py - e["cy"]) / e["ry"]) ** 2
        eye_hard |= valid_uv & (d <= 1.0)
        t = np.clip((d - 1.0) / 1.15, 0.0, 1.0)
        eye_soft = np.maximum(eye_soft, 1.0 - (t * t * (3.0 - 2.0 * t)))
    eye_soft[~valid_uv] = 0.0
    fit_weight = np.clip(1.0 - eye_soft, 0.0, 1.0)

    x, y, _ = canonical_coords(vertices)
    nose_bridge = valid_uv & uv_ellipse(px, py, 0.500 * w, 0.435 * h, 0.048 * w, 0.105 * h)
    nose_dorsum = valid_uv & uv_ellipse(px, py, 0.500 * w, 0.505 * h, 0.066 * w, 0.125 * h)
    nose_tip = valid_uv & uv_ellipse(px, py, 0.500 * w, 0.585 * h, 0.082 * w, 0.078 * h)
    alar = valid_uv & uv_ellipse(px, py, 0.500 * w, 0.595 * h, 0.105 * w, 0.058 * h)
    philtrum = valid_uv & uv_ellipse(px, py, 0.500 * w, 0.665 * h, 0.075 * w, 0.035 * h)
    mouth = valid_uv & uv_ellipse(px, py, 0.500 * w, 0.725 * h, 0.205 * w, 0.070 * h)
    lower_lip = valid_uv & uv_ellipse(px, py, 0.500 * w, 0.760 * h, 0.170 * w, 0.052 * h)
    midface = valid_uv & uv_ellipse(px, py, 0.500 * w, 0.555 * h, 0.215 * w, 0.155 * h) & (py < 0.655 * h)
    outer = (np.abs(x) > 0.430) | (y > 0.775) | (y < 0.160)

    fit_weight[midface] *= 1.20
    fit_weight[nose_bridge | nose_dorsum] *= 2.05
    fit_weight[nose_tip | alar] *= 2.35
    fit_weight[philtrum] *= 1.04
    fit_weight[mouth | lower_lip] *= 0.42
    fit_weight[outer] *= 0.62
    fit_weight = np.clip(fit_weight, 0.0, 3.0)

    full_no_eye = fit_weight > 0.08
    nose = (nose_bridge | nose_dorsum | nose_tip | alar) & full_no_eye
    masks = {
        "eye": eye_hard,
        "eye_soft": eye_soft > 0.08,
        "full_no_eye": full_no_eye,
        "midface": midface & full_no_eye,
        "nose": nose,
        "nasal_bridge": nose_bridge & full_no_eye,
        "nasal_dorsum": nose_dorsum & full_no_eye,
        "nose_tip": nose_tip & full_no_eye,
        "alar": alar & full_no_eye,
        "subnasal": philtrum & full_no_eye,
        "philtrum": philtrum & full_no_eye,
        "mouth": (mouth | lower_lip) & full_no_eye,
    }
    return fit_weight, masks, ellipses


def build_edges(faces):
    edges = set()
    for tri in faces:
        a, b, c = map(int, tri[:3])
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            edges.add((u, v))
    return np.asarray(sorted(edges), dtype=np.int64)


def solve(v_ref, edges, target_idx, target_pos, target_w, fixed_idx, fixed_pos, edge_w, fixed_w):
    n = len(v_ref)
    mat = lil_matrix((n, n), dtype=np.float64)
    rhs = np.zeros((n, 3), dtype=np.float64)
    for u, v in edges:
        d0 = v_ref[u] - v_ref[v]
        mat[u, u] += edge_w
        mat[v, v] += edge_w
        mat[u, v] -= edge_w
        mat[v, u] -= edge_w
        rhs[u] += edge_w * d0
        rhs[v] -= edge_w * d0
    diag_add = np.zeros(n, dtype=np.float64)
    diag_add[target_idx] += target_w
    rhs[target_idx] += target_w[:, None] * target_pos
    diag_add[fixed_idx] += fixed_w
    rhs[fixed_idx] += fixed_w * fixed_pos
    mat = mat.tocsr() + diags(diag_add + 1e-8, 0, shape=(n, n), format="csr")
    out = np.empty_like(v_ref)
    for dim in range(3):
        out[:, dim] = spsolve(mat, rhs[:, dim])
    return out


def deterministic_sample(points, n):
    if len(points) <= n:
        return points.copy()
    return points[np.linspace(0, len(points) - 1, n, dtype=np.int64)]


def sample_idx(n, max_n, seed):
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, max_n, replace=False)


def depth_contours(current, masks, thresholds):
    nose_seed = masks["nose"] | masks["midface"]
    movable = masks["full_no_eye"]
    if int(nose_seed.sum()) < 80:
        nose_seed = movable
    depth = -current[:, 1]
    vals = depth[nose_seed & movable]
    lo = float(np.quantile(vals, 0.08))
    hi = float(np.quantile(vals, 0.985))
    protrusion = (depth - lo) / max(hi - lo, 1e-8)
    neighborhood = (nose_seed | masks["nasal_bridge"] | masks["nasal_dorsum"] | masks["nose_tip"] | masks["alar"]) & movable
    return [(t, neighborhood & (protrusion >= t)) for t in thresholds], protrusion


def nn_metrics(vertices, target, mask):
    if int(mask.sum()) == 0:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None}
    tree = cKDTree(target)
    d, _ = tree.query(vertices[mask], k=1, workers=-1)
    return {
        "n": int(mask.sum()),
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "p90": float(np.quantile(d, 0.90)),
        "p95": float(np.quantile(d, 0.95)),
    }


def all_metrics(vertices, target_sample, masks):
    regions = ["full_no_eye", "midface", "nose", "nasal_bridge", "nasal_dorsum", "nose_tip", "alar", "subnasal", "philtrum", "mouth"]
    return {name: nn_metrics(vertices, target_sample, masks[name]) for name in regions}


def run_nonrigid(init_vertices, faces, target_sample, masks, thresholds):
    current = init_vertices.copy()
    start = init_vertices.copy()
    edges = build_edges(faces)
    tree = cKDTree(target_sample)
    fixed_idx = np.flatnonzero(masks["eye_soft"])
    if len(fixed_idx) < 50:
        fixed_idx = np.flatnonzero(masks["eye"])

    history = []
    contour_sets, protrusion = depth_contours(current, masks, thresholds)
    local_sets = contour_sets + [
        ("bridge", masks["nasal_bridge"]),
        ("dorsum", masks["nasal_dorsum"]),
        ("tip_alar", masks["nose_tip"] | masks["alar"]),
        ("subnasal", masks["subnasal"]),
        ("philtrum", masks["philtrum"]),
    ]
    for pass_i, (name, active_mask) in enumerate(local_sets, 1):
        target_idx = np.flatnonzero(active_mask & masks["full_no_eye"] & ~masks["eye_soft"])
        if len(target_idx) < 20:
            continue
        d, nn = tree.query(current[target_idx], k=1, workers=-1)
        delta = target_sample[nn] - current[target_idx]
        norm = np.linalg.norm(delta, axis=1)
        max_step = 0.018 if pass_i <= len(contour_sets) else 0.014
        step = np.minimum(1.0, max_step / np.maximum(norm, 1e-8))
        target = current[target_idx] + delta * step[:, None]
        weights = 18.0 + np.clip(d, 0.0, 0.20) * 85.0
        if name == "tip_alar" or (isinstance(name, float) and name >= 0.75):
            weights *= 1.25
        solved = solve(current, edges, target_idx, target, weights, fixed_idx, start[fixed_idx], edge_w=190.0, fixed_w=32000.0)
        current = 0.43 * solved + 0.57 * current
        current[fixed_idx] = start[fixed_idx]
        history.append({"phase": "nasal_depth", "pass": pass_i, "name": str(name), "active_vertices": int(len(target_idx)), "median_nn": float(np.median(d)), "p90_nn": float(np.quantile(d, 0.90))})

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
        for key, factor in [("nasal_bridge", 1.30), ("nasal_dorsum", 1.35), ("nose_tip", 1.42), ("alar", 1.35), ("subnasal", 1.18), ("philtrum", 1.12)]:
            local = masks[key][target_idx]
            weights[local] *= factor
        solved = solve(current, edges, target_idx, target, weights, fixed_idx, start[fixed_idx], edge_w=edge_w, fixed_w=32000.0)
        current = (1.0 - gain) * current + gain * solved
        current[fixed_idx] = start[fixed_idx]
        history.append({"phase": "full_no_eye", "pass": pass_i, "active_vertices": int(len(target_idx)), "median_nn": float(np.median(d)), "p90_nn": float(np.quantile(d, 0.90))})

    eye_disp = np.linalg.norm(current[fixed_idx] - start[fixed_idx], axis=1) if len(fixed_idx) else np.zeros(0)
    return local_refined, current, history, protrusion, {
        "fixed_eye_vertices": int(len(fixed_idx)),
        "eye_fixed_max": float(eye_disp.max()) if len(eye_disp) else 0.0,
        "eye_fixed_mean": float(eye_disp.mean()) if len(eye_disp) else 0.0,
    }


def write_obj_like(src_obj, out_obj, vertices):
    out_obj.parent.mkdir(parents=True, exist_ok=True)
    src_obj = Path(src_obj)
    src_dir = src_obj.parent
    out_dir = out_obj.parent
    lines = []
    vi = 0
    for line in src_obj.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("v "):
            v = vertices[vi]
            lines.append(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}")
            vi += 1
        elif line.startswith("mtllib "):
            name = line.split(maxsplit=1)[1].strip()
            side = src_dir / name
            if side.exists():
                shutil.copy2(side, out_dir / side.name)
            lines.append(line)
        else:
            lines.append(line)
    out_obj.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for suffix in [".jpg", ".png", ".jpeg"]:
        side = src_obj.with_suffix(suffix)
        if side.exists():
            shutil.copy2(side, out_obj.with_suffix(suffix))


def projection_matrix(points_list, view_name):
    pts = np.vstack(points_list)
    center = np.median(pts, axis=0)
    radius = np.percentile(np.linalg.norm(pts - center, axis=1), 96)
    radius = max(float(radius), 1e-8)
    fig = plt.figure(figsize=(2, 2), dpi=50)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    if view_name == "front":
        elev, azim = 8.0, -88.0
    elif view_name == "side":
        elev, azim = 8.0, -8.0
    else:
        elev, azim = 18.0, -48.0
    ax.view_init(elev=elev, azim=azim)
    matrix = ax.get_proj()
    plt.close(fig)
    return matrix


def project_points(points, matrix):
    x, y, z = proj3d.proj_transform(points[:, 0], points[:, 1], points[:, 2], matrix)
    return np.column_stack([x, y]), np.asarray(z)


def bounds_for(points_list, matrix, pad_ratio=0.08):
    pts = np.vstack([project_points(p, matrix)[0] for p in points_list])
    lo = np.percentile(pts, 0.8, axis=0)
    hi = np.percentile(pts, 99.2, axis=0)
    pad = np.maximum(hi - lo, 1e-8) * pad_ratio
    return lo - pad, hi + pad


def to_canvas(points2, lo, hi, size):
    w, h = size
    x = (points2[:, 0] - lo[0]) / max(float(hi[0] - lo[0]), 1e-8) * (w - 1)
    y = (1.0 - (points2[:, 1] - lo[1]) / max(float(hi[1] - lo[1]), 1e-8)) * (h - 1)
    return np.c_[x, y]


def render_textured(vertices, uvs, faces, face_uvs, texture_path, target_points, view_name, size=(760, 760)):
    target_sample = target_points[sample_idx(len(target_points), 80000, 11)]
    matrix = projection_matrix([vertices, target_sample], view_name)
    lo, hi = bounds_for([vertices, target_sample], matrix, pad_ratio=0.075)
    tex = cv2.cvtColor(cv2.imread(str(texture_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    th, tw = tex.shape[:2]
    out = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
    target_2d = to_canvas(project_points(target_sample, matrix)[0], lo, hi, size).astype(np.int32)
    for x, y in target_2d:
        if 0 <= x < size[0] and 0 <= y < size[1]:
            out[max(0, y - 1):min(size[1], y + 2), max(0, x - 1):min(size[0], x + 2)] = (208, 208, 208)
    projected_vertices, depth = project_points(vertices, matrix)
    pts2 = to_canvas(projected_vertices, lo, hi, size).astype(np.float32)
    uv_px = np.empty_like(uvs, dtype=np.float32)
    uv_px[:, 0] = uvs[:, 0] * (tw - 1)
    uv_px[:, 1] = (1.0 - uvs[:, 1]) * (th - 1)
    order = np.argsort(depth[faces].mean(axis=1))
    for fi in order:
        tri = pts2[faces[fi]].astype(np.float32)
        if not np.isfinite(tri).all() or cv2.contourArea(tri) < 0.35:
            continue
        src = uv_px[face_uvs[fi]].astype(np.float32)
        x, y, w, h = cv2.boundingRect(tri)
        if x >= size[0] or y >= size[1] or x + w <= 0 or y + h <= 0:
            continue
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, size[0]), min(y + h, size[1])
        if x1 <= x0 or y1 <= y0:
            continue
        local_tri = tri - np.array([x0, y0], dtype=np.float32)
        affine = cv2.getAffineTransform(src, local_tri)
        patch = cv2.warpAffine(tex, affine, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.round(local_tri).astype(np.int32), 255)
        out[y0:y1, x0:x1][mask > 0] = patch[mask > 0]
    return Image.fromarray(out)


def render_overlay(target_points, init_vertices, local_vertices, final_vertices, view_name, size=(760, 760)):
    target_sample = target_points[sample_idx(len(target_points), 80000, 21)]
    init_sample = init_vertices[sample_idx(len(init_vertices), 30000, 22)]
    local_sample = local_vertices[sample_idx(len(local_vertices), 30000, 23)]
    final_sample = final_vertices[sample_idx(len(final_vertices), 30000, 24)]
    matrix = projection_matrix([target_sample, init_sample, local_sample, final_sample], view_name)
    lo, hi = bounds_for([target_sample, init_sample, local_sample, final_sample], matrix)
    fig, ax = plt.subplots(figsize=(4.3, 4.3), dpi=180)
    for pts, color, alpha, s, label in [
        (target_sample, "#222222", 0.30, 0.35, "eye-closed target"),
        (init_sample, "#2364aa", 0.48, 0.30, "rigid init"),
        (local_sample, "#f08c00", 0.62, 0.30, "nasal local"),
        (final_sample, "#d64032", 0.70, 0.30, "non-rigid final"),
    ]:
        xy = to_canvas(project_points(pts, matrix)[0], lo, hi, size)
        ax.scatter(xy[:, 0], xy[:, 1], s=s, c=color, alpha=alpha, linewidths=0, label=label)
    ax.set_xlim(0, size[0])
    ax.set_ylim(size[1], 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="lower left", fontsize=6, frameon=True)
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(rgba[:, :, :3])


def title_panel(img, title):
    canvas = Image.new("RGB", (img.width, img.height + 44), "white")
    canvas.paste(img, (0, 44))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 14), title, fill=(0, 0, 0))
    return canvas


def make_review(case_dir, case, schedule_name, src_obj, target_reg, init_v, local_v, final_v):
    v, uvs, faces, face_uvs, _, _ = parse_obj_with_uv(src_obj)
    texture = Path(src_obj).with_suffix(".jpg")
    panels = []
    for view in ["front", "side", "oblique"]:
        panels.append(title_panel(render_overlay(target_reg, init_v, local_v, final_v, view), f"{view}: point overlay"))
        panels.append(title_panel(render_textured(final_v, uvs, faces, face_uvs, texture, target_reg, view), f"{view}: final textured"))
    w, h = panels[0].size
    canvas = Image.new("RGB", (w * 2 + 24, h * 3 + 62), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 18), f"{case} non-rigid nasal depth contour: {schedule_name}", fill=(0, 0, 0))
    for i, panel in enumerate(panels):
        x = 8 + (i % 2) * (w + 8)
        y = 54 + (i // 2) * (h + 8)
        canvas.paste(panel, (x, y))
    out = case_dir / f"{case}_{schedule_name}_nonrigid_review.png"
    canvas.save(out)
    return out


def process_case(case, manifest):
    target_id = f"{int(case[:3]):03d}_18_eye_closed"
    target_mesh = Path(manifest[target_id]["mesh"])
    _, target_world, _ = load_mesh(target_mesh)
    target_reg = target_registration_frame(target_world, target_mesh.parent / "selected_camera.json")
    target_sample = deterministic_sample(target_reg, 140000)

    src_obj = RIGID_ROOT / case / f"{case}_similarity_icp_regframe.obj"
    init_v, _, faces, _, _, _ = parse_obj_with_uv(src_obj)
    fit_weight, masks, ellipses = build_masks(src_obj, init_v)
    case_dir = OUT / case
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / f"{case}_eye_soft_mask.npy", masks["eye_soft"])
    np.save(case_dir / f"{case}_nose_mask.npy", masks["nose"])

    reports = []
    baseline_metrics = all_metrics(init_v, target_sample, masks)
    for schedule_name, thresholds in SCHEDULES.items():
        local_v, final_v, history, protrusion, eye_report = run_nonrigid(init_v, faces, target_sample, masks, thresholds)
        schedule_dir = case_dir / schedule_name
        local_obj = schedule_dir / f"{case}_{schedule_name}_nasal_local_regframe.obj"
        final_obj = schedule_dir / f"{case}_{schedule_name}_nonrigid_final_regframe.obj"
        write_obj_like(src_obj, local_obj, local_v)
        write_obj_like(src_obj, final_obj, final_v)
        local_metrics = all_metrics(local_v, target_sample, masks)
        final_metrics = all_metrics(final_v, target_sample, masks)
        disp = np.linalg.norm(final_v - init_v, axis=1)
        report = {
            "case": case,
            "schedule": schedule_name,
            "thresholds": thresholds,
            "method": "final rigid initialization -> UV eye/orbit exclusion -> weighted edge-preserving nasal depth-contour refinement -> weighted edge-preserving full no-eye propagation, source-to-target one-way nearest neighbor, no target ROI",
            "source_rigid_obj": str(src_obj),
            "target_mesh": str(target_mesh),
            "target_policy": "full eye-closed target mesh converted to registration frame, no ROI crop",
            "nasal_local_obj": str(local_obj),
            "nonrigid_final_obj": str(final_obj),
            "fit_vertices": int((fit_weight > 0.08).sum()),
            "mask_vertices": {k: int(v.sum()) for k, v in masks.items() if v.dtype == bool},
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
        review = make_review(schedule_dir, case, schedule_name, src_obj, target_reg, init_v, local_v, final_v)
        report["review_png"] = str(review)
        (schedule_dir / f"{case}_{schedule_name}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        reports.append(report)
        b = baseline_metrics["nose"]
        f = final_metrics["nose"]
        print(
            f"{case} {schedule_name} nose median {b['median']:.6f}->{f['median']:.6f} "
            f"p90 {b['p90']:.6f}->{f['p90']:.6f} eye_max={eye_report['eye_fixed_max']:.8f}",
            flush=True,
        )
    return reports


def summarize(reports):
    rows = []
    for r in reports:
        for stage_key, stage_name in [
            ("baseline_metrics", "rigid_init"),
            ("nasal_local_metrics", "nasal_local"),
            ("nonrigid_final_metrics", "nonrigid_final"),
        ]:
            for region, metrics in r[stage_key].items():
                row = {
                    "case": r["case"],
                    "schedule": r["schedule"],
                    "stage": stage_name,
                    "region": region,
                    **metrics,
                    "eye_fixed_max": r["eye_fixed_max"],
                    "displacement_p90": r["displacement_p90"],
                }
                rows.append(row)
    fields = sorted({k for row in rows for k in row.keys()})
    with (OUT / "nonrigid_nasal_depth_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    best = []
    for case in CASES:
        case_reports = [r for r in reports if r["case"] == case]
        best_report = min(case_reports, key=lambda r: (
            r["nonrigid_final_metrics"]["nose"]["median"],
            r["nonrigid_final_metrics"]["nose"]["p90"],
        ))
        best.append({
            "case": case,
            "best_by_nose_median": best_report["schedule"],
            "nose_baseline_median": best_report["baseline_metrics"]["nose"]["median"],
            "nose_final_median": best_report["nonrigid_final_metrics"]["nose"]["median"],
            "nose_baseline_p90": best_report["baseline_metrics"]["nose"]["p90"],
            "nose_final_p90": best_report["nonrigid_final_metrics"]["nose"]["p90"],
            "review_png": best_report["review_png"],
            "final_obj": best_report["nonrigid_final_obj"],
        })
    (OUT / "summary.json").write_text(json.dumps({"best": best, "reports": reports}, indent=2), encoding="utf-8")
    lines = ["case,best_schedule,nose_median_before,nose_median_after,nose_p90_before,nose_p90_after,review_png"]
    for row in best:
        lines.append(
            f"{row['case']},{row['best_by_nose_median']},{row['nose_baseline_median']:.6f},"
            f"{row['nose_final_median']:.6f},{row['nose_baseline_p90']:.6f},{row['nose_final_p90']:.6f},{row['review_png']}"
        )
    (OUT / "best_by_case.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest()
    all_reports = []
    for case in CASES:
        all_reports.extend(process_case(case, manifest))
    summarize(all_reports)
    print(OUT, flush=True)


if __name__ == "__main__":
    main()


