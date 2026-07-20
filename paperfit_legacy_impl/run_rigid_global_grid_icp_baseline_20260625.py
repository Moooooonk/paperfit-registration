#!/usr/bin/env python3
"""Global-grid rigid ICP baseline for HRN-to-FaceScape pairs.

This is not an official Go-ICP implementation. It is a deterministic
Go-ICP-style baseline: broad proper-rotation grid, coarse nearest-neighbor
scoring, fixed-scale ICP refinement, and the same anatomical QC gate used by
the existing rigid baselines.
"""

import argparse
import os
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(os.environ.get("PAPERFIT_ROOT", "/path/to/prepared/facescape_pipeline"))
TOOLS = ROOT / "research_tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_scratch_surface_registration_3case_final_attempt as rigid  # noqa: E402
import run_rigid_upright_hardgate_3case_20260619 as hard3  # noqa: E402
import run_rigid_global_upright_failed11_20260617 as global_rigid  # noqa: E402
import run_rigid_open3d_baselines_allpairs_20260623 as allpairs_base  # noqa: E402


MANIFEST = ROOT / "prepared_001_020_cv" / "facescape_frontal_pairs_manifest.csv"
OUT = ROOT / "research_rigid_global_grid_icp_baseline_20260625"
BASE40 = ROOT / "research_rigid_expanded_qc_40case_20260611" / "rigid_expanded_qc_compact_summary.csv"


def read_manifest():
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return {r["pair_id"]: r for r in csv.DictReader(f)}


def read_cases(scope):
    if scope == "allpairs":
        return allpairs_base.read_cases()
    if scope == "base40":
        with BASE40.open("r", encoding="utf-8", newline="") as f:
            return [r["case"] for r in csv.DictReader(f)]
    return [c.strip() for c in scope.split(",") if c.strip()]


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def proper_axis_bases():
    base = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    out = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                r = np.diag([sx, sy, sz]) @ base
                if np.linalg.det(r) > 0:
                    out.append(r)
    return out


def rotation_grid(mode):
    if mode == "coarse":
        yaw_deg = [-45, -25, 0, 25, 45]
        pitch_deg = [-25, 0, 25]
        roll_deg = [-25, 0, 25]
    elif mode == "medium":
        yaw_deg = [-60, -40, -20, 0, 20, 40, 60]
        pitch_deg = [-35, -18, 0, 18, 35]
        roll_deg = [-35, -18, 0, 18, 35]
    else:
        raise ValueError(f"unknown grid mode: {mode}")

    seen = set()
    for base in proper_axis_bases():
        for yaw in np.deg2rad(yaw_deg):
            for pitch in np.deg2rad(pitch_deg):
                for roll in np.deg2rad(roll_deg):
                    r = rot_z(roll) @ rot_x(pitch) @ rot_y(yaw) @ base
                    if np.linalg.det(r) <= 0:
                        continue
                    key = tuple(np.round(r.reshape(-1), 5))
                    if key in seen:
                        continue
                    seen.add(key)
                    yield r


def sample_idx(n, max_n, seed):
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, max_n, replace=False)


def weighted_quantile(values, weights, q):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    if cdf[-1] <= 0:
        return float(np.quantile(values, q))
    cdf = cdf / cdf[-1]
    return float(np.interp(q, cdf, values))


def make_candidates(src_v, target_reg, fit_weight, region_masks, eye_mask, nose_anchor, mode, keep):
    fit_mask = fit_weight > 0.35
    src_fit = src_v[fit_mask]
    src_fit_weights = fit_weight[fit_mask]
    eval_idx = sample_idx(len(src_fit), 2600 if mode == "coarse" else 3600, 20260625)
    target_eval = target_reg[sample_idx(len(target_reg), 45000 if mode == "coarse" else 65000, 20260626)]
    tree = cKDTree(target_eval)

    y = target_reg[:, 1]
    front = target_reg[y >= np.quantile(y, 0.78)]
    if len(front) < 5000:
        front = target_reg
    src_ext = np.maximum(np.quantile(src_fit, 0.95, axis=0) - np.quantile(src_fit, 0.05, axis=0), 1e-8)
    front_ext = np.maximum(np.quantile(front, 0.92, axis=0) - np.quantile(front, 0.08, axis=0), 1e-8)
    target_center = np.median(front, axis=0)
    target_front_y = float(np.quantile(y, 0.94))

    scales = np.linspace(0.52, 1.50, 18 if mode == "coarse" else 24)
    candidates = []
    for rot in rotation_grid(mode):
        mapped_ext = np.maximum(np.abs(rot) @ src_ext, 1e-8)
        scale0 = float(np.median([front_ext[0] / mapped_ext[0], front_ext[2] / mapped_ext[2]]))
        for scale in scale0 * scales:
            moved_all = rigid.transform(src_v, float(scale), rot, np.zeros(3))
            moved_eval = rigid.transform(src_fit[eval_idx], float(scale), rot, np.zeros(3))
            trans = target_center - np.median(moved_all[fit_mask], axis=0)
            trans[1] = target_front_y - np.max(moved_all[fit_mask, 1])
            aligned_eval = moved_eval + trans
            d, _ = tree.query(aligned_eval, k=1, workers=-1)
            ew = src_fit_weights[eval_idx]
            aligned_all = moved_all + trans
            anchor = rigid.source_nose_anchor_distance(aligned_all, region_masks, nose_anchor)
            mouth_guard = rigid.source_mouth_to_nose_guard(aligned_all, region_masks, nose_anchor)
            upright = global_rigid.anatomical_upright_penalty(aligned_all, region_masks, eye_mask)
            score = (
                weighted_quantile(d, ew, 0.50)
                + 0.25 * weighted_quantile(d, ew, 0.90)
                + 0.10 * weighted_quantile(d, ew, 0.98)
                + 0.65 * anchor
                + 0.45 * mouth_guard
                + 3.0 * upright
            )
            candidates.append({
                "score": float(score),
                "scale": float(scale),
                "rot": rot,
                "trans": trans,
                "anchor": float(anchor),
                "mouth_guard": float(mouth_guard),
                "upright": float(upright),
            })
    candidates.sort(key=lambda x: x["score"])
    return candidates[:keep], len(candidates)


def process(case, by_pair, args):
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

    case_dir = OUT / args.scope / args.grid / case
    case_dir.mkdir(parents=True, exist_ok=True)
    candidates, total_candidates = make_candidates(
        src_v, target_reg, fit_weight, region_masks, eye_mask, nose_anchor,
        args.grid, args.keep
    )
    best = None
    tried = []
    for cand in candidates:
        scale, rot, trans, hist = rigid.refine_rigid_fixed_scale(
            src_v, target_reg, fit_weight, cand["scale"], cand["rot"], cand["trans"],
            iterations=args.icp_iter, region_masks=region_masks, nose_anchor=nose_anchor
        )
        scale, rot, trans = rigid.translation_only_polish(src_v, target_reg, fit_weight, scale, rot, trans)
        aligned = rigid.transform(src_v, scale, rot, trans)
        m = rigid.metrics(aligned, target_reg, fit_weight, region_masks=region_masks, nose_anchor=nose_anchor)
        o = hard3.orientation_metrics(aligned, region_masks, eye_mask)
        upright = global_rigid.anatomical_upright_penalty(aligned, region_masks, eye_mask)
        score = float(m["selection_score"] + 3.0 * upright)
        rec = {
            "initial_score": cand["score"],
            "initial_anchor": cand["anchor"],
            "initial_mouth_guard": cand["mouth_guard"],
            "initial_upright": cand["upright"],
            "scale": float(scale),
            "score": score,
            "median": float(m["median"]),
            "p90": float(m["p90"]),
            "nose_weighted_median": float(m.get("nose_weighted_median", m["median"])),
            "source_nose_anchor_distance": float(m.get("source_nose_anchor_distance", 999.0)),
            "upside_down": int(o.get("upside_down", 0)),
        }
        tried.append(rec)
        if best is None or score < best[0]:
            best = (score, scale, rot, trans, aligned, m, o, rec)

    _, scale, rot, trans, aligned, m, o, best_rec = best
    obj_path = case_dir / f"{case}_global_grid_icp_regframe.obj"
    rigid.write_obj_like(src_obj, obj_path, aligned)
    report = {
        "case": case,
        "method": f"global_grid_icp_{args.grid}",
        "note": "Deterministic Go-ICP-style exhaustive rotation/scale grid plus fixed-scale ICP; not official Go-ICP.",
        "grid": args.grid,
        "keep": args.keep,
        "icp_iter": args.icp_iter,
        "total_coarse_candidates": total_candidates,
        "source_obj": str(src_obj),
        "target_mesh": str(target_mesh),
        "similarity_metrics": m,
        "similarity_orientation_metrics": o,
        "rigid_scale": float(scale),
        "final_similarity_scale": float(scale),
        "similarity_regframe_obj": str(obj_path),
        "diagnostic_png": "",
        "tried_refined_candidates": tried,
        "runtime_sec": float(time.time() - t0),
    }
    qc = hard3.qc_row(report)
    row_out = {
        **qc,
        "method": report["method"],
        "runtime_sec": report["runtime_sec"],
        "grid": args.grid,
        "keep": args.keep,
        "total_coarse_candidates": total_candidates,
        "regframe_obj": str(obj_path),
        "report_json": str(case_dir / f"{case}_global_grid_icp_report.json"),
    }
    (case_dir / f"{case}_global_grid_icp_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "case": case,
        "qc_pass": row_out["qc_pass"],
        "median": row_out["similarity_median"],
        "p90": row_out["similarity_p90"],
        "anchor": row_out["source_nose_anchor_distance"],
        "upside_down": row_out.get("upside_down", 0),
        "elapsed": row_out["runtime_sec"],
    }), flush=True)
    return row_out


def write_summary(rows, args):
    out_dir = OUT / args.scope / args.grid
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "global_grid_icp_summary.csv"
    fields = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    valid = [r for r in rows if not r.get("error")]
    aggregate = {
        "method": f"global_grid_icp_{args.grid}",
        "note": "Deterministic Go-ICP-style exhaustive rotation/scale grid plus fixed-scale ICP; not official Go-ICP.",
        "scope": args.scope,
        "cases": len(valid),
        "qc_pass": int(sum(int(r.get("qc_pass", 0)) for r in valid)),
        "qc_fail": int(sum(1 - int(r.get("qc_pass", 0)) for r in valid)),
        "qc_pass_rate_pct": float(100.0 * sum(int(r.get("qc_pass", 0)) for r in valid) / max(len(valid), 1)),
        "upside_down": int(sum(int(r.get("upside_down", 0)) for r in valid)),
        "mean_median": float(np.mean([float(r["similarity_median"]) for r in valid])) if valid else None,
        "mean_p90": float(np.mean([float(r["similarity_p90"]) for r in valid])) if valid else None,
        "mean_anchor": float(np.mean([float(r["source_nose_anchor_distance"]) for r in valid])) if valid else None,
        "pass_cases": [r["case"] for r in valid if int(r.get("qc_pass", 0))],
        "fail_cases": [r["case"] for r in valid if not int(r.get("qc_pass", 0))],
    }
    (out_dir / "global_grid_icp_aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default="base40", help="base40, allpairs, or comma-separated case ids")
    parser.add_argument("--grid", default="coarse", choices=["coarse", "medium"])
    parser.add_argument("--keep", type=int, default=6)
    parser.add_argument("--icp-iter", type=int, default=18)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    by_pair = read_manifest()
    cases = read_cases(args.scope)
    if args.limit:
        cases = cases[: args.limit]
    out_dir = OUT / args.scope / args.grid
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        try:
            rows.append(process(case, by_pair, args))
        except Exception as exc:
            print(json.dumps({"case": case, "error": str(exc)}), flush=True)
            rows.append({"case": case, "method": f"global_grid_icp_{args.grid}", "error": str(exc), "qc_pass": 0})
        write_summary(rows, args)
    aggregate = write_summary(rows, args)
    print(json.dumps({"output_root": str(out_dir), "aggregate": aggregate}, indent=2), flush=True)


if __name__ == "__main__":
    main()


