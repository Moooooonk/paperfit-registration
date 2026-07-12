#!/usr/bin/env python3
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("PAPERFIT_ROOT", "/path/to/prepared/facescape_pipeline"))
TOOLS = ROOT / "research_tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_scratch_surface_registration_3case_final_attempt as rigid  # noqa: E402
import run_rigid_upright_hardgate_3case_20260619 as hard3  # noqa: E402
import run_rigid_fail6_nose_anchor_targeted_fast_20260622 as targeted  # noqa: E402


MANIFEST = ROOT / "prepared_001_020_cv" / "facescape_frontal_pairs_manifest.csv"
BASE_RECOVERY_ROOT = ROOT / "research_rigid_upright_hardgate_fail8_precision_recovery_20260621"
OUT = ROOT / "research_rigid_fail6_nose_tip_gradient_candidate_only_20260622"


def read_manifest():
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return {r["pair_id"]: r for r in csv.DictReader(f)}


def read_cases():
    aggregate = json.loads((BASE_RECOVERY_ROOT / "fail8_precision_recovery_aggregate.json").read_text(encoding="utf-8"))
    return list(aggregate["fail_cases"])


def apply_nose_tip_gradient_weight(obj_path, vertices, fit_weight):
    x, y, _ = rigid.canonical_coords(vertices)
    vertex_uv, valid_uv = rigid.parse_vertex_uv(obj_path)
    image = rigid.Image.open(Path(obj_path).with_suffix(".jpg"))
    w, h = image.size
    px = vertex_uv[:, 0] * (w - 1)
    py = (1.0 - vertex_uv[:, 1]) * (h - 1)

    nose_bridge = valid_uv & rigid.uv_ellipse(px, py, 0.500 * w, 0.435 * h, 0.048 * w, 0.105 * h)
    nose_dorsum = valid_uv & rigid.uv_ellipse(px, py, 0.500 * w, 0.505 * h, 0.066 * w, 0.125 * h)
    nose_tip = valid_uv & rigid.uv_ellipse(px, py, 0.500 * w, 0.585 * h, 0.082 * w, 0.078 * h)
    alar = valid_uv & rigid.uv_ellipse(px, py, 0.500 * w, 0.595 * h, 0.105 * w, 0.058 * h)
    philtrum = valid_uv & rigid.uv_ellipse(px, py, 0.500 * w, 0.665 * h, 0.075 * w, 0.035 * h)
    mouth = valid_uv & rigid.uv_ellipse(px, py, 0.500 * w, 0.725 * h, 0.205 * w, 0.070 * h)
    lower_lip = valid_uv & rigid.uv_ellipse(px, py, 0.500 * w, 0.760 * h, 0.170 * w, 0.052 * h)
    midface = valid_uv & rigid.uv_ellipse(px, py, 0.500 * w, 0.555 * h, 0.215 * w, 0.155 * h) & (py < 0.655 * h)
    outer = (np.abs(x) > 0.430) | (y > 0.775) | (y < 0.160)

    nose_union = nose_bridge | nose_dorsum | nose_tip | alar
    # Continuous UV-y gradient along the nasal ridge. Near bridge stays close
    # to the old dorsum weight, while tip/alar approaches the maximum weight.
    t = np.clip((py - 0.435 * h) / max(0.170 * h, 1.0), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    nose_gradient = 1.75 + 1.15 * smooth
    nose_gradient[nose_tip | alar] = np.maximum(nose_gradient[nose_tip | alar], 2.75)

    region_weight = np.ones(len(vertices), dtype=np.float64)
    region_weight[midface] *= 1.16
    region_weight[nose_bridge | nose_dorsum | nose_tip | alar] *= nose_gradient[nose_union]
    region_weight[philtrum] *= 1.00
    region_weight[mouth | lower_lip] *= 0.40
    region_weight[outer] *= 0.60

    weighted = np.clip(fit_weight * region_weight, 0.0, 3.0)
    masks = {
        "midface": midface,
        "nose_bridge": nose_bridge,
        "nose_dorsum": nose_dorsum,
        "nose_tip": nose_tip,
        "alar": alar,
        "philtrum": philtrum,
        "mouth_downweighted": mouth | lower_lip,
        "outer_downweighted": outer,
        "nose_tip_gradient": nose_union,
    }
    return weighted, masks


def qc_from_metrics(case, m, o, scale):
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
        "case": case,
        "qc_pass": int(qc_pass),
        "upside_down": int(o["upside_down"]),
        "eye_over_mouth_norm": float(o["eye_over_mouth_norm"]),
        "nose_over_mouth_norm": float(o["nose_over_mouth_norm"]),
        "vertical_z": float(o["vertical_z"]),
        "similarity_median": med,
        "similarity_p90": p90,
        "nose_weighted_median": nose,
        "source_nose_anchor_distance": anchor,
        "final_similarity_scale": float(scale),
    }


def process(case):
    by_pair = read_manifest()
    row = by_pair[case]
    target_row = by_pair["%03d_18_eye_closed" % int(row["subject"])]
    src_obj = rigid.hrn_obj(case)
    _, src_v, _ = rigid.load_mesh(src_obj)
    target_mesh = Path(target_row["mesh"])
    _, target_world, _ = rigid.load_mesh(target_mesh)
    target_reg = rigid.target_registration_frame(target_world, target_mesh.parent / "selected_camera.json")
    nose_anchor = rigid.target_nose_anchor(target_reg)
    fit_weight_base, eye_mask, _ = rigid.texture_eye_weight(src_obj)
    fit_weight, region_masks = apply_nose_tip_gradient_weight(src_obj, src_v, fit_weight_base)

    candidates = targeted.nose_anchor_initial_candidates(
        src_v, target_reg, fit_weight, region_masks, eye_mask, nose_anchor, keep=10
    )
    tried = []
    best = None
    for ci, cand in enumerate(candidates, start=1):
        scale, rot, trans, hist = rigid.refine_rigid_fixed_scale(
            src_v,
            target_reg,
            fit_weight,
            cand["scale"],
            cand["rot"],
            cand["trans"],
            iterations=20,
            region_masks=region_masks,
            nose_anchor=nose_anchor,
        )
        scale, rot, trans = rigid.translation_only_polish(
            src_v, target_reg, fit_weight, scale, rot, trans, iterations=8
        )
        aligned = rigid.transform(src_v, scale, rot, trans)
        m = rigid.metrics(aligned, target_reg, fit_weight, region_masks=region_masks, nose_anchor=nose_anchor)
        o = hard3.orientation_metrics(aligned, region_masks, eye_mask)
        score = hard3.combined_score(m, o)
        row_out = {
            **qc_from_metrics(case, m, o, scale),
            "candidate": ci,
            "score": float(score),
            "initial_score": float(cand["score"]),
            "initial_anchor": float(cand["anchor"]),
            "initial_upright_penalty": float(cand["upright_penalty"]),
            "initial_family": cand.get("candidate_family", "original"),
            "rigid_last": float(hist[-1]) if hist else None,
            "weight_policy": "nose_tip_gradient_1.75_to_2.90_tip_min_2.75",
        }
        tried.append(row_out)
        print(json.dumps(row_out), flush=True)
        if best is None or score < best["score"]:
            best = row_out
    return case, best, tried


def write_outputs(best_rows, tried_rows):
    OUT.mkdir(parents=True, exist_ok=True)
    if tried_rows:
        with (OUT / "fail6_nose_tip_gradient_all_candidates.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(tried_rows[0].keys()))
            writer.writeheader()
            writer.writerows(tried_rows)
    if best_rows:
        with (OUT / "fail6_nose_tip_gradient_best.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(best_rows[0].keys()))
            writer.writeheader()
            writer.writerows(best_rows)
    aggregate = {
        "attempted": len(best_rows),
        "qc_pass": int(sum(r["qc_pass"] for r in best_rows)),
        "qc_fail": int(len(best_rows) - sum(r["qc_pass"] for r in best_rows)),
        "upside_down_remaining": [r["case"] for r in best_rows if r["upside_down"]],
        "pass_cases": [r["case"] for r in best_rows if r["qc_pass"]],
        "fail_cases": [r["case"] for r in best_rows if not r["qc_pass"]],
        "policy": "candidate-only fast audit: nose-anchor targeted candidates plus UV nose-tip gradient weights, keep=10, rigid_iters=20, translation_polish=8",
    }
    (OUT / "fail6_nose_tip_gradient_aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return aggregate


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = read_cases()
    workers = int(os.environ.get("RIGID_TIP_GRADIENT_WORKERS", "3"))
    best_rows = []
    tried_rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process, case): case for case in cases}
        for future in as_completed(futures):
            case, best, tried = future.result()
            best_rows.append(best)
            tried_rows.extend(tried)
            aggregate = write_outputs(sorted(best_rows, key=lambda r: r["case"]), tried_rows)
            print(json.dumps({"case_done": case, **aggregate}), flush=True)
    aggregate = write_outputs(sorted(best_rows, key=lambda r: r["case"]), tried_rows)
    print(json.dumps({"output_root": str(OUT), **aggregate}, indent=2), flush=True)


if __name__ == "__main__":
    main()


