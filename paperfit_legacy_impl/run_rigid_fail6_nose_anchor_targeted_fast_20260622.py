#!/usr/bin/env python3
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(os.environ.get("PAPERFIT_ROOT", "/path/to/prepared/facescape_pipeline"))
TOOLS = ROOT / "research_tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_scratch_surface_registration_3case_final_attempt as rigid  # noqa: E402
import run_rigid_global_upright_failed11_20260617 as global_rigid  # noqa: E402
import run_rigid_upright_hardgate_full40_20260619 as full40  # noqa: E402


BASE_RECOVERY_ROOT = ROOT / "research_rigid_upright_hardgate_fail8_precision_recovery_20260621"
OUT = ROOT / "research_rigid_fail6_nose_anchor_targeted_fast_20260622"


ORIGINAL_GLOBAL_INITIAL = global_rigid.global_initial_candidates


def read_cases():
    aggregate = json.loads((BASE_RECOVERY_ROOT / "fail8_precision_recovery_aggregate.json").read_text(encoding="utf-8"))
    return list(aggregate["fail_cases"])


def nose_mask_from_regions(region_masks):
    return (
        region_masks["nose_bridge"]
        | region_masks["nose_dorsum"]
        | region_masks["nose_tip"]
        | region_masks["alar"]
    )


def nose_anchor_initial_candidates(src, target_full, fit_weight, region_masks, eye_mask, nose_anchor, keep=80):
    original = ORIGINAL_GLOBAL_INITIAL(src, target_full, fit_weight, region_masks, eye_mask, nose_anchor, keep=200)
    fit_mask = fit_weight > 0.35
    src_fit = src[fit_mask]
    src_fit_weights = fit_weight[fit_mask]
    eval_idx = rigid.sample_idx(len(src_fit), 5000, 241)
    target_eval = target_full[rigid.sample_idx(len(target_full), 60000, 242)]
    tree = cKDTree(target_eval)

    nose_mask = nose_mask_from_regions(region_masks)
    src_nose = np.median(src[nose_mask], axis=0)
    src_ext = np.maximum(np.quantile(src_fit, 0.95, axis=0) - np.quantile(src_fit, 0.05, axis=0), 1e-8)
    front = target_full[target_full[:, 1] >= np.quantile(target_full[:, 1], 0.78)]
    if len(front) < 5000:
        front = target_full
    front_ext = np.maximum(np.quantile(front, 0.92, axis=0) - np.quantile(front, 0.08, axis=0), 1e-8)

    base = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    base_rots = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                rot = np.diag([sx, sy, sz]) @ base
                if np.linalg.det(rot) > 0.0:
                    base_rots.append(rot)

    candidates = list(original)
    for base_rot in base_rots:
        for rot in global_rigid.rotation_grid(base_rot):
            mapped_ext = np.maximum(np.abs(rot) @ src_ext, 1e-8)
            scale0 = float(np.median([front_ext[0] / mapped_ext[0], front_ext[2] / mapped_ext[2]]))
            for scale in scale0 * np.linspace(0.50, 1.52, 27):
                moved_fit = rigid.transform(src_fit[eval_idx], float(scale), rot, np.zeros(3))
                trans = nose_anchor - float(scale) * (src_nose @ rot.T)
                aligned_eval = moved_fit + trans
                d, _ = tree.query(aligned_eval, k=1, workers=-1)
                ew = src_fit_weights[eval_idx]
                aligned_all = rigid.transform(src, float(scale), rot, trans)
                anchor = rigid.source_nose_anchor_distance(aligned_all, region_masks, nose_anchor)
                mouth_guard = rigid.source_mouth_to_nose_guard(aligned_all, region_masks, nose_anchor)
                upright = global_rigid.anatomical_upright_penalty(aligned_all, region_masks, eye_mask)
                coarse = float(
                    rigid.weighted_quantile(d, ew, 0.50)
                    + 0.25 * rigid.weighted_quantile(d, ew, 0.90)
                    + 0.10 * rigid.weighted_quantile(d, ew, 0.98)
                    + 1.25 * anchor
                    + 0.65 * mouth_guard
                    + 3.0 * upright
                )
                candidates.append({
                    "score": coarse,
                    "scale": float(scale),
                    "rot": rot.copy(),
                    "trans": trans.copy(),
                    "upright_penalty": float(upright),
                    "anchor": float(anchor),
                    "mouth_guard": float(mouth_guard),
                    "candidate_family": "nose_anchor_targeted",
                })
    seen = set()
    unique = []
    for cand in sorted(candidates, key=lambda r: r["score"]):
        key = (
            round(cand["scale"], 5),
            tuple(np.round(cand["rot"].reshape(-1), 4)),
            tuple(np.round(cand["trans"], 4)),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)
    return unique[:keep]


def configure():
    full40.OUT = OUT
    global_rigid.global_initial_candidates = nose_anchor_initial_candidates
    full40.global_rigid.global_initial_candidates = nose_anchor_initial_candidates


def report_path(case):
    return OUT / case / f"{case}_scratch_surface_registration_report.json"


def load_report(case):
    path = report_path(case)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def run_one(case):
    configure()
    existing = load_report(case)
    if existing is not None:
        return case, "skipped", existing
    by_pair = full40.read_manifest()
    report = full40.process(case, by_pair, keep=16, rigid_iters=32)
    report["method"] = "fail6 nose-anchor-targeted fast recovery: top original/explicit source-nose-to-target-anchor initial translations"
    report["recovery_policy"] = {
        "input_root": str(BASE_RECOVERY_ROOT),
        "candidate_keep": 16,
        "rigid_iters": 32,
        "added_candidates": "nose anchor targeted translation across proper rotations and scale grid",
        "qc": "same hard orientation gate and strict full40 thresholds",
    }
    report_path(case).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return case, "processed", report


def write_summary(cases, reports_by_case):
    configure()
    reports = [reports_by_case[case] for case in cases if case in reports_by_case]
    rows, aggregate = full40.write_summary(reports)
    aggregate["input_rigid_root"] = str(BASE_RECOVERY_ROOT)
    aggregate["recovery_policy"] = "nose-anchor-targeted candidates, keep=16, rigid_iters=32"
    (OUT / "fail6_nose_anchor_targeted_aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    if rows:
        with (OUT / "fail6_nose_anchor_targeted_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return rows, aggregate


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    configure()
    cases = read_cases()
    reports_by_case = {case: load_report(case) for case in cases if load_report(case) is not None}
    write_summary(cases, reports_by_case)
    todo = [case for case in cases if case not in reports_by_case]
    workers = int(os.environ.get("RIGID_NOSE_TARGET_WORKERS", "3"))
    print(json.dumps({"total": len(cases), "existing": len(reports_by_case), "todo": len(todo), "workers": workers}), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, case): case for case in todo}
        for future in as_completed(futures):
            case = futures[future]
            _, status, report = future.result()
            reports_by_case[case] = report
            _, aggregate = write_summary(cases, reports_by_case)
            print(json.dumps({"case": case, "status": status, **aggregate}), flush=True)
    reports = [reports_by_case[case] for case in cases]
    (OUT / "fail6_nose_anchor_targeted_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    _, aggregate = write_summary(cases, reports_by_case)
    print(json.dumps({"output_root": str(OUT), **aggregate}, indent=2), flush=True)


if __name__ == "__main__":
    main()


