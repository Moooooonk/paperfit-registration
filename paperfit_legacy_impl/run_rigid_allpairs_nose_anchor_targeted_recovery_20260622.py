#!/usr/bin/env python3
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(os.environ.get("PAPERFIT_ROOT", "/path/to/prepared/facescape_pipeline"))
TOOLS = ROOT / "research_tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_rigid_upright_hardgate_full40_20260619 as full40  # noqa: E402
import run_rigid_upright_hardgate_3case_20260619 as hard3  # noqa: E402
import run_rigid_fail6_nose_anchor_targeted_fast_20260622 as targeted  # noqa: E402


BASE_RIGID_ROOT = ROOT / "research_rigid_upright_hardgate_allpairs_20260622"
BASE_AGG = BASE_RIGID_ROOT / "upright_hardgate_allpairs_aggregate.json"
OUT = Path(os.environ.get(
    "RIGID_TARGETED_OUT",
    str(ROOT / "research_rigid_allpairs_nose_anchor_targeted_recovery_20260622"),
))
_ORIGINAL_WRITE_OBJ_LIKE = full40.rigid.write_obj_like
_ORIGINAL_PLOT_REVIEW = full40.rigid.plot_review
_ORIGINAL_FINAL_SMALL_POSE_SEARCH = full40.rigid.final_small_pose_search


def configure():
    full40.OUT = OUT
    targeted.global_rigid.global_initial_candidates = targeted.nose_anchor_initial_candidates
    full40.global_rigid.global_initial_candidates = targeted.nose_anchor_initial_candidates
    if os.environ.get("RIGID_TARGETED_FAST_OUTPUTS", "1") == "0":
        full40.rigid.write_obj_like = _ORIGINAL_WRITE_OBJ_LIKE
        full40.rigid.plot_review = _ORIGINAL_PLOT_REVIEW
    else:
        def write_similarity_only(src_obj, out_obj, vertices):
            out_obj = Path(out_obj)
            if out_obj.name.endswith("_similarity_icp_regframe.obj"):
                return _ORIGINAL_WRITE_OBJ_LIKE(src_obj, out_obj, vertices)
            out_obj.parent.mkdir(parents=True, exist_ok=True)
            return None

        def skip_review(case_dir, case, *args, **kwargs):
            return Path(case_dir) / f"{case}_targeted_fast_no_review.png"

        full40.rigid.write_obj_like = write_similarity_only
        full40.rigid.plot_review = skip_review
    if os.environ.get("RIGID_TARGETED_FAST_FINAL", "1") == "0":
        full40.rigid.final_small_pose_search = _ORIGINAL_FINAL_SMALL_POSE_SEARCH
    else:
        def skip_final_small_pose_search(src, target_full, fit_weight, scale, rot, trans, region_masks=None, nose_anchor=None):
            return scale, rot, trans, [{
                "round": 0,
                "move": "skipped_for_allpairs_targeted_recovery",
                "scale": float(scale),
            }]

        full40.rigid.final_small_pose_search = skip_final_small_pose_search


def selected_subjects():
    value = os.environ.get("RIGID_TARGETED_SUBJECTS", "")
    if not value.strip():
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def read_cases():
    aggregate = json.loads(BASE_AGG.read_text(encoding="utf-8"))
    cases = list(aggregate.get("fail_cases", []))
    subjects = selected_subjects()
    if subjects:
        cases = [case for case in cases if case.split("_", 1)[0] in subjects]
    explicit = os.environ.get("RIGID_TARGETED_CASES", "").strip()
    if explicit:
        wanted = [item.strip() for item in explicit.split(",") if item.strip()]
        cases = [case for case in wanted if case in set(aggregate.get("fail_cases", []))]
    return sorted(set(cases))


def report_path(case):
    return OUT / case / f"{case}_scratch_surface_registration_report.json"


def load_report(case):
    path = report_path(case)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def candidate_qc(metric, orient):
    med = float(metric["median"])
    p90 = float(metric["p90"])
    nose = float(metric.get("nose_weighted_median", med))
    anchor = float(metric.get("source_nose_anchor_distance", 999.0))
    return (
        not bool(orient["upside_down"])
        and med <= 0.023
        and p90 <= 0.070
        and nose <= 0.040
        and anchor <= 0.120
    )


def process_candidate_only(case, by_pair, keep, rigid_iters):
    print(f"{case} candidate_only start", flush=True)
    row = by_pair[case]
    target_row = by_pair["%03d_18_eye_closed" % int(row["subject"])]
    src_obj = full40.rigid.hrn_obj(case)
    _, src_v, _ = full40.rigid.load_mesh(src_obj)
    target_mesh = Path(target_row["mesh"])
    _, target_world, _ = full40.rigid.load_mesh(target_mesh)
    cam_json = target_mesh.parent / "selected_camera.json"
    target_reg = full40.rigid.target_registration_frame(target_world, cam_json)
    nose_anchor = full40.rigid.target_nose_anchor(target_reg)
    fit_weight_base, eye_mask, eye_ellipses = full40.rigid.texture_eye_weight(src_obj)
    fit_weight, region_masks = full40.rigid.apply_nose_midface_weight(src_obj, src_v, fit_weight_base)
    initial = full40.global_rigid.global_initial_candidates(
        src_v, target_reg, fit_weight, region_masks, eye_mask, nose_anchor, keep=keep
    )
    tried = []
    best = None
    translation_iters = int(os.environ.get("RIGID_TARGETED_TRANSLATION_ITERS", "4"))
    for ci, cand in enumerate(initial, start=1):
        scale, rot, trans, hist = full40.rigid.refine_rigid_fixed_scale(
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
        scale, rot, trans = full40.rigid.translation_only_polish(
            src_v, target_reg, fit_weight, scale, rot, trans, iterations=translation_iters
        )
        aligned = full40.rigid.transform(src_v, scale, rot, trans)
        metric = full40.rigid.metrics(aligned, target_reg, fit_weight, region_masks=region_masks, nose_anchor=nose_anchor)
        orient = hard3.orientation_metrics(aligned, region_masks, eye_mask)
        score = hard3.combined_score(metric, orient)
        is_pass = int(candidate_qc(metric, orient))
        item = {
            "candidate": ci,
            "score": float(score),
            "qc_pass": is_pass,
            "scale": float(scale),
            "history_last": float(hist[-1]) if hist else None,
            **metric,
            **orient,
        }
        tried.append(item)
        rank = (
            int(not is_pass),
            float(metric["median"]) + 0.35 * float(metric["p90"]) + 0.45 * float(metric.get("source_nose_anchor_distance", 999.0)),
            float(score),
        )
        if best is None or rank < best[0]:
            best = (rank, scale, rot, trans, aligned, metric, orient, hist, ci)
        print(
            f"{case} candidate_only cand={ci}/{len(initial)} pass={is_pass} "
            f"score={score:.6f} med={metric['median']:.6f} p90={metric['p90']:.6f} "
            f"anchor={metric['source_nose_anchor_distance']:.6f} up={orient['upside_down']} scale={scale:.6f}",
            flush=True,
        )

    _, scale, rot, trans, aligned, metric, orient, hist, best_ci = best
    case_dir = OUT / case
    case_dir.mkdir(parents=True, exist_ok=True)
    sim_reg = case_dir / f"{case}_similarity_icp_regframe.obj"
    full40.rigid.write_obj_like(src_obj, sim_reg, aligned)
    fig = case_dir / f"{case}_targeted_candidate_only_no_review.png"
    report = {
        "case": case,
        "method": "allpairs nose-anchor-targeted candidate-only recovery: strict-QC best refined candidate",
        "selected_stage": "targeted_candidate_only",
        "selected_candidate": int(best_ci),
        "source_obj": str(src_obj),
        "target_mesh": str(target_mesh),
        "similarity_regframe_obj": str(sim_reg),
        "diagnostic_png": str(fig),
        "rigid_scale": float(scale),
        "adaptive_scale_only_scale": float(scale),
        "similarity_scale": float(scale),
        "final_similarity_scale": float(scale),
        "eye_ellipses": eye_ellipses,
        "tried_initializations": tried,
        "stage_scores": [{"stage": "targeted_candidate_only", "score": float(best[0][2]), **metric, **orient}],
        "rigid_metrics": metric,
        "rigid_orientation_metrics": orient,
        "adaptive_scale_only_metrics": metric,
        "adaptive_scale_only_orientation_metrics": orient,
        "similarity_metrics": metric,
        "similarity_orientation_metrics": orient,
        "scale_only_history": [],
        "similarity_history": hist,
        "final_small_pose_history": [],
    }
    report_path(case).write_text(json.dumps(report, indent=2), encoding="utf-8")
    row = full40.qc_row(report)
    print(
        f"{case} candidate_only selected={best_ci} qc={row['qc_pass']} "
        f"med={row['similarity_median']:.6f} p90={row['similarity_p90']:.6f} "
        f"anchor={row['source_nose_anchor_distance']:.6f}",
        flush=True,
    )
    return report


def run_one(case):
    configure()
    existing = load_report(case)
    rerun_fail = os.environ.get("RIGID_TARGETED_RERUN_FAIL", "1") != "0"
    if existing is not None:
        try:
            row = full40.qc_row(existing)
            if row["qc_pass"] or not rerun_fail:
                return case, "skipped", existing
        except Exception:
            if not rerun_fail:
                return case, "skipped", existing

    by_pair = full40.read_manifest()
    keep = int(os.environ.get("RIGID_TARGETED_KEEP", "24"))
    rigid_iters = int(os.environ.get("RIGID_TARGETED_ITERS", "32"))
    if os.environ.get("RIGID_TARGETED_CANDIDATE_ONLY", "0") != "0":
        report = process_candidate_only(case, by_pair, keep=keep, rigid_iters=rigid_iters)
    else:
        report = full40.process(case, by_pair, keep=keep, rigid_iters=rigid_iters)
    report["method"] = "allpairs nose-anchor-targeted recovery: original hardgate plus explicit source-nose-to-target-anchor initial translations"
    report["recovery_policy"] = {
        "input_root": str(BASE_RIGID_ROOT),
        "candidate_keep": keep,
        "rigid_iters": rigid_iters,
        "candidate_generator": "run_rigid_fail6_nose_anchor_targeted_fast_20260622.nose_anchor_initial_candidates",
        "qc": "same hard orientation gate and strict full40 thresholds",
    }
    report_path(case).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return case, "processed", report


def write_summary(cases, reports_by_case):
    configure()
    reports = [reports_by_case[case] for case in cases if case in reports_by_case]
    rows, aggregate = full40.write_summary(reports)
    aggregate["input_rigid_root"] = str(BASE_RIGID_ROOT)
    aggregate["input_aggregate"] = str(BASE_AGG)
    aggregate["target_subjects"] = sorted(selected_subjects()) if selected_subjects() else "all_failed"
    aggregate["recovery_policy"] = "allpairs nose-anchor-targeted candidates with strict full40 QC"
    (OUT / "allpairs_nose_anchor_targeted_aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    if rows:
        with (OUT / "allpairs_nose_anchor_targeted_summary.csv").open("w", encoding="utf-8", newline="") as f:
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
    todo = []
    rerun_fail = os.environ.get("RIGID_TARGETED_RERUN_FAIL", "1") != "0"
    for case in cases:
        report = reports_by_case.get(case)
        if report is None:
            todo.append(case)
            continue
        try:
            if rerun_fail and not full40.qc_row(report)["qc_pass"]:
                todo.append(case)
        except Exception:
            if rerun_fail:
                todo.append(case)
    workers = int(os.environ.get("RIGID_TARGETED_WORKERS", "2"))
    print(json.dumps({"total": len(cases), "existing": len(reports_by_case), "todo": len(todo), "workers": workers}), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, case): case for case in todo}
        for future in as_completed(futures):
            case = futures[future]
            _, status, report = future.result()
            reports_by_case[case] = report
            _, aggregate = write_summary(cases, reports_by_case)
            print(json.dumps({"case": case, "status": status, **aggregate}), flush=True)
    reports = [reports_by_case[case] for case in cases if case in reports_by_case]
    (OUT / "allpairs_nose_anchor_targeted_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    _, aggregate = write_summary(cases, reports_by_case)
    print(json.dumps({"output_root": str(OUT), **aggregate}, indent=2), flush=True)


if __name__ == "__main__":
    main()


