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


MANIFEST = ROOT / "prepared_001_020_cv" / "facescape_frontal_pairs_manifest.csv"
OUT = ROOT / "research_rigid_upright_hardgate_allpairs_20260622"
_ORIGINAL_WRITE_OBJ_LIKE = full40.rigid.write_obj_like
_ORIGINAL_PLOT_REVIEW = full40.rigid.plot_review
_ORIGINAL_FINAL_SMALL_POSE_SEARCH = full40.rigid.final_small_pose_search
_ORIGINAL_ADAPTIVE_SCALE_PREFIT_ONLY = full40.rigid.adaptive_scale_prefit_only
_ORIGINAL_SCALE_FIRST_THEN_RIGID_ICP = full40.rigid.scale_first_then_rigid_icp


def enable_fast_outputs():
    if os.environ.get("RIGID_ALLPAIRS_FAST_OUTPUTS", "1") == "0":
        full40.rigid.write_obj_like = _ORIGINAL_WRITE_OBJ_LIKE
        full40.rigid.plot_review = _ORIGINAL_PLOT_REVIEW
        return

    def write_similarity_only(src_obj, out_obj, vertices):
        out_obj = Path(out_obj)
        if out_obj.name.endswith("_similarity_icp_regframe.obj"):
            return _ORIGINAL_WRITE_OBJ_LIKE(src_obj, out_obj, vertices)
        out_obj.parent.mkdir(parents=True, exist_ok=True)
        return None

    def skip_review(case_dir, case, *args, **kwargs):
        return Path(case_dir) / f"{case}_allpairs_fast_no_review.png"

    full40.rigid.write_obj_like = write_similarity_only
    full40.rigid.plot_review = skip_review


def enable_fast_final_polish():
    if os.environ.get("RIGID_ALLPAIRS_FAST_FINAL", "1") == "0":
        full40.rigid.final_small_pose_search = _ORIGINAL_FINAL_SMALL_POSE_SEARCH
        return

    def skip_final_small_pose_search(src, target_full, fit_weight, scale, rot, trans, region_masks=None, nose_anchor=None):
        return scale, rot, trans, [{
            "round": 0,
            "move": "skipped_for_allpairs_main",
            "scale": float(scale),
        }]

    full40.rigid.final_small_pose_search = skip_final_small_pose_search


def enable_fast_scale_refinement():
    if os.environ.get("RIGID_ALLPAIRS_FAST_SCALE", "1") == "0":
        full40.rigid.adaptive_scale_prefit_only = _ORIGINAL_ADAPTIVE_SCALE_PREFIT_ONLY
        full40.rigid.scale_first_then_rigid_icp = _ORIGINAL_SCALE_FIRST_THEN_RIGID_ICP
        return

    def skip_adaptive_scale_prefit_only(src, target_full, fit_weight, scale, rot, trans, region_masks=None, nose_anchor=None, rounds=6):
        return scale, rot, trans, [{
            "round": 0,
            "move": "skipped_for_allpairs_main",
            "scale": float(scale),
        }]

    def skip_scale_first_then_rigid_icp(src, target_full, fit_weight, scale, rot, trans, region_masks=None, nose_anchor=None, rounds=6):
        return scale, rot, trans, [{
            "round": 0,
            "move": "skipped_for_allpairs_main",
            "scale": float(scale),
        }]

    full40.rigid.adaptive_scale_prefit_only = skip_adaptive_scale_prefit_only
    full40.rigid.scale_first_then_rigid_icp = skip_scale_first_then_rigid_icp


def configure(fast_final=None, fast_scale=None):
    full40.OUT = OUT
    old_fast_final = os.environ.get("RIGID_ALLPAIRS_FAST_FINAL")
    old_fast_scale = os.environ.get("RIGID_ALLPAIRS_FAST_SCALE")
    if fast_final is not None:
        os.environ["RIGID_ALLPAIRS_FAST_FINAL"] = "1" if fast_final else "0"
    if fast_scale is not None:
        os.environ["RIGID_ALLPAIRS_FAST_SCALE"] = "1" if fast_scale else "0"
    enable_fast_outputs()
    enable_fast_final_polish()
    enable_fast_scale_refinement()
    if fast_final is not None:
        if old_fast_final is None:
            os.environ.pop("RIGID_ALLPAIRS_FAST_FINAL", None)
        else:
            os.environ["RIGID_ALLPAIRS_FAST_FINAL"] = old_fast_final
    if fast_scale is not None:
        if old_fast_scale is None:
            os.environ.pop("RIGID_ALLPAIRS_FAST_SCALE", None)
        else:
            os.environ["RIGID_ALLPAIRS_FAST_SCALE"] = old_fast_scale


def read_manifest_rows():
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_cases():
    cases = []
    for row in read_manifest_rows():
        pair_id = row["pair_id"]
        if pair_id.endswith("_18_eye_closed"):
            continue
        if not Path(row["mesh"]).exists():
            continue
        if not full40.rigid.hrn_obj(pair_id).exists():
            continue
        target = f"{int(row['subject']):03d}_18_eye_closed"
        cases.append(pair_id)
    return cases


def report_path(case):
    return OUT / case / f"{case}_scratch_surface_registration_report.json"


def load_report(case):
    path = report_path(case)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def run_one(case):
    inline_recovery = os.environ.get("RIGID_ALLPAIRS_INLINE_RECOVERY", "1") != "0"
    rerun_existing_fail = os.environ.get("RIGID_ALLPAIRS_RERUN_EXISTING_FAIL", "1") != "0"
    configure(fast_final=True, fast_scale=True)
    existing = load_report(case)
    if existing is not None:
        if not inline_recovery or not rerun_existing_fail or qc_row(existing)["qc_pass"]:
            return case, "skipped", existing
        print(f"{case} rerun existing QC-fail with inline full recovery", flush=True)
    by_pair = full40.read_manifest()
    report = existing
    if report is None:
        report = full40.process(case, by_pair, keep=8, rigid_iters=20)
        report["method"] = "allpairs main rigid: upright hard-gated multi-start similarity registration"
    if inline_recovery and not qc_row(report)["qc_pass"]:
        recovery_keep = int(os.environ.get("RIGID_ALLPAIRS_INLINE_KEEP", "8"))
        recovery_iters = int(os.environ.get("RIGID_ALLPAIRS_INLINE_ITERS", "20"))
        inline_fast_final = os.environ.get("RIGID_ALLPAIRS_INLINE_FAST_FINAL", "1") != "0"
        print(f"{case} fast QC-fail -> inline full recovery", flush=True)
        configure(fast_final=inline_fast_final, fast_scale=False)
        report = full40.process(case, by_pair, keep=recovery_keep, rigid_iters=recovery_iters)
        report["method"] = (
            "allpairs rigid: fast upright hard-gated screening followed by inline "
            "scale refinement for strict-QC failures"
        )
        report["inline_recovery_keep"] = recovery_keep
        report["inline_recovery_rigid_iters"] = recovery_iters
        report["inline_recovery_fast_final"] = inline_fast_final
    report_path(case).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return case, "processed", report


def qc_row(report):
    configure()
    row = full40.qc_row(report)
    row["report_json"] = str(report_path(report["case"]))
    return row


def write_summary(cases, reports_by_case):
    rows = [qc_row(reports_by_case[case]) for case in cases if case in reports_by_case]
    if rows:
        with (OUT / "upright_hardgate_allpairs_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    aggregate = {
        "attempted": len(rows),
        "total_cases": len(cases),
        "qc_pass": int(sum(row["qc_pass"] for row in rows)),
        "qc_fail": int(len(rows) - sum(row["qc_pass"] for row in rows)),
        "upside_down_remaining": [row["case"] for row in rows if row["upside_down"]],
        "pass_cases": [row["case"] for row in rows if row["qc_pass"]],
        "fail_cases": [row["case"] for row in rows if not row["qc_pass"]],
        "policy": (
            "all non-eye-closed source expressions; fast keep=8 screening; "
            "strict-QC failures rerun with full scale refinement and optional fast final; "
            "remaining failures are reserved for expanded keep=32 recovery"
        ),
    }
    (OUT / "upright_hardgate_allpairs_aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return rows, aggregate


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    configure()
    cases = read_cases()
    reports_by_case = {case: load_report(case) for case in cases if load_report(case) is not None}
    write_summary(cases, reports_by_case)
    inline_recovery = os.environ.get("RIGID_ALLPAIRS_INLINE_RECOVERY", "1") != "0"
    rerun_existing_fail = os.environ.get("RIGID_ALLPAIRS_RERUN_EXISTING_FAIL", "1") != "0"
    todo = []
    for case in cases:
        report = reports_by_case.get(case)
        if report is None:
            todo.append(case)
        elif inline_recovery and rerun_existing_fail and not qc_row(report)["qc_pass"]:
            todo.append(case)
    workers = int(os.environ.get("RIGID_ALLPAIRS_WORKERS", "4"))
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
    (OUT / "upright_hardgate_allpairs_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    _, aggregate = write_summary(cases, reports_by_case)
    print(json.dumps({"output_root": str(OUT), **aggregate}, indent=2), flush=True)


if __name__ == "__main__":
    main()


