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


BASE_RIGID_ROOT = ROOT / "research_rigid_upright_hardgate_allpairs_20260622"
BASE_AGG = BASE_RIGID_ROOT / "upright_hardgate_allpairs_aggregate.json"
OUT = ROOT / "research_rigid_upright_hardgate_allpairs_recovery_20260622"
_ORIGINAL_WRITE_OBJ_LIKE = full40.rigid.write_obj_like
_ORIGINAL_PLOT_REVIEW = full40.rigid.plot_review


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
    if os.environ.get("RIGID_RECOVERY_FAST_FINAL", "0") == "0":
        return

    def skip_final_small_pose_search(src, target_full, fit_weight, scale, rot, trans, region_masks=None, nose_anchor=None):
        return scale, rot, trans, [{
            "round": 0,
            "move": "skipped_for_allpairs_recovery",
            "scale": float(scale),
        }]

    full40.rigid.final_small_pose_search = skip_final_small_pose_search


def configure():
    full40.OUT = OUT
    enable_fast_outputs()
    enable_fast_final_polish()


def read_cases():
    aggregate = json.loads(BASE_AGG.read_text(encoding="utf-8"))
    return list(aggregate["fail_cases"])


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
    report = full40.process(case, by_pair, keep=32, rigid_iters=36)
    report["method"] = "allpairs fail recovery: expanded candidate retention and longer rigid refinement"
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
        with (OUT / "allpairs_recovery_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    aggregate = {
        "input_rigid_root": str(BASE_RIGID_ROOT),
        "attempted": len(rows),
        "total_cases": len(cases),
        "qc_pass": int(sum(row["qc_pass"] for row in rows)),
        "qc_fail": int(len(rows) - sum(row["qc_pass"] for row in rows)),
        "upside_down_remaining": [row["case"] for row in rows if row["upside_down"]],
        "pass_cases": [row["case"] for row in rows if row["qc_pass"]],
        "fail_cases": [row["case"] for row in rows if not row["qc_pass"]],
        "recovery_policy": "keep=32, rigid_iters=36, same strict QC thresholds as main allpairs rigid",
    }
    (OUT / "allpairs_recovery_aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return rows, aggregate


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    configure()
    cases = read_cases()
    reports_by_case = {case: load_report(case) for case in cases if load_report(case) is not None}
    write_summary(cases, reports_by_case)
    todo = [case for case in cases if case not in reports_by_case]
    workers = int(os.environ.get("RIGID_RECOVERY_WORKERS", "4"))
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
    (OUT / "allpairs_recovery_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    _, aggregate = write_summary(cases, reports_by_case)
    print(json.dumps({"output_root": str(OUT), **aggregate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
