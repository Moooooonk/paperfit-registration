#!/usr/bin/env python3
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(os.environ.get("PAPERFIT_ROOT", "/path/to/prepared/facescape_pipeline"))
TOOLS = ROOT / "research_tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_scratch_surface_registration_3case_final_attempt as rigid  # noqa: E402
import run_rigid_upright_hardgate_3case_20260619 as hard3  # noqa: E402
import run_rigid_upright_hardgate_full40_20260619 as full40  # noqa: E402


MANIFEST = ROOT / "prepared_001_020_cv" / "facescape_frontal_pairs_manifest.csv"
HRN_ROOT = ROOT / "hrn_outputs_001_020"
BASE_RIGID_ROOT = ROOT / "research_rigid_upright_hardgate_allpairs_20260622"
SUBJECT_ID = os.environ.get("RIGID_SUBJECT_PRIOR_SUBJECT", "004")
OUT = Path(os.environ.get(
    "RIGID_SUBJECT_PRIOR_OUT",
    str(ROOT / "research_rigid_allpairs_subject004_prior_recovery_20260622"),
))

DEFAULT_DONOR_REPORTS = [
    ROOT / "research_rigid_proper_rotation_failed14_20260614" / "004_1_neutral" / "004_1_neutral_scratch_surface_registration_report.json",
    ROOT / "research_rigid_global_upright_failed11_20260617" / "004_10_dimpler" / "004_10_dimpler_scratch_surface_registration_report.json",
]
if os.environ.get("RIGID_SUBJECT_PRIOR_DONOR_REPORTS", "").strip():
    DONOR_REPORTS = [
        Path(item.strip())
        for item in os.environ["RIGID_SUBJECT_PRIOR_DONOR_REPORTS"].split(",")
        if item.strip()
    ]
else:
    DONOR_REPORTS = DEFAULT_DONOR_REPORTS

_DONOR_CACHE = None


def read_manifest():
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return {r["pair_id"]: r for r in csv.DictReader(f)}


def hrn_obj(pair_id):
    return HRN_ROOT / pair_id / f"{pair_id}_0_hrn_mid_mesh.obj"


def load_vertices(path):
    mesh = trimesh.load(str(path), process=False)
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def solve_global_similarity(src, dst, weights=None):
    if weights is None:
        weights = np.ones(len(src), dtype=np.float64)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    weights = weights / max(float(weights.sum()), 1e-12)
    src_c = np.sum(src * weights[:, None], axis=0)
    dst_c = np.sum(dst * weights[:, None], axis=0)
    a = src - src_c
    b = dst - dst_c
    cov = a.T @ (b * weights[:, None])
    u, svals, vt = np.linalg.svd(cov)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1] *= -1
        svals[-1] *= -1
        rot = vt.T @ u.T
    var = float(np.sum(weights * np.sum(a * a, axis=1)))
    scale = float(np.sum(svals) / max(var, 1e-12))
    trans = dst_c - scale * (src_c @ rot.T)
    return scale, rot, trans


def report_similarity_obj(report):
    for key in ("similarity_regframe_obj", "recovered_regframe_obj"):
        value = report.get(key)
        if value and Path(value).exists():
            return Path(value)
    case = report["case"]
    root = Path(report.get("report_json", "")).parent
    for name in (
        f"{case}_similarity_icp_regframe.obj",
        f"{case}_subject_prior_recovered_regframe.obj",
    ):
        path = root / name
        if path.exists():
            return path
    return None


def donor_transforms():
    global _DONOR_CACHE
    if _DONOR_CACHE is not None:
        return _DONOR_CACHE
    donors = []
    for report_path in DONOR_REPORTS:
        print(f"donor_transform load {report_path}", flush=True)
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        case = report["case"]
        aligned_obj = report_similarity_obj(report)
        src_obj = hrn_obj(case)
        if aligned_obj is None or not src_obj.exists():
            continue
        src_v, _ = load_vertices(src_obj)
        aligned_v, _ = load_vertices(aligned_obj)
        sample = rigid.sample_idx(len(src_v), min(len(src_v), 12000), 914)
        scale, rot, trans = solve_global_similarity(src_v[sample], aligned_v[sample])
        fit_weight_base, eye_mask, _ = rigid.texture_eye_weight(src_obj)
        _, region_masks = rigid.apply_nose_midface_weight(src_obj, src_v, fit_weight_base)
        self_orient = hard3.orientation_metrics(rigid.transform(src_v, scale, rot, trans), region_masks, eye_mask)
        print(
            f"donor_transform ready {case} scale={scale:.6f} "
            f"upside_down={self_orient.get('upside_down')}",
            flush=True,
        )
        donors.append({
            "case": case,
            "report": str(report_path),
            "aligned_obj": str(aligned_obj),
            "scale": scale,
            "rot": rot,
            "trans": trans,
            "self_orientation": self_orient,
        })
    if not donors:
        raise RuntimeError("No usable subject-004 donor transforms were found.")
    _DONOR_CACHE = donors
    return donors


def read_cases():
    cases = []
    for path in sorted(BASE_RIGID_ROOT.glob(f"{SUBJECT_ID}_*/*_scratch_surface_registration_report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        row = full40.qc_row(report)
        if not row["qc_pass"]:
            cases.append(path.parent.name)
    return cases


def report_path(case):
    return OUT / case / f"{case}_scratch_surface_registration_report.json"


def load_report(case):
    path = report_path(case)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def fit_from_donor(case, donor, by_pair):
    print(f"{case} donor={donor['case']} fit start", flush=True)
    row = by_pair[case]
    target_row = by_pair[f"{int(row['subject']):03d}_18_eye_closed"]
    src_obj = hrn_obj(case)
    _, src_v, _ = rigid.load_mesh(src_obj)
    target_mesh = Path(target_row["mesh"])
    _, target_world, _ = rigid.load_mesh(target_mesh)
    cam_json = target_mesh.parent / "selected_camera.json"
    target_reg = rigid.target_registration_frame(target_world, cam_json)
    nose_anchor = rigid.target_nose_anchor(target_reg)
    fit_weight_base, eye_mask, eye_ellipses = rigid.texture_eye_weight(src_obj)
    fit_weight, region_masks = rigid.apply_nose_midface_weight(src_obj, src_v, fit_weight_base)

    best = None
    tried = []
    scale_steps = int(os.environ.get("RIGID_SUBJECT004_PRIOR_SCALE_STEPS", "9"))
    rigid_iters = int(os.environ.get("RIGID_SUBJECT004_PRIOR_ITERS", "28"))
    final_polish = os.environ.get("RIGID_SUBJECT004_PRIOR_FINAL", "0") != "0"
    for scale_factor in np.linspace(0.88, 1.12, scale_steps):
        scale0 = float(donor["scale"] * scale_factor)
        rot0 = donor["rot"].copy()
        trans0 = donor["trans"].copy()

        aligned0 = rigid.transform(src_v, scale0, rot0, trans0)
        idx = np.flatnonzero(fit_weight > 0.35)
        idx = idx[rigid.sample_idx(len(idx), min(len(idx), int(os.environ.get("RIGID_SUBJECT004_PRIOR_SRC_SAMPLE", "8000"))), 701)]
        target_eval = target_reg[
            rigid.sample_idx(len(target_reg), min(len(target_reg), int(os.environ.get("RIGID_SUBJECT004_PRIOR_TARGET_SAMPLE", "50000"))), 702)
        ]
        tree = rigid.cKDTree(target_eval)
        d, nn = tree.query(aligned0[idx], k=1, workers=-1)
        keep = d <= np.quantile(d, 0.62)
        if int(keep.sum()) >= 300:
            delta = target_eval[nn[keep]] - aligned0[idx][keep]
            trans0 = trans0 + np.average(delta, axis=0, weights=fit_weight[idx][keep])

        scale, rot, trans, hist = rigid.refine_rigid_fixed_scale(
            src_v,
            target_reg,
            fit_weight,
            scale0,
            rot0,
            trans0,
            iterations=rigid_iters,
            region_masks=region_masks,
            nose_anchor=nose_anchor,
        )
        scale, rot, trans = rigid.translation_only_polish(
            src_v,
            target_reg,
            fit_weight,
            scale,
            rot,
            trans,
            iterations=int(os.environ.get("RIGID_SUBJECT004_PRIOR_TRANSLATION_ITERS", "8")),
        )
        aligned = rigid.transform(src_v, scale, rot, trans)
        metric = rigid.metrics(aligned, target_reg, fit_weight, region_masks=region_masks, nose_anchor=nose_anchor)
        orient = hard3.orientation_metrics(aligned, region_masks, eye_mask)
        score = float(
            metric["median"]
            + 0.35 * metric["p90"]
            + 1.9 * max(0.0, metric["source_nose_anchor_distance"] - 0.08)
            + 1.0 * max(0.0, metric.get("source_mouth_to_nose_guard", 0.0) - 0.01)
            + 10.0 * int(bool(orient.get("upside_down", 0)))
            + 3.5 * max(0.0, float(orient.get("upright_penalty", 0.0)))
            + 2.0 * max(0.0, 0.08 - float(orient.get("eye_over_mouth_norm", 0.0)))
        )
        tried.append({
            "donor": donor["case"],
            "scale_factor": float(scale_factor),
            "score": score,
            "history_last": float(hist[-1]) if hist else None,
            **metric,
            **orient,
        })
        if best is None or score < best[0]:
            best = (score, donor, scale, rot, trans, aligned, metric, hist)
        print(
            f"{case} donor={donor['case']} scale_factor={scale_factor:.3f} "
            f"score={score:.6f} med={metric['median']:.6f} "
            f"p90={metric['p90']:.6f} anchor={metric['source_nose_anchor_distance']:.6f} "
            f"upside_down={orient.get('upside_down')}",
            flush=True,
        )

    _, donor, scale, rot, trans, aligned, metric, hist = best
    final_hist = []
    if final_polish:
        final_scale, final_rot, final_trans, final_hist = rigid.final_small_pose_search(
            src_v, target_reg, fit_weight, scale, rot, trans, region_masks=region_masks, nose_anchor=nose_anchor
        )
        final_aligned = rigid.transform(src_v, final_scale, final_rot, final_trans)
        final_metric = rigid.metrics(final_aligned, target_reg, fit_weight, region_masks=region_masks, nose_anchor=nose_anchor)
        if final_metric["source_nose_anchor_distance"] <= metric["source_nose_anchor_distance"] + 0.012:
            scale, rot, trans, aligned, metric = final_scale, final_rot, final_trans, final_aligned, final_metric
        else:
            final_hist = []
    orient = hard3.orientation_metrics(aligned, region_masks, eye_mask)

    case_dir = OUT / case
    case_dir.mkdir(parents=True, exist_ok=True)
    sim_reg = case_dir / f"{case}_similarity_icp_regframe.obj"
    rigid.write_obj_like(src_obj, sim_reg, aligned)
    fig = rigid.plot_review(
        case_dir,
        case,
        src_v,
        target_reg,
        aligned,
        aligned,
        aligned,
        fit_weight,
        eye_mask,
        region_masks,
        metric,
        metric,
        metric,
    )
    report = {
        "case": case,
        "method": f"allpairs subject-{SUBJECT_ID} prior recovery: prior successful same-subject pose transferred across expressions and guarded by strict QC",
        "source_obj": str(src_obj),
        "target_mesh": str(target_mesh),
        "similarity_regframe_obj": str(sim_reg),
        "diagnostic_png": str(fig),
        "donor_case": donor["case"],
        "donor_report": donor["report"],
        "rigid_scale": float(scale),
        "adaptive_scale_only_scale": float(scale),
        "similarity_scale": float(scale),
        "final_similarity_scale": float(scale),
        "eye_ellipses": eye_ellipses,
        "tried_subject_priors": tried,
        "stage_scores": [{"stage": "subject004_prior", **metric, **orient}],
        "rigid_metrics": metric,
        "rigid_orientation_metrics": orient,
        "adaptive_scale_only_metrics": metric,
        "adaptive_scale_only_orientation_metrics": orient,
        "similarity_metrics": metric,
        "similarity_orientation_metrics": orient,
        "final_small_pose_history": final_hist,
    }
    report_path(case).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_one(case):
    print(f"{case} subject004_prior start", flush=True)
    existing = load_report(case)
    if existing is not None:
        rerun_fail = os.environ.get("RIGID_SUBJECT004_PRIOR_RERUN_FAIL", "1") != "0"
        try:
            existing_pass = bool(full40.qc_row(existing)["qc_pass"])
        except Exception:
            existing_pass = False
        if existing_pass or not rerun_fail:
            print(f"{case} subject004_prior skip existing qc={int(existing_pass)}", flush=True)
            return case, "skipped", existing
        print(f"{case} subject004_prior rerun existing fail", flush=True)
    by_pair = read_manifest()
    donors = donor_transforms()
    donors = sorted(
        donors,
        key=lambda d: (
            int(bool(d.get("self_orientation", {}).get("upside_down", 1))),
            0 if d["case"].startswith(f"{SUBJECT_ID}_") else 1,
        ),
    )
    upright_donors = [d for d in donors if not bool(d.get("self_orientation", {}).get("upside_down", 1))]
    if upright_donors:
        donors = upright_donors
    donor_limit = int(os.environ.get("RIGID_SUBJECT004_PRIOR_DONORS", "1"))
    donors = donors[:max(1, donor_limit)]
    report = fit_from_donor(case, donors[0], by_pair)
    if len(donors) > 1:
        reports = [report]
        for donor in donors:
            if donor["case"] == report["donor_case"]:
                continue
            alt = fit_from_donor(case, donor, by_pair)
            reports.append(alt)
        def report_rank(r):
            row = full40.qc_row(r)
            return (
                int(not bool(row["qc_pass"])),
                int(bool(row["upside_down"])),
                float(row["source_nose_anchor_distance"]),
                float(row["similarity_median"]) + 0.35 * float(row["similarity_p90"]),
            )

        report = min(reports, key=report_rank)
        report_path(case).write_text(json.dumps(report, indent=2), encoding="utf-8")
    row = full40.qc_row(report)
    print(
        f"{case} subject004_prior done donor={report.get('donor_case')} "
        f"qc={row['qc_pass']} med={row['similarity_median']:.6f} "
        f"p90={row['similarity_p90']:.6f} anchor={row['source_nose_anchor_distance']:.6f}",
        flush=True,
    )
    return case, "processed", report


def qc_row(report):
    row = full40.qc_row(report)
    row["report_json"] = str(report_path(report["case"]))
    return row


def write_summary(cases, reports_by_case):
    rows = [qc_row(reports_by_case[case]) for case in cases if case in reports_by_case]
    if rows:
        with (OUT / "subject004_prior_recovery_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    aggregate = {
        "input_rigid_root": str(BASE_RIGID_ROOT),
        "subject_id": SUBJECT_ID,
        "donor_reports": [str(p) for p in DONOR_REPORTS],
        "attempted": len(rows),
        "total_cases": len(cases),
        "qc_pass": int(sum(row["qc_pass"] for row in rows)),
        "qc_fail": int(len(rows) - sum(row["qc_pass"] for row in rows)),
        "pass_cases": [row["case"] for row in rows if row["qc_pass"]],
        "fail_cases": [row["case"] for row in rows if not row["qc_pass"]],
        "recovery_policy": f"subject-{SUBJECT_ID} prior transform from successful same-subject rigid alignments",
    }
    (OUT / f"subject{SUBJECT_ID}_prior_recovery_aggregate.json").write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )
    if SUBJECT_ID == "004":
        (OUT / "subject004_prior_recovery_aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return rows, aggregate


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = read_cases()
    reports_by_case = {case: load_report(case) for case in cases if load_report(case) is not None}
    write_summary(cases, reports_by_case)
    rerun_fail = os.environ.get("RIGID_SUBJECT004_PRIOR_RERUN_FAIL", "1") != "0"
    todo = []
    for case in cases:
        report = reports_by_case.get(case)
        if report is None:
            todo.append(case)
            continue
        if rerun_fail:
            try:
                if not full40.qc_row(report)["qc_pass"]:
                    todo.append(case)
            except Exception:
                todo.append(case)
    workers = int(os.environ.get("RIGID_SUBJECT004_PRIOR_WORKERS", "2"))
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
    (OUT / "subject004_prior_recovery_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    _, aggregate = write_summary(cases, reports_by_case)
    print(json.dumps({"output_root": str(OUT), **aggregate}, indent=2), flush=True)


if __name__ == "__main__":
    main()


