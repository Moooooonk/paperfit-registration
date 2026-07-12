#!/usr/bin/env python3
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(os.environ.get("PAPERFIT_ROOT", "/path/to/prepared/facescape_pipeline"))
TOOLS = ROOT / "research_tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_nonrigid_nasal_depth_ablation_3case as nonrigid  # noqa: E402


RIGID_ROOT = ROOT / "research_rigid_upright_hardgate_full40_20260619"
RIGID_AGG = RIGID_ROOT / "upright_hardgate_full40_aggregate.json"
OUT = ROOT / "research_nonrigid_proposed_s8_hardgate_full40_pass32_20260619"
SCHEDULES = {"proposed_S8": [0.00, 0.22, 0.40, 0.55, 0.68, 0.78, 0.87, 0.94]}


def read_cases():
    aggregate = json.loads(RIGID_AGG.read_text(encoding="utf-8"))
    return list(aggregate["pass_cases"])


def report_path(case):
    return OUT / case / "proposed_S8" / f"{case}_proposed_S8_report.json"


def load_existing(case):
    path = report_path(case)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def configure():
    nonrigid.RIGID_ROOT = RIGID_ROOT
    nonrigid.OUT = OUT
    nonrigid.SCHEDULES = SCHEDULES


def run_one(case):
    configure()
    existing = load_existing(case)
    if existing is not None:
        return case, "skipped", existing
    manifest = nonrigid.read_manifest()
    reports = nonrigid.process_case(case, manifest)
    return case, "processed", reports[0]


def compact_row(report):
    base = report["baseline_metrics"]
    local = report["nasal_local_metrics"]
    final = report["nonrigid_final_metrics"]
    return {
        "case": report["case"],
        "schedule": report["schedule"],
        "full_median_before": base["full_no_eye"]["median"],
        "full_median_after": final["full_no_eye"]["median"],
        "midface_median_before": base["midface"]["median"],
        "midface_median_after": final["midface"]["median"],
        "nose_median_before": base["nose"]["median"],
        "nose_median_local": local["nose"]["median"],
        "nose_median_after": final["nose"]["median"],
        "nose_p90_before": base["nose"]["p90"],
        "nose_p90_after": final["nose"]["p90"],
        "nose_tip_median_before": base["nose_tip"]["median"],
        "nose_tip_median_after": final["nose_tip"]["median"],
        "eye_fixed_max": report["eye_fixed_max"],
        "displacement_p90": report["displacement_p90"],
        "review_png": report["review_png"],
        "final_obj": report["nonrigid_final_obj"],
    }


def write_outputs(cases, reports_by_case):
    rows = [compact_row(reports_by_case[case]) for case in cases if case in reports_by_case]
    if rows:
        fields = list(rows[0].keys())
        with (OUT / "nonrigid_proposed_s8_hardgate_full40_pass32_compact_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    aggregate = {
        "rigid_root": str(RIGID_ROOT),
        "processed": len(rows),
        "cases": [row["case"] for row in rows],
        "mean_nose_median_before": float(np.mean([row["nose_median_before"] for row in rows])) if rows else None,
        "mean_nose_median_after": float(np.mean([row["nose_median_after"] for row in rows])) if rows else None,
        "mean_nose_p90_before": float(np.mean([row["nose_p90_before"] for row in rows])) if rows else None,
        "mean_nose_p90_after": float(np.mean([row["nose_p90_after"] for row in rows])) if rows else None,
        "mean_full_median_before": float(np.mean([row["full_median_before"] for row in rows])) if rows else None,
        "mean_full_median_after": float(np.mean([row["full_median_after"] for row in rows])) if rows else None,
        "max_eye_fixed_max": float(np.max([row["eye_fixed_max"] for row in rows])) if rows else None,
        "full_median_improved_cases": int(sum(row["full_median_after"] <= row["full_median_before"] for row in rows)),
        "nose_median_improved_cases": int(sum(row["nose_median_after"] <= row["nose_median_before"] for row in rows)),
        "worst_after_full_cases": [row["case"] for row in sorted(rows, key=lambda row: row["full_median_after"], reverse=True)[:8]],
        "worst_after_nose_cases": [row["case"] for row in sorted(rows, key=lambda row: row["nose_median_after"], reverse=True)[:8]],
    }
    (OUT / "nonrigid_proposed_s8_hardgate_full40_pass32_aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return rows, aggregate


def make_contact_sheet(rows):
    selected = []
    seen = set()
    for key in ["full_median_after", "nose_median_after", "displacement_p90"]:
        for row in sorted(rows, key=lambda item: item[key], reverse=True)[:8]:
            if row["case"] not in seen:
                selected.append(row)
                seen.add(row["case"])
    selected = selected[:18]
    if not selected:
        return None
    font = ImageFont.load_default()
    thumb_w = 420
    pad = 14
    label_h = 30
    tiles = []
    for row in selected:
        path = Path(row["review_png"])
        if not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        scale = thumb_w / image.width
        image = image.resize((thumb_w, int(image.height * scale)), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (thumb_w, image.height + label_h), "white")
        draw = ImageDraw.Draw(tile)
        label = (
            f"{row['case']} | full {row['full_median_before']:.4f}->{row['full_median_after']:.4f} "
            f"nose {row['nose_median_before']:.4f}->{row['nose_median_after']:.4f}"
        )
        draw.text((6, 8), label, fill=(0, 0, 0), font=font)
        tile.paste(image, (0, label_h))
        tiles.append(tile)
    cols = 3
    cell_h = max(tile.height for tile in tiles)
    rows_n = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w + (cols - 1) * pad, rows_n * cell_h + (rows_n - 1) * pad), "white")
    for idx, tile in enumerate(tiles):
        x = (idx % cols) * (thumb_w + pad)
        y = (idx // cols) * (cell_h + pad)
        sheet.paste(tile, (x, y))
    out = OUT / "nonrigid_proposed_s8_hardgate_full40_pass32_worstcase_contact_sheet.png"
    sheet.save(out)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    configure()
    cases = read_cases()
    nonrigid.CASES = cases
    reports_by_case = {case: load_existing(case) for case in cases if load_existing(case) is not None}
    write_outputs(cases, reports_by_case)
    todo = [case for case in cases if case not in reports_by_case]
    workers = int(os.environ.get("NONRIGID_WORKERS", "4"))
    print(json.dumps({"total": len(cases), "existing": len(reports_by_case), "todo": len(todo), "workers": workers}), flush=True)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, case): case for case in todo}
        for future in as_completed(futures):
            case = futures[future]
            _, status, report = future.result()
            reports_by_case[case] = report
            _, aggregate = write_outputs(cases, reports_by_case)
            print(json.dumps({"case": case, "status": status, **aggregate}), flush=True)

    rows, aggregate = write_outputs(cases, reports_by_case)
    contact = make_contact_sheet(rows)
    if contact is not None:
        aggregate["worstcase_contact_sheet"] = str(contact)
        (OUT / "nonrigid_proposed_s8_hardgate_full40_pass32_aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    reports = [reports_by_case[case] for case in cases]
    (OUT / "nonrigid_proposed_s8_hardgate_full40_pass32_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(OUT), **aggregate}, indent=2), flush=True)


if __name__ == "__main__":
    main()


