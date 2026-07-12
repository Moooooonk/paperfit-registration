#!/usr/bin/env python3
import csv
import json
import os
import sys
from pathlib import Path


ROOT = Path(os.environ.get("PAPERFIT_ROOT", "/path/to/prepared/facescape_pipeline"))
TOOLS = ROOT / "research_tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_nonrigid_proposed_s8_hardgate_full40_pass32_20260619 as pass32  # noqa: E402


MAIN_RIGID_ROOT = ROOT / "research_rigid_upright_hardgate_allpairs_20260622"
MAIN_AGG = MAIN_RIGID_ROOT / "upright_hardgate_allpairs_aggregate.json"
ANCHOR_RECOVERY_AGG = (
    ROOT
    / "research_nonrigid_proposed_s8_allpairs_rough_anchorfail_20260623"
    / "nonrigid_proposed_s8_hardgate_full40_pass32_aggregate.json"
)
OUT = ROOT / "research_nonrigid_proposed_s8_allpairs_broadfail_probe_20260623"
SELECTION_JSON = OUT / "broadfail_probe_selection.json"
SELECTION_CSV = OUT / "broadfail_probe_selection.csv"
LIMIT = int(os.environ.get("NONRIGID_BROADFAIL_LIMIT", "0"))


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def aux_aggregate_paths():
    paths = [
        ROOT / "research_rigid_upright_hardgate_allpairs_recovery_20260622" / "allpairs_recovery_aggregate.json",
        ROOT / "research_rigid_allpairs_subject004_prior_recovery_20260622" / "subject004_prior_recovery_aggregate.json",
        ROOT / "research_rigid_allpairs_subject007_prior_recovery_20260622" / "subject007_prior_recovery_aggregate.json",
        ROOT / "research_rigid_allpairs_subject007_prior_recovery_20260622" / "subject004_prior_recovery_aggregate.json",
    ]
    for root in ROOT.glob("research_rigid_allpairs_nose_anchor_targeted*_20260622"):
        paths.extend(sorted(root.glob("*_aggregate.json")))
    return paths


def effective_remaining_cases():
    main = read_json(MAIN_AGG)
    failed = set(main.get("fail_cases", []))
    recovered = set(main.get("pass_cases", []))
    for path in aux_aggregate_paths():
        recovered.update(read_json(path).get("pass_cases", []))
    recovered.update(read_json(ANCHOR_RECOVERY_AGG).get("cases", []))
    return sorted(failed - recovered)


def selected_rows():
    rows = []
    for case in effective_remaining_cases():
        report = read_json(MAIN_RIGID_ROOT / case / f"{case}_scratch_surface_registration_report.json")
        metrics = report.get("similarity_metrics") or {}
        orient = report.get("similarity_orientation_metrics") or {}
        row = {
            "case": case,
            "subject": case.split("_", 1)[0],
            "median": float(metrics.get("median", 9.0)),
            "p90": float(metrics.get("p90", 9.0)),
            "nose_weighted_median": float(metrics.get("nose_weighted_median", 9.0)),
            "source_nose_anchor_distance": float(metrics.get("source_nose_anchor_distance", 9.0)),
            "upside_down": int(orient.get("upside_down", 1)),
            "vertical_z": float(orient.get("vertical_z", 0.0)),
        }
        row["failure_bucket"] = "orientation" if row["upside_down"] else "surface_or_nose_broad_error"
        rows.append(row)
    rows.sort(key=lambda row: (row["subject"], row["median"] + row["p90"], row["case"]))
    if LIMIT > 0:
        return rows[:LIMIT]
    return rows


def write_selection(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_rigid_root": str(MAIN_RIGID_ROOT),
        "anchor_recovery_aggregate": str(ANCHOR_RECOVERY_AGG),
        "selection_policy": {
            "effective_fail_after_rigid_and_anchor_nonrigid": True,
            "limit": LIMIT,
        },
        "selected": len(rows),
        "cases": [row["case"] for row in rows],
        "rows": rows,
    }
    SELECTION_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if rows:
        with SELECTION_CSV.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return payload


def read_cases():
    if not SELECTION_JSON.exists():
        write_selection(selected_rows())
    return list(read_json(SELECTION_JSON).get("cases", []))


def main():
    rows = selected_rows()
    selection = write_selection(rows)
    print(json.dumps(selection, indent=2), flush=True)
    if not rows:
        return
    pass32.RIGID_ROOT = MAIN_RIGID_ROOT
    pass32.RIGID_AGG = SELECTION_JSON
    pass32.OUT = OUT
    pass32.read_cases = read_cases
    pass32.main()


if __name__ == "__main__":
    main()


