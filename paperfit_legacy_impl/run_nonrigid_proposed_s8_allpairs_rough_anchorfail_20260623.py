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
OUT = ROOT / "research_nonrigid_proposed_s8_allpairs_rough_anchorfail_20260623"
SELECTION_JSON = OUT / "rough_anchorfail_selection.json"
SELECTION_CSV = OUT / "rough_anchorfail_selection.csv"

MEDIAN_MAX = float(os.environ.get("ROUGH_ANCHORFAIL_MEDIAN_MAX", "0.023"))
P90_MAX = float(os.environ.get("ROUGH_ANCHORFAIL_P90_MAX", "0.070"))
NOSE_MEDIAN_MAX = float(os.environ.get("ROUGH_ANCHORFAIL_NOSE_MEDIAN_MAX", "0.040"))
ANCHOR_MAX = float(os.environ.get("ROUGH_ANCHORFAIL_ANCHOR_MAX", "999"))
LIMIT = int(os.environ.get("NONRIGID_ROUGH_LIMIT", "16"))


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


def effective_fail_cases(main):
    failed = set(main.get("fail_cases", []))
    recovered = set(main.get("pass_cases", []))
    for path in aux_aggregate_paths():
        aggregate = read_json(path)
        recovered.update(aggregate.get("pass_cases", []))
    return sorted(failed - recovered)


def case_report(case):
    path = MAIN_RIGID_ROOT / case / f"{case}_scratch_surface_registration_report.json"
    return read_json(path)


def selected_rows():
    main = read_json(MAIN_AGG)
    rows = []
    for case in effective_fail_cases(main):
        report = case_report(case)
        metrics = report.get("similarity_metrics") or {}
        orient = report.get("similarity_orientation_metrics") or {}
        median = metrics.get("median")
        p90 = metrics.get("p90")
        nose_median = metrics.get("nose_weighted_median")
        anchor = metrics.get("source_nose_anchor_distance")
        upside_down = int(orient.get("upside_down", 1))
        if None in (median, p90, nose_median, anchor):
            continue
        if upside_down:
            continue
        if median <= MEDIAN_MAX and p90 <= P90_MAX and nose_median <= NOSE_MEDIAN_MAX and anchor <= ANCHOR_MAX:
            rows.append(
                {
                    "case": case,
                    "subject": case.split("_", 1)[0],
                    "median": float(median),
                    "p90": float(p90),
                    "nose_weighted_median": float(nose_median),
                    "source_nose_anchor_distance": float(anchor),
                    "vertical_z": float(orient.get("vertical_z", 0.0)),
                }
            )
    rows.sort(key=lambda row: (row["median"] + row["p90"] + row["nose_weighted_median"], row["case"]))
    if LIMIT > 0:
        return rows[:LIMIT]
    return rows


def write_selection(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_rigid_root": str(MAIN_RIGID_ROOT),
        "selection_policy": {
            "effective_fail_only": True,
            "upside_down": 0,
            "median_max": MEDIAN_MAX,
            "p90_max": P90_MAX,
            "nose_weighted_median_max": NOSE_MEDIAN_MAX,
            "anchor_max": ANCHOR_MAX,
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
    aggregate = read_json(SELECTION_JSON)
    return list(aggregate.get("cases", []))


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


