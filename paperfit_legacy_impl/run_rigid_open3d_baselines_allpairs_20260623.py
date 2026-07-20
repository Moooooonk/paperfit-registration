#!/usr/bin/env python3
import csv
import os
import json
import sys
from pathlib import Path


ROOT = Path(os.environ.get("PAPERFIT_ROOT", "/path/to/prepared/facescape_pipeline"))
TOOLS = ROOT / "research_tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_rigid_open3d_baselines_40_20260622 as base40  # noqa: E402


OUT = ROOT / "research_rigid_open3d_baselines_allpairs_20260623"
MANIFEST = ROOT / "prepared_001_020_cv" / "facescape_frontal_pairs_manifest.csv"


def read_cases():
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    cases = [
        row["pair_id"]
        for row in rows
        if row["pair_id"].endswith("_18_eye_closed") is False
    ]
    return sorted(cases)


def main():
    base40.OUT = OUT
    base40.read_cases = read_cases
    base40.main()
    agg = json.loads((OUT / "open3d_baselines_40_aggregate.json").read_text(encoding="utf-8"))
    (OUT / "open3d_baselines_allpairs_aggregate.json").write_text(
        json.dumps(agg, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()


