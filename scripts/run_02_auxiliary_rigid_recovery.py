#!/usr/bin/env python3
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
IMPL = ROOT / "paperfit_legacy_impl"
sys.path.insert(0, str(IMPL))

for script in [
    "run_rigid_upright_hardgate_allpairs_recovery_20260622.py",
    "run_rigid_allpairs_subject004_prior_recovery_20260622.py",
    "run_rigid_allpairs_nose_anchor_targeted_recovery_20260622.py",
    "make_allpairs_effective_status_20260623.py",
]:
    runpy.run_path(str(IMPL / script), run_name="__main__")

