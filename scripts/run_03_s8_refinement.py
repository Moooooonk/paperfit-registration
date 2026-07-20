#!/usr/bin/env python3
from pathlib import Path
import os
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
IMPL = ROOT / "paperfit_legacy_impl"
sys.path.insert(0, str(IMPL))

# The paper reports all 84 anchor-only cases. The legacy experiment script used
# 16 only as an early smoke-test default, so disable that limit for reproduction.
os.environ.setdefault("NONRIGID_ROUGH_LIMIT", "0")

for script in [
    "run_nonrigid_proposed_s8_allpairs_20260622.py",
    "run_nonrigid_proposed_s8_allpairs_rough_anchorfail_20260623.py",
    "run_nonrigid_proposed_s8_allpairs_broadfail_probe_20260623.py",
]:
    runpy.run_path(str(IMPL / script), run_name="__main__")

