#!/usr/bin/env python3
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
IMPL = ROOT / "paperfit_legacy_impl"
sys.path.insert(0, str(IMPL))

for script in [
    "run_rigid_open3d_baselines_allpairs_20260623.py",
    "run_rigid_open3d_facecrop_baselines_allpairs_20260704.py",
    "run_dai_like_adaptive_template_allpairs_20260624.py",
]:
    runpy.run_path(str(IMPL / script), run_name="__main__")

