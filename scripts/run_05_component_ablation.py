#!/usr/bin/env python3
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
IMPL = ROOT / "paperfit_legacy_impl"
sys.path.insert(0, str(IMPL))
runpy.run_path(str(IMPL / "run_nonrigid_component_ablation_representative40_20260624.py"), run_name="__main__")

