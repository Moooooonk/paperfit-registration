#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(os.environ.get("PAPERFIT_ROOT", "/path/to/prepared/facescape_pipeline"))
TOOLS = ROOT / "research_tools"
LOG_ROOT = ROOT / "research_allpairs_pipeline_20260622"


STEPS = [
    (
        "rigid_main",
        "run_rigid_upright_hardgate_allpairs_20260622.py",
        {"RIGID_ALLPAIRS_WORKERS": os.environ.get("RIGID_ALLPAIRS_WORKERS", "4")},
    ),
    (
        "rigid_recovery",
        "run_rigid_upright_hardgate_allpairs_recovery_20260622.py",
        {"RIGID_RECOVERY_WORKERS": os.environ.get("RIGID_RECOVERY_WORKERS", "4")},
    ),
    (
        "subject004_prior_recovery",
        "run_rigid_allpairs_subject004_prior_recovery_20260622.py",
        {"RIGID_SUBJECT004_PRIOR_WORKERS": os.environ.get("RIGID_SUBJECT004_PRIOR_WORKERS", "2")},
    ),
    (
        "nonrigid_s8",
        "run_nonrigid_proposed_s8_allpairs_20260622.py",
        {"NONRIGID_WORKERS": os.environ.get("NONRIGID_WORKERS", "4")},
    ),
]


def run_step(name, script, env_extra):
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{name}.log"
    env = os.environ.copy()
    env.update(env_extra)
    cmd = [sys.executable, str(TOOLS / script)]
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    print(json.dumps({"step": name, "status": "start", "script": script, "log": str(log_path), "started": started}), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"step": name, "event": "start", "started": started, "cmd": cmd}) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        ended = time.strftime("%Y-%m-%d %H:%M:%S")
        log.write(json.dumps({"step": name, "event": "end", "returncode": proc.returncode, "ended": ended}) + "\n")
    print(json.dumps({"step": name, "status": "end", "returncode": proc.returncode, "ended": ended}), flush=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main():
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    for name, script, env_extra in STEPS:
        run_step(name, script, env_extra)
    print(json.dumps({"status": "complete", "log_root": str(LOG_ROOT)}, indent=2), flush=True)


if __name__ == "__main__":
    main()


