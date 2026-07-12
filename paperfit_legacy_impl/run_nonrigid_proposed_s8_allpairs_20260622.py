#!/usr/bin/env python3
import json
import os
import sys
import os
from pathlib import Path


ROOT = Path(os.environ.get("PAPERFIT_ROOT", "/path/to/prepared/facescape_pipeline"))
TOOLS = ROOT / "research_tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_nonrigid_proposed_s8_hardgate_full40_pass32_20260619 as pass32  # noqa: E402
import os


MAIN_RIGID_ROOT = ROOT / "research_rigid_upright_hardgate_allpairs_20260622"
MAIN_AGG = MAIN_RIGID_ROOT / "upright_hardgate_allpairs_aggregate.json"
RECOVERY_ROOT = ROOT / "research_rigid_upright_hardgate_allpairs_recovery_20260622"
RECOVERY_AGG = RECOVERY_ROOT / "allpairs_recovery_aggregate.json"
SUBJECT004_PRIOR_ROOT = ROOT / "research_rigid_allpairs_subject004_prior_recovery_20260622"
SUBJECT004_PRIOR_AGG = SUBJECT004_PRIOR_ROOT / "subject004_prior_recovery_aggregate.json"
SUBJECT007_PRIOR_ROOT = ROOT / "research_rigid_allpairs_subject007_prior_recovery_20260622"
SUBJECT007_PRIOR_AGG = SUBJECT007_PRIOR_ROOT / "subject007_prior_recovery_aggregate.json"
SUBJECT007_PRIOR_LEGACY_AGG = SUBJECT007_PRIOR_ROOT / "subject004_prior_recovery_aggregate.json"
TARGETED_ROOT = ROOT / "research_rigid_allpairs_nose_anchor_targeted_recovery_20260622"
TARGETED_AGG = TARGETED_ROOT / "allpairs_nose_anchor_targeted_aggregate.json"
MERGED_RIGID_ROOT = ROOT / "research_rigid_upright_hardgate_allpairs_mergedpass_20260622"
MERGED_AGG = MERGED_RIGID_ROOT / "allpairs_mergedpass_aggregate.json"
OUT = ROOT / "research_nonrigid_proposed_s8_allpairs_20260622"


def read_cases():
    aggregate = json.loads(MERGED_AGG.read_text(encoding="utf-8"))
    return list(aggregate["pass_cases"])


def link_case(src_root, case):
    src = src_root / case
    dst = MERGED_RIGID_ROOT / case
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src, target_is_directory=True)
    except OSError:
        import shutil
        shutil.copytree(src, dst)


def read_aggregate(path):
    path = Path(path)
    if not path.exists():
        return {}, [], []
    d = json.loads(path.read_text(encoding="utf-8"))
    return d, list(d.get("pass_cases", [])), list(d.get("fail_cases", []))


def read_first_existing(*paths):
    for path in paths:
        d, pass_cases, fail_cases = read_aggregate(path)
        if d:
            return d, pass_cases, fail_cases
    return {}, [], []


def targeted_roots():
    roots = [TARGETED_ROOT]
    roots.extend(sorted(ROOT.glob("research_rigid_allpairs_nose_anchor_targeted*_20260622")))
    seen = set()
    unique = []
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        unique.append(root)
    return unique


def prepare_merged_rigid_root():
    MERGED_RIGID_ROOT.mkdir(parents=True, exist_ok=True)
    main = json.loads(MAIN_AGG.read_text(encoding="utf-8"))
    main_pass = list(main["pass_cases"])
    _, recovery_pass, recovery_fail = read_aggregate(RECOVERY_AGG)
    _, subject004_prior_pass, subject004_prior_fail = read_aggregate(SUBJECT004_PRIOR_AGG)
    _, subject007_prior_pass, subject007_prior_fail = read_first_existing(
        SUBJECT007_PRIOR_AGG,
        SUBJECT007_PRIOR_LEGACY_AGG,
    )
    targeted_pass = []
    targeted_fail = []
    targeted_root_pass_counts = {}
    for root in targeted_roots():
        _, pass_cases, fail_cases = read_aggregate(root / "allpairs_nose_anchor_targeted_aggregate.json")
        targeted_pass.extend(pass_cases)
        targeted_fail.extend(fail_cases)
        if pass_cases or fail_cases:
            targeted_root_pass_counts[str(root)] = len(pass_cases)
    for case in main_pass:
        link_case(MAIN_RIGID_ROOT, case)
    for case in recovery_pass:
        link_case(RECOVERY_ROOT, case)
    for case in subject004_prior_pass:
        link_case(SUBJECT004_PRIOR_ROOT, case)
    for case in subject007_prior_pass:
        link_case(SUBJECT007_PRIOR_ROOT, case)
    for root in targeted_roots():
        _, pass_cases, _ = read_aggregate(root / "allpairs_nose_anchor_targeted_aggregate.json")
        for case in pass_cases:
            link_case(root, case)
    all_pass = main_pass + recovery_pass + subject004_prior_pass + subject007_prior_pass + targeted_pass
    cases = sorted(set(all_pass))
    remaining = recovery_fail if recovery_fail else list(main.get("fail_cases", []))
    remaining = sorted(set(remaining) - set(all_pass))
    remaining = sorted(set(remaining + subject004_prior_fail + subject007_prior_fail + targeted_fail) - set(all_pass))
    aggregate = {
        "main_rigid_root": str(MAIN_RIGID_ROOT),
        "recovery_rigid_root": str(RECOVERY_ROOT),
        "subject004_prior_rigid_root": str(SUBJECT004_PRIOR_ROOT),
        "subject007_prior_rigid_root": str(SUBJECT007_PRIOR_ROOT),
        "targeted_rigid_roots": [str(root) for root in targeted_roots()],
        "attempted_main": main.get("attempted"),
        "main_pass": len(main_pass),
        "recovery_pass": len(recovery_pass),
        "subject004_prior_pass": len(subject004_prior_pass),
        "subject007_prior_pass": len(subject007_prior_pass),
        "targeted_pass": len(targeted_pass),
        "targeted_root_pass_counts": targeted_root_pass_counts,
        "merged_pass": len(cases),
        "pass_cases": cases,
        "remaining_fail_cases": remaining,
    }
    MERGED_AGG.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return aggregate


def main():
    prepare_merged_rigid_root()
    pass32.RIGID_ROOT = MERGED_RIGID_ROOT
    pass32.RIGID_AGG = MERGED_AGG
    pass32.OUT = OUT
    pass32.read_cases = read_cases
    pass32.main()


if __name__ == "__main__":
    main()


