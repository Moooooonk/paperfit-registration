#!/usr/bin/env python3
import csv
import os
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(os.environ.get("PAPERFIT_ROOT", "/path/to/prepared/facescape_pipeline"))
MAIN_AGG = ROOT / "research_rigid_upright_hardgate_allpairs_20260622" / "upright_hardgate_allpairs_aggregate.json"
OUT = ROOT / "research_allpairs_submission_assets_20260622"


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def aux_aggregate_paths():
    paths = [
        ROOT / "research_rigid_upright_hardgate_allpairs_recovery_20260622" / "allpairs_recovery_aggregate.json",
        ROOT / "research_rigid_allpairs_subject004_prior_recovery_20260622" / "subject004_prior_recovery_aggregate.json",
        ROOT / "research_rigid_allpairs_subject007_prior_recovery_20260622" / "subject007_prior_recovery_aggregate.json",
        ROOT / "research_rigid_allpairs_subject007_prior_recovery_20260622" / "subject004_prior_recovery_aggregate.json",
    ]
    for root in sorted(ROOT.glob("research_rigid_allpairs_nose_anchor_targeted*_20260622")):
        paths.extend(sorted(root.glob("*_aggregate.json")))
    seen = set()
    for path in paths:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        yield path


def subject(case):
    return str(case).split("_", 1)[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    main_agg = read_json(MAIN_AGG)
    if not main_agg:
        raise SystemExit(f"Missing main aggregate: {MAIN_AGG}")

    main_pass = set(map(str, main_agg.get("pass_cases") or []))
    main_fail = set(map(str, main_agg.get("fail_cases") or []))
    attempted = main_pass | main_fail

    pass_sources = defaultdict(list)
    fail_sources = defaultdict(list)
    for case in main_pass:
        pass_sources[case].append("main")
    for case in main_fail:
        fail_sources[case].append("main")

    aux_sources = []
    for path in aux_aggregate_paths():
        agg = read_json(path) or {}
        source = str(path.relative_to(ROOT))
        aux_sources.append({
            "source": source,
            "attempted": agg.get("attempted"),
            "qc_pass": agg.get("qc_pass"),
            "qc_fail": agg.get("qc_fail"),
            "target_subjects": agg.get("target_subjects"),
        })
        for case in agg.get("pass_cases") or []:
            pass_sources[str(case)].append(source)
        for case in agg.get("fail_cases") or []:
            fail_sources[str(case)].append(source)

    effective_pass = attempted & set(pass_sources)
    effective_fail = attempted - effective_pass
    recovered_from_main_fail = main_fail & set(pass_sources)
    pending_total = int(main_agg.get("total_cases") or 0) - len(attempted)

    rows = []
    for case in sorted(attempted):
        rows.append({
            "case": case,
            "subject": subject(case),
            "main_status": "pass" if case in main_pass else "fail",
            "effective_status": "pass" if case in effective_pass else "fail",
            "pass_sources": ";".join(pass_sources.get(case, [])),
            "fail_sources": ";".join(fail_sources.get(case, [])),
        })

    by_subject = []
    for subj in sorted({subject(c) for c in attempted}):
        cases = [r for r in rows if r["subject"] == subj]
        by_subject.append({
            "subject": subj,
            "attempted": len(cases),
            "effective_pass": sum(1 for r in cases if r["effective_status"] == "pass"),
            "effective_fail": sum(1 for r in cases if r["effective_status"] == "fail"),
            "main_fail": sum(1 for r in cases if r["main_status"] == "fail"),
        })

    remaining_by_subject = Counter(subject(c) for c in effective_fail)
    summary = {
        "input_main_aggregate": str(MAIN_AGG),
        "main_attempted": len(attempted),
        "main_total_cases": main_agg.get("total_cases"),
        "main_qc_pass": len(main_pass),
        "main_qc_fail": len(main_fail),
        "pending_total_cases": pending_total,
        "aux_sources": aux_sources,
        "aux_unique_pass_cases": len(set(pass_sources) - main_pass),
        "recovered_from_main_fail": len(recovered_from_main_fail),
        "effective_pass_among_attempted": len(effective_pass),
        "effective_fail_among_attempted": len(effective_fail),
        "effective_pass_rate_attempted": (len(effective_pass) / len(attempted)) if attempted else None,
        "remaining_fail_by_subject": dict(sorted(remaining_by_subject.items())),
        "remaining_fail_cases": sorted(effective_fail),
        "status_note": "Live effective rigid status. Final Q1 readiness requires all 380 main cases, recovery/targeted merge, nonrigid aggregate, figures, and manuscript assets.",
    }

    (OUT / "allpairs_effective_rigid_status_live.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (OUT / "allpairs_effective_rigid_status_cases_live.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["case"])
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "allpairs_effective_rigid_status_by_subject_live.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(by_subject[0].keys()) if by_subject else ["subject"])
        writer.writeheader()
        writer.writerows(by_subject)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()


