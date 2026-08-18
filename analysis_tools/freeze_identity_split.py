"""Create a metric-blind deterministic subject-level development/test split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


SEED = "CMES-89502-revision-20260816"
SUBJECTS = [f"{index:03d}" for index in range(1, 21)]
OUTPUT = Path(__file__).resolve().parents[1] / "identity_disjoint_split.json"


def rank_key(subject: str) -> str:
    return hashlib.sha256(f"{SEED}:{subject}".encode("ascii")).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen split: {OUTPUT}")
    ranked = sorted(SUBJECTS, key=rank_key)
    development = sorted(ranked[:10])
    test = sorted(ranked[10:])
    payload = {
        "frozen_date": "2026-08-16",
        "seed": SEED,
        "assignment_method": "SHA-256 rank of '<seed>:<three-digit subject ID>'",
        "metric_blind": True,
        "development_subjects": development,
        "test_subjects": test,
        "ranked_subjects": ranked,
        "subject_count": len(SUBJECTS),
        "development_count": len(development),
        "test_count": len(test),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
