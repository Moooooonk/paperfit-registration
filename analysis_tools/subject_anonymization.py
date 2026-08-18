"""Stable partition-aware subject labels that do not expose dataset IDs."""

from __future__ import annotations

from typing import Any


def normalize_subject(value: object) -> str:
    return f"{int(value):03d}"


def build_anonymized_subject_labels(split: dict[str, Any]) -> dict[str, str]:
    development = [normalize_subject(value) for value in split["development_subjects"]]
    test = [normalize_subject(value) for value in split["test_subjects"]]
    if len(development) != len(set(development)) or len(test) != len(set(test)):
        raise ValueError("Duplicate subject in frozen split")
    if set(development) & set(test):
        raise ValueError("Development and held-out subjects overlap")

    all_subjects = set(development) | set(test)
    ranked_raw = split.get("ranked_subjects", development + test)
    ranked = [normalize_subject(value) for value in ranked_raw]
    if len(ranked) != len(set(ranked)) or set(ranked) != all_subjects:
        raise ValueError("ranked_subjects does not match the frozen subject set")

    development_set = set(development)
    test_set = set(test)
    development_order = [value for value in ranked if value in development_set]
    test_order = [value for value in ranked if value in test_set]
    labels = {
        subject: f"D{index:02d}"
        for index, subject in enumerate(development_order, start=1)
    }
    labels.update(
        {
            subject: f"H{index:02d}"
            for index, subject in enumerate(test_order, start=1)
        }
    )
    return labels
