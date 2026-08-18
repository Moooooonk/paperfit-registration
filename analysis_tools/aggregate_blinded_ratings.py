#!/usr/bin/env python3
"""Validate masked perceptual ratings and create final visual-reference labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


BINARY_FIELDS = (
    "anatomical_orientation",
    "visible_deformation_artifact",
    "overall_anatomical_usability",
)
ORDINAL_FIELDS = (
    "nasal_midface_alignment_1to5",
    "eye_orbit_preservation_1to5",
    "broader_facial_placement_1to5",
)
PRIMARY_ORDINAL_FIELDS = (
    "nasal_midface_alignment_1to5",
    "broader_facial_placement_1to5",
)
EYE_AUDIT_FIELD = "eye_orbit_preservation_1to5"
EXPRESSIONS_PER_SUBJECT = 19
EXPECTED_EXPRESSION_INDICES = {*range(1, 18), 19, 20}
RATING_SCOPES = {"full_independent_rater", "partial_blinded_adjudicator"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-csv", type=Path, required=True)
    parser.add_argument("--rater-csv", type=Path, action="append", required=True)
    parser.add_argument("--rater-metadata-csv", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument(
        "--subset", choices=("development", "heldout"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def read(path: Path) -> list[dict[str, str]]:
    with path.resolve(strict=True).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_case_partition(
    case_key: dict[str, str], split: dict[str, Any], subset: str
) -> list[str]:
    split_key = "development_subjects" if subset == "development" else "test_subjects"
    expected_subjects = {f"{int(value):03d}" for value in split[split_key]}
    if len(expected_subjects) != 10:
        raise ValueError(f"Frozen {subset} split must contain exactly 10 identities")
    counts: Counter[str] = Counter()
    expression_indices: dict[str, set[int]] = {
        subject: set() for subject in expected_subjects
    }
    expression_suffixes: dict[str, set[str]] = {
        subject: set() for subject in expected_subjects
    }
    cases = list(case_key.values())
    if len(cases) != len(set(cases)):
        raise ValueError("Private rating key contains duplicate case IDs")
    for case in cases:
        try:
            subject_token, suffix = str(case).split("_", 1)
            subject = f"{int(subject_token):03d}"
            expression_index = int(suffix.split("_", 1)[0])
        except (TypeError, ValueError, IndexError) as error:
            raise ValueError(f"Invalid case ID in private rating key: {case}") from error
        counts[subject] += 1
        expression_indices.setdefault(subject, set()).add(expression_index)
        expression_suffixes.setdefault(subject, set()).add(suffix)
    bad_counts = {
        subject: counts.get(subject, 0)
        for subject in sorted(expected_subjects | set(counts))
        if counts.get(subject, 0) != EXPRESSIONS_PER_SUBJECT
    }
    expected_cases = len(expected_subjects) * EXPRESSIONS_PER_SUBJECT
    if set(counts) != expected_subjects or len(cases) != expected_cases or bad_counts:
        raise ValueError(
            f"Rating key does not match the frozen {subset} partition: "
            f"cases={len(cases)}/{expected_cases}, counts={bad_counts}"
        )
    for subject in sorted(expected_subjects):
        if expression_indices[subject] != EXPECTED_EXPRESSION_INDICES:
            raise ValueError(
                f"Rating key has an invalid expression grid for {subject}: "
                f"observed={sorted(expression_indices[subject])}"
            )
    expected_suffixes = expression_suffixes[sorted(expected_subjects)[0]]
    for subject in sorted(expected_subjects):
        if expression_suffixes[subject] != expected_suffixes:
            raise ValueError(
                f"Rating key expression names differ for subject {subject}"
            )
    return sorted(expected_subjects)


def yes_no(value: str, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"yes", "y", "1", "true"}:
        return True
    if normalized in {"no", "n", "0", "false"}:
        return False
    raise ValueError(f"{field} must be Yes or No, found {value!r}")


def validate_rater_metadata(path: Path, rater_count: int) -> list[dict[str, str]]:
    rows = read(path)
    required = {
        "rater_code",
        "discipline_or_specialty",
        "current_role",
        "years_relevant_experience",
        "three_dimensional_face_or_craniofacial_experience",
        "registration_method_development_involvement",
        "access_to_automatic_metrics_or_status_during_rating",
        "facescape_access_confirmed",
        "independent_rating_confirmed",
        "rating_scope",
        "rating_completion_date",
    }
    if not rows or not required.issubset(rows[0]):
        raise KeyError(f"Rater metadata fields missing: {required - set(rows[0] if rows else ())}")
    by_code = {str(row["rater_code"]).strip(): row for row in rows}
    expected_codes = {f"R{index}" for index in range(1, rater_count + 1)}
    if len(by_code) != len(rows) or set(by_code) != expected_codes:
        raise ValueError(
            f"Rater metadata must contain exactly {sorted(expected_codes)}"
        )
    for code in sorted(by_code):
        row = by_code[code]
        for field in (
            "discipline_or_specialty",
            "current_role",
            "three_dimensional_face_or_craniofacial_experience",
            "rating_completion_date",
        ):
            if not str(row[field]).strip():
                raise ValueError(f"Rater metadata {code} is missing {field}")
        try:
            years = float(row["years_relevant_experience"])
        except ValueError as exc:
            raise ValueError(f"Invalid experience years for {code}") from exc
        if not np.isfinite(years) or years < 0.0:
            raise ValueError(f"Invalid experience years for {code}")
        yes_no(
            row["registration_method_development_involvement"],
            "registration_method_development_involvement",
        )
        if yes_no(
            row["access_to_automatic_metrics_or_status_during_rating"],
            "access_to_automatic_metrics_or_status_during_rating",
        ):
            raise ValueError(f"Rater {code} was not blinded to automatic evidence")
        if not yes_no(row["facescape_access_confirmed"], "facescape_access_confirmed"):
            raise ValueError(f"Rater {code} lacks confirmed lawful FaceScape access")
        if not yes_no(
            row["independent_rating_confirmed"], "independent_rating_confirmed"
        ):
            raise ValueError(f"Rater {code} did not confirm independent rating")
        scope = str(row["rating_scope"]).strip().lower().replace(" ", "_")
        if scope not in RATING_SCOPES:
            raise ValueError(
                f"Rater {code} has invalid rating_scope {row['rating_scope']!r}"
            )
        row["rating_scope"] = scope
        if yes_no(
            row["registration_method_development_involvement"],
            "registration_method_development_involvement",
        ):
            raise ValueError(
                f"Rater {code} was involved in development of the evaluated method"
            )
    ordered = [by_code[f"R{index}"] for index in range(1, rater_count + 1)]
    if any(row["rating_scope"] != "full_independent_rater" for row in ordered[:2]):
        raise ValueError("R1 and R2 must rate the complete case set independently")
    if rater_count == 2 and any(
        row["rating_scope"] != "full_independent_rater" for row in ordered
    ):
        raise ValueError("Two-rater aggregation requires two full independent raters")
    if rater_count == 3 and ordered[2]["rating_scope"] not in RATING_SCOPES:
        raise ValueError("R3 must be a full independent rater or blinded adjudicator")
    return ordered


def binary(value: str) -> int | None:
    normalized = value.strip().lower().replace(" ", "_")
    if normalized in {"yes", "y", "1", "true"}:
        return 1
    if normalized in {"no", "n", "0", "false"}:
        return 0
    if normalized in {"cannot_judge", "cannotjudge", "na", "n/a", ""}:
        return None
    raise ValueError(f"Invalid binary rating: {value!r}")


def ordinal(value: str) -> int | None:
    normalized = value.strip().lower().replace(" ", "_")
    if normalized in {"cannot_judge", "cannotjudge", "na", "n/a", ""}:
        return None
    numeric = float(value)
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"Ordinal rating must be an integer from 1 to 5: {value!r}")
    result = int(numeric)
    if result not in range(1, 6):
        raise ValueError(f"Ordinal rating outside 1-5: {value!r}")
    return result


def majority(values: list[int | None]) -> int | None:
    valid = [value for value in values if value is not None]
    if len(valid) < 2:
        return None
    counts = Counter(valid)
    if counts[0] == counts[1]:
        return None
    return 1 if counts[1] > counts[0] else 0


def ordinal_adequacy_consensus(
    values: list[int | None], threshold: int = 4
) -> int | None:
    """Resolve the prespecified pass/fail interpretation of an ordinal score.

    With two raters, opposite sides of the threshold remain unresolved. With
    three raters, the thresholded judgments are resolved by majority vote.
    This prevents a 3 and a 5 from being converted into a passing median of 4.
    """
    return majority(
        [None if value is None else int(value >= threshold) for value in values]
    )


def primary_perceptual_decision(
    binary_consensus: dict[str, int | None],
    ordinal_adequacy: dict[str, int | None],
) -> tuple[bool, int]:
    """Return the direct overall visual-usability decision.

    Region-specific ratings remain diagnostic outcomes. They do not silently
    replace the overall judgment with an uncommunicated derived conjunction.
    """
    overall = binary_consensus["overall_anatomical_usability"]
    return overall is not None, int(overall == 1)


def strict_multidomain_decision(
    binary_consensus: dict[str, int | None],
    ordinal_adequacy: dict[str, int | None],
) -> tuple[bool, int]:
    """Return the secondary strict multidomain sensitivity outcome."""
    resolved = all(value is not None for value in binary_consensus.values()) and all(
        ordinal_adequacy[field] is not None for field in PRIMARY_ORDINAL_FIELDS
    )
    usable = int(
        resolved
        and binary_consensus["anatomical_orientation"] == 1
        and ordinal_adequacy["nasal_midface_alignment_1to5"] == 1
        and ordinal_adequacy["broader_facial_placement_1to5"] == 1
        and binary_consensus["visible_deformation_artifact"] == 0
        and binary_consensus["overall_anatomical_usability"] == 1
    )
    return resolved, usable


def is_complete_independent_form(
    form: dict[str, dict[str, Any]], case_ids: set[str]
) -> bool:
    return all(
        blinded_id in form and form[blinded_id][field] is not None
        for blinded_id in case_ids
        for field in (*BINARY_FIELDS, *ORDINAL_FIELDS)
    )


def cohen_kappa(left: list[int], right: list[int]) -> float | None:
    if not left:
        return None
    left_array = np.asarray(left, dtype=np.int64)
    right_array = np.asarray(right, dtype=np.int64)
    observed = float(np.mean(left_array == right_array))
    categories = sorted(set(left) | set(right))
    expected = sum(
        float(np.mean(left_array == category))
        * float(np.mean(right_array == category))
        for category in categories
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else None
    return (observed - expected) / (1.0 - expected)


def quadratic_weighted_kappa(left: list[int], right: list[int]) -> float | None:
    if not left:
        return None
    categories = list(range(1, 6))
    left_array = np.asarray(left, dtype=np.int64)
    right_array = np.asarray(right, dtype=np.int64)
    observed = 0.0
    expected = 0.0
    denominator = float((len(categories) - 1) ** 2)
    for left_category in categories:
        for right_category in categories:
            weight = ((left_category - right_category) ** 2) / denominator
            observed += weight * float(
                np.mean((left_array == left_category) & (right_array == right_category))
            )
            expected += (
                weight
                * float(np.mean(left_array == left_category))
                * float(np.mean(right_array == right_category))
            )
    if expected <= 1e-15:
        return 1.0 if observed <= 1e-15 else None
    return 1.0 - observed / expected


def percentile_interval(values: list[float]) -> tuple[float | None, float | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(finite):
        return None, None
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def cluster_bootstrap_pairwise(
    parsed_forms: list[dict[str, dict[str, Any]]],
    case_key: dict[str, str],
    field: str,
    left_index: int,
    right_index: int,
    repetitions: int,
    seed: int,
) -> dict[str, tuple[float | None, float | None]]:
    grouped: dict[str, list[tuple[int, int]]] = {}
    for blinded_id in sorted(case_key):
        left = parsed_forms[left_index][blinded_id][field]
        right = parsed_forms[right_index][blinded_id][field]
        if left is None or right is None:
            continue
        subject = f"{int(case_key[blinded_id].split('_', 1)[0]):03d}"
        grouped.setdefault(subject, []).append((int(left), int(right)))
    subjects = sorted(grouped)
    if not subjects:
        return {"unweighted": (None, None), "quadratic_weighted": (None, None)}
    rng = np.random.default_rng(seed)
    unweighted_values: list[float] = []
    weighted_values: list[float] = []
    for _ in range(repetitions):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        pairs = [pair for subject in sampled for pair in grouped[str(subject)]]
        left_values = [pair[0] for pair in pairs]
        right_values = [pair[1] for pair in pairs]
        unweighted = cohen_kappa(left_values, right_values)
        if unweighted is not None:
            unweighted_values.append(unweighted)
        if field in ORDINAL_FIELDS:
            weighted = quadratic_weighted_kappa(left_values, right_values)
            if weighted is not None:
                weighted_values.append(weighted)
    return {
        "unweighted": percentile_interval(unweighted_values),
        "quadratic_weighted": percentile_interval(weighted_values),
    }


def fleiss_kappa(
    parsed_forms: list[dict[str, dict[str, Any]]],
    field: str,
    categories: tuple[int, ...],
) -> dict[str, Any]:
    complete = []
    for blinded_id in sorted(parsed_forms[0]):
        values = [form[blinded_id][field] for form in parsed_forms]
        if any(value is None for value in values):
            continue
        complete.append([int(value) for value in values])
    if not complete:
        return {"field": field, "complete_cases": 0, "fleiss_kappa": None}
    category_index = {category: index for index, category in enumerate(categories)}
    counts = np.zeros((len(complete), len(categories)), dtype=np.float64)
    for row_index, values in enumerate(complete):
        for value in values:
            counts[row_index, category_index[value]] += 1.0
    raters = float(len(parsed_forms))
    item_agreement = (
        np.sum(counts * counts, axis=1) - raters
    ) / (raters * (raters - 1.0))
    observed = float(np.mean(item_agreement))
    proportions = np.sum(counts, axis=0) / float(len(complete) * raters)
    expected = float(np.sum(proportions * proportions))
    value = None if expected >= 1.0 else (observed - expected) / (1.0 - expected)
    return {
        "field": field,
        "complete_cases": len(complete),
        "fleiss_kappa": value,
        "observed_agreement": observed,
        "chance_agreement": expected,
    }


def pairwise_kappa(
    parsed_forms: list[dict[str, dict[str, Any]]],
    case_key: dict[str, str],
    field: str,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for left_index in range(len(parsed_forms)):
        for right_index in range(left_index + 1, len(parsed_forms)):
            left_values = []
            right_values = []
            for blinded_id in sorted(parsed_forms[left_index]):
                left = parsed_forms[left_index][blinded_id][field]
                right = parsed_forms[right_index][blinded_id][field]
                if left is None or right is None:
                    continue
                left_values.append(int(left))
                right_values.append(int(right))
            intervals = cluster_bootstrap_pairwise(
                parsed_forms,
                case_key,
                field,
                left_index,
                right_index,
                repetitions,
                seed + 1000 * left_index + 100 * right_index + len(rows),
            )
            rows.append(
                {
                    "field": field,
                    "rater_pair": f"R{left_index + 1}-R{right_index + 1}",
                    "complete_cases": len(left_values),
                    "cohen_kappa_unweighted": cohen_kappa(left_values, right_values),
                    "cohen_kappa_unweighted_ci_low": intervals["unweighted"][0],
                    "cohen_kappa_unweighted_ci_high": intervals["unweighted"][1],
                    "cohen_kappa_quadratic_weighted": (
                        quadratic_weighted_kappa(left_values, right_values)
                        if field in ORDINAL_FIELDS
                        else None
                    ),
                    "cohen_kappa_quadratic_weighted_ci_low": (
                        intervals["quadratic_weighted"][0]
                        if field in ORDINAL_FIELDS
                        else None
                    ),
                    "cohen_kappa_quadratic_weighted_ci_high": (
                        intervals["quadratic_weighted"][1]
                        if field in ORDINAL_FIELDS
                        else None
                    ),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    if len(args.rater_csv) not in (2, 3):
        raise ValueError("Exactly two or three blinded rating forms are required")
    if args.bootstrap_repetitions < 1000:
        raise ValueError("At least 1000 bootstrap repetitions are required")
    key_rows = read(args.key_csv)
    if not key_rows or not {"blinded_case_id", "case"}.issubset(key_rows[0]):
        raise KeyError("Key CSV requires blinded_case_id and case")
    case_key = {row["blinded_case_id"]: row["case"] for row in key_rows}
    if len(case_key) != len(key_rows):
        raise ValueError("Duplicate blinded IDs in key")
    if any(not str(key).strip() or not str(case).strip() for key, case in case_key.items()):
        raise ValueError("Private rating key contains a blank blinded ID or case")
    split_path = args.split_json.resolve(strict=True)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    partition_subjects = validate_case_partition(case_key, split, args.subset)
    metadata_path = args.rater_metadata_csv.resolve(strict=True)
    metadata_rows = validate_rater_metadata(metadata_path, len(args.rater_csv))

    parsed_forms = []
    for form_index, path in enumerate(args.rater_csv):
        rows = read(path)
        required_form_fields = {"blinded_case_id", *BINARY_FIELDS, *ORDINAL_FIELDS}
        if not rows or not required_form_fields.issubset(rows[0]):
            raise KeyError(
                f"Rating form fields missing in {path}: "
                f"{required_form_fields - set(rows[0] if rows else ())}"
            )
        by_id = {}
        for row in rows:
            blinded_id = str(row["blinded_case_id"]).strip()
            if not blinded_id:
                raise ValueError(f"Blank blinded ID in {path}")
            if blinded_id in by_id:
                raise ValueError(f"Duplicate ID in {path}: {blinded_id}")
            by_id[blinded_id] = {
                **{field: binary(row[field]) for field in BINARY_FIELDS},
                **{field: ordinal(row[field]) for field in ORDINAL_FIELDS},
            }
        scope = metadata_rows[form_index]["rating_scope"]
        if scope == "full_independent_rater":
            if set(by_id) != set(case_key):
                raise ValueError(f"Full rating form case set differs from key: {path}")
        else:
            if form_index != 2 or len(parsed_forms) != 2:
                raise ValueError("A partial adjudication form is allowed only for R3")
            expected_disagreements = {
                blinded_id
                for blinded_id in case_key
                if parsed_forms[0][blinded_id]["overall_anatomical_usability"]
                != parsed_forms[1][blinded_id]["overall_anatomical_usability"]
            }
            if set(by_id) != expected_disagreements:
                raise ValueError(
                    "Partial R3 form must contain exactly the R1/R2 "
                    f"overall-usability disagreements: {path}"
                )
        parsed_forms.append(by_id)

    consensus_rows = []
    unresolved = 0
    strict_multidomain_unresolved = 0
    eye_audit_unresolved = 0
    cannot_counts = Counter()
    for blinded_id in sorted(case_key):
        binary_consensus = {}
        for field in BINARY_FIELDS:
            values = [
                form[blinded_id][field]
                for form in parsed_forms
                if blinded_id in form
            ]
            cannot_counts[field] += sum(value is None for value in values)
            binary_consensus[field] = majority(values)
        ordinal_medians = {}
        ordinal_adequacy = {}
        for field in ORDINAL_FIELDS:
            values = [
                form[blinded_id][field]
                for form in parsed_forms
                if blinded_id in form
            ]
            cannot_counts[field] += sum(value is None for value in values)
            valid = [int(value) for value in values if value is not None]
            ordinal_medians[field] = float(np.median(valid)) if len(valid) >= 2 else None
            ordinal_adequacy[field] = ordinal_adequacy_consensus(values)
        all_fields_resolved = all(
            value is not None for value in binary_consensus.values()
        ) and all(value is not None for value in ordinal_adequacy.values())
        resolved, usable = primary_perceptual_decision(
            binary_consensus, ordinal_adequacy
        )
        strict_resolved, strict_usable = strict_multidomain_decision(
            binary_consensus, ordinal_adequacy
        )
        if not resolved:
            unresolved += 1
        if not strict_resolved:
            strict_multidomain_unresolved += 1
        if ordinal_adequacy[EYE_AUDIT_FIELD] is None:
            eye_audit_unresolved += 1
        consensus_rows.append(
            {
                "case": case_key[blinded_id],
                "subject": f"{int(case_key[blinded_id].split('_', 1)[0]):03d}",
                "blinded_case_id": blinded_id,
                "orientation_consensus": binary_consensus["anatomical_orientation"],
                "nasal_midface_median": ordinal_medians[
                    "nasal_midface_alignment_1to5"
                ],
                "nasal_midface_adequacy_consensus": ordinal_adequacy[
                    "nasal_midface_alignment_1to5"
                ],
                "eye_orbit_median": ordinal_medians[
                    "eye_orbit_preservation_1to5"
                ],
                "eye_orbit_adequacy_consensus": ordinal_adequacy[
                    "eye_orbit_preservation_1to5"
                ],
                "broader_placement_median": ordinal_medians[
                    "broader_facial_placement_1to5"
                ],
                "broader_placement_adequacy_consensus": ordinal_adequacy[
                    "broader_facial_placement_1to5"
                ],
                "artifact_consensus": binary_consensus[
                    "visible_deformation_artifact"
                ],
                "overall_usability_consensus": binary_consensus[
                    "overall_anatomical_usability"
                ],
                "all_fields_resolved": int(all_fields_resolved),
                "eye_orbit_audit_resolved": int(
                    ordinal_adequacy[EYE_AUDIT_FIELD] is not None
                ),
                "strict_multidomain_resolved": int(strict_resolved),
                "strict_multidomain_usable": strict_usable,
                "consensus_resolved": int(resolved),
                "consensus_usable": usable,
            }
        )

    third_form_role = None
    third_form_all_cells_complete = None
    agreement_forms = parsed_forms
    if len(parsed_forms) == 3:
        third_form_all_cells_complete = is_complete_independent_form(
            parsed_forms[2], set(case_key)
        )
        third_form_role = metadata_rows[2]["rating_scope"]
        if third_form_role == "partial_blinded_adjudicator":
            agreement_forms = parsed_forms[:2]

    kappa_rows = []
    for field in (*BINARY_FIELDS, *ORDINAL_FIELDS):
        kappa_rows.extend(
            pairwise_kappa(
                agreement_forms,
                case_key,
                field,
                args.bootstrap_repetitions,
                args.seed,
            )
        )
    fleiss_rows = []
    if len(parsed_forms) == 3 and third_form_role == "full_independent_rater":
        for field in BINARY_FIELDS:
            fleiss_rows.append(fleiss_kappa(parsed_forms, field, (0, 1)))
        for field in ORDINAL_FIELDS:
            fleiss_rows.append(fleiss_kappa(parsed_forms, field, (1, 2, 3, 4, 5)))
    consensus_by_subject: dict[str, list[int]] = {}
    for row in consensus_rows:
        consensus_by_subject.setdefault(str(row["subject"]), []).append(
            int(row["consensus_usable"])
        )
    consensus_subject_rates = np.asarray(
        [
            np.mean(consensus_by_subject[subject])
            for subject in sorted(consensus_by_subject)
        ],
        dtype=float,
    )
    consensus_rng = np.random.default_rng(args.seed)
    consensus_bootstrap = np.mean(
        consensus_rng.choice(
            consensus_subject_rates,
            size=(args.bootstrap_repetitions, len(consensus_subject_rates)),
            replace=True,
        ),
        axis=1,
    )
    consensus_ci = np.quantile(consensus_bootstrap, (0.025, 0.975))

    summary = {
        "subset": args.subset,
        "development_only": args.subset == "development",
        "heldout_inspected": args.subset == "heldout",
        "partition_subjects": partition_subjects,
        "raters": len(parsed_forms),
        "primary_independent_raters": sum(
            row["rating_scope"] == "full_independent_rater"
            for row in metadata_rows
        ),
        "third_form_role": third_form_role,
        "third_form_all_cells_complete": third_form_all_cells_complete,
        "cases": len(consensus_rows),
        "unresolved_cases": unresolved,
        "strict_multidomain_unresolved_cases": strict_multidomain_unresolved,
        "eye_orbit_audit_unresolved_cases": eye_audit_unresolved,
        "positive_consensus_cases": sum(
            int(row["consensus_usable"]) for row in consensus_rows
        ),
        "positive_consensus_rate": float(np.mean(consensus_subject_rates)),
        "positive_consensus_subject_cluster_bootstrap_95_ci": [
            float(consensus_ci[0]),
            float(consensus_ci[1]),
        ],
        "cannot_judge_counts": dict(cannot_counts),
        "three_rater_fleiss_kappa": fleiss_rows,
        "subject_cluster_bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.seed,
        "qualified_rater_metadata_complete": True,
        "split_sha256": sha256(split_path),
        "private_key_sha256": sha256(args.key_csv.resolve(strict=True)),
        "rater_metadata_sha256": sha256(metadata_path),
        "rater_form_sha256": {
            f"R{index}": sha256(path.resolve(strict=True))
            for index, path in enumerate(args.rater_csv, start=1)
        },
        "rater_qualification_summary": [
            {
                "rater_code": row["rater_code"],
                "discipline_or_specialty": row["discipline_or_specialty"],
                "current_role": row["current_role"],
                "years_relevant_experience": row["years_relevant_experience"],
                "registration_method_development_involvement": row[
                    "registration_method_development_involvement"
                ],
                "rating_scope": row["rating_scope"],
            }
            for row in metadata_rows
        ],
        "rule": (
            "Primary reference: final overall visual-registration-usability label. "
            "The shared label is retained when R1 and R2 agree; a two-of-three "
            "majority label is used after a third masked rating when they disagree. "
            "Region-specific, orientation, deformation, and eye/orbit "
            "ratings are reported as diagnostic outcomes. A strict multidomain "
            "conjunction is retained only as a secondary sensitivity outcome."
        ),
        "note": (
            "Eye/orbit preservation is reported as a secondary constraint-fidelity "
            "check and is excluded from the primary visual-usability conjunction "
            "because S8 explicitly restores the fixed eye/orbit vertices. "
            "The primary endpoint is the direct overall-usability label; the "
            "regional scores explain that judgment rather than mechanically "
            "redefining it. "
            "For binary fields and the prespecified ordinal >=4 adequacy threshold, "
            "two-rater disagreements remain unresolved until a third masked form "
            "is supplied. Three-rater decisions use majority vote. Unresolved cases "
            "are never positive. A partial third form is used only for disagreement "
            "rating; agreement statistics then use the two complete primary "
            "raters only. Rater scope is fixed in metadata and is not inferred from "
            "Cannot-judge cells. Kappa intervals resample whole subjects."
        ),
    }

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "consensus_ratings.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(consensus_rows[0].keys()))
        writer.writeheader()
        writer.writerows(consensus_rows)
    with (args.output_dir / "pairwise_agreement.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(kappa_rows[0].keys()))
        writer.writeheader()
        writer.writerows(kappa_rows)
    if fleiss_rows:
        with (args.output_dir / "fleiss_agreement.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fleiss_rows[0].keys()))
            writer.writeheader()
            writer.writerows(fleiss_rows)
    summary["consensus_ratings_sha256"] = sha256(
        args.output_dir / "consensus_ratings.csv"
    )
    (args.output_dir / "rating_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
