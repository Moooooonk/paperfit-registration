#!/usr/bin/env python3
"""Run the four S8 component conditions on every available rigid case."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

import run_anchor_aware_s8_pilot as s8
import run_mm_s8_from_rigid as s8_runner
import run_pairwise_mm_rigid as registration
import frozen_nonrigid_nasal_solver as frozen_nonrigid


CONDITIONS = (
    {
        "condition": "proposed_s8",
        "disable_eye_exclusion": False,
        "disable_contours": False,
        "disable_region_weights": False,
    },
    {
        "condition": "no_eye_exclusion",
        "disable_eye_exclusion": True,
        "disable_contours": False,
        "disable_region_weights": False,
    },
    {
        "condition": "no_nasal_depth_contours",
        "disable_eye_exclusion": False,
        "disable_contours": True,
        "disable_region_weights": False,
    },
    {
        "condition": "no_region_weights",
        "disable_eye_exclusion": False,
        "disable_contours": False,
        "disable_region_weights": True,
    },
)


def condition_failure_row(
    case: str,
    source_method: str,
    pre_s8_branch: str,
    condition: dict[str, Any],
    failure_reason: str,
    exception: Exception | None = None,
) -> dict[str, Any]:
    expression = case.split("_", 1)[1]
    expression_index, expression_name = expression.split("_", 1)
    row: dict[str, Any] = {
        "case": case,
        "subject": f"{int(case.split('_', 1)[0]):03d}",
        "expression": expression,
        "expression_index": int(expression_index),
        "expression_name": expression_name,
        "source_method": source_method,
        "pre_s8_branch": pre_s8_branch,
        "condition": condition["condition"],
        "disable_eye_exclusion": int(condition["disable_eye_exclusion"]),
        "disable_contours": int(condition["disable_contours"]),
        "disable_region_weights": int(condition["disable_region_weights"]),
        "completed": 0,
        "execution_failure_reason": failure_reason,
    }
    if exception is not None:
        row["execution_exception_type"] = type(exception).__name__
        row["execution_exception_message"] = str(exception)
    return row


def failure_rows(
    case: str,
    source_method: str,
    pre_s8_branch: str,
    failure_reason: str,
    exception: Exception | None = None,
) -> list[dict[str, Any]]:
    return [
        condition_failure_row(
            case,
            source_method,
            pre_s8_branch,
            condition,
            failure_reason,
            exception,
        )
        for condition in CONDITIONS
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scale-dict", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument(
        "--subset", choices=("development", "heldout", "all"), required=True
    )
    parser.add_argument("--target-anchor-json", type=Path, required=True)
    parser.add_argument("--source-method", choices=("hrn", "3ddfa"), required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--topology-file", type=Path)
    parser.add_argument("--rigid-output", type=Path, required=True)
    parser.add_argument("--branch-assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contour-stages", type=int, choices=(4, 8, 12), default=8)
    parser.add_argument("--step-multiplier", type=float, default=1.0)
    parser.add_argument("--gain-multiplier", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--anchor-step-mm", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def metric_row(
    case: str,
    source_method: str,
    condition: dict[str, Any],
    initial: np.ndarray,
    final: np.ndarray,
    faces: np.ndarray,
    masks: dict[str, np.ndarray],
    target: np.ndarray,
    target_anchor: np.ndarray,
    history: list[dict[str, Any]],
    expected_updates: int,
    nonrigid: Any,
) -> dict[str, Any]:
    tree = cKDTree(target)
    distances, _ = tree.query(final, k=1, workers=1)
    full_median, full_p90 = s8_runner.quantiles(distances, masks["full_no_eye"])
    nose_median, nose_p90 = s8_runner.quantiles(distances, masks["nose"])
    anchor_surface = registration.anchor_consistency_distance(
        final, masks, target_anchor
    )
    anchor = float(
        np.linalg.norm(registration.source_anchor(final, masks) - target_anchor)
    )
    orientation = registration.orientation_metrics(final, masks)
    eye_displacement = np.linalg.norm(
        final[masks["eye_soft"]] - initial[masks["eye_soft"]], axis=1
    )
    displacement = np.linalg.norm(final - initial, axis=1)
    strain = s8.edge_strain(initial, final, faces, nonrigid)
    expression = case.split("_", 1)[1]
    expression_index, expression_name = expression.split("_", 1)
    return {
        "case": case,
        "subject": f"{int(case.split('_', 1)[0]):03d}",
        "expression": expression,
        "expression_index": int(expression_index),
        "expression_name": expression_name,
        "source_method": source_method,
        "condition": condition["condition"],
        "disable_eye_exclusion": int(condition["disable_eye_exclusion"]),
        "disable_contours": int(condition["disable_contours"]),
        "disable_region_weights": int(condition["disable_region_weights"]),
        "total_scheduled_passes": len(history),
        "expected_scheduled_passes": expected_updates,
        "executed_constrained_solves": sum(
            bool(item.get("executed_constrained_solve", True)) for item in history
        ),
        "skipped_scheduled_passes": sum(
            not bool(item.get("executed_constrained_solve", True))
            for item in history
        ),
        "source_anchor_definition": registration.source_anchor_definition(
            source_method
        ),
        "nose_anchor_metric_definition": (
            f"transformed {registration.source_anchor_definition(source_method)} "
            "to target nose-tip anchor"
        ),
        "post_full_median_mm": full_median,
        "post_full_p90_mm": full_p90,
        "post_nose_median_mm": nose_median,
        "post_nose_p90_mm": nose_p90,
        "post_anchor_mm": anchor,
        "post_anchor_point_mm": anchor,
        "post_anchor_surface_mm": anchor_surface,
        "post_orientation_pass": int(not bool(orientation["upside_down"])),
        "eye_fixed_max_mm": (
            float(np.max(eye_displacement)) if len(eye_displacement) else 0.0
        ),
        "displacement_p90_mm": float(np.quantile(displacement, 0.90)),
        "displacement_max_mm": float(np.max(displacement)),
        **strain,
    }


def process_case(
    case: str,
    root_string: str,
    scale_path_string: str,
    anchor_path_string: str,
    source_method: str,
    source_root_string: str | None,
    topology_string: str | None,
    rigid_output_string: str,
    output_string: str,
    contour_stages: int,
    step_multiplier: float,
    gain_multiplier: float,
    anchor_weight: float,
    anchor_step_mm: float,
    pre_s8_branch: str,
) -> list[dict[str, Any]]:
    root = Path(root_string)
    output = Path(output_string)
    detail_path = output / "case_details" / f"{case}.json"
    if detail_path.exists():
        return json.loads(detail_path.read_text(encoding="utf-8"))["rows"]

    scale_dict = json.loads(Path(scale_path_string).read_text(encoding="utf-8"))
    anchor_path = Path(anchor_path_string)
    anchor_payload = json.loads(anchor_path.read_text(encoding="utf-8"))
    source_root = Path(source_root_string) if source_root_string else None
    topology = Path(topology_string) if topology_string else None
    _, _, faces, _, masks = registration.load_source(
        case, source_method, root, source_root, topology
    )
    initial = np.asarray(
        np.load(Path(rigid_output_string) / "rigid_vertices_mm" / f"{case}.npz")[
            "vertices_mm"
        ],
        dtype=np.float64,
    )
    target, target_anchor, _, _, _ = s8_runner.target_for_case(
        case, root, scale_dict, anchor_payload, anchor_path
    )
    rows = []
    details = []
    for condition in CONDITIONS:
        try:
            variant = {
                **condition,
                "name": condition["condition"],
                "anchor_weight": anchor_weight,
                "anchor_step_mm": anchor_step_mm,
                "contours": s8_runner.contour_schedule(contour_stages),
                "step_multiplier": step_multiplier,
                "gain_multiplier": gain_multiplier,
            }
            final, history = s8.run_variant(
                initial, faces, target, masks, target_anchor, frozen_nonrigid, variant
            )
            expected_updates = (
                8
                if condition["disable_contours"]
                else s8_runner.expected_scheduled_passes(contour_stages)
            )
            if len(history) != expected_updates:
                raise RuntimeError(
                    f"{case}/{condition['condition']}: expected {expected_updates} "
                    f"scheduled S8 passes, found {len(history)}"
                )
            row = metric_row(
                case,
                source_method,
                condition,
                initial,
                final,
                faces,
                masks,
                target,
                target_anchor,
                history,
                expected_updates,
                frozen_nonrigid,
            )
            row["pre_s8_branch"] = pre_s8_branch
            row["completed"] = 1
            details.append(
                {"condition": condition["condition"], "history": history}
            )
        except Exception as error:
            row = condition_failure_row(
                case,
                source_method,
                pre_s8_branch,
                condition,
                "s8_ablation_condition_failure",
                error,
            )
            details.append(
                {
                    "condition": condition["condition"],
                    "history": [],
                    "execution_failed": True,
                    "execution_exception_type": type(error).__name__,
                    "execution_exception_message": str(error),
                }
            )
        rows.append(row)
    detail_path.write_text(
        json.dumps({"rows": rows, "details": details}, indent=2), encoding="utf-8"
    )
    return rows


def write_summary(
    output: Path, rows: list[dict[str, Any]], config: dict[str, Any]
) -> None:
    if not rows:
        return
    ordered = sorted(rows, key=lambda row: (str(row["case"]), str(row["condition"])))
    fields = sorted({key for row in ordered for key in row})
    with (output / "s8_component_ablation_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)
    def mean_or_none(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    summaries = {}
    for condition in (item["condition"] for item in CONDITIONS):
        subset = [
            row
            for row in ordered
            if row["condition"] == condition
            and int(float(row.get("completed", 1))) == 1
        ]
        summaries[condition] = {
            "attempted_cases": sum(
                row["condition"] == condition for row in ordered
            ),
            "cases": len(subset),
            "subjects": len({str(row["subject"]) for row in subset}),
            "mean_full_median_mm": mean_or_none(
                [float(row["post_full_median_mm"]) for row in subset]
            ),
            "mean_nose_median_mm": mean_or_none(
                [float(row["post_nose_median_mm"]) for row in subset]
            ),
            "mean_nose_p90_mm": mean_or_none(
                [float(row["post_nose_p90_mm"]) for row in subset]
            ),
            "mean_eye_displacement_max_mm": mean_or_none(
                [float(row["eye_fixed_max_mm"]) for row in subset]
            ),
        }
    completed_rows = sum(
        int(float(row.get("completed", 1))) == 1 for row in ordered
    )
    (output / "s8_component_ablation_summary.json").write_text(
        json.dumps(
            {
                **config,
                "recorded_rows": len(ordered),
                "completed_rows": completed_rows,
                "invalid_pre_s8_evidence_rows": sum(
                    row.get("execution_failure_reason")
                    == "invalid_pre_s8_evidence_not_processed"
                    for row in ordered
                ),
                "s8_common_setup_failure_rows": sum(
                    row.get("execution_failure_reason")
                    == "s8_ablation_execution_failure"
                    for row in ordered
                ),
                "s8_condition_failure_rows": sum(
                    row.get("execution_failure_reason")
                    == "s8_ablation_condition_failure"
                    for row in ordered
                ),
                "s8_execution_failure_rows": sum(
                    row.get("execution_failure_reason")
                    in {
                        "s8_ablation_execution_failure",
                        "s8_ablation_condition_failure",
                    }
                    for row in ordered
                ),
                "conditions": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.anchor_weight < 0.0 or args.anchor_step_mm < 0.0:
        raise ValueError("Anchor parameters must be nonnegative")
    if (args.anchor_weight == 0.0) != (args.anchor_step_mm == 0.0):
        raise ValueError("Anchor weight and step must both be zero or both positive")
    root = args.root.expanduser().resolve(strict=True)
    revision = registration.checked_existing(
        root / "cmes_revision_20260816", root, "revision"
    )
    scale = registration.checked_existing(args.scale_dict, revision, "scale dictionary")
    split = registration.checked_existing(args.split, revision, "identity split")
    anchor = registration.checked_existing(
        args.target_anchor_json, revision, "target-anchor JSON"
    )
    rigid = registration.checked_existing(args.rigid_output, revision, "rigid output")
    branch_assignments_path = registration.checked_existing(
        args.branch_assignments, revision, "pre-S8 branch assignments"
    )
    manifest = registration.checked_existing(
        root / "prepared_cohort" / "facescape_frontal_pairs_manifest.csv",
        root,
        "pair manifest",
    )
    source_root = (
        registration.checked_existing(args.source_root, revision, "source root")
        if args.source_root
        else None
    )
    topology = (
        registration.checked_existing(args.topology_file, revision, "topology")
        if args.topology_file
        else None
    )
    if args.source_method == "3ddfa" and (source_root is None or topology is None):
        raise ValueError("3DDFA requires --source-root and --topology-file")
    output = args.output_dir.expanduser().resolve(strict=False)
    if output == revision or not registration.is_relative_to(output, revision):
        raise ValueError(f"Output escapes revision root: {output}")
    if output.exists() and not args.resume and not args.dry_run:
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    cases = sorted(path.stem for path in (rigid / "case_rows").glob("*.json"))
    if args.case:
        requested = set(args.case)
        missing = sorted(requested - set(cases))
        if missing:
            raise ValueError(f"Cases missing from rigid output: {missing}")
        cases = sorted(requested)
    if not cases:
        raise ValueError("No rigid cases available")
    expected_cases, _ = registration.read_selected_cases(
        manifest, split, args.subset, args.case
    )
    if cases != expected_cases:
        missing = sorted(set(expected_cases) - set(cases))
        extra = sorted(set(cases) - set(expected_cases))
        raise ValueError(
            f"Rigid cases do not match frozen {args.subset} split; "
            f"missing={missing}, extra={extra}"
        )
    branch_assignments = s8_runner.load_branch_assignments(
        branch_assignments_path, cases, args.source_method
    )
    eligible_cases = [
        case
        for case in cases
        if int(float(branch_assignments[case]["s8_eligible"])) == 1
    ]
    config = {
        "source_method": args.source_method,
        "subset": args.subset,
        "case_count": len(cases),
        "conditions": list(CONDITIONS),
        "contour_stages": args.contour_stages,
        "step_multiplier": args.step_multiplier,
        "gain_multiplier": args.gain_multiplier,
        "anchor_weight": args.anchor_weight,
        "anchor_step_mm": args.anchor_step_mm,
        "anchor_force_used": bool(
            args.anchor_weight > 0.0 and args.anchor_step_mm > 0.0
        ),
        "rigid_output": str(rigid),
        "branch_assignments": str(branch_assignments_path),
        "branch_assignments_sha256": registration.sha256(branch_assignments_path),
        "s8_eligible_cases": len(eligible_cases),
        "target_anchor_json": str(anchor),
        "target_anchor_json_sha256": registration.sha256(anchor),
        "identity_split": str(split),
        "identity_split_sha256": registration.sha256(split),
        "pair_manifest": str(manifest),
        "pair_manifest_sha256": registration.sha256(manifest),
        "stores_final_meshes": False,
        "acceptance_threshold_applied": False,
    }
    if args.dry_run:
        print(json.dumps({**config, "first_case": cases[0], "last_case": cases[-1]}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=args.resume)
    (output / "case_details").mkdir(exist_ok=True)
    eligible_set = set(eligible_cases)
    rows: list[dict[str, Any]] = []
    for case in cases:
        if case in eligible_set:
            continue
        rows.extend(
            failure_rows(
                case,
                args.source_method,
                branch_assignments[case]["pre_s8_branch"],
                "invalid_pre_s8_evidence_not_processed",
            )
        )
        case_rows = [row for row in rows if row["case"] == case]
        (output / "case_details" / f"{case}.json").write_text(
            json.dumps(
                {
                    "rows": case_rows,
                    "details": [],
                    "not_processed_due_to_invalid_pre_s8_evidence": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def arguments(case: str) -> tuple[Any, ...]:
        return (
            case,
            str(root),
            str(scale),
            str(anchor),
            args.source_method,
            str(source_root) if source_root else None,
            str(topology) if topology else None,
            str(rigid),
            str(output),
            args.contour_stages,
            args.step_multiplier,
            args.gain_multiplier,
            args.anchor_weight,
            args.anchor_step_mm,
            branch_assignments[case]["pre_s8_branch"],
        )

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_case, *arguments(case)): case
            for case in eligible_cases
        }
        for future in as_completed(futures):
            case = futures[future]
            try:
                case_rows = future.result()
            except Exception as error:
                case_rows = failure_rows(
                    case,
                    args.source_method,
                    branch_assignments[case]["pre_s8_branch"],
                    "s8_ablation_execution_failure",
                    error,
                )
                (output / "case_details" / f"{case}.json").write_text(
                    json.dumps(
                        {
                            "rows": case_rows,
                            "details": [],
                            "s8_ablation_execution_failed": True,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            rows.extend(case_rows)
            write_summary(output, rows, config)
            print(
                json.dumps({"completed": case, "cases": len(rows) // len(CONDITIONS), "total": len(cases)}),
                flush=True,
            )
    write_summary(output, rows, config)
    if len(rows) != len(cases) * len(CONDITIONS):
        raise RuntimeError(
            "Ablation denominator mismatch: "
            f"recorded {len(rows)} of {len(cases) * len(CONDITIONS)} rows"
        )


if __name__ == "__main__":
    main()
