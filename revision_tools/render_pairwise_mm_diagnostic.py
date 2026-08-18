#!/usr/bin/env python3
"""Render internal-only alignment and semantic-mask diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_pairwise_mm_rigid as registration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scale-dict", type=Path, required=True)
    parser.add_argument("--source-method", choices=("hrn", "3ddfa"), required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--topology-file", type=Path)
    parser.add_argument("--rigid-output", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sample(points: np.ndarray, maximum: int) -> np.ndarray:
    return points[registration.deterministic_indices(len(points), maximum)]


def scatter_view(
    axis: plt.Axes,
    target: np.ndarray,
    source: np.ndarray,
    masks: dict[str, np.ndarray],
    coordinates: tuple[int, int],
    title: str,
) -> None:
    target_indices = registration.deterministic_indices(len(target), 50000)
    source_indices = registration.deterministic_indices(len(source), 20000)
    tx, ty = coordinates
    axis.scatter(
        target[target_indices, tx], target[target_indices, ty], s=0.15,
        color="#b8b8b8", alpha=0.28, linewidths=0,
    )
    axis.scatter(
        source[source_indices, tx], source[source_indices, ty], s=0.28,
        color="#2f6fb0", alpha=0.48, linewidths=0,
    )
    for name, color in (("eye_soft", "#2b9a66"), ("nose", "#d64c3f"), ("mouth", "#d9902f")):
        indices = np.flatnonzero(masks[name])
        indices = indices[registration.deterministic_indices(len(indices), 5000)]
        axis.scatter(
            source[indices, tx], source[indices, ty], s=0.9,
            color=color, alpha=0.80, linewidths=0, label=name,
        )
    axis.set_aspect("equal")
    axis.set_title(title)
    axis.set_xlabel(("x", "y", "z")[tx] + " (mm)")
    axis.set_ylabel(("x", "y", "z")[ty] + " (mm)")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve(strict=True)
    rigid_output = registration.checked_existing(args.rigid_output, root, "rigid output")
    source_root = (
        registration.checked_existing(args.source_root, root, "source root")
        if args.source_root else None
    )
    topology = (
        registration.checked_existing(args.topology_file, root, "topology")
        if args.topology_file else None
    )
    output = args.output.expanduser().resolve(strict=False)
    revision = registration.checked_existing(root / "cmes_revision_20260816", root, "revision")
    if not registration.is_relative_to(output, revision) or output.exists():
        raise ValueError(f"Invalid or existing diagnostic output: {output}")

    manifest = root / "prepared_cohort" / "facescape_frontal_pairs_manifest.csv"
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        by_pair = {row["pair_id"]: row for row in csv.DictReader(handle)}
    subject = f"{int(by_pair[args.case]['subject']):03d}"
    target_row = by_pair[f"{subject}_18_eye_closed"]
    target_mesh = registration.checked_existing(Path(target_row["mesh"]), root, "target")
    target_world, _ = registration.load_trimesh(target_mesh)
    scales = json.loads(args.scale_dict.read_text(encoding="utf-8"))
    mm_per_unit = float(scales[str(int(subject))]["18"][0])
    target = registration.target_registration_frame(
        target_world, target_mesh.parent / "selected_camera.json"
    ) * mm_per_unit
    _, _, _, _, masks = registration.load_source(
        args.case, args.source_method, root, source_root, topology
    )
    payload = np.load(rigid_output / "rigid_vertices_mm" / f"{args.case}.npz")
    source = np.asarray(payload["vertices_mm"], dtype=np.float64)

    figure, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=180)
    scatter_view(axes[0], target, source, masks, (0, 2), "Frontal view")
    scatter_view(axes[1], target, source, masks, (1, 2), "Profile view")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle(f"Internal diagnostic: {args.case} / {args.source_method}")
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
