#!/usr/bin/env python3
"""Run the pinned official 3DDFA-V2 model on a frozen identity subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


# 3DDFA-V2 predates removal of NumPy's deprecated scalar aliases.
for _name, _value in {
    "long": np.int64,
    "int": int,
    "float": float,
    "bool": bool,
    "complex": complex,
    "object": object,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument(
        "--subset", choices=("development", "heldout", "all"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def checked_existing(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not is_relative_to(resolved, root):
        raise ValueError(f"{label} escapes project root: {resolved}")
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_cases(
    manifest_path: Path, split_path: Path, subset: str, root: Path
) -> list[dict[str, str]]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if subset == "all":
        subjects = set(split["development_subjects"]) | set(split["test_subjects"])
    elif subset == "heldout":
        subjects = set(split["test_subjects"])
    else:
        subjects = set(split["development_subjects"])
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for row in rows:
        pair_id = str(row["pair_id"])
        subject = f"{int(row['subject']):03d}"
        if subject not in subjects or pair_id.endswith("_18_eye_closed"):
            continue
        image = checked_existing(Path(row["image"]), root, f"image for {pair_id}")
        selected.append({**row, "subject_id": subject, "image": str(image)})
    selected.sort(key=lambda row: str(row["pair_id"]))
    expected = 380 if subset == "all" else 190
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} {subset} cases, found {len(selected)}")
    return selected


def write_rows(output: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with (output / "3ddfa_reconstruction_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: str(row["case"])))


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve(strict=True)
    revision_root = checked_existing(root / "cmes_revision_20260816", root, "revision root")
    repo = checked_existing(args.repo, revision_root, "3DDFA-V2 repository")
    manifest = checked_existing(args.manifest, root, "pair manifest")
    split = checked_existing(args.split, revision_root, "identity split")
    output = args.output_dir.expanduser().resolve(strict=False)
    if output == revision_root or not is_relative_to(output, revision_root):
        raise ValueError(f"Output escapes revision root: {output}")
    if output.exists() and not args.resume and not args.dry_run:
        raise FileExistsError(f"Refusing to overwrite output: {output}")

    cases = read_cases(manifest, split, args.subset, root)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "subset": args.subset,
                    "case_count": len(cases),
                    "first_case": cases[0]["pair_id"],
                    "last_case": cases[-1]["pair_id"],
                    "output": str(output),
                },
                indent=2,
            )
        )
        return

    output.mkdir(parents=True, exist_ok=args.resume)
    objects = output / "objects"
    details = output / "case_metadata"
    objects.mkdir(exist_ok=True)
    details.mkdir(exist_ok=True)

    sys.path.insert(0, str(repo))
    os.chdir(repo)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    os.environ["OMP_NUM_THREADS"] = "4"
    from FaceBoxes import FaceBoxes  # noqa: PLC0415
    from TDDFA import TDDFA  # noqa: PLC0415
    from utils.serialization import ser_to_obj  # noqa: PLC0415

    config_path = repo / "configs" / "mb1_120x120.yml"
    config = yaml.load(config_path.read_text(encoding="utf-8"), Loader=yaml.SafeLoader)
    face_boxes = FaceBoxes()
    tddfa = TDDFA(gpu_mode=True, **config)
    triangles = np.asarray(tddfa.tri, dtype=np.int32)
    raw_keypoints = np.asarray(tddfa.bfm.keypoints, dtype=np.int64).reshape(-1, 3)
    landmark_vertex_ids = raw_keypoints[:, 0] // 3
    if len(landmark_vertex_ids) != 68:
        raise ValueError(f"Expected 68 BFM landmarks, found {len(landmark_vertex_ids)}")
    np.savez_compressed(
        output / "3ddfa_topology_and_landmarks.npz",
        triangles=triangles,
        landmark_vertex_ids=landmark_vertex_ids.astype(np.int32),
    )

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(cases, start=1):
        case = str(item["pair_id"])
        obj_path = objects / f"{case}.obj"
        detail_path = details / f"{case}.json"
        if args.resume and obj_path.exists() and detail_path.exists():
            resumed_row = json.loads(detail_path.read_text(encoding="utf-8"))
            if "detection_mode" not in resumed_row:
                resumed_row["detection_mode"] = "FaceBoxes-largest"
                detail_path.write_text(
                    json.dumps(resumed_row, indent=2), encoding="utf-8"
                )
            rows.append(resumed_row)
            print(json.dumps({"skipped": case, "count": index}), flush=True)
            continue

        started = time.time()
        image = cv2.imread(str(item["image"]))
        if image is None:
            raise ValueError(f"Could not read image: {item['image']}")
        boxes = face_boxes(image)
        detected_face_count = len(boxes)
        if boxes:
            boxes = sorted(
                boxes,
                key=lambda box: float(
                    max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                ),
                reverse=True,
            )
            detection_mode = "FaceBoxes-largest"
        else:
            # FaceScape frontal frames are portrait-centered. 3DDFA-V2 accepts
            # a supplied ROI, so retain the case with a deterministic central
            # box when its bundled detector misses an extreme expression.
            height, width = image.shape[:2]
            box_width = 0.56 * width
            box_height = 0.86 * height
            center_x = 0.52 * width
            center_y = 0.50 * height
            boxes = [[
                center_x - 0.5 * box_width,
                center_y - 0.5 * box_height,
                center_x + 0.5 * box_width,
                center_y + 0.5 * box_height,
                1.0,
            ]]
            detection_mode = "deterministic-central-fallback"
        parameters, roi_boxes = tddfa(image, [boxes[0]])
        vertices_list = tddfa.recon_vers(parameters, roi_boxes, dense_flag=True)
        ser_to_obj(
            image,
            vertices_list,
            triangles,
            height=image.shape[0],
            wfp=str(obj_path),
        )
        vertices = np.asarray(vertices_list[0], dtype=np.float64).T
        row = {
            "case": case,
            "subject": str(item["subject_id"]),
            "expression": case.split("_", 1)[1],
            "status": "success",
            "image": str(item["image"]),
            "output_obj": str(obj_path),
            "detected_faces": int(detected_face_count),
            "detection_mode": detection_mode,
            "selected_box": [float(value) for value in boxes[0]],
            "vertices": int(vertices.shape[0]),
            "triangles": int(triangles.shape[0]),
            "runtime_seconds": float(time.time() - started),
        }
        detail_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        write_rows(output, rows)
        print(
            json.dumps(
                {
                    "completed": case,
                    "count": index,
                    "total": len(cases),
                    "runtime_seconds": row["runtime_seconds"],
                }
            ),
            flush=True,
        )

    write_rows(output, rows)
    summary = {
        "method": "official 3DDFA-V2 PyTorch inference",
        "subset": args.subset,
        "case_count": len(rows),
        "subject_count": len({str(row["subject"]) for row in rows}),
        "pinned_repository_commit": "1b6c67601abffc1e9f248b291708aef0e43b55ae",
        "repository": str(repo),
        "config": str(config_path),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "identity_split": str(split),
        "identity_split_sha256": sha256(split),
        "topology_file": str(output / "3ddfa_topology_and_landmarks.npz"),
        "note": "Only reconstructed meshes are stored; FaceScape images are not copied.",
    }
    (output / "3ddfa_reconstruction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
