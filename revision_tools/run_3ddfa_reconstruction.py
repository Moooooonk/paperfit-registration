#!/usr/bin/env python3
"""Run official 3DDFA-V2 ONNX inference with an explicit output path."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-obj", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--backend", choices=("pytorch", "onnx"), default="pytorch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.expanduser().resolve(strict=True)
    image_path = args.image.expanduser().resolve(strict=True)
    output_obj = args.output_obj.expanduser().resolve(strict=False)
    metadata_path = args.metadata.expanduser().resolve(strict=False)
    if output_obj.exists() or metadata_path.exists():
        raise FileExistsError("Refusing to overwrite an existing output")
    if output_obj.suffix.lower() != ".obj":
        raise ValueError("--output-obj must end in .obj")

    sys.path.insert(0, str(repo))
    os.chdir(repo)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    os.environ["OMP_NUM_THREADS"] = "4"

    from utils.serialization import ser_to_obj  # noqa: E402

    config_path = repo / "configs" / "mb1_120x120.yml"
    config = yaml.load(config_path.read_text(encoding="utf-8"), Loader=yaml.SafeLoader)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    if args.backend == "onnx":
        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX  # noqa: E402
        from TDDFA_ONNX import TDDFA_ONNX  # noqa: E402

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**config)
    else:
        from FaceBoxes import FaceBoxes  # noqa: E402
        from TDDFA import TDDFA  # noqa: E402

        face_boxes = FaceBoxes()
        tddfa = TDDFA(gpu_mode=True, **config)
    boxes = face_boxes(image)
    if not boxes:
        raise RuntimeError(f"No face detected in {image_path}")
    boxes = sorted(
        boxes,
        key=lambda box: float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])),
        reverse=True,
    )
    selected = [boxes[0]]
    parameters, roi_boxes = tddfa(image, selected)
    vertices = tddfa.recon_vers(parameters, roi_boxes, dense_flag=True)

    output_obj.parent.mkdir(parents=True, exist_ok=True)
    ser_to_obj(
        image,
        vertices,
        tddfa.tri,
        height=image.shape[0],
        wfp=str(output_obj),
    )
    mesh = np.asarray(vertices[0], dtype=np.float64).T
    triangles = np.asarray(tddfa.tri)
    triangle_count = int(
        triangles.shape[0] if triangles.ndim == 2 and triangles.shape[1] == 3
        else triangles.shape[1]
    )
    metadata = {
        "method": f"official 3DDFA-V2 {args.backend} inference",
        "backend": args.backend,
        "repo": str(repo),
        "config": str(config_path),
        "image": str(image_path),
        "output_obj": str(output_obj),
        "detected_faces": len(boxes),
        "selected_box": [float(value) for value in boxes[0]],
        "vertices": int(mesh.shape[0]),
        "triangles": triangle_count,
        "coordinate_min": mesh.min(axis=0).tolist(),
        "coordinate_max": mesh.max(axis=0).tolist(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
