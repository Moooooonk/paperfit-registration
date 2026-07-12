# PaperFit Registration

This repository contains the public code package for the Sensors manuscript experiments.

The public entry points are in `scripts/`. The original experiment implementation files are kept once in `paperfit_legacy_impl/` so that manuscript results remain traceable without duplicating similar scripts across folders.

## Public Entry Points

```text
scripts/run_01_main_rigid_qc.py
scripts/run_02_auxiliary_rigid_recovery.py
scripts/run_03_s8_refinement.py
scripts/run_04_baselines.py
scripts/run_05_component_ablation.py
```

## What Is Directly Used

- `run_01_main_rigid_qc.py`: main all-pairs rigid candidate generation and strict QC.
- `run_02_auxiliary_rigid_recovery.py`: auxiliary rigid recovery and effective rigid-status merge.
- `run_03_s8_refinement.py`: S8 refinement for rigid-pass, anchor-only, and broad-failure branches.
- `run_04_baselines.py`: full-target Open3D, cropped-target Open3D, and adaptive-template-inspired baselines.
- `run_05_component_ablation.py`: representative component ablation.

Files in `paperfit_legacy_impl/` are implementation dependencies used by these entry points. They are not all separate manuscript stages.

## Key Result

The final accepted set is 346/380 source-target pairs, or 91.05%. The rigid-pass set also undergoes S8 refinement and fixed-eye displacement audit.

Distances in the manuscript tables are reported in millimeters. Raw experiment CSV files keep the FaceScape coordinate-unit values unless the column name explicitly ends with `_mm`; the manuscript conversion is `distance_mm = raw_distance * 100`.

## Data

The FaceScape dataset, HRN weights, generated mesh outputs, local quick-test meshes, private server files, and figure asset packages are not included. Prepare the dataset separately under the applicable FaceScape license terms and set `PAPERFIT_ROOT` to the prepared experiment workspace before running scripts.

Do not add FaceScape-derived OBJ, PLY, texture images, HRN output meshes, local figure assets, or local quick-test data to this repository.

## Environment

Use Python 3.10 or 3.11. The baseline scripts require `open3d==0.19.0`, which is not available for Python 3.13 in the standard PyPI wheels.

```powershell
$env:PAPERFIT_ROOT = "D:\path\to\prepared\facescape_pipeline"
python scripts\run_01_main_rigid_qc.py
```

## Notes

The original implementation filenames include dates because they correspond to the experiment runs used to derive the manuscript tables. Public-facing scripts use stable names; legacy names are kept only for traceability.

This repository does not include manuscript figure source files or PPT diagrams. The submitted manuscript package remains the source of truth for PDF, LaTeX, and final figure files.
