# Final Code Audit

This folder is the public code package for the Sensors manuscript.

## Verified contents

- Public entry scripts:
  - `scripts/run_01_main_rigid_qc.py`
  - `scripts/run_02_auxiliary_rigid_recovery.py`
  - `scripts/run_03_s8_refinement.py`
  - `scripts/run_04_baselines.py`
  - `scripts/run_05_component_ablation.py`
- Original implementation dependencies:
  - `paperfit_legacy_impl/*.py`
- Manuscript result tables and summaries:
  - `results/tables/*.csv`
  - `results/summaries/*.json`

## Validation performed

- Source-level compile check over all Python files in `scripts/` and `paperfit_legacy_impl/`.
- Compile errors: 0.
- `__pycache__` directories were removed after validation.
- Manuscript table values were cross-checked against the included aggregate CSV/JSON files.

## Data note

FaceScape raw data, HRN weights, generated private mesh outputs, local quick-test meshes, figure assets, and server-only files are not included. Prepare the dataset separately and set `PAPERFIT_ROOT` as described in `README.md`.

## Manuscript linkage

This package supports the final Sensors manuscript submission. The manuscript package remains the source of truth for submitted PDF, LaTeX, and figure filenames.
