# Final Code Audit

This folder is the public code package for the CMES manuscript.

## Verified contents

- Public entry scripts:
  - `scripts/run_01_main_rigid_qc.py`
  - `scripts/run_02_auxiliary_rigid_recovery.py`
  - `scripts/run_03_s8_refinement.py`
  - `scripts/run_04_baselines.py`
  - `scripts/run_05_component_ablation.py`
  - `scripts/run_06_final_decision.py`
- Original implementation dependencies:
  - `paperfit_legacy_impl/*.py`
- Manuscript-level numerical result tables:
  - `results/tables/*.csv`

## Validation performed

- Source-level compile check over all Python files in `scripts/` and `paperfit_legacy_impl/`.
- Compile errors: 0.
- Unit tests for branch-specific final acceptance and post-hoc FaceScape millimeter conversion passed.
- `__pycache__` directories were removed after validation.
- Manuscript table values were cross-checked against the included aggregate CSV files.

## Data note

FaceScape raw data, per-case result rows, HRN weights, private mesh outputs, local quick-test meshes, figure assets, and server-only files are not included. Prepare the dataset separately and set `PAPERFIT_ROOT` as described in `README.md`.

## Manuscript linkage

This package supports the final CMES manuscript submission. The manuscript package remains the source of truth for submitted PDF, LaTeX, and figure filenames.
