# PaperFit Registration: CMES Revision Code

This package contains the sanitized code and aggregate numerical results for
the revised CMES evaluation of eye-preserving cross-source 3D face
registration. It corresponds to the identity-disjoint, millimeter-scale,
branch-independent evaluation described in the revised manuscript.

## What Is Included

- `revision_tools/`: target-anchor and face-ROI construction, HRN/3DDFA-V2
  rigid registration, S8 refinement, Open3D rigid controls, the shared-cues
  ICP control, ARAP, and full-case ablation.
- `analysis_tools/`: identity splitting, development-only QC calibration,
  fail-closed evaluation, subject-clustered statistics, paired tests, Holm
  correction, sensitivity analysis, and masked-rating aggregation.
- `results/aggregate/`: manuscript-level automatic results only. These files
  contain no FaceScape geometry, image, subject mapping, or per-case row.
- `docs/`: physical-unit provenance and data-handling restrictions.

## External Inputs

The package does not redistribute FaceScape data, FaceScape-derived meshes or
images, HRN or 3DDFA-V2 model weights, reconstruction outputs, target scans,
rating panels, or private case keys. Obtain and use each external resource
under its provider's terms:

- FaceScape: <https://nju-3dv.github.io/projects/FaceScape/>
- HRN: <https://github.com/younglbw/hrn>
- 3DDFA-V2: <https://github.com/cleardusk/3DDFA_V2>

The revised 3DDFA-V2 run used commit
`1b6c67601abffc1e9f248b291708aef0e43b55ae` and
`configs/mb1_120x120.yml` from the official repository.

## Environment

Use Python 3.10 or 3.11. Open3D 0.19.0 does not provide the same standard
wheel coverage for every newer Python release.

```text
python -m venv .venv
python -m pip install -r requirements.txt
```

Set `PAPERFIT_ROOT` to a separately prepared, licensed workspace. The scripts
also accept explicit input and output paths; use `--help` for the full command
line of each stage. Outputs are required to remain under the configured root,
and existing result directories are not silently overwritten.

`run_anchor_aware_s8_pilot.py` retains its original development filename for
provenance and code-map continuity. The revised production scripts import
only its frozen S8 core functions; its historical direct-execution pilot is
not a revised headline experiment.

## Revised Evaluation Order

1. Build the target face ROI and semantic target anchors with
   `build_mediapipe_target_anchors.py`.
2. Reconstruct the secondary source with the official 3DDFA-V2 code, if that
   comparison is required.
3. Run `run_pairwise_mm_rigid.py` with the frozen identity split and official
   FaceScape target scale dictionary.
4. Run `run_mm_s8_from_rigid.py` for every evaluable pre-S8 stratum.
5. Apply the frozen branch-independent conjunction with
   `analysis_tools/apply_frozen_common_qc.py`.
6. Run the conventional common-ROI, shared-cues ICP, ARAP, and ablation tools
   under the same target support, scale, denominator, and final QC.
7. Use `analysis_tools/` to reproduce subject-clustered intervals, paired
   tests, sensitivity analyses, and aggregate tables.

The frozen common final limits are 6 mm full-surface median, 30 mm full-surface p90, 3 mm
nasal median, 10 mm nasal p90, 30 mm semantic nose-anchor distance, 0.35 p99
edge strain, valid facial orientation, and a `1e-6` mm fixed-eye implementation
audit. Invalid evidence and native solver failures remain failed attempts in
the denominator.

## Reproducibility Boundary

The automatic held-out experiment was executed once after all development
settings and hashes were frozen. Do not use held-out outcomes to revise a
threshold or method setting. Human-rating panels remain private because they
are FaceScape-derived. The completed, locked held-out assessment is released
only as aggregate agreement and QC-predictive statistics in
`results/aggregate/`.

## Current Automatic Result

On the 10-identity, 190-pair HRN held-out partition, the proposed method
accepted 168/190 attempts. 3DDFA-V2 accepted 164/190, shared-cues Open3D ICP
accepted 101/190, common-ROI FPFH-RANSAC+ICP accepted 4/190, and ARAP-A
accepted 0/190 under the same common final QC. See `results/aggregate/` for
subject-aware intervals and the complete automatic summaries. The two primary
raters gave the same overall-usability label for 185/190 HRN held-out outputs.
A third evaluator rated only the five disagreements without seeing the primary
ratings or hidden case metadata and QC results. The resulting final visual
reference classified 169/190 outputs as usable; the frozen QC had 168 true
positives, no false positives, 21 true negatives, and one false negative.
