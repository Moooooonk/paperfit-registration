# PaperFit Registration: CMES Revision Code

This package contains the sanitized code and aggregate numerical results for
the revised CMES evaluation of eye-preserving cross-source 3D face
registration. It corresponds to the identity-disjoint, millimeter-scale,
branch-independent evaluation described in the revised manuscript.

## What Is Included

- `revision_tools/`: target-anchor and face-ROI construction, HRN/3DDFA-V2
  rigid registration, S8 refinement, Open3D rigid controls, the shared-cues
  ICP control, full-resolution ARAP, and all-attempt ablation.
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

After installing the dependencies, the public package can be checked with:

```text
python -m compileall -q analysis_tools revision_tools tests
python -m pytest -q -rs
```

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
4. Run `run_mm_s8_from_rigid.py` for every processable pre-S8 stratum. A
   labeled visible-point fallback may preserve diagnostic processing when a
   direct target nose-ray intersection is unavailable.
5. Validate and record target-anchor provenance with
   `analysis_tools/validate_target_anchor_evidence.py`. Only a direct
   perspective ray--mesh intersection is protocol-valid for final acceptance.
6. Build the immutable fail-closed analysis view with
   `analysis_tools/build_fail_closed_analysis_view.py`; fallback-based cases
   remain in the denominator but cannot be accepted.
7. Apply the frozen branch-independent conjunction to that analysis view with
   `analysis_tools/apply_frozen_common_qc.py`.
8. Run the conventional common-ROI and shared-cues ICP controls under the
   frozen common final QC. Run the full-resolution ARAP comparison from the
   common rigid initialization and evaluate it with the prespecified paired
   continuous fit and deformation metrics; the S8-calibrated binary QC is not
   used as a method-neutral ARAP endpoint.
9. Use `analysis_tools/` to reproduce subject-clustered intervals, paired
   tests, sensitivity analyses, and aggregate tables.

The frozen common final rule requires protocol-valid direct target-anchor
evidence, valid facial orientation, at most 6 mm full-surface median, 30 mm
full-surface p90, 3 mm nasal median, 10 mm nasal p90, 30 mm semantic
nose-anchor distance, 0.35 p99 edge strain, and a `1e-6` mm fixed-eye
implementation audit. Invalid evidence and native solver failures remain
failed attempts in the denominator.

Throughout this package, `full-surface` denotes the facial evaluation mask
after the soft eye/orbit exclusion; it does not include the protected
eye/orbit vertices.

## Reproducibility Boundary

The automatic evaluation experiment was executed once after all development
settings and hashes were frozen. Do not use evaluation outcomes to revise a
threshold or method setting. All 20 identities had appeared in the submitted
pooled analysis, so this identity-disjoint revision-stage partition is not an
untouched external cohort. Human-rating panels remain private because they
are FaceScape-derived. The completed, locked evaluation assessment is released
only as aggregate agreement and QC-predictive statistics in
`results/aggregate/`.

## Current Automatic Result

On the 10-identity, 190-pair HRN evaluation partition, the proposed method
accepted 168/190 attempts. 3DDFA-V2 accepted 164/190, shared-cues Open3D ICP
accepted 101/190, and common-ROI FPFH-RANSAC+ICP accepted 4/190. The
development-selected ARAP-FR setting completed all 190 attempts at full source
resolution. Relative to S8, it had larger full-surface and nasal distances but
lower edge strain and displacement; this is reported as a fit--deformation
tradeoff, not as a binary-QC comparison or uniform superiority claim. See
`results/aggregate/` for subject-aware intervals and the complete automatic
summaries. The two primary
raters gave the same overall-usability label for 185/190 HRN evaluation outputs.
A third evaluator rated only the five disagreements without seeing the primary
ratings or hidden case metadata and QC results. The resulting final visual
reference classified 169/190 outputs as usable; the frozen QC had 168 true
positives, no false positives, 21 true negatives, and one false negative.
