# PaperFit Registration

This repository contains the public code package for the CMES manuscript experiments.

The public entry points are in `scripts/`. The original experiment implementation files are kept once in `paperfit_legacy_impl/` so that manuscript results remain traceable without duplicating similar scripts across folders.

## Public Entry Points

```text
scripts/run_01_main_rigid_qc.py
scripts/run_02_auxiliary_rigid_recovery.py
scripts/run_03_s8_refinement.py
scripts/run_04_baselines.py
scripts/run_05_component_ablation.py
scripts/run_06_final_decision.py
```

## What Is Directly Used

- `run_01_main_rigid_qc.py`: main all-pairs rigid candidate generation and strict QC.
- `run_02_auxiliary_rigid_recovery.py`: auxiliary rigid recovery and effective rigid-status merge.
- `run_03_s8_refinement.py`: S8 refinement for rigid-pass, anchor-only, and broad-failure branches.
- `run_04_baselines.py`: full-target Open3D, cropped-target Open3D, and adaptive-template-inspired baselines.
- `run_05_component_ablation.py`: representative component ablation.
- `run_06_final_decision.py`: branch-wise final acceptance and one auditable status per pair.

Files in `paperfit_legacy_impl/` are implementation dependencies used by these entry points. They are not all separate manuscript stages.

## Key Result

The final accepted set is 346/380 source-target pairs, or 91.05%. The rigid-pass set also undergoes S8 refinement and fixed-eye displacement audit. In S8, the name refers to the eight-stage nasal depth-contour schedule. The implementation then performs five anatomical local updates and three full-face propagation updates outside the fixed eye/orbit region, for an 8+5+3 constrained-update sequence.

QC thresholds and aggregate experiment tables retain the numerical coordinate units of the FaceScape target registration frame used by the evaluated code. For physical interpretation, a case-level distance can be converted with the official FaceScape subject/expression scale, `distance_mm = scale(subject, expression) * distance_registration_unit`. A fixed global factor such as `raw * 100` is not used. See `docs/DISTANCE_UNITS.md`.

## External Data and Reconstruction Dependencies

This repository does not redistribute FaceScape data, FaceScape-derived meshes or images, HRN weights, HRN output meshes, local quick-test data, or figure asset packages.

To reproduce the experiments, prepare the following external resources separately:

- FaceScape dataset: <https://nju-3dv.github.io/projects/FaceScape/>
- FaceScape license agreement: <https://facescape.nju.edu.cn/static/License_Agreement.pdf>
- HRN official implementation: <https://github.com/younglbw/hrn>
- HRN project page: <https://younglbw.github.io/HRN-homepage/>

Use FaceScape only under the provider's license terms. In particular, do not redistribute FaceScape meshes, scans, textures, rendered portraits, or FaceScape-derived mesh/image files through this repository.

Set `PAPERFIT_ROOT` to a local prepared workspace that contains the licensed FaceScape data and HRN reconstruction outputs required by the scripts.

## Environment

Use Python 3.10 or 3.11. The baseline scripts require `open3d==0.19.0`, which is not available for Python 3.13 in the standard PyPI wheels.

```powershell
$env:PAPERFIT_ROOT = "D:\path\to\prepared\facescape_pipeline"
python scripts\run_01_main_rigid_qc.py
python scripts\run_02_auxiliary_rigid_recovery.py
python scripts\run_03_s8_refinement.py
python scripts\run_06_final_decision.py --expect-paper-counts
python scripts\run_06_final_decision.py --expect-paper-counts --facescape-scale-dict D:\path\to\facescape\toolkit\predef\Rt_scale_dict.json
```

## Notes

The original implementation filenames include dates because they correspond to the experiment runs used to derive the manuscript tables. Public-facing scripts use stable names; legacy names are kept only for traceability. `run_03_s8_refinement.py` explicitly disables the legacy 16-case anchor-only smoke-test limit so that all 84 reported anchor-only cases are processed.

This repository does not include manuscript figure source files or PPT diagrams. The submitted manuscript package remains the source of truth for PDF, LaTeX, and final figure files.
