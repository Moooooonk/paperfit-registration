# GitHub Release Audit

## Release decision

This clean folder is suitable as the GitHub repository for code and manuscript-level aggregate-result sharing.

## What is included

- Source code needed to trace the manuscript pipeline.
- Manuscript-level aggregate result tables without per-case rows.
- Documentation describing excluded data and required environment variables.

## What is excluded

- FaceScape raw meshes, scans, textures, rendered portraits, and derived mesh/image assets.
- HRN model weights and HRN output mesh caches.
- Local quick-test OBJ/PLY/JPG data.
- Local figure asset packages and PPT diagrams.
- Manuscript PDF/LaTeX files.
- Private server paths and machine-specific configuration.

## Checks completed

- `python -m compileall -q <repo>` passed.
- Public wrapper scripts point to existing implementation files.
- Repository scan found no private local paths or server paths.
- Repository scan found no raw mesh, scan, model-weight, archive, or image files.
- External FaceScape and HRN access links are documented in `README.md` and `docs/DATASET_AND_PRIVATE_FILES.md`.
- The current default local Python environment is Python 3.13.5 and does not have `open3d` installed. Full baseline execution therefore requires a separate Python 3.10/3.11 environment with `requirements.txt` installed.

## Manuscript consistency checks

- Final coverage table matches the manuscript values: 214 main strict rigid QC, 22 rigid recovery merged, 84 anchor-only S8 recovery, 26 broad-failure S8 threshold accepted, 346/380 final accepted, and 34 residual failures.
- Stage-level S8 metric table matches the manuscript in target-frame coordinate units.
- Face-ROI Open3D baseline table matches the manuscript values, including the best baseline result of 68/380 strict QC passes (17.89%).
- Geometry-landmark adaptive-template-inspired baseline values match the manuscript in target-frame coordinate units.
- Representative Figure 3 table values match the local Figure 3 numeric report after applying the official target-specific FaceScape scale factors.
- Representative 35-case component ablation values match the manuscript table in target-frame coordinate units.

## Runtime scope

The public repository does not include FaceScape data, HRN weights, HRN output mesh caches, or local figure assets. Therefore, a full 380-pair rerun requires a prepared licensed workspace, `PAPERFIT_ROOT`, and a Python 3.10/3.11 environment with the required packages installed. In this clean release folder, source-level compilation, wrapper-target existence checks, dependency availability checks, and manuscript-result consistency checks were completed; full end-to-end rerun was not performed inside the public folder alone.

## Manuscript link

The public repository URL is included in the manuscript Data Availability statement.
