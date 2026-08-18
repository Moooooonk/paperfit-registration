# Revised Method-to-Code Map

| Manuscript operation | Public script |
|---|---|
| Target face ROI, target nose anchor, target eye centers | `revision_tools/build_mediapipe_target_anchors.py` |
| Secondary 3DDFA-V2 source generation | `revision_tools/run_3ddfa_reconstruction.py`, `revision_tools/run_3ddfa_batch.py` |
| Interocular-scaled eight-candidate rigid stage | `revision_tools/run_pairwise_mm_rigid.py` |
| Eight-stage contour plus five local and three propagation updates | `revision_tools/run_mm_s8_from_rigid.py` |
| Core constrained non-rigid operations | `revision_tools/run_anchor_aware_s8_pilot.py`, `revision_tools/frozen_nonrigid_nasal_solver.py` |
| Common branch-independent post-S8 QC | `analysis_tools/apply_frozen_common_qc.py` |
| Conventional common-ROI controls | `revision_tools/run_open3d_common_roi_baselines.py` |
| Shared-cues ICP control | `revision_tools/run_open3d_shared_cues_icp_baseline.py` |
| Eye-constrained ARAP comparison | `revision_tools/run_arap_baseline_from_rigid.py` |
| Complete-case S8 ablation | `revision_tools/run_s8_component_ablation.py` |
| Development-only QC calibration | `analysis_tools/calibrate_common_qc_development.py` |
| Frozen common-QC application | `analysis_tools/apply_frozen_common_qc.py` |
| Subject-clustered statistics | `analysis_tools/analyze_subject_clustered_results.py`, `analysis_tools/analyze_subject_clustered_metrics.py` |
| Paired comparisons and Holm adjustment | `analysis_tools/compare_subject_clustered_methods.py`, `analysis_tools/summarize_paired_comparisons.py` |
| Blinded-rating aggregation and QC prediction | `analysis_tools/aggregate_blinded_ratings.py`, `analysis_tools/analyze_qc_against_blinded_ratings.py` |

`S8` names the eight nasal depth-contour stages. The complete scheduled update
sequence is 8 contour + 5 fixed facial-region + 3 full-face propagation
updates. A contour or fixed facial-region update with fewer than 20 eligible
active vertices is a recorded no-op, so 16 scheduled updates do not
necessarily imply 16 executed linear solves.
