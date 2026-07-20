# Included Code Map

The public scripts in `scripts/` are the files a reader should run.

The files in `paperfit_legacy_impl/` are included because the public scripts call them directly or because those implementation files import each other. They are not separate manuscript claims.

## Direct manuscript stages

- Main rigid QC: `run_rigid_upright_hardgate_allpairs_20260622.py`
- Auxiliary rigid recovery: `run_rigid_upright_hardgate_allpairs_recovery_20260622.py`, `run_rigid_allpairs_subject004_prior_recovery_20260622.py`, `run_rigid_allpairs_nose_anchor_targeted_recovery_20260622.py`
- Effective status merge: `make_allpairs_effective_status_20260623.py`
- S8 rigid-pass branch: `run_nonrigid_proposed_s8_allpairs_20260622.py`
- S8 anchor-only branch: `run_nonrigid_proposed_s8_allpairs_rough_anchorfail_20260623.py`
- S8 broad-failure branch: `run_nonrigid_proposed_s8_allpairs_broadfail_probe_20260623.py`
- Open3D baselines: `run_rigid_open3d_baselines_allpairs_20260623.py`, `run_rigid_open3d_facecrop_baselines_allpairs_20260704.py`
- Adaptive-template-inspired baseline: `run_dai_like_adaptive_template_allpairs_20260624.py`
- Component ablation: `run_nonrigid_component_ablation_representative40_20260624.py`

## Implementation dependencies

- `run_nonrigid_nasal_depth_ablation_3case.py`
- `run_nonrigid_proposed_s8_hardgate_full40_pass32_20260619.py`
- `run_scratch_surface_registration_3case_final_attempt.py`
- `run_rigid_upright_hardgate_3case_20260619.py`
- `run_rigid_upright_hardgate_full40_20260619.py`
- `run_rigid_fail6_nose_anchor_targeted_fast_20260622.py`
- `run_rigid_global_upright_failed11_20260617.py`
- `run_rigid_fail6_nose_tip_gradient_candidate_only_20260622.py`
- `run_rigid_open3d_baselines_40_20260622.py`
- `run_nonrigid_component_ablation_goodrigid_6case_20260611.py`
- `run_rigid_global_grid_icp_baseline_20260625.py`
