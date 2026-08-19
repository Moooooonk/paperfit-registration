# Aggregate Results

These files contain manuscript-level automatic and development-rating
summaries. No actual FaceScape subject identifier, per-case row, geometry,
image, panel, or private case key is included.

All acceptance denominators include invalid evidence and native solver
failures. Confidence intervals resample complete subjects. ARAP-FR completed
all 190 evaluation attempts. Its paired full-surface, nasal, and deformation
metrics therefore use 190 cases from 10 subjects; only the semantic-anchor row
uses 171 cases from nine subjects with protocol-valid direct target-anchor
evidence. The S8-calibrated binary QC is not used as a cross-method ARAP
endpoint.

The `heldout_*` filenames are retained from the frozen internal pipeline for
machine-readable compatibility; in the manuscript they denote the
revision-stage evaluation partition, not an untouched external cohort.

`heldout_component_ablation.csv` distinguishes final acceptance from the
no-eye diagnostic. Because that condition deliberately removes eye/orbit
fixation, its count is passage of the remaining geometric gates rather than a
replacement final-acceptance count.

`heldout_perceptual_qc_summary.csv` reports only locked cohort-level agreement,
final visual-reference, and frozen-QC statistics. The private rating forms,
panels, case mapping, and individual case decisions are not distributed.

The `consensus_*` field names are retained for machine-readable compatibility.
For the evaluation-partition overall-usability label, they denote the shared label when the
two primary raters agreed and the two-of-three majority label after a third
masked rating for the five disagreements. No group discussion or negotiated
post hoc consensus was used.
