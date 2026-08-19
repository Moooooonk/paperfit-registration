# Full-Resolution ARAP Re-audit Protocol

Frozen on 2026-08-19 before inspecting cohort-wide full-resolution ARAP results.

## Purpose

Define the finalized, numerically stable, face-adapted ARAP baseline at the original
source-mesh resolution. The baseline uses the same rigid initialization, target
face samples, source facial masks, and target-frame millimeter conversion as the proposed
pipeline. All source eye/orbit vertices are hard positional constraints.

Three development cases were used only as implementation smoke tests. They established
that the full-resolution solver completed and that constrained-eye displacement could be
measured without post-solve vertex restoration. They are not excluded from the 190-case
development run and are not used separately in configuration selection.

## Frozen split and provenance

- Split file SHA-256: `bac06897e25ff75df4dce95e43e5b25ee991aa6e735e2c75f67a169992867c46`
- Scale file SHA-256: `b40a929b11a2ba99bc14ba9497740c0214a14979ab2dac6794a7a3c8bb729d02`
- Development: the ten identities under `development_subjects`, 190 pairs.
- Evaluation: the ten identities under `test_subjects`, 190 pairs.
- The evaluation partition is identity-disjoint from revision-stage development, but all
  20 identities appeared in the originally submitted pooled analysis. It is therefore
  described as a revision-stage evaluation partition, not an untouched external cohort.

## Common implementation

- Original HRN mesh resolution; no decimation and no displacement transfer.
- Open3D spokes ARAP energy, ten solver iterations per round.
- Deterministic facial control-vertex sampling.
- All eye/orbit vertices constrained to their rigid-stage positions.
- No explicit eye-vertex restoration after a full-resolution solve.
- One numerical thread and one isolated case process at a time.
- A failed or incomplete case remains in the 190-attempt denominator.

## Development candidates

| Setting | Controls | Rounds | Maximum target step per round |
|---|---:|---:|---:|
| FULL_C300_R1_S05 | 300 | 1 | 0.5 mm |
| FULL_C300_R1_S10 | 300 | 1 | 1.0 mm |
| FULL_C600_R1_S10 | 600 | 1 | 1.0 mm |
| FULL_C600_R2_S10 | 600 | 2 | 1.0 mm |

## Frozen selection rule

Only candidates completing all 190 development attempts with no orientation failure are
eligible. For each eligible candidate and each metric, the mean is first calculated within
each subject and the median of the ten subject means is then used as the candidate value.

Fit metrics are post-ARAP full-surface median, full-surface 90th-percentile, nasal median,
nasal 90th-percentile, and nose-anchor error. Here, full-surface denotes the facial
evaluation region after the soft eye/orbit exclusion, matching the manuscript and S8
comparison. Deformation metrics are non-eye edge-strain p99 and
eye-boundary edge-strain p99. Each metric is min-max normalized across eligible candidates,
where zero is best. The selection score is the equally weighted mean of the average fit and
average deformation scores. Exact ties are resolved by lower `controls x rounds x step`,
then fewer controls, fewer rounds, and smaller step.

No S8-calibrated acceptance threshold or evaluation-partition result is used for selection.
After selection, exactly one frozen setting is run once on all 190 evaluation attempts.

## Comparison policy

Cross-method conclusions do not rely on the S8-calibrated binary QC gate. The ARAP
comparison uses prespecified paired continuous fit and deformation measures on the full
evaluation partition. Proposed-method QC remains an internal fail-closed policy and is
reported separately.
