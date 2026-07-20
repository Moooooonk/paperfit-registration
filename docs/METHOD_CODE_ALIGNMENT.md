# Method-to-Code Alignment

## Rigid candidate search and recovery

The coarse rigid search evaluates four proper-rotation axis conventions and
17 scale values from `0.58` to `1.38` times an extent-based initial scale,
producing 68 initializations. The eight lowest-scoring candidates are retained
for 20 iterations of trimmed, source-weighted rigid ICP and translation
polishing. Strict-QC failures are rerun with scale refinement. The expanded
general recovery retains 32 candidates and uses 36 rigid iterations;
same-subject prior and nose-anchor-targeted recovery provide additional
candidates. Every recovered output is merged only if it satisfies the same
strict rigid QC as the main pass.

The source-side rigid weights begin with the soft eye/orbit exclusion weight.
The midface, nasal bridge/dorsum, tip/alar, philtrum, mouth/lower lip, and
outer-face regions are multiplied by `1.20`, `2.05`, `2.35`, `1.04`, `0.42`,
and `0.62`, respectively, before clipping to `[0, 3]`.

## S8 update sequence

`S8` names the eight-stage nasal depth-contour schedule:

```text
0.00, 0.22, 0.40, 0.55, 0.68, 0.78, 0.87, 0.94
```

The evaluated implementation performs three consecutive phases:

1. Eight nasal depth-contour updates.
2. Five anatomical local updates: nasal bridge, nasal dorsum, tip/alar,
   subnasal region, and philtrum.
3. Three full-face propagation updates outside the fixed eye/orbit region.

The constrained solve preserves edge offsets relative to the mesh at the start
of the current update. Its solution is blended with the current mesh using a
gain of `0.43` in the first 13 updates and gains of `0.12`, `0.14`, and `0.13`
in the final three updates. Fixed eye/orbit vertices are restored after every
update.

## S8 solve parameters

For the eight contour updates, the maximum step is `0.018` target-frame units.
For the five anatomical local updates, it is `0.014`. Their target-fitting
weight is `18 + 85 * clip(distance, 0, 0.20)`, with a factor of `1.25` for the
tip/alar update and contour thresholds of at least `0.75`. These 13 updates use
an edge coefficient of `190`, a fixed-eye coefficient of `32000`, and a blend
gain of `0.43`.

For the three full-face propagation updates, the tuples
`(edge coefficient, base fitting weight, gain, maximum step)` are:

```text
(210,  9, 0.12, 0.018)
(185, 11, 0.14, 0.016)
(165, 12, 0.13, 0.014)
```

Their target-fitting weight is
`base + 65 * clip(distance, 0, 0.25)`. Multipliers for the nasal bridge, nasal
dorsum, nose tip, alar, subnasal, and philtrum masks are `1.30`, `1.35`,
`1.42`, `1.35`, `1.18`, and `1.12`, respectively. The fixed-eye coefficient
remains `32000`. A value of `1e-8` is used both as the clipping denominator
lower bound and as a sparse-system diagonal stabilizer.

## Distance aggregation

Strict rigid QC and the rigid/global comparison tables use source-side fitting
weights. The full evaluation set contains weights above `0.08`, and the nasal
set contains weights above `1.75`. Weighted quantiles are computed by sorting
distances and linearly interpolating them against normalized cumulative
weights. The S8 before/after, representative-example, and component-ablation
tables instead use ordinary quantiles on binary evaluation masks. This is why
the manuscript distinguishes source-weighted rigid metrics from unweighted
pre/post-S8 metrics.

## Final decision

- Rigid pass: fixed-eye audit must pass after S8.
- Anchor only: full and nasal median distances must not increase, and the
  fixed-eye audit must pass.
- Broad failure: orientation, full median, nasal median, nasal p90, and
  fixed-eye criteria must all pass.

Run `python scripts/run_06_final_decision.py --expect-paper-counts` after the
three S8 branches. The command fails if the reproduced branch counts differ
from the manuscript values `236 + 84 + 26 = 346`, with 34 residual failures.
