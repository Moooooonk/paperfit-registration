# Method-to-Code Alignment

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

## Final decision

- Rigid pass: fixed-eye audit must pass after S8.
- Anchor only: full and nasal median distances must not increase, and the
  fixed-eye audit must pass.
- Broad failure: orientation, full median, nasal median, nasal p90, and
  fixed-eye criteria must all pass.

Run `python scripts/run_06_final_decision.py --expect-paper-counts` after the
three S8 branches. The command fails if the reproduced branch counts differ
from the manuscript values `236 + 84 + 26 = 346`, with 34 residual failures.
