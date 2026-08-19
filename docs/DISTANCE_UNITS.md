# Distance Units in the Revised Evaluation

The revised rigid, S8, baseline, QC, and statistical outputs are computed and
reported in millimeters. FaceScape multi-view PLY models have scan-dependent
Structure-from-Motion scale. The official FaceScape alignment example provides
a subject- and expression-specific factor for alignment to the topologically
uniform canonical models, whose unit is millimeters.

For subject `j` and target expression `e`, the conversion is:

```text
distance_mm = Rt_scale_dict[j][e][0] * distance_registration_unit
```

Expression 18 is the target in this study. Across the 20 evaluated targets,
the official factors range from 144.59 to 517.93 mm per registration unit,
with a median of 235.04. A fixed multiplier such as `raw * 100` is therefore
not used.

Every revised target mesh, target face ROI, and target anchor is converted
before optimization. Source scale is then fixed by the target-to-source 3D
interocular-distance ratio. Distance-valued inherited optimizer constants are
converted with the median factor from development identities only
(262.581050 mm per registration unit). Final case measurements and QC evidence
always use the official factor for that case's target; no factor or outcome
from the evaluation partition is used to select a threshold or method setting.

Official source:
<https://github.com/zhuhao-nju/facescape/blob/master/toolkit/demo_align.ipynb>

Preserved `Rt_scale_dict.json` SHA-256:
`b40a929b11a2ba99bc14ba9497740c0214a14979ab2dac6794a7a3c8bb729d02`.
