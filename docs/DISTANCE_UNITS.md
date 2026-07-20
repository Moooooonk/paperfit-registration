# Distance Units

All distance thresholds, CSV values, and manuscript tables use the numerical
coordinate units of the target registration frame used by the evaluated code.

The FaceScape multi-view PLY models are reconstructed at an uncertain
Structure-from-Motion scale. The official FaceScape alignment example states
that the topologically uniform canonical models use millimeters and provides
subject/expression-specific scale metadata for transforming a multi-view PLY
model into that canonical coordinate system. A case-level distance can
therefore be reported as:

```text
distance_mm = Rt_scale_dict[subject][target_expression][0] * distance_registration_unit
```

The evaluated experiment used expression 18 as the target. Across subjects
001--020, the corresponding factors range from 144.59 to 517.93 mm/unit, with
a median of 235.04 mm/unit. Thus, 0.01 registration units corresponds to a
median of about 2.35 mm (range 1.45--5.18 mm). A fixed conversion such as
`raw * 100` is not scientifically justified and is not used.

Reproducing the reported 380-pair experiment requires retaining the same
target-frame coordinates and thresholds. Millimeter columns may be added
post hoc for interpretation without changing the registration or acceptance
decisions. Operating the pipeline itself with subject-independent millimeter
thresholds would be a different protocol and would require scale-aware
recalibration and validation.

Official source: <https://github.com/zhuhao-nju/facescape/blob/master/toolkit/demo_align.ipynb>
