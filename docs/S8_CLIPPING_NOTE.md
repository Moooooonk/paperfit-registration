# S8 Maximum Step Clipping

The implementation clips the local target step as:

```python
step = np.minimum(1.0, max_step / np.maximum(norm, 1e-8))
target = current[target_idx] + delta * step[:, None]
```

Therefore, the exact mathematical form is:

```tex
\hat{x}_i^{(k)}
=
x_i^{(k)}
+
\min\left(
1,
\frac{\delta_k}{\max(\|\Delta_i\|_2,\epsilon)}
\right)\Delta_i .
```

This clips the local target used by the constrained solve, not the final vertex displacement after the solve/blending step.

