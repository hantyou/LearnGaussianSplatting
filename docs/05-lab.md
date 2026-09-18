# 5. Learn by predicting, implementing, and testing

Use the workspace `.venv` commands in the README. The shorter `python` commands
below assume that interpreter is active. Suggested timings are study-session
estimates, not requirements; pause whenever you can no longer explain a line.

## Session 1: understand the forward pass (45–60 minutes)

Read chapters 1 and 2 through compositing. Open `splatlab/core.py` and write
the input/output shapes on paper before reading its operations. Run:

```text
python -m splatlab.demo
python -m pytest -q
```

Open the comparison image and orbit animation. Identify a Gaussian that becomes
partially hidden as the camera moves. Explain which parameters are shared across
views, and which intermediate quantities are recomputed for each camera.

**Checkpoint questions:** Why square the scales? Why is t not the camera center?
Why does depth variance affect an off-axis footprint? Why must the blend be ordered?

## Session 2: implement the missing mathematics (60–90 minutes)

Edit only `exercises/core.py`. The quaternion rotation helper is supplied, so
your attention stays on the four essential operations. `splatlab/core.py` is
the complete answer key; use it after making a serious attempt.

| Exercise | Implement | Run | Main pitfall |
|---|---|---|---|
| 1 | Positive definite 3D covariance | `python -m pytest --student -q -k covariance` | Standard deviation versus variance |
| 2 | Camera transform + projection Jacobian | `python -m pytest --student -q -k projection` | Transform direction and missing z derivatives |
| 3 | Elliptical pixel footprint | `python -m pytest --student -q -k footprint` | x/y versus row/column; unwanted PDF normalization |
| 4 | Ordered compositing | `python -m pytest --student -q -k composite` | Updating T before adding color; forgetting background |

Then run all checks and train with your functions:

```text
python -m pytest --student -q
python -m splatlab.demo --student --output outputs/student
```

The gradient check uses small double-precision tensors and compares autograd
with finite differences. A function can produce correct-looking images and still
break gradients, for example by converting tensors to NumPy or using `.detach()`.
The empty-scene check ensures a camera looking away yields the background.
The two-layer opacity test also checks that increasing foreground opacity hides
the colored suffix and background. Detaching transmittance gives a correct
forward image but fails that derivative check.

Once all four functions pass, use them on the larger street with
`python -m splatlab.fit_street --student`. Follow [chapter 7](07-viewer.md) to
inspect the resulting 755 Gaussians and compare Before / Learned / Target.

## Session 3: distinguish fitting from understanding (45–60 minutes)

Compare five-view and one-view learning with the same initialization:

```text
python -m splatlab.demo --output outputs/reference
python -m splatlab.demo --train-views 1 --output outputs/single-view
```

**Measured in this workspace**, seed 7, 300 steps, 48×48 images, CPU:

| Experiment | Final training PSNR | Final unseen-view PSNR |
|---|---:|---:|
| Five training cameras | 49.51 dB | 50.36 dB |
| One training camera | 50.47 dB | 26.73 dB |

The common unseen cameras are at -37.5°, 12.5°, and 37.5°. The one-view experiment
fits only 0°; the five-view experiment uses -50°, -25°, 0°, 25°, 50°. Training PSNR
therefore averages different camera sets. The **unseen** comparison is shared.

The numerical result is a demonstration, not a dataset benchmark. Targets come
from the same rendering model; initialization uses perturbed teacher parameters;
Gaussian count is known; appearance is view-independent. The five-view result
does not establish that individual parameters uniquely recover the teacher.
PSNR is \(-10\log_{10}(\operatorname{MSE})\) for RGB in [0,1], computed from
mean MSE over the relevant images, not mean of per-image PSNRs.

Both checked runs took about 3.6 seconds for the optimization loop on this
machine. Startup, imports, plotting, and GIF generation take additional time.
Exact values may change with versions, hardware, or seed. Inspect `metrics.json`.
The five-view loss had a small late increase; fixed learning rates, alternating
views, and discrete ordering allow nonmonotonic behavior. We report the final
iterate, not a checkpoint selected using unseen-view results.

## Small experiments with predictions

Make one change at a time and use a separate output directory.

1. **Freeze geometry.** After creating `student`, set `requires_grad_(False)` on
   `means`, `log_scales`, and `quaternions`. Predict whether color and opacity can
   repair misplaced footprints. Compare residual patterns with the reference.
2. **Force isotropy.** In an experimental copy, replace each log-scale triple by
   its mean expanded to three entries before forming covariance. You now have
   one effective scale per splat. Predict which elongated features fit poorly.
3. **Reverse sorting.** Change `argsort(depth)` to descending order. Which
   overlaps change color? Restore the reference ordering afterwards.
4. **Change initialization.** Try `--seed 19` and `--seed 31`. Remember that this
   changes perturbations around the known scene, not a fully random reconstruction.
5. **Inspect parameters.** Load `learned_scene.npz` using `numpy.load`. Are good
   image metrics accompanied by identical individual parameters? Explain why not.

## Next extensions, in a deliberate order

**A. Add view-dependent color.** Implement real degree-one SH with a documented
direction convention. Generate a teacher with directional color too; constant
RGB targets cannot demonstrate its necessity. Unit-test basis values on axes.

**B. Add density control.** Start by measuring gradients of projected centers
(`uv.retain_grad()`), accumulate them only for visible splats, and inspect the
statistics before changing topology. Clone small/high-gradient splats, split
large/high-gradient ones, and prune low-opacity ones. When parameter arrays
change, update optimizer groups and Adam moments. Simply assigning a new
`nn.Parameter` can leave the optimizer updating obsolete tensors.

**C. Use real multiview images.** Move to gsplat or the original repository.
Begin with a small static capture with calibrated poses and an SfM point cloud.
Check transform conventions, image resizing/intrinsics, masks, and train/test
views before touching losses. Do not feed a city dataset to this dense renderer.

**D. Connect to radar.** Derive polar-to-Cartesian covariance propagation from
chapter 4 and implement a tiny BEV feature-splat experiment. Only after that
choose between radar image synthesis and camera–radar perception.

## Debugging by symptom

| Symptom | First thing to inspect |
|---|---|
| Mirrored or upside-down result | Camera handedness, axis directions, pixel row/column ordering |
| Everything is invisible | World-to-camera versus camera-to-world; sign of z |
| Splats explode near camera | Division by z, near-plane handling, scales |
| Covariance inversion fails | Positive scales, quaternion validity, 2D variance floor |
| Correct colors but wrong occlusion | Depth ordering and transmittance update |
| Gradients absent | `.detach()`, NumPy conversion, tensor reconstruction, `no_grad()` |
| Good training view, bad orbit | Insufficient views, geometry ambiguity, initialization |

## Answer sketches for self-checking

- Covariance squares axis standard deviations because its eigenvalues are variances.
- t=-RC maps the camera center to zero; a translation does not change covariance.
- Off-axis projection changes horizontal/vertical position when depth changes.
- Opacity consumes transmittance; reversing layers changes each layer's weight.
- A low loss says rendered measurements fit, not that the inverse problem has a
  unique solution. That distinction should feel familiar from Bayesian inference.
