# 7. Reconstruct a scene, move through it, inspect its primitives

Start with **[the real Room scene](http://127.0.0.1:8765/?scene=room)** while the
local server is running. Then use the Scene menu for **Train** and **Miniature
street**. The single-Gaussian projection exercise is a stepping stone; this lab
lets you investigate a complete scene, occlusion, generalization, and geometry.

## What is actually included

| Scene | What you are exploring | Was it trained by this lab? |
|---|---|---|
| Room | Real capture, 1,593,376 Gaussians | No; downloaded pretrained model |
| Train | Real capture, 1,026,508 Gaussians | No; downloaded pretrained model |
| Miniature street | Road, two cars, building, trees; 755 Gaussians | Yes; complete CPU image-fitting experiment |
| Nine Gaussians | Small, easily traced mathematical exercise | Yes; complete CPU image-fitting experiment |

The synthetic targets are procedural Gaussian scenes. Initialization perturbs
their known parameters; both the count and approximate geometry are supplied.
Only rendered training-image error enters the optimization objective. This is
deliberately easier than reconstructing unknown geometry from photographs.
There is no claim that the readable CPU trainer can fit the million-splat real
captures in this viewer. That next step needs an efficient implementation and
calibrated images; see [the repository reading guide](06-reading-code.md).

## Five things to try

1. **Move inside a real scene.** Drag to orbit, right-drag to pan, scroll to zoom.
   Double-click a visible surface to change the orbit focus. Choose **Fly through**,
   click the canvas, and use W/S forward/back, A/D sideways, Q/E down/up, Shift
   for faster motion. Left-drag to look. Reset view returns to the starting pose.
   There is no collision detection; you can pass through walls.
2. **Expose the representation.** Switch to **3D ellipsoids** without moving the
   camera. Choose up to 10,000 samples for real scenes. Their sparse appearance
   reflects diagnostic sampling; rendered mode still uses the whole model.
   The miniature street shows all 755 primitives at the default sample budget.
3. **Inspect one primitive.** Click a solid ellipsoid, then **Focus selected
   Gaussian** and **Isolate selected Gaussian**. Read its center, opacity, and
   three axis standard deviations. Compare 1σ, 2σ, and 3σ contours. This changes
   the diagnostic surface, not the model's learned scale or opacity.
4. **Compare the learned street to its target.** Switch Before / Learned / Target
   while keeping the viewpoint. Look from between the training cameras and then
   from an unusual elevation. Identify errors the training images did not resolve.
5. **Look for evidence of ambiguity.** A sharp image can be formed by thin,
   stretched, overlapping primitives. Fly behind a reconstructed surface. Explain
   why a convincing front view does not guarantee complete geometry or occupancy.

For a less cluttered inspection, first double-click the region of interest,
then enable **Only near the orbit focus** and adjust the radius. If no ellipsoids
remain, enlarge the radius or turn the filter off. On million-splat scenes this
filters a deterministic pool of up to 50,000 primitives, not every primitive in
that region. It is a teaching inspection tool, not an exhaustive local query.

## Connect what you see to the math

The same primitive has three views:

- **Center:** a point at μ. It hides scale, rotation, opacity, and image coverage.
- **Ellipsoid:** the contour `(x−μ)ᵀΣ⁻¹(x−μ) = k²`, with semi-axis lengths `k s`.
  The underlying function has value `exp(−k²/2)` relative to its peak there.
  These are shape contours, not uncertainty bounds; a 3D 1σ ellipsoid does not
  enclose the one-dimensional 68% probability mass.
- **Rendered scene:** project the covariance with the camera Jacobian, evaluate
  image footprints, and composite by depth. The opaque diagnostic ellipsoids are
  not what the splat renderer draws.

The inspector uses the loaded model's decoded centers, rotations, scales, colors,
and opacity. Spark packs/quantizes some parameters; this is not a full-precision
checkpoint browser. Python quaternions use **w,x,y,z**, whereas Three.js uses
**x,y,z,w**. See the explicit reorder in `viewer/app.js`. The street is Y-up;
the other examples are rotated 180° around X for display. Inspector coordinates
remain in the source scene frame, and scene units are not necessarily meters.

## Run or reproduce

In this workspace the dependencies, downloaded real models, and trained synthetic
results are already present. Start the server from the repository root:

```powershell
.\.venv\Scripts\python.exe -m splatlab.serve_viewer
```

Open `http://127.0.0.1:8765/?scene=room` in a browser with WebGL2 enabled. Keep the
terminal running; Ctrl+C stops it. The server binds only to this computer.

For a fresh checkout, first install the Python requirements using the README,
then install the pinned browser dependencies and prepare data:

```powershell
npm --prefix viewer ci
.\.venv\Scripts\python.exe -m splatlab.demo
.\.venv\Scripts\python.exe -m splatlab.prepare_viewer --download
.\.venv\Scripts\python.exe -m splatlab.fit_street
.\.venv\Scripts\python.exe -m splatlab.serve_viewer
```

The two real downloads total about 84 MB. The browser loads dependencies locally;
it does not need a CDN once setup is complete. `prepare_viewer` exports the nine-
Gaussian checkpoint and the street target. `fit_street` creates the street's
Before and Learned files. Refresh the browser after rerunning training.

The complete street trainer optimizes positions, log scales, rotations, opacity
logits, and RGB logits. Defaults are 350 steps, six 48×48 training images, and
six held-out camera angles. One checked CPU run took about six minutes. It improved
held-out PSNR from **22.89 to 36.34 dB**; training PSNR reached **44.56 dB**.
Those numbers describe this controlled synthetic experiment only.

Metrics and a checkpoint are in `outputs/street/`. The comparison image is rendered
at **96×96** for inspection; reported metrics use the **48×48** experiment images.
The browser uses its viewport resolution and the efficient
[Spark renderer](https://github.com/sparkjsdev/spark), so filtering, packing, and
display behavior need not match the PyTorch rasterizer pixel for pixel. Compute
quantitative metrics with the Python renderer, not screenshots of the viewer.

After completing the exercises, use your own core in either experiment:

```powershell
.\.venv\Scripts\python.exe -m pytest --student -q
.\.venv\Scripts\python.exe -m splatlab.demo --student --output outputs/student
.\.venv\Scripts\python.exe -m splatlab.fit_street --student
```

The last command replaces the street checkpoint and viewer stages with your run.
Until the exercises are filled, student commands intentionally raise
`NotImplementedError`. The complete version is always in `splatlab/core.py`.

## Read the added code in this order

| File | Learning purpose |
|---|---|
| `splatlab/street.py` | Construct anisotropic primitives on recognizable surfaces; Y-up camera geometry |
| `splatlab/fit_street.py` | Fit multiple images with the same four core functions; hold out cameras |
| `splatlab/export.py` | Turn optimization parameters into centers, scales, rotation, color, and opacity |
| `viewer/inspector.js` | Draw actual 3D covariance contours and inspect their parameters |
| `viewer/app.js` | Load models and keep rendering and diagnostics in a shared frame |
| `viewer/navigation.js` | Move a camera; independent of Gaussian mathematics |

Read the Python trainer before the viewer plumbing. Spark supplies efficient
rendering of large models; rewriting that library is not part of the exercises.

## Real-model provenance

The files are public pretrained examples distributed by
[cakewalk/splat-data](https://huggingface.co/cakewalk/splat-data). The manifest
`viewer/data/sources.json` records their URLs, sizes, and SHA-256 hashes. The source
card says its assets have varying licenses and does not identify a per-file license
for these samples. Their rights are separate from Spark's MIT license; check the
original asset terms before redistribution. This kit keeps the large downloads
out of version control and provides the retrieval script for local study.

These compact `.splat` files store position, scale, quaternion, opacity, and
constant RGB. They do not contain the original training images, cameras, or full
view-dependent SH appearance. We do not infer reconstruction accuracy or dataset
provenance beyond what the distributor establishes.
