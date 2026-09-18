# Learn Gaussian Splatting

A small course for someone comfortable with linear algebra, Bayesian inference,
and training a PyTorch model, but new to 3D vision. The goal is to understand and
implement the rendering equations before dealing with CUDA or a driving dataset.

**Start by [exploring the real Room scene](http://127.0.0.1:8765/?scene=room)**
and reading [the ten-minute explanation](docs/01-intuition.md). The local viewer
includes two real reconstructions, a trained miniature street, fly-through
navigation, and inspection of individual 3D Gaussian ellipsoids.
[Viewer setup and guided experiments](docs/07-viewer.md) explain how to start it.

The **complete reference implementation** is in [core.py](splatlab/core.py);
the separate [student version](exercises/core.py) leaves four functions for you.
Connect both to [the mathematical walkthrough](docs/02-math.md).

## What you will learn

1. **[Intuition](docs/01-intuition.md):** what is learned, why splatting works,
   and the connection to Gaussian basis functions and sparse Bayesian learning.
2. **[Math](docs/02-math.md):** coordinates, covariance, perspective Jacobians,
   alpha compositing, gradients, spherical harmonics, and density control.
3. **[How the field developed](docs/03-field-map.md):** a selective, sourced map
   from the 2023 paper through developments checked on **18 September 2026**.
4. **[Driving, perception, and radar](docs/04-driving-radar.md):** reconstruction
   versus perception; radar rendering versus fusion; nuScenes, RADIATE, and VoD.
5. **[Exercises and experiments](docs/05-lab.md):** fill four core functions,
   test them, and investigate what images can and cannot constrain.
6. **[Reading existing repositories](docs/06-reading-code.md):** where to look
   in the original implementation and gsplat after this lab.
7. **[Navigate and inspect whole scenes](docs/07-viewer.md):** two real captures,
   a 755-Gaussian reconstruction experiment, and 3D shape inspection.
8. **[CPU, AMD ROCm, and NVIDIA CUDA](docs/08-hardware.md):** one `--device` flag,
   one environment per backend, and why AMD GPUs also answer to `--device cuda`.
9. **[Agent handoff](docs/09-agent-handoff.md):** exact current state, verified
   real-scene results, boundaries, dirty worktree, and next-machine checklist.

## Run it

Python 3.10 or later. The experiments need no external data and run on the CPU by
default; a GPU is optional. The browser viewer uses WebGL2; the real scenes are
downloaded pretrained models. On this workspace a CPU-only `.venv` has already
been prepared; the tested commands are:

```powershell
.\.venv\Scripts\python.exe -m splatlab.demo
.\.venv\Scripts\python.exe -m pytest -q
```

To run on a GPU instead, add `--device cuda` from an environment built with the
matching wheels — see [the hardware guide](docs/08-hardware.md), which also
explains why an AMD ROCm GPU is selected with `cuda` (a PyTorch API name) while
the metrics still report its backend as `rocm`. `python -m splatlab.device`
prints what the current machine supports.

To exercise public real-scene splats through the CPU renderer at a manageable
size, run:

```powershell
.\.venv\Scripts\python.exe -m splatlab.real_scene_smoke --download `
  --gaussians 1024 --size 64 --steps 12
```

This uses low-resolution diagnostic renders and a 1,024-Gaussian subset of the
downloaded `room` and `train` checkpoints. It checks real-scene decoding,
finite rendering, gradients, and short parameter optimization; it is not a
full reconstruction benchmark because the compact checkpoints do not include
the original images and camera calibration.

For a fresh installation on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m splatlab.demo
```

On Linux/macOS, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.
No activation is needed. The general requirements are intentionally broad;
`requirements-tested.txt` records the packages used for the checked run.

The experiment fits **nine anisotropic 3D Gaussians to five 48×48 images**. It
optimizes positions, scale, rotation, opacity, and RGB. Three cameras are held
out from optimization. Open these generated files:

| File under `outputs/reference/` | What to inspect |
|---|---|
| `comparison.png` | Target, before, and after for a training and an unseen camera |
| `orbit.gif` | Target and learned scene as the camera moves |
| `loss.png` | Training and unseen-view error |
| `metrics.json` | Exact settings, timing, PSNR, and loss history |
| `learned_scene.npz` | Learned raw parameter arrays; keys match `state_dict()` |

PSNR measures image agreement. It does **not** establish correct geometry.
The target images come from our own renderer, and initialization perturbs the
known target parameters. This deliberately easy experiment isolates the math;
it is not evidence of reconstruction performance on real photographs.

## Read the code in this order

```text
splatlab/core.py      four rendering equations, each with tensor shapes
splatlab/camera.py    world-to-camera coordinates and a pinhole camera
splatlab/scene.py     nn.Parameter: learn a scene directly, with no CNN/MLP
splatlab/render.py    cull, project, sort, evaluate, composite
splatlab/demo.py      image loss -> backward() -> Adam -> unseen-view evaluation
exercises/core.py    the same four equations, left for you to implement
tests/               numerical examples, gradients, and a learning check
```

For the student version:

```powershell
.\.venv\Scripts\python.exe -m pytest --student -q -k covariance
.\.venv\Scripts\python.exe -m pytest --student -q
.\.venv\Scripts\python.exe -m splatlab.demo --student --output outputs/student
```

Student checks are **supposed to fail with `NotImplementedError` initially**.
The normal tests use the complete reference implementation. Both paths use the
same renderer and optimizer, so filling the exercises produces a working system.

## How close is this to the original paper?

| Component | This lab | Full original system |
|---|---|---|
| Representation | 3D centers, rotated anisotropic covariance, opacity | Same essential parameters |
| Appearance | View-independent RGB | View-dependent spherical harmonics |
| Projection | Pinhole + local Jacobian | Same basic approximation, extra clipping/filter details |
| Occlusion | Center-depth sorting and alpha blending | Tile-based sorted GPU rasterization |
| Learning | PyTorch autograd + MSE | Specialized backward pass, L1 + structural image loss |
| Initialization | Perturbed known synthetic scene | Typically calibrated cameras and SfM points |
| Number of Gaussians | Fixed nine or 755 (street) | Adaptive cloning, splitting, and pruning |
| Complexity | O(NHW), dense footprint tensors | Tile culling, compact support, early termination |

The omitted parts are explained in the notes. This renderer will **not** scale
to a city or millions of Gaussians; increasing resolution and N quickly increases
memory use. The separate viewer uses Spark to explore the larger pretrained
models. There is no radar simulator, SLAM system, or production driving stack
hidden behind the demo.

Recommended first session: read chapter 1, run the demo, derive the 2×3 projection
Jacobian, and implement exercise 2. You already know enough optimization to focus
your effort on the geometry and image formation.
