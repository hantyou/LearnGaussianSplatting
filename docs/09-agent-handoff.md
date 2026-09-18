# Agent handoff: current state and boundaries

Last updated: **18 September 2026**
Workspace: `K:\code\GaussianSplattingLearn`
Latest committed change: **30b42fc Document agent handoff and benchmark limits**
Previous real-scene change: **23457a0 Add CPU smoke test for real Gaussian splats**

Read this file before changing the repository. The working tree is intentionally
dirty: there is a newer, uncommitted device/GPU portability effort in progress.
Preserve it until its owner decides whether to commit, revise, or discard it.

## One-minute status

The project is a readable PyTorch Gaussian-splatting teaching lab. The CPU path
works. It has:

- a complete differentiable renderer in `splatlab/core.py`;
- synthetic nine-Gaussian and 755-Gaussian learning experiments;
- a WebGL viewer with two downloaded real pretrained splat scenes;
- a committed CPU smoke test for real `.splat` data;
- a new, uncommitted device abstraction and GPU-port changes that have only been
  verified on CPU so far.

The most important limitation is that the real-scene smoke test uses pretrained
Gaussian parameters. It does **not** reconstruct photographs, because the compact
`.splat` files do not contain the original images, camera poses, or calibration.

## Git and worktree state

Committed history relevant to future work:

~~~
30b42fc Document agent handoff and benchmark limits
23457a0 Add CPU smoke test for real Gaussian splats
2780bc1 Initial commit
~~~

At handoff, the following changes are uncommitted:

~~~
 M .gitignore
 M README.md
 M splatlab/camera.py
 M splatlab/core.py
 M splatlab/demo.py
 M splatlab/export.py
 M splatlab/fit_street.py
 M splatlab/real_scene_smoke.py
 M splatlab/scene.py
 M splatlab/street.py
?? PLAN-linux-rocm.md
?? docs/08-hardware.md
?? splatlab/device.py
?? tests/test_device.py
~~~

The uncommitted changes currently do the following:

- add `splatlab/device.py` with `auto`, `cpu`, `cuda`, and `mps` selection;
- report `rocm` separately when PyTorch exposes an AMD GPU through its `cuda`
  API namespace;
- add `--device` to `splatlab.demo` and `splatlab.fit_street`;
- move cameras/scenes to the selected device while keeping seeded initialization
  on CPU for reproducible starting parameters;
- chunk the large footprint scratch tensors in `core.gaussian_alpha`;
- add CPU-safe GPU agreement/optimization tests in `tests/test_device.py`;
- add `docs/08-hardware.md` and `PLAN-linux-rocm.md` for future Linux/ROCm or
  NVIDIA work;
- change the local default of `real_scene_smoke.py` to 128 pixels. The verified
  real-scene run used explicit `--size 64`, so do not treat the new default as
  benchmarked.

Do not use `git reset --hard`, `git checkout --`, or broad cleanup commands here.
Those changes belong to the user and may be the beginning of the next task.

## Environments and data actually present

The existing Windows environment is:

~~~
.\\.venv\\Scripts\\python.exe
~~~

It is a CPU-only PyTorch environment. The downloaded real scene files currently
exist locally but are ignored by Git:

~~~
viewer/data/room.splat   50,988,032 bytes   1,593,376 Gaussians
viewer/data/train.splat  32,848,256 bytes   1,026,508 Gaussians
~~~

Their source and verified hashes are recorded in `viewer/data/sources.json`:

- Room: `b751ede3e3e58d0345de3b4a83abc948112301e72dd1e8a6543b3aef14a7c226`
- Train: `d371f544e9e6a0a10b4a9d75c48d66f3c47f484559c33883c33c97ad7d2aa6af`

On a fresh machine, reacquire them with:

~~~
.\\.venv\\Scripts\\python.exe -m splatlab.prepare_viewer --download
~~~

or let the smoke test repair a missing file with `--download`. Do not commit the
large `.splat` files. The source repository says its assets came from different
sources/licenses; check the terms before redistribution.

The browser dependencies may also be local under `viewer/node_modules/`, but
they are ignored and should be recreated with `npm --prefix viewer ci` when
needed.

## What is implemented

### Educational CPU renderer

- `splatlab/core.py`: quaternion covariance, perspective projection, 2D Gaussian
  footprint, and front-to-back alpha compositing.
- `splatlab/render.py`: camera-space culling, projection, center-depth sorting,
  footprint evaluation, and compositing.
- `splatlab/scene.py`: learnable means, log-scales, quaternions, RGB logits, and
  opacity logits.
- `splatlab/demo.py`: nine-Gaussian synthetic fitting with held-out views.
- `splatlab/fit_street.py`: 755-Gaussian synthetic miniature street fitting.
- `splatlab/prepare_viewer.py`: exports teaching scenes and downloads Room/Train
  `.splat` viewer assets.
- `splatlab/serve_viewer.py` plus `viewer/`: browser inspection of synthetic and
  real pretrained scenes.

### Real-scene CPU smoke test

`splatlab/real_scene_smoke.py` decodes the 32-byte-per-Gaussian `.splat` format,
selects a deterministic subset, creates diagnostic look-at cameras, renders
low-resolution views, perturbs the real parameters, and runs a short Adam
optimization. This tests real-scene decoding, covariance projection, alpha
compositing, autograd, and parameter updates.

It deliberately does **not** claim benchmark reconstruction quality. The cameras
are diagnostic cameras made from the point cloud, not the original capture poses.

The reproducible CPU command is:

~~~
.\\.venv\\Scripts\\python.exe -m splatlab.real_scene_smoke ^
  --download --gaussians 1024 --size 64 --steps 12 ^
  --learning-rate 0.003 --output outputs\\real-smoke
~~~

The dense teaching renderer is `O(N·H·W)`, so this is intentionally modest. A
larger default or full million-Gaussian run is not a sensible CPU benchmark.

## Verified results

The successful runs used 1,024 Gaussians, 64×64 renders, three diagnostic
cameras, 12 Adam steps, and learning rate `0.003`:

| Scene | Full scene | Initial PSNR | Final PSNR | Time |
|---|---:|---:|---:|---:|
| Room | 1,593,376 Gaussians | 39.02 dB | **45.62 dB** | 74.6 s |
| Train | 1,026,508 Gaussians | 29.57 dB | **41.43 dB** | 72.8 s |

Useful artifacts are outside Git and should be regenerated if absent:

- [Room metrics](../outputs/real-smoke-room-lr003/room/metrics.json)
- [Room comparison](../outputs/real-smoke-room-lr003/room/comparison.png)
- [Train metrics](../outputs/real-smoke-train-lr003/train/metrics.json)
- [Train comparison](../outputs/real-smoke-train-lr003/train/comparison.png)

The repository test command completed with:

~~~
30 passed, 3 skipped
~~~

The three skips are GPU-only tests in the uncommitted hardware-port work; the
current Windows machine has no usable GPU backend in its CPU environment.

## What we do not have

Do not describe the current repository as any of the following:

- a full 3DGS trainer for real photographs;
- a COLMAP/SfM pipeline or a calibrated multi-view dataset loader;
- a CPU-capable million-Gaussian renderer;
- an official Mip-NeRF 360, Tanks and Temples, or other raw benchmark evaluation;
- a geometry-accurate or novel-view-accurate real-scene metric;
- an implemented radar, SLAM, perception, or driving stack;
- a GPU-verified ROCm or CUDA implementation.

The real Room/Train assets are pretrained viewer checkpoints from
`cakewalk/splat-data`, not raw benchmark image sets. There are no original
training images, camera files, COLMAP database, view-dependent spherical
harmonics, adaptive cloning/splitting/pruning, tile-based rasterization, or
official ground-truth comparison metrics in this workspace.

## Future-machine startup checklist

Run these in order:

~~~
# 1. Inspect before touching anything.
git status --short
git diff --stat

# 2. Verify the installed backend.
python -m splatlab.device

# 3. Install/use a separate environment for the backend.
# CPU: .venv or .venv/bin/python
# AMD: .venv-rocm or .venv-rocm/bin/python
# NVIDIA: .venv-cuda or .venv-cuda/bin/python

# 4. Run the suite.
.venv/bin/python -m pytest -q

# 5. Reacquire ignored real data if needed.
.venv/bin/python -m splatlab.prepare_viewer --download

# 6. Run the honest CPU smoke test.
.venv/bin/python -m splatlab.real_scene_smoke --download \
  --gaussians 1024 --size 64 --steps 12 --learning-rate 0.003
~~~

On Windows, replace `.venv/bin/python` with
`\.venv\Scripts\python.exe`. On Linux AMD, read [the hardware guide](08-hardware.md)
and [the Linux ROCm plan](../PLAN-linux-rocm.md) before installing wheels. Never
install ROCm or CUDA wheels into the CPU `.venv`; native libraries are kept in
separate environments.

## Recommended next work

1. Decide whether the uncommitted device/GPU port should be committed as one
   change. First run `pytest` and inspect the CPU baseline; do not discard it.
2. On Linux, follow `PLAN-linux-rocm.md` and verify `rocminfo`, `/dev/kfd`, the
   detected `gfx` target, and the separate `.venv-rocm` before running GPU tests.
3. If the goal is real reconstruction, obtain a raw dataset with images and
   calibrated poses, or add a COLMAP/NeRF-format loader. Then define a small
   image-based experiment before attempting full scene training.
4. If the goal is scale, replace the dense educational rasterizer with a tiled
   renderer or use a production implementation such as gsplat. Chunking reduces
   scratch memory but does not remove the full `N×H×W` output/compositing cost.
5. Keep CPU as the reference backend and record `requested_device`, actual
   device, backend, PyTorch version, Gaussian count, resolution, and timing in
   every future benchmark.
