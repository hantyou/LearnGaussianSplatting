# COLMAP photo-preparation integration

This directory is intentionally outside `splatlab/`.

`splatlab/` is the self-contained, readable Gaussian-splatting implementation:
its teaching experiments and renderer do not invoke, import, install, or require
COLMAP.  Keep it usable with synthetic or already-calibrated cameras.

This integration is the separate *photo calibration* path.  It calls the COLMAP
executable and writes only COLMAP's public on-disk interface:

```text
<output>/database.db
<output>/sparse/0/{cameras.bin,images.bin,points3D.bin}
```

Any real-photo Gaussian trainer belongs in its own integration/trainer directory
and may consume those files.  Reusable readers for the COLMAP binary format are
fine, but a training script must not make `splatlab` depend on this directory or
on the `colmap` executable.

`train_own_renderer.py` is the deliberately separate example of that rule.  It
uses reusable rendering classes from `splatlab`, but it lives here because its
data contract is COLMAP calibration.  Run its public-Garden example with:

```bash
conda activate gslab-rocm
python -m integrations.colmap.train_own_renderer --download --device cuda
```

It is not imported by `splatlab` and it does not change the self-contained
synthetic training scripts.

## Run a sparse reconstruction

Activate the existing ROCm environment, then run:

```bash
conda activate gslab-rocm
bash integrations/colmap/reconstruct.sh /absolute/path/to/garden-photos \
  /absolute/path/to/garden-colmap
```

This workspace has `colmap 3.8` installed in `gslab-rocm` from conda-forge.  It
is a CPU build (`colmap help` reports `without CUDA`), which is intentional: the
same environment keeps ROCm PyTorch available for a later GPU training step
without pretending that this CPU COLMAP package can use the AMD GPU.

The script intentionally runs CPU SIFT extraction and matching (`use_gpu=0`).
This is portable to the Radeon 780M and is the reliable setting for the CPU
COLMAP package installed in this environment.  It performs sparse SfM only,
which is the calibration/point-cloud input a Gaussian trainer needs.  It does
not train Gaussians and does not edit `splatlab/`.

The output directory must be new or empty.  That guard prevents a fresh image
capture from accidentally overwriting a previous reconstruction.

For a garden, use a stationary scene with substantial overlap between adjacent
images.  Start with 150--400 images; wind-driven leaves, changing exposure, and
moving people reduce geometric consistency.
