# 10. From local photographs to a Gaussian reconstruction

This is the end-to-end path that the compact synthetic exercises deliberately
avoid:

```text
JPG photographs → COLMAP features/matches/SfM → sparse points + camera poses
                 → splatlab Gaussian initialization → our differentiable renderer
                 → optimized Gaussian parameters + novel-view comparison
```

`splatlab.garden_from_photos` uses the project's `GaussianScene` and
`splatlab.render.render` for the optimization. It does not read a `.splat`
checkpoint, use a pretrained neural model, or outsource rendering to a Gaussian
Splatting library. COLMAP is used only for classical camera calibration and a
sparse starting point cloud.

## Repeat the checked Garden run

The Garden images are public Mip-NeRF 360 photographs. Download them once:

```bash
GPU_PY=/home/pzhai/anaconda3/envs/gslab-rocm/bin/python
env HSA_OVERRIDE_GFX_VERSION=11.0.0 "$GPU_PY" -m splatlab.garden_from_photos --download --steps 1
```

The `--steps 1` command only obtains the input files and confirms that they can
be read. Then produce a *new local* COLMAP reconstruction from all 185 JPGs:

```bash
COLMAP=/home/pzhai/anaconda3/envs/gslab-rocm/bin/colmap
OUT=outputs/garden-colmap-local
mkdir -p "$OUT/sparse"
"$COLMAP" feature_extractor --database_path "$OUT/database.db" \
  --image_path data/mipnerf360/garden/images_8 --ImageReader.single_camera 1 \
  --SiftExtraction.use_gpu 0 --SiftExtraction.num_threads 8
"$COLMAP" sequential_matcher --database_path "$OUT/database.db" \
  --SiftMatching.use_gpu 0 --SiftMatching.num_threads 8 --SequentialMatching.overlap 10
"$COLMAP" mapper --database_path "$OUT/database.db" \
  --image_path data/mipnerf360/garden/images_8 --output_path "$OUT/sparse"
```

Finally train the lab renderer from that local model. This is the exact
configuration checked on the AMD GPU in this workspace:

```bash
env HSA_OVERRIDE_GFX_VERSION=11.0.0 AMD_SERIALIZE_KERNEL=3 "$GPU_PY" \
  -m splatlab.garden_from_photos --device cuda \
  --colmap-model outputs/garden-colmap-local/sparse/0 \
  --gaussians 1000 --width 96 --steps 400 --train-views 18 --holdout-views 3 \
  --output outputs/garden-from-local-photos
```

It writes these files:

| Output | Meaning |
|---|---|
| `comparison.png` | two held-out original photographs beside rendered reconstructions |
| `metrics.json` | input/image counts, GPU/backend, timing, and train/held-out PSNR |
| `learned_scene.npz` | learned centers, scales, rotations, colors, and opacities |
| `outputs/garden-colmap-local/sparse/0/` | the locally generated calibration and sparse point cloud |

The checked baseline processed all 185 photos in COLMAP, then trained from 18
spaced camera views and held out three. At 1,000 fixed Gaussians and 96-pixel
wide images it forms the table region only coarsely; that is intentional enough
to make the whole computation fit the readable dense renderer. Increase
`--train-views`, `--width`, `--gaussians`, and `--steps` gradually. The renderer
has dense `O(NHW)` footprint tensors, so thousands of Gaussians at larger image
sizes already become substantially slower.

For larger counts, add `--tile-size 32`. The tiled path projects the full scene
once, then renders only Gaussians whose three-standard-deviation screen-space
bounds overlap each 32×32 tile, while retaining their original depth order. This
keeps the peak footprint tensor bounded by one tile rather than a full image.
The photo initializer also reduces initial scale and opacity by `N^(-1/3)` as
the requested Gaussian count grows, preventing a denser point cloud from
starting as an opaque sheet.

## Use your own photos

Take overlapping photos while walking around a mostly static subject. Avoid
moving people, shiny reflections, strong auto-exposure changes, and large blank
walls. Place the JPGs in a directory, run the same three COLMAP commands with
that directory as `--image_path`, then train with both locations:

```bash
env HSA_OVERRIDE_GFX_VERSION=11.0.0 AMD_SERIALIZE_KERNEL=3 "$GPU_PY" \
  -m splatlab.garden_from_photos --device cuda \
  --image-dir /absolute/path/to/my-jpgs \
  --colmap-model /absolute/path/to/my-colmap/sparse/0 \
  --output outputs/my-reconstruction
```

The image names must remain the same as the names COLMAP records (normally the
original file names).

For a production-quality result, the next algorithmic extension is density
control: periodically clone/split high-gradient Gaussians and prune transparent
or oversized ones. That changes only the scene-management loop; the covariance,
projection, compositing, and image-loss core remain the code in this lab.
