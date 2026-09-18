# 6. From the teaching implementation to research code

The code here is an independently written educational implementation of the
standard equations. I inspected the following primary implementations to check
parameterization, projection, blending, and density-control conventions. No
upstream repository is installed or required by this lab; no CUDA source has
been copied into it. Links to `main` were inspected on 18 September 2026 and can
change. Pin a commit before conducting a reproducibility experiment.

## First: the original system

[graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
is the official reference. Read these pieces by responsibility:

| Your lab | Upstream file or symbol | What to compare |
|---|---|---|
| `scene.py` | [`scene/gaussian_model.py`](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py), `setup_functions` | Exponential scales, sigmoid opacity, normalized quaternions |
| Fixed Gaussian count | Same file, `densify_and_clone`, `densify_and_split`, `densify_and_prune` | How capacity changes, and why optimizer state needs care |
| `project_gaussians` | [`cuda_rasterizer/forward.cu`](https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu), `computeCov2D` | Jacobian projection, clipping, covariance floor |
| `composite` | Same CUDA file, `renderCUDA` | Front-to-back weights, alpha cap, thresholds, early stopping |

Do not start with CUDA thread indexing. First identify our four equations inside
the implementation. Then ask which surrounding operations avoid unnecessary
work. Also distinguish the original 2023 algorithm from features added to the
repository since publication. Inspect upstream licenses before redistributing
their code or integrating it into another project.

## Next: gsplat

[nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) provides
an optimized rasterization library with Python bindings. Its
[PyTorch reference implementation](https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/_torch_impl.py)
is a useful bridge from this lab to batched production operations. Look for
`_persp_proj`; expect more batch dimensions and clipping rules than our version.
The [rasterization API documentation](https://docs.gsplat.studio/main/apis/rasterization.html)
explains accepted parameter shapes and rendering modes.

Read its COLMAP training example only after you understand the renderer. Camera
loading, image sampling, exposure correction, schedules, and densification can
otherwise obscure the small number of core geometric operations. Installation
has PyTorch/CUDA/platform constraints that our CPU lab deliberately avoids;
follow the current upstream instructions when you reach that stage.

## Then choose the branch that matches the question

| Your question | Starting repository |
|---|---|
| How do background and moving vehicles compose? | [Street Gaussians](https://github.com/zju3dv/street_gaussians) |
| What changes for LiDAR and camera simulation? | [SplatAD](https://github.com/carlinds/splatad) |
| What changes for scanning-radar image synthesis? | [RadarSplat](https://github.com/umautobots/radarsplat) |
| How do Gaussians represent semantic occupancy? | [GaussianFormer](https://github.com/huang-yh/GaussianFormer) |
| How do distorted cameras and ray tracing change rendering? | [3DGRUT](https://github.com/nv-tlabs/3dgrut) |

These are follow-on references, not promises that every project is easy to build
on your current Windows environment. No large driving dataset was downloaded
as part of preparing this course.

## Your useful stopping criterion

You are ready for one of those repositories when you can explain:

1. What every column of the Gaussian parameter table means.
2. Why projection requires a Jacobian and is approximate.
3. Why compositing depends on order and differs from a Gaussian mixture density.
4. How a photometric residual reaches a 3D center through backpropagation.
5. What the renderer can identify from multiple images, and what remains ambiguous.
6. Which measurement model would have to change for your chosen radar data.
