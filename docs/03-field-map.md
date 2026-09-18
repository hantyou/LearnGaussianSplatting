# 3. How the technology developed

This is a selective reading map, checked against primary sources on **18 September
2026**, not an exhaustive survey or a claim about which method is currently best.
The most useful organizing question is: **which limitation is a method changing?**

## The timeline in one table

| Period | Change in emphasis | Representative primary sources |
|---|---|---|
| Before 2023 | Point/surfel rendering, Gaussian filtering, and differentiable radiance fields provide ingredients | [NeRF, 2020](https://www.matthewtancik.com/nerf); related work in the [3DGS paper](https://arxiv.org/abs/2308.04079) |
| 2023 | Optimize an explicit scene and render it efficiently with anisotropic splats | [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) |
| 2024 | Fix image-quality and geometry weaknesses; support motion, SLAM, semantics, and prediction from sparse views | [Mip-Splatting](https://github.com/autonomousvision/mip-splatting), [2DGS](https://arxiv.org/abs/2403.17888), [Street Gaussians](https://arxiv.org/abs/2401.01339), [SplaTAM](https://spla-tam.github.io/), [GaussianFormer](https://arxiv.org/abs/2405.17429), [MVSplat](https://donydchen.github.io/mvsplat/) |
| 2025 | Broaden camera/sensor models; use Gaussians for sensor simulation and fast inference | [3DGUT](https://research.nvidia.com/labs/toronto-ai/3DGUT/res/3DGUT_ready_main.pdf), [SplatAD](https://arxiv.org/abs/2411.16816), [RadarSplat](https://arxiv.org/abs/2506.01379), [SHARP, Dec. 2025](https://machinelearning.apple.com/research/sharp-monocular-view) |
| 2026, checked through September | More integration into sensor/perception pipelines and rendering libraries | [GaussianCaR, ICRA 2026](https://arxiv.org/abs/2602.08784); the dated development log in [gsplat](https://github.com/nerfstudio-project/gsplat) |

Publication and first preprint dates differ: for example, SplatAD appeared on
arXiv in 2024 and at CVPR in 2025. The table uses the stated publication period
where relevant. Library `main` can contain features not in a released package.

## Seven branches worth recognizing

**1. Image formation and sampling.** Tiny splats can alias, and changes in image
resolution or viewing distance can reveal artifacts. Mip-Splatting introduces
3D smoothing and a 2D Mip filter. The conceptual lesson is that evaluating a smooth
function at a pixel center is not identical to integrating over a pixel's area.
[Paper](https://arxiv.org/abs/2311.16493).

**2. Geometry.** Rendering quality and surface accuracy are different objectives.
2DGS represents oriented 2D Gaussian surfels embedded in 3D and adds geometric
constraints. “2D” here does not mean fitting independent blobs to a flat image.
[Paper and official code](https://github.com/hbb1/2d-gaussian-splatting).

**3. Dynamics.** A static set cannot represent a moving car consistently across
time. Approaches attach Gaussians to tracked objects, deform a canonical scene,
or use time-dependent representations. Street Gaussians is a concrete driving
example of separating background from moving vehicle representations.
[Paper](https://arxiv.org/abs/2401.01339).

**4. Mapping and pose estimation.** Reconstruction with known camera poses is
easier than estimating both the map and the poses. SplaTAM uses RGB-D data for
tracking and mapping; Splat-SLAM investigates globally optimized RGB-only SLAM.
Ask what is assumed about depth, pose initialization, and loop closure before
comparing a SLAM system with a novel-view renderer.
[SplaTAM](https://spla-tam.github.io/), [Splat-SLAM](https://github.com/google-research/Splat-SLAM).

**5. Predict instead of optimize each scene from scratch.** pixelSplat predicts
Gaussians from image pairs; MVSplat uses sparse multiview inputs. SHARP predicts
a Gaussian representation from one image for nearby-view synthesis. Such systems
learn cross-scene priors: the ambiguous parts of a new scene are filled partly
by learned assumptions. That differs from the per-scene optimization in our lab.
[pixelSplat](https://github.com/dcharatan/pixelsplat), [MVSplat](https://donydchen.github.io/mvsplat/),
[SHARP](https://machinelearning.apple.com/research/sharp-monocular-view).

**6. Features and semantics.** A Gaussian can carry a feature vector or class
information instead of RGB. GaussianFormer predicts a sparse semantic Gaussian
representation for occupancy; GaussianCaR uses splatting to fuse camera/radar
features in BEV. A renderer and a perception model may share geometric operations
while solving different learning problems.
[GaussianFormer](https://arxiv.org/abs/2405.17429), [GaussianCaR](https://arxiv.org/abs/2602.08784).

**7. Camera models, sensor physics, and systems.** 3DGUT uses the unscented transform
to accommodate more complex projection behavior; the 3DGRUT repository also
provides Gaussian ray tracing. SplatAD adds camera/LiDAR simulation, while
RadarSplat changes image formation for radar. gsplat supplies reusable optimized
building blocks, with its 2026 development log including additional sensor and
rendering support. Check the specific release before relying on an API.
[3DGRUT](https://github.com/nv-tlabs/3dgrut), [SplatAD](https://github.com/carlinds/splatad),
[RadarSplat](https://github.com/umautobots/radarsplat), [gsplat](https://github.com/nerfstudio-project/gsplat).

Compression, large-scene partitioning, level of detail, editing, and relighting
are also useful questions to investigate. They are outside this first course's
implementation. A practical checklist is whether a representation fits memory,
can stream the visible region, remains temporally stable, and behaves correctly
when the viewpoint or illumination changes.

## How to read a new paper without getting lost

Write down five answers before reading implementation details:

1. **Input:** posed RGB, unposed RGB, depth, LiDAR, radar images, or detections?
2. **Output:** RGB views, geometry, sensor returns, features, or occupancy?
3. **Optimization:** one scene at a time, a pretrained predictor, or both?
4. **Changed component:** representation, projection, blending, loss, capacity, or runtime?
5. **Evidence:** what benchmark, hardware, resolution, and held-out protocol?

My recommended order for your interests is original 3DGS → this lab →
Street Gaussians → SplatAD → RadarSplat → GaussianCaR. Read GaussianFormer when
you want the occupancy/perception branch. Mip-Splatting and 3DGUT are especially
useful after the projection approximation feels concrete.
