"""Wire the four core functions into a renderer; students replace those functions."""

import torch
from . import core as reference


def render(scene, camera, core=reference):
    background = scene.means.new_tensor([0.035, 0.045, 0.065])
    # Cull BEFORE dividing by z. This discrete visibility decision has no gradient.
    z = (scene.means @ camera.R.T + camera.t)[:, 2]
    visible = z > camera.near
    cov3 = core.covariance_3d(scene.log_scales[visible], scene.quaternions[visible])
    uv, cov2, depth = core.project_gaussians(
        scene.means[visible], cov3, camera.R, camera.t, camera.K)
    # Center-depth order is an approximation; sorting itself is not differentiated.
    order = torch.argsort(depth, stable=True)
    alpha = core.gaussian_alpha(uv[order], cov2[order], scene.opacities[visible][order],
                                camera.height, camera.width)
    image, accumulated_alpha = core.composite(alpha, scene.colors[visible][order], background)
    return image, accumulated_alpha
