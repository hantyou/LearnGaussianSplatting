"""Wire the four core functions into a renderer; students replace those functions."""

import torch
from . import core as reference


def _render_tiled(uv, cov2, depth, opacities, colors, camera, background,
                  core, tile_size, sigma_cutoff):
    """Render tiles without ever allocating a full ``[N,H,W]`` alpha tensor.

    A tile keeps globally depth-sorted Gaussians whose three-sigma screen-space
    bounds overlap it. The ignored Gaussian tail is the standard finite-support
    approximation used to make practical splat rasterizers possible.
    """
    order = torch.argsort(depth, stable=True)
    extent = sigma_cutoff * torch.sqrt(
        torch.diagonal(cov2, dim1=-2, dim2=-1).clamp_min(0))
    rows, alpha_rows = [], []
    for top in range(0, camera.height, tile_size):
        bottom = min(camera.height, top + tile_size)
        images, alphas = [], []
        for left in range(0, camera.width, tile_size):
            right = min(camera.width, left + tile_size)
            touches = ((uv[:, 0] + extent[:, 0] >= left) & (uv[:, 0] - extent[:, 0] < right) &
                       (uv[:, 1] + extent[:, 1] >= top) & (uv[:, 1] - extent[:, 1] < bottom))
            indices = order[touches[order]]  # Keep global front-to-back ordering.
            height, width = bottom-top, right-left
            if len(indices):
                alpha = core.gaussian_alpha(uv[indices] - uv.new_tensor((left, top)),
                                            cov2[indices], opacities[indices], height, width)
                image, accumulated_alpha = core.composite(alpha, colors[indices], background)
            else:
                image = background.expand(height, width, 3)
                accumulated_alpha = background.new_zeros((height, width))
            images.append(image)
            alphas.append(accumulated_alpha)
        rows.append(torch.cat(images, dim=1))
        alpha_rows.append(torch.cat(alphas, dim=1))
    return torch.cat(rows, dim=0), torch.cat(alpha_rows, dim=0)


def render(scene, camera, core=reference, tile_size: int | None = None,
           sigma_cutoff: float = 3.0):
    """Render exactly (default) or with memory-bounded screen tiles.

    ``tile_size=None`` keeps the original dense teaching implementation.
    ``tile_size=32`` is a useful starting point for larger photo reconstructions.
    """
    if tile_size is not None and tile_size < 1:
        raise ValueError("tile_size must be positive or None")
    if sigma_cutoff <= 0:
        raise ValueError("sigma_cutoff must be positive")
    background = scene.means.new_tensor([0.035, 0.045, 0.065])
    # Cull BEFORE dividing by z. This discrete visibility decision has no gradient.
    z = (scene.means @ camera.R.T + camera.t)[:, 2]
    visible = z > camera.near
    cov3 = core.covariance_3d(scene.log_scales[visible], scene.quaternions[visible])
    uv, cov2, depth = core.project_gaussians(
        scene.means[visible], cov3, camera.R, camera.t, camera.K)
    colors, opacities = scene.colors[visible], scene.opacities[visible]
    if tile_size is not None:
        return _render_tiled(uv, cov2, depth, opacities, colors, camera, background,
                             core, tile_size, sigma_cutoff)
    # Center-depth order is an approximation; sorting itself is not differentiated.
    order = torch.argsort(depth, stable=True)
    alpha = core.gaussian_alpha(uv[order], cov2[order], opacities[order],
                                camera.height, camera.width)
    image, accumulated_alpha = core.composite(alpha, colors[order], background)
    return image, accumulated_alpha
