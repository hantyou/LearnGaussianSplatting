"""The four equations of our renderer, written in ordinary PyTorch.

Read in order: covariance_3d -> project_gaussians -> gaussian_alpha -> composite.
All tensors use a leading Gaussian dimension N. Cameras look along positive Z.
This is teaching code, not a CUDA renderer or an exact paper reproduction.
"""

import torch
from torch import Tensor


def quaternion_to_matrix(quaternions: Tensor) -> Tensor:
    """[N,4] nonzero quaternions in (w,x,y,z) order -> [N,3,3] rotations.

    A quaternion is just a convenient four-number rotation parameterization.
    Normalization makes it a unit quaternion; q and -q describe the same rotation.
    """
    q = torch.nn.functional.normalize(quaternions, dim=-1)
    w, x, y, z = q.unbind(-1)
    entries = (
        1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
        2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y),
    )
    return torch.stack(entries, dim=-1).reshape(-1, 3, 3)


def covariance_3d(log_scales: Tensor, quaternions: Tensor) -> Tensor:
    """[N,3], [N,4] -> positive definite [N,3,3] spatial covariances.

    The scales are STANDARD DEVIATIONS, so covariance uses their SQUARES.
    Learning log(s) ensures positive s without constrained optimization.
    """
    rotation = quaternion_to_matrix(quaternions)
    variance = torch.exp(2 * log_scales)
    return rotation @ torch.diag_embed(variance) @ rotation.transpose(-1, -2)


def project_gaussians(means: Tensor, covariances: Tensor, R: Tensor,
                      t: Tensor, K: Tensor, pixel_variance: float = 0.1):
    """World -> camera -> pixels. Inputs must already be in front of camera.

    R [3,3], t [3] are WORLD-TO-CAMERA: x_camera = R @ x_world + t.
    K [3,3] is a zero-skew pinhole intrinsic matrix in pixel units.
    Returns uv [N,2], covariance_2d [N,2,2], camera_z [N].
    Perspective projection is nonlinear: the covariance uses a local Jacobian.
    pixel_variance is in pixels squared, not pixels or world units.
    """
    camera_means = means @ R.T + t  # Row storage implements column-vector R @ x.
    camera_cov = R @ covariances @ R.T
    x, y, z = camera_means.unbind(-1)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    uv = torch.stack((fx*x/z + cx, fy*y/z + cy), dim=-1)
    zero = torch.zeros_like(z)
    J = torch.stack((fx/z, zero, -fx*x/z.square(),
                     zero, fy/z, -fy*y/z.square()), dim=-1).reshape(-1, 2, 3)
    cov2 = J @ camera_cov @ J.transpose(-1, -2)
    cov2 = cov2 + pixel_variance * torch.eye(2, dtype=means.dtype, device=means.device)
    return uv, cov2, z


# Target size of the [n,H,W] block evaluated per pass. The [n,H,W,2] pixel
# offsets and the einsum workspace dwarf the alpha itself, so bound n, not N.
FOOTPRINT_CHUNK_ELEMENTS = 4_000_000


def gaussian_alpha(uv: Tensor, cov2: Tensor, opacities: Tensor,
                   height: int, width: int, chunk: int | None = None) -> Tensor:
    """Evaluate N elliptical footprints at all pixels -> [N,H,W] alpha.

    G = exp(-0.5 * delta^T @ inverse(cov2) @ delta); alpha = opacity * G.
    There is NO probability-density normalization factor. G peaks at one.
    Pixel centers use integer coordinates in this laboratory.

    Gaussians never interact here, so at most `chunk` of them are evaluated per
    pass (default: as many as keep the offsets near FOOTPRINT_CHUNK_ELEMENTS).
    Each slice does the identical arithmetic on its own rows, so chunking moves
    peak memory and nothing else. The returned [N,H,W] alpha is still built in
    full because composite() needs every sorted layer; what shrinks is the far
    larger scratch space around it.
    """
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=uv.dtype, device=uv.device),
        torch.arange(width, dtype=uv.dtype, device=uv.device), indexing="ij")
    pixels = torch.stack((xs, ys), dim=-1)  # [H,W,2]: coordinate order is x,y.
    inverse = torch.linalg.inv(cov2)  # Tiny 2x2 SPD matrices; explicit for teaching.
    if chunk is None:
        chunk = max(1, FOOTPRINT_CHUNK_ELEMENTS // max(1, height*width))
    parts = []
    for start in range(0, len(uv), chunk):
        stop = start + chunk
        delta = pixels[None] - uv[start:stop, None, None, :]
        distance_squared = torch.einsum("nhwi,nij,nhwj->nhw",
                                        delta, inverse[start:stop], delta)
        parts.append(opacities[start:stop, None, None] * torch.exp(-0.5*distance_squared))
    if not parts:  # Everything was culled; composite() still expects [0,H,W].
        return uv.new_zeros((0, height, width))
    return torch.cat(parts).clamp(max=0.99)


def composite(alpha: Tensor, colors: Tensor, background: Tensor):
    """Front-to-back alpha [N,H,W], colors [N,3], background [3].

    Inputs MUST be sorted near-to-far before calling. Returns RGB [H,W,3]
    and accumulated alpha [H,W]. The explicit loop makes occlusion visible.
    Never normalize the color weights: unused weight belongs to the background.
    """
    height, width = alpha.shape[1:]
    transmittance = torch.ones((height, width), dtype=alpha.dtype, device=alpha.device)
    image = torch.zeros((height, width, 3), dtype=alpha.dtype, device=alpha.device)
    for i in range(len(alpha)):
        weight = transmittance * alpha[i]
        image = image + weight[..., None] * colors[i]
        transmittance = transmittance * (1 - alpha[i])
    image = image + transmittance[..., None] * background
    return image, 1 - transmittance
