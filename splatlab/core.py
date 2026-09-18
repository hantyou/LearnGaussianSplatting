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
                      t: Tensor, K: Tensor, pixel_variance: float = 0.3):
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


def gaussian_alpha(uv: Tensor, cov2: Tensor, opacities: Tensor,
                   height: int, width: int) -> Tensor:
    """Evaluate N elliptical footprints at all pixels -> [N,H,W] alpha.

    G = exp(-0.5 * delta^T @ inverse(cov2) @ delta); alpha = opacity * G.
    There is NO probability-density normalization factor. G peaks at one.
    Pixel centers use integer coordinates in this laboratory.
    """
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=uv.dtype, device=uv.device),
        torch.arange(width, dtype=uv.dtype, device=uv.device), indexing="ij")
    pixels = torch.stack((xs, ys), dim=-1)  # [H,W,2]: coordinate order is x,y.
    delta = pixels[None] - uv[:, None, None, :]
    inverse = torch.linalg.inv(cov2)  # Tiny 2x2 SPD matrices; explicit for teaching.
    distance_squared = torch.einsum("nhwi,nij,nhwj->nhw", delta, inverse, delta)
    return (opacities[:, None, None] * torch.exp(-0.5*distance_squared)).clamp(max=0.99)


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
