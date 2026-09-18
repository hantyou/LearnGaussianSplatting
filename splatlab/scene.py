"""A scene is a table of learnable numbers, not a neural network of layers."""

import torch
from torch import nn


class GaussianScene(nn.Module):
    def __init__(self, means, scales, quaternions, colors, opacities):
        super().__init__()
        # nn.Parameter registers each tensor with model.parameters()/Adam.
        self.means = nn.Parameter(means.clone())
        self.log_scales = nn.Parameter(scales.log())
        self.quaternions = nn.Parameter(quaternions.clone())
        self.color_logits = nn.Parameter(torch.logit(colors.clamp(0.001, 0.999)))
        self.opacity_logits = nn.Parameter(torch.logit(opacities.clamp(0.001, 0.999)))

    @property
    def colors(self):
        return self.color_logits.sigmoid()

    @property
    def opacities(self):
        return self.opacity_logits.sigmoid()


def make_teacher(device=None) -> GaussianScene:
    """Nine overlapping, oriented ellipsoids at different depths.

    We generate images from a known scene so no downloads/calibration are needed.
    The optimization objective uses only rendered training images; initialization
    perturbs teacher parameters. This is a controlled optimization lab, not reconstruction
    from arbitrary photographs. Teacher geometry is never an optimization loss.
    Built on the CPU and then moved, so every backend starts from the same bits.
    """
    means = torch.tensor([
        [-0.55, -0.40, 0.10], [0., -0.48, 0.24], [0.55, -0.32, 0.04],
        [-0.55, 0.08, -0.08], [0., 0.02, -0.42], [0.55, 0.12, 0.18],
        [-0.46, 0.53, 0.25], [0.10, 0.48, 0.12], [0.56, 0.54, -0.10]])
    scales = torch.tensor([[0.29, 0.12, 0.13], [0.17, 0.26, 0.10], [0.26, 0.13, 0.12],
                           [0.14, 0.24, 0.15], [0.25, 0.20, 0.13], [0.16, 0.27, 0.12],
                           [0.24, 0.15, 0.10], [0.28, 0.11, 0.15], [0.14, 0.21, 0.10]])
    colors = torch.tensor([[0.92, 0.27, 0.13], [0.97, 0.65, 0.10], [0.40, 0.75, 0.21],
                           [0.18, 0.62, 0.85], [0.90, 0.28, 0.58], [0.25, 0.82, 0.69],
                           [0.39, 0.37, 0.87], [0.95, 0.49, 0.17], [0.61, 0.30, 0.77]])
    angles = torch.linspace(-0.7, 0.8, len(means))
    quaternions = torch.stack((torch.cos(angles/2), torch.zeros_like(angles),
                               torch.zeros_like(angles), torch.sin(angles/2)), dim=1)
    scene = GaussianScene(means, scales, quaternions, colors, torch.full((len(means),), 0.85))
    return scene.to(device) if device is not None else scene


def make_student(teacher: GaussianScene, seed: int = 7, device=None) -> GaussianScene:
    """Perturb the teacher on the CPU, then move the result to `device`.

    The seeded CPU generator is deliberate: CUDA and ROCm draw a different random
    stream from the same seed, so drawing here keeps a GPU run comparable with the
    CPU run instead of merely similar.
    """
    generator = torch.Generator().manual_seed(seed)
    def noise(tensor, amount):
        return amount * torch.randn(tensor.shape, generator=generator)
    with torch.no_grad():
        cpu = {name: value.detach().cpu() for name, value in teacher.state_dict().items()}
        student = GaussianScene(
            cpu["means"] + noise(cpu["means"], 0.13),
            (cpu["log_scales"] + noise(cpu["log_scales"], 0.25)).exp(),
            cpu["quaternions"] + noise(cpu["quaternions"], 0.18),
            (cpu["color_logits"] + noise(cpu["color_logits"], 0.9)).sigmoid(),
            torch.full_like(cpu["opacity_logits"], 0.60))
    return student.to(device) if device is not None else student
