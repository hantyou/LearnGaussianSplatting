"""Fixed pinhole cameras. See docs/02-math.md for coordinate conventions."""

from dataclasses import dataclass
import math
import torch


@dataclass
class Camera:
    R: torch.Tensor  # [3,3] world-to-camera rotation
    t: torch.Tensor  # [3] world-to-camera translation, NOT the camera center
    K: torch.Tensor  # [3,3] intrinsic matrix
    height: int = 256
    width: int = 256
    near: float = 0.1


def orbit_camera(angle_degrees: float, size: int = 48) -> Camera:
    """Orbit around world origin; camera +x right, +y down, +z forward.

    World +y is down too. Coordinates have arbitrary world units.
    Camera center C maps to the origin because t = -R @ C.
    """
    angle = math.radians(angle_degrees)
    center = torch.tensor([3.2*math.sin(angle), -0.25, -3.2*math.cos(angle)])
    forward = torch.nn.functional.normalize(-center, dim=0)
    world_down = torch.tensor([0., 1., 0.])
    right = torch.nn.functional.normalize(torch.linalg.cross(world_down, forward), dim=0)
    down = torch.linalg.cross(forward, right)
    R = torch.stack((right, down, forward))
    t = -R @ center
    focal = 1.1 * size
    K = torch.tensor([[focal, 0., (size-1)/2],
                      [0., focal, (size-1)/2], [0., 0., 1.]])
    return Camera(R, t, K, size, size)
