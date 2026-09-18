"""A recognizable 3D scene made of Gaussians, in a Y-up world.

The street is a procedural target, not a scan. fit_street.py fits perturbed
Gaussians to images of it using the SAME four equations as the first lesson.
"""
import math
import numpy as np
import torch
from .camera import Camera
from .scene import GaussianScene


def make_street():
    means, scales, colors = [], [], []

    def add(p, s, c):
        means.append(p); scales.append(s); colors.append(c)

    def box(center, size, color, spacing=0.22):
        # Sample each box surface with flat, axis-aligned Gaussian ellipsoids.
        for normal in range(3):
            tangent = [axis for axis in range(3) if axis != normal]
            for sign in [-1, 1]:
                ranges = [np.linspace(-size[a]/2, size[a]/2,
                                      max(2, int(math.ceil(size[a]/spacing))+1)) for a in tangent]
                for u in ranges[0]:
                    for v in ranges[1]:
                        p = list(center); p[normal] += sign*size[normal]/2
                        p[tangent[0]] += u; p[tangent[1]] += v
                        s = [spacing*0.65]*3; s[normal] = 0.035
                        shade = 0.78 + 0.10*normal
                        add(p, s, [value*shade for value in color])

    # Road, pavement, and broken center line.
    for x in np.linspace(-1.8, 1.8, 15):
        for z in np.linspace(-2.1, 2.1, 18):
            pavement = abs(x) > 0.85
            c = [0.52, 0.55, 0.57] if pavement else [0.16, 0.19, 0.23]
            add([x, 0, z], [0.18, 0.025, 0.16], c)
    for z in [-1.7, -0.85, 0., 0.85, 1.7]:
        add([0, 0.035, z], [0.045, 0.012, 0.21], [0.97, 0.82, 0.28])
    for x, z, color in [(-0.45, -0.65, [0.94, 0.19, 0.12]),
                         (0.46, 0.65, [0.10, 0.60, 0.92])]:
        box([x, 0.23, z], [0.51, 0.27, 0.88], color)
        box([x, 0.45, z+0.03], [0.43, 0.22, 0.46], [0.22, 0.40, 0.49])
        for dx in [-0.27, 0.27]:
            for dz in [-0.27, 0.27]:
                add([x+dx, 0.14, z+dz], [0.055, 0.13, 0.13], [0.055]*3)
        for dx in [-0.15, 0.15]:
            add([x+dx, 0.25, z-0.46], [0.07, 0.05, 0.025], [0.95, 0.91, 0.64])
    box([-1.40, 0.53, 1.0], [0.62, 1.02, 0.93], [0.83, 0.61, 0.35], spacing=0.3)
    for y in [0.33, 0.72]:
        for z in [0.72, 1.21]:
            add([-1.07, y, z], [0.022, 0.11, 0.13], [0.13, 0.40, 0.55])
    rng = np.random.default_rng(11)
    for x,z in [(1.30,-1.1),(1.35,1.2),(-1.38,-1.28)]:
        add([x, .32, z], [.08,.32,.08], [.34,.23,.12])
        for _ in range(13):
            p = rng.normal(size=3); p /= np.linalg.norm(p)
            p = np.array([x,.83,z]) + p * [.30,.29,.30]
            add(p.tolist(), [.20,.20,.20], [0.12, float(rng.uniform(.38,.61)),.20])
    n = len(means)
    quaternions = torch.zeros(n,4); quaternions[:,0] = 1
    return GaussianScene(torch.tensor(means, dtype=torch.float32),
                         torch.tensor(scales, dtype=torch.float32), quaternions,
                         torch.tensor(colors, dtype=torch.float32), torch.full((n,),.84))


def street_camera(angle_degrees, size=48):
    angle = math.radians(angle_degrees)
    center = torch.tensor([4.6*math.sin(angle), 3.1, 4.6*math.cos(angle)])
    forward = torch.nn.functional.normalize(torch.tensor([0.,.35,0.])-center, dim=0)
    right = torch.nn.functional.normalize(torch.linalg.cross(forward, torch.tensor([0.,1.,0.])), dim=0)
    down = torch.linalg.cross(forward, right)
    R = torch.stack((right, down, forward))
    K = torch.tensor([[size*1.0,0.,(size-1)/2],[0.,size*1.0,(size-1)/2],[0.,0.,1.]])
    return Camera(R,-R@center,K,size,size)
