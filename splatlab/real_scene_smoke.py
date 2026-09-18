"""Run a small CPU smoke test on public real-scene Gaussian splats.

The downloaded ``.splat`` files are already trained scene representations.  This
command does not reconstruct the original photographs: it decodes a deterministic
small subset, renders it from diagnostic cameras, and then checks that the
differentiable renderer can improve a perturbed copy of the real parameters.

Run from the repository root::

    python -m splatlab.real_scene_smoke --download

The original captures contain roughly one million Gaussians each.  The default
test uses 2,048 splats and 96x96 CPU renders so it remains practical on a machine
without CUDA.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .camera import Camera
from .prepare_viewer import DATA, SAMPLES, download
from .render import render
from .scene import GaussianScene


ROOT = Path(__file__).resolve().parents[1]


def psnr(mse: float) -> float:
    return -10.0 * math.log10(max(float(mse), 1e-12))


def read_splat(path: Path) -> tuple[torch.Tensor, ...]:
    """Decode the 32-byte antimatter15 ``.splat`` records used by the viewer."""
    raw = np.fromfile(path, dtype=np.uint8)
    if len(raw) == 0 or len(raw) % 32:
        raise ValueError(f"{path} is not a complete 32-byte .splat stream")
    records = raw.reshape(-1, 32)

    def floats(start: int) -> np.ndarray:
        # ``tobytes`` avoids alignment/stride surprises on Windows.
        return np.frombuffer(records[:, start:start + 12].tobytes(),
                             dtype="<f4").reshape(-1, 3).copy()

    means = floats(0)
    scales = floats(12)
    colors = records[:, 24:27].astype(np.float32) / 255.0
    opacities = records[:, 27].astype(np.float32) / 255.0
    # The compact format stores normalized (w, x, y, z) in four bytes.
    quaternions = (records[:, 28:32].astype(np.float32) - 128.0) / 128.0
    tensors = tuple(torch.from_numpy(array) for array in
                    (means, scales, quaternions, colors, opacities))
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError(f"{path} contains non-finite decoded values")
    if (scales <= 0).any():
        raise ValueError(f"{path} contains non-positive decoded scales")
    return tensors


def choose_subset(tensors: tuple[torch.Tensor, ...], count: int, seed: int):
    """Choose a reproducible spatially broad subset without loading extra data."""
    total = len(tensors[0])
    if count >= total:
        return tensors
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator)[:count]
    return tuple(tensor[indices] for tensor in tensors)


def make_scene(tensors: tuple[torch.Tensor, ...]) -> GaussianScene:
    means, scales, quaternions, colors, opacities = tensors
    # The file stores already-activated scales, RGB, and alpha; GaussianScene
    # converts them to the learnable parameterization used by the renderer.
    return GaussianScene(means, scales, quaternions, colors, opacities)


def diagnostic_cameras(means: torch.Tensor, size: int) -> list[Camera]:
    """Build three stable look-at cameras around the decoded real scene."""
    target = means.median(dim=0).values
    distances = torch.linalg.vector_norm(means - target, dim=1)
    radius = float(torch.quantile(distances, 0.90).clamp_min(1.0))
    cameras = []
    world_down = torch.tensor([0.0, -1.0, 0.0])
    for angle in (0.0, 120.0, 240.0):
        radians = math.radians(angle)
        center = target + torch.tensor([
            1.8 * radius * math.sin(radians),
            0.25 * radius,
            1.8 * radius * math.cos(radians),
        ])
        forward = torch.nn.functional.normalize(target - center, dim=0)
        right = torch.nn.functional.normalize(torch.linalg.cross(world_down, forward), dim=0)
        down = torch.linalg.cross(forward, right)
        rotation = torch.stack((right, down, forward))
        translation = -rotation @ center
        focal = 0.82 * size
        intrinsics = torch.tensor([
            [focal, 0.0, (size - 1) / 2],
            [0.0, focal, (size - 1) / 2],
            [0.0, 0.0, 1.0],
        ])
        cameras.append(Camera(rotation, translation, intrinsics, size, size, near=0.01))
    return cameras


def perturb(scene: GaussianScene, seed: int) -> GaussianScene:
    generator = torch.Generator().manual_seed(seed)

    def noise(value: torch.Tensor, amount: float) -> torch.Tensor:
        return amount * torch.randn(value.shape, generator=generator, dtype=value.dtype)

    with torch.no_grad():
        return GaussianScene(
            scene.means + noise(scene.means, 0.015),
            (scene.log_scales + noise(scene.log_scales, 0.04)).exp(),
            scene.quaternions + noise(scene.quaternions, 0.025),
            (scene.color_logits + noise(scene.color_logits, 0.15)).sigmoid(),
            scene.opacities.clamp(0.02, 0.98),
        )


@torch.no_grad()
def evaluate(scene, cameras, targets, core):
    images = [render(scene, camera, core)[0] for camera in cameras]
    mse = torch.stack([(image - target).square().mean()
                       for image, target in zip(images, targets)]).mean().item()
    return images, mse


def save_comparison(path: Path, target, before, after, scene_name: str, size: int):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), constrained_layout=True)
    for axis, image, title in zip(axes, (target, before, after),
                                  ("Real splat subset", "Perturbed", "After CPU steps")):
        axis.imshow(image.detach().cpu().numpy(), vmin=0, vmax=1)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(f"{scene_name} · {size}×{size} · CPU differentiable-renderer smoke test")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_scene(name: str, args, core) -> dict:
    path = DATA / f"{name}.splat"
    expected_bytes = SAMPLES[name]
    if not path.exists() or path.stat().st_size != expected_bytes:
        download(name, expected_bytes)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    tensors = read_splat(path)
    original_count = len(tensors[0])
    # Avoid Python's process-randomized ``hash()`` so repeated runs choose the
    # same real-scene subset even when PYTHONHASHSEED changes.
    scene_seed = args.seed + sum((index + 1) * ord(char)
                                 for index, char in enumerate(name))
    subset = choose_subset(tensors, args.gaussians, scene_seed)
    reference = make_scene(subset).requires_grad_(False)
    cameras = diagnostic_cameras(reference.means, args.size)
    with torch.no_grad():
        targets = [render(reference, camera, core)[0] for camera in cameras]

    student = perturb(reference, args.seed + 17)
    before, initial_mse = evaluate(student, cameras, targets, core)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)
    start = time.perf_counter()
    first_grad_norm = None
    for step in range(args.steps):
        camera_index = step % len(cameras)
        optimizer.zero_grad()
        image, _ = render(student, cameras[camera_index], core)
        loss = (image - targets[camera_index]).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name}: non-finite loss at step {step + 1}")
        loss.backward()
        if first_grad_norm is None:
            first_grad_norm = float(torch.linalg.vector_norm(student.means.grad).item())
        optimizer.step()
    seconds = time.perf_counter() - start
    after, final_mse = evaluate(student, cameras, targets, core)

    output = args.output / name
    output.mkdir(parents=True, exist_ok=True)
    save_comparison(output / "comparison.png", targets[0], before[0], after[0], name, args.size)
    report = {
        "scene": name,
        "source": f"https://huggingface.co/cakewalk/splat-data/resolve/main/{name}.splat",
        "sha256": digest,
        "original_bytes": path.stat().st_size,
        "original_gaussians": original_count,
        "tested_gaussians": len(reference.means),
        "render_size": [args.size, args.size],
        "cameras": 3,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "training_seconds": seconds,
        "initial_mse": initial_mse,
        "final_mse": final_mse,
        "initial_psnr_db": psnr(initial_mse),
        "final_psnr_db": psnr(final_mse),
        "first_means_gradient_norm": first_grad_norm,
        "loss_decreased": final_mse < initial_mse,
        "device": "cpu",
        "note": "Real pretrained splat subset; this is not reconstruction from original photographs.",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{name}: {len(reference.means):,}/{original_count:,} Gaussians, "
          f"{args.size}x{args.size}, {args.steps} steps, "
          f"{psnr(initial_mse):.2f} -> {psnr(final_mse):.2f} dB "
          f"({seconds:.1f}s)", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("room", "train", "all"), default="all")
    parser.add_argument("--download", action="store_true",
                        help="download missing public checkpoints and verify their size")
    parser.add_argument("--gaussians", type=int, default=2048)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "real-smoke")
    args = parser.parse_args()
    if args.gaussians < 1 or args.size < 8 or args.steps < 1:
        parser.error("--gaussians, --size, and --steps must be positive")

    # The download flag is explicit for the user-facing command; the test also
    # repairs a missing checkpoint so a fresh checkout remains reproducible.
    names = ("room", "train") if args.scene == "all" else (args.scene,)
    if args.download:
        for name in names:
            download(name, SAMPLES[name])
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    from . import core
    reports = [run_scene(name, args, core) for name in names]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"Saved real-scene smoke-test results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
