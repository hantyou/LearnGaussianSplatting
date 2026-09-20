"""Reconstruct the classic Mip-NeRF 360 Garden from its real photographs.

This is deliberately a small, readable end-to-end reconstruction run.  It downloads
the public ``images_8`` photographs plus COLMAP's camera poses and sparse point
cloud, initializes learnable Gaussians from that point cloud, then trains entirely
through :mod:`splatlab.core` and :mod:`splatlab.render`.

The supplied COLMAP files are the camera-calibration stage for this public photo
set.  Running feature extraction and bundle adjustment from scratch is intentionally
outside this lab, but no pretrained Gaussian model or rendered target is used.

Run from the repository root::

    python -m integrations.colmap.train_own_renderer --download --device cuda

The defaults are sized for a consumer integrated GPU.  They make a visibly
recognizable but not benchmark-quality reconstruction; increase ``--gaussians``,
``--width``, and ``--steps`` only after a successful first run.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
import struct
import time
from urllib.request import urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from splatlab.camera import Camera
from splatlab.device import CHOICES, describe_device, format_report, select_device, synchronize
from splatlab.export import export_scene
from splatlab.render import render
from splatlab.scene import GaussianScene


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "mipnerf360" / "garden"
OUTPUT = ROOT / "outputs" / "garden-from-photos"
REPO = "https://huggingface.co/datasets/mileleap/mipnerf360/resolve/main/garden"
API = "https://huggingface.co/api/datasets/mileleap/mipnerf360/tree/main/garden"

# Values are the count of double parameters after COLMAP's camera header.
CAMERA_PARAM_COUNTS = {
    0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12,
}


@dataclass(frozen=True)
class ColmapImage:
    name: str
    camera_id: int
    R: np.ndarray  # World -> COLMAP camera, where +x right, +y down, +z forward.
    t: np.ndarray


def _download(url: str, destination: Path) -> None:
    """Download one immutable public asset atomically and without extra packages."""
    if destination.exists() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with urlopen(url, timeout=120) as response, partial.open("wb") as file:
        while chunk := response.read(1024 * 1024):
            file.write(chunk)
    partial.replace(destination)


def download_garden() -> None:
    """Fetch 1/8-resolution photographs and COLMAP calibration, not a splat model."""
    for relative in ("sparse/0/cameras.bin", "sparse/0/images.bin", "sparse/0/points3D.bin"):
        _download(f"{REPO}/{relative}", DATA / relative)
    with urlopen(f"{API}/images_8?recursive=false&expand=false", timeout=60) as response:
        files = json.load(response)
    names = [entry["path"].rsplit("/", 1)[-1] for entry in files if entry["type"] == "file"]
    print(f"Downloading {len(names)} Garden photographs at 1/8 resolution…", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda name: _download(f"{REPO}/images_8/{name}", DATA / "images_8" / name), names))


def _read_exact(file, count: int) -> bytes:
    value = file.read(count)
    if len(value) != count:
        raise ValueError("Unexpected end of COLMAP binary file")
    return value


def read_cameras(path: Path) -> dict[int, tuple[int, int, np.ndarray]]:
    """Read enough of COLMAP's cameras.bin for a pinhole projection matrix."""
    cameras = {}
    with path.open("rb") as file:
        count, = struct.unpack("<Q", _read_exact(file, 8))
        for _ in range(count):
            camera_id, model_id, width, height = struct.unpack("<iiQQ", _read_exact(file, 24))
            params = np.array(struct.unpack(f"<{CAMERA_PARAM_COUNTS[model_id]}d",
                                            _read_exact(file, CAMERA_PARAM_COUNTS[model_id] * 8)))
            # Garden uses a single PINHOLE camera. Supporting simple pinhole makes
            # the reader useful for similarly exported public COLMAP sets too.
            if model_id == 0:
                fx = fy = params[0]; cx, cy = params[1:3]
            elif model_id == 1:
                fx, fy, cx, cy = params[:4]
            else:
                raise ValueError(f"Unsupported COLMAP camera model {model_id}; use PINHOLE images")
            cameras[camera_id] = (int(width), int(height), np.array([fx, fy, cx, cy]))
    return cameras


def quaternion_matrix(qvec: np.ndarray) -> np.ndarray:
    """COLMAP's w,x,y,z world-to-camera quaternion as a 3×3 rotation."""
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float32)


def read_images(path: Path) -> list[ColmapImage]:
    """Read image pose records while skipping the large 2D feature-track payload."""
    records = []
    with path.open("rb") as file:
        count, = struct.unpack("<Q", _read_exact(file, 8))
        for _ in range(count):
            image_id, = struct.unpack("<i", _read_exact(file, 4))
            qvec = np.array(struct.unpack("<4d", _read_exact(file, 32)))
            tvec = np.array(struct.unpack("<3d", _read_exact(file, 24)), dtype=np.float32)
            camera_id, = struct.unpack("<i", _read_exact(file, 4))
            name = bytearray()
            while (char := _read_exact(file, 1)) != b"\x00":
                name.extend(char)
            feature_count, = struct.unpack("<Q", _read_exact(file, 8))
            file.seek(feature_count * 24, 1)  # x, y, point3D_id
            records.append(ColmapImage(name.decode("utf-8"), camera_id, quaternion_matrix(qvec), tvec))
    return records


def read_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read COLMAP's sparse colored point cloud, our geometric initialization."""
    positions, colors = [], []
    with path.open("rb") as file:
        count, = struct.unpack("<Q", _read_exact(file, 8))
        for _ in range(count):
            _read_exact(file, 8)  # point id
            positions.append(struct.unpack("<3d", _read_exact(file, 24)))
            colors.append(struct.unpack("<3B", _read_exact(file, 3)))
            _read_exact(file, 8)  # reprojection error
            track_length, = struct.unpack("<Q", _read_exact(file, 8))
            file.seek(track_length * 8, 1)
    return np.asarray(positions, dtype=np.float32), np.asarray(colors, dtype=np.float32) / 255.0


def make_targets(records: list[ColmapImage], cameras: dict[int, tuple[int, int, np.ndarray]],
                 width: int, device: torch.device) -> list[tuple[str, Camera, torch.Tensor]]:
    """Turn real JPGs and COLMAP poses into the exact tensors consumed by our renderer."""
    samples = []
    for record in sorted(records, key=lambda item: item.name):
        path = DATA / "images_8" / record.name
        if not path.exists() or record.camera_id not in cameras:
            continue
        original_width, original_height, intrinsics = cameras[record.camera_id]
        with Image.open(path) as source:
            source = source.convert("RGB")
            # The downloaded images are already 1/8, but derive scale from their
            # actual dimensions to keep the calibration correct across sources.
            image_width, image_height = source.size
            height = round(image_height * width / image_width)
            image = source.resize((width, height), Image.Resampling.LANCZOS)
        scale_x, scale_y = image_width / original_width, image_height / original_height
        fx, fy, cx, cy = intrinsics * np.array([scale_x, scale_y, scale_x, scale_y])
        final_scale_x, final_scale_y = width / image_width, height / image_height
        K = torch.tensor([[fx * final_scale_x, 0.0, cx * final_scale_x],
                          [0.0, fy * final_scale_y, cy * final_scale_y],
                          [0.0, 0.0, 1.0]], dtype=torch.float32)
        camera = Camera(torch.from_numpy(record.R), torch.from_numpy(record.t), K, height, width, near=0.01).to(device)
        target = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).to(device)
        samples.append((record.name, camera, target))
    if not samples:
        raise RuntimeError(f"No images in {DATA / 'images_8'} matched the COLMAP records")
    return samples


def choose_views(samples, train_count: int, holdout_count: int):
    """Evenly distribute training cameras and interleave held-out real photographs."""
    wanted = min(len(samples), train_count + holdout_count)
    indices = np.linspace(0, len(samples) - 1, wanted, dtype=int)
    picked = [samples[index] for index in indices]
    train = picked[::2][:train_count]
    heldout = picked[1::2][:holdout_count]
    # Small requested counts need this fallback to remain a valid run.
    if len(train) < train_count:
        train = picked[:train_count]
        heldout = picked[train_count:train_count + holdout_count]
    return train, heldout


def image_stratified_indices(points: np.ndarray, samples, count: int, seed: int) -> np.ndarray:
    """Select sparse points evenly over the *photographs*, not just over XYZ space.

    A uniformly random 3D sample spends most of its budget on the distant hedge.
    Selecting one observed point per image cell gives the tabletop, patio, and
    foliage a useful initial density before the image loss changes any parameter.
    """
    rng = np.random.default_rng(seed)
    per_view = max(1, math.ceil(count / len(samples)))
    cells = max(1, math.ceil(math.sqrt(per_view)))
    selected = []
    for _, camera, _ in samples:
        R = camera.R.detach().cpu().numpy()
        t = camera.t.detach().cpu().numpy()
        K = camera.K.detach().cpu().numpy()
        projected = points @ R.T + t
        z = projected[:, 2]
        u = K[0, 0] * projected[:, 0] / np.maximum(z, 1e-8) + K[0, 2]
        v = K[1, 1] * projected[:, 1] / np.maximum(z, 1e-8) + K[1, 2]
        visible = (z > camera.near) & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
        point_indices = np.flatnonzero(visible)
        if not len(point_indices):
            continue
        bins = np.minimum(cells - 1, (v[point_indices] / camera.height * cells).astype(int)) * cells
        bins += np.minimum(cells - 1, (u[point_indices] / camera.width * cells).astype(int))
        for cell in range(cells * cells):
            candidates = point_indices[bins == cell]
            if len(candidates):
                selected.append(rng.choice(candidates))
    selected = np.unique(selected)
    if len(selected) >= count:
        return rng.choice(selected, size=count, replace=False)
    remaining = np.setdiff1d(np.arange(len(points)), selected, assume_unique=False)
    return np.concatenate((selected, rng.choice(remaining, size=count - len(selected), replace=False)))


def initialize_scene(points: np.ndarray, colors: np.ndarray, count: int, seed: int,
                     device: torch.device, samples):
    if len(points) < count:
        raise ValueError(f"COLMAP provides only {len(points)} points, fewer than --gaussians {count}")
    generator = np.random.default_rng(seed)
    indices = image_stratified_indices(points, samples, count, seed)
    chosen_points, chosen_colors = points[indices], colors[indices]
    low, high = np.quantile(points, [0.05, 0.95], axis=0)
    scene_radius = float(np.linalg.norm(high - low) / 2)
    # One scale value is intentionally simple: subsequent image loss changes all
    # three axes and rotations through the same core equations as the lessons.
    # The sparse cloud already has reliable geometry.  Start with small footprints
    # (about a few pixels at the 1/8-resolution captures) so the table and patio
    # are visible before optimization rather than being washed into one green blob.
    scales = np.full((count, 3), max(scene_radius * 0.0025, 1e-3), dtype=np.float32)
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (count, 1))
    scene = GaussianScene(torch.from_numpy(chosen_points), torch.from_numpy(scales),
                          torch.from_numpy(quaternions), torch.from_numpy(chosen_colors),
                          # Sparse COLMAP points are far fewer than a final 3DGS
                          # model; a moderately opaque starting cloud makes their
                          # measured RGB visible before opacity optimization.
                          torch.full((count,), 0.35)).to(device)
    return scene, scene_radius


@torch.no_grad()
def mean_mse(scene, samples):
    return torch.stack([(render(scene, camera)[0] - target).square().mean()
                        for _, camera, target in samples]).mean().item()


def psnr(mse: float) -> float:
    return -10 * math.log10(max(float(mse), 1e-12))


@torch.no_grad()
def save_preview(path: Path, scene: GaussianScene, samples) -> None:
    """Save real photo alongside the reconstruction for two held-out camera views."""
    shown = samples[:2] if len(samples) >= 2 else samples
    figure, axes = plt.subplots(len(shown), 2, figsize=(8, 3.5 * len(shown)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row, (name, camera, target) in enumerate(shown):
        prediction, _ = render(scene, camera)
        for column, (image, title) in enumerate(((target, "Original photograph"), (prediction, "Our Gaussian reconstruction"))):
            axes[row, column].imshow(image.detach().cpu().numpy().clip(0, 1))
            axes[row, column].set_title(title)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(name)
    figure.suptitle("Mip-NeRF 360 Garden · real photos → COLMAP points → our splat core")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="download public photos and COLMAP calibration if absent")
    parser.add_argument("--device", choices=CHOICES, default="cuda")
    parser.add_argument("--gaussians", type=int, default=3000)
    parser.add_argument("--width", type=int, default=128, help="training photograph width; aspect ratio is preserved")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--train-views", type=int, default=18)
    parser.add_argument("--holdout-views", type=int, default=3)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if min(args.gaussians, args.width, args.steps, args.train_views, args.holdout_views) < 1:
        parser.error("--gaussians, --width, --steps, and view counts must be positive")
    if args.download:
        download_garden()
    required = [DATA / "sparse" / "0" / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
    if not all(path.exists() for path in required) or not (DATA / "images_8").exists():
        parser.error("Garden photos are absent. Re-run with --download.")

    device = select_device(args.device)
    info = describe_device(device, args.device)
    print(format_report(info), flush=True)
    torch.manual_seed(args.seed)
    cameras = read_cameras(required[0])
    records = read_images(required[1])
    points, point_colors = read_points(required[2])
    samples = make_targets(records, cameras, args.width, device)
    train, heldout = choose_views(samples, args.train_views, args.holdout_views)
    scene, radius = initialize_scene(points, point_colors, args.gaussians, args.seed, device, train)
    args.output.mkdir(parents=True, exist_ok=True)
    initial_train, initial_holdout = mean_mse(scene, train), mean_mse(scene, heldout)
    optimizer = torch.optim.Adam([
        {"params": [scene.means], "lr": radius * 0.0008},
        {"params": [scene.log_scales, scene.quaternions], "lr": 0.005},
        {"params": [scene.color_logits, scene.opacity_logits], "lr": 0.03},
    ])
    synchronize(device)
    started = time.perf_counter()
    history = []
    print(f"Training {args.gaussians:,} learnable Gaussians from {len(train)} real photographs at {args.width}px wide", flush=True)
    for step in range(1, args.steps + 1):
        _, camera, target = train[(step - 1) % len(train)]
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = render(scene, camera)
        loss = (prediction - target).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == args.steps:
            record = {"step": step, "current_view_mse": float(loss.item())}
            history.append(record)
            print(f"step {step:4d}/{args.steps}: current view PSNR {psnr(loss.item()):.2f} dB", flush=True)
    synchronize(device)
    seconds = time.perf_counter() - started
    final_train, final_holdout = mean_mse(scene, train), mean_mse(scene, heldout)
    save_preview(args.output / "comparison.png", scene, heldout)
    np.savez(args.output / "learned_scene.npz", **{key: value.detach().cpu().numpy()
             for key, value in scene.state_dict().items()})
    export_scene(scene, args.output / "learned_scene.json", "Garden reconstructed from photographs", y_up=False)
    report = {
        "description": "Real Mip-NeRF 360 Garden JPGs plus supplied COLMAP calibration; trained with splatlab.core/render only",
        **info, "source": "https://huggingface.co/datasets/mileleap/mipnerf360/tree/main/garden",
        "image_directory": str(DATA / "images_8"), "input_photos": len(samples),
        "training_views": [name for name, _, _ in train], "heldout_views": [name for name, _, _ in heldout],
        "gaussians": args.gaussians, "training_width": args.width, "steps": args.steps,
        "scene_radius": radius, "training_seconds": seconds,
        "initial_train_psnr_db": psnr(initial_train), "final_train_psnr_db": psnr(final_train),
        "initial_heldout_psnr_db": psnr(initial_holdout), "final_heldout_psnr_db": psnr(final_holdout),
        "loss_decreased": final_train < initial_train, "history": history,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
