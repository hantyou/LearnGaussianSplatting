"""Train this lab's Gaussian renderer on real Mip-NeRF 360 Garden photographs.

The input is ``data/mipnerf360/garden/images_8`` plus COLMAP's sparse camera
calibration and point cloud.  No pretrained splat model is read by this command.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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

from .camera import Camera
from .device import CHOICES, describe_device, format_report, select_device, synchronize
from .export import export_scene
from .render import render
from .scene import GaussianScene

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "mipnerf360" / "garden"
SOURCE = "https://huggingface.co/datasets/mileleap/mipnerf360/resolve/main/garden"
API = "https://huggingface.co/api/datasets/mileleap/mipnerf360/tree/main/garden"
CAMERA_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}


@dataclass(frozen=True)
class Pose:
    name: str
    camera_id: int
    R: np.ndarray
    t: np.ndarray


def read_exact(file, count):
    data = file.read(count)
    if len(data) != count:
        raise ValueError("truncated COLMAP binary")
    return data


def download(url, destination):
    if destination.exists() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with urlopen(url, timeout=120) as response, partial.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
    partial.replace(destination)


def download_data():
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        download(f"{SOURCE}/sparse/0/{name}", DATA / "sparse" / "0" / name)
    with urlopen(f"{API}/images_8?recursive=false&expand=false", timeout=60) as response:
        listing = json.load(response)
    names = [item["path"].rsplit("/", 1)[-1] for item in listing if item["type"] == "file"]
    print(f"Downloading {len(names)} real Garden JPGs…", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda name: download(f"{SOURCE}/images_8/{name}", DATA / "images_8" / name), names))


def q_to_R(q):
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(((1 - 2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)),
                     (2*(x*y+w*z), 1 - 2*(x*x+z*z), 2*(y*z-w*x)),
                     (2*(x*z-w*y), 2*(y*z+w*x), 1 - 2*(x*x+y*y))), dtype=np.float32)


def load_cameras(path):
    cameras = {}
    with path.open("rb") as file:
        total, = struct.unpack("<Q", read_exact(file, 8))
        for _ in range(total):
            camera_id, model, width, height = struct.unpack("<iiQQ", read_exact(file, 24))
            params = np.array(struct.unpack(f"<{CAMERA_PARAMS[model]}d", read_exact(file, 8 * CAMERA_PARAMS[model])))
            if model in (0, 2, 3, 8, 9):
                focal, cx, cy = params[:3]; fx = fy = focal
            elif model == 1:
                fx, fy, cx, cy = params[:4]
            else:
                raise ValueError(f"Garden needs a pinhole camera, got COLMAP model {model}")
            cameras[camera_id] = (width, height, np.array((fx, fy, cx, cy)))
    return cameras


def load_poses(path):
    poses = []
    with path.open("rb") as file:
        total, = struct.unpack("<Q", read_exact(file, 8))
        for _ in range(total):
            read_exact(file, 4)
            q = np.array(struct.unpack("<4d", read_exact(file, 32)))
            t = np.array(struct.unpack("<3d", read_exact(file, 24)), dtype=np.float32)
            camera_id, = struct.unpack("<i", read_exact(file, 4))
            name = bytearray()
            while (char := read_exact(file, 1)) != b"\0":
                name.extend(char)
            count, = struct.unpack("<Q", read_exact(file, 8))
            file.seek(count * 24, 1)
            poses.append(Pose(name.decode(), camera_id, q_to_R(q), t))
    return poses


def load_points(path):
    xyz, rgb = [], []
    with path.open("rb") as file:
        total, = struct.unpack("<Q", read_exact(file, 8))
        for _ in range(total):
            read_exact(file, 8)
            xyz.append(struct.unpack("<3d", read_exact(file, 24)))
            rgb.append(struct.unpack("<3B", read_exact(file, 3)))
            read_exact(file, 8)
            track, = struct.unpack("<Q", read_exact(file, 8))
            file.seek(track * 8, 1)
    return np.asarray(xyz, np.float32), np.asarray(rgb, np.float32) / 255


def load_samples(poses, cameras, image_dir, width, device):
    samples = []
    for pose in sorted(poses, key=lambda x: x.name):
        path = image_dir / pose.name
        if not path.exists() or pose.camera_id not in cameras:
            continue
        original_w, original_h, values = cameras[pose.camera_id]
        with Image.open(path) as file:
            file = file.convert("RGB")
            source_w, source_h = file.size
            height = round(source_h * width / source_w)
            image = file.resize((width, height), Image.Resampling.LANCZOS)
        fx, fy, cx, cy = values * np.array((source_w/original_w, source_h/original_h,
                                             source_w/original_w, source_h/original_h))
        fx, cx = fx * width/source_w, cx * width/source_w
        fy, cy = fy * height/source_h, cy * height/source_h
        K = torch.tensor(((fx, 0., cx), (0., fy, cy), (0., 0., 1.)), dtype=torch.float32)
        camera = Camera(torch.from_numpy(pose.R), torch.from_numpy(pose.t), K, height, width, near=.01).to(device)
        target = torch.from_numpy(np.asarray(image, np.float32) / 255).to(device)
        samples.append((pose.name, camera, target))
    if not samples:
        raise RuntimeError("No JPGs matched the COLMAP poses")
    return samples


def choose_views(samples, train_count, heldout_count):
    indices = np.linspace(0, len(samples)-1, min(len(samples), train_count+heldout_count), dtype=int)
    chosen = [samples[i] for i in indices]
    return chosen[:train_count], chosen[train_count:train_count+heldout_count]


def stratified_points(points, samples, count, seed):
    """Seed points evenly over input images so the table is not starved by foliage."""
    rng = np.random.default_rng(seed)
    cells = max(1, math.ceil(math.sqrt(math.ceil(count / len(samples)))))
    selected = []
    for _, camera, _ in samples:
        R, t, K = (x.detach().cpu().numpy() for x in (camera.R, camera.t, camera.K))
        p = points @ R.T + t
        z = p[:, 2]
        u = K[0, 0]*p[:, 0]/np.maximum(z, 1e-8) + K[0, 2]
        v = K[1, 1]*p[:, 1]/np.maximum(z, 1e-8) + K[1, 2]
        ok = (z > camera.near) & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
        ids = np.flatnonzero(ok)
        bins = np.minimum(cells-1, (v[ids]/camera.height*cells).astype(int))*cells
        bins += np.minimum(cells-1, (u[ids]/camera.width*cells).astype(int))
        for cell in range(cells*cells):
            choices = ids[bins == cell]
            if len(choices):
                selected.append(rng.choice(choices))
    selected = np.unique(selected)
    if len(selected) >= count:
        return rng.choice(selected, count, replace=False)
    rest = np.setdiff1d(np.arange(len(points)), selected)
    return np.concatenate((selected, rng.choice(rest, count-len(selected), replace=False)))


def make_scene(points, colors, samples, count, seed, device):
    ids = stratified_points(points, samples, count, seed)
    low, high = np.quantile(points, (.05, .95), axis=0)
    radius = float(np.linalg.norm(high-low)/2)
    # More points have closer neighbours. Scaling both footprint width and
    # opacity by N^(-1/3) keeps a denser initialization from becoming an
    # opaque screen before its colors and geometry can be optimized.
    density_scale = (1000 / count) ** (1 / 3)
    scale = max(radius*.0025 * density_scale, 1e-3)
    opacity = min(.35, .35 * density_scale)
    scene = GaussianScene(torch.from_numpy(points[ids]), torch.full((count, 3), scale),
                          torch.tensor((1., 0., 0., 0.)).repeat(count, 1), torch.from_numpy(colors[ids]),
                          torch.full((count,), opacity)).to(device)
    return scene, radius, scale, opacity


def psnr(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().item()
    return -10*math.log10(max(float(value), 1e-12))


@torch.no_grad()
def score(scene, samples, tile_size=None):
    return torch.stack([(render(scene, camera, tile_size=tile_size)[0]-target).square().mean()
                        for _, camera, target in samples]).mean().item()


@torch.no_grad()
def preview(path, scene, samples, tile_size=None):
    samples = samples[:2]
    fig, axes = plt.subplots(len(samples), 2, figsize=(8, 3.5*len(samples)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row, (name, camera, target) in enumerate(samples):
        predicted, _ = render(scene, camera, tile_size=tile_size)
        for col, (image, title) in enumerate(((target, "Original photograph"), (predicted, "Our Gaussian reconstruction"))):
            axes[row, col].imshow(image.detach().cpu().numpy().clip(0, 1)); axes[row, col].set_title(title)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(name)
    fig.suptitle("Mip-NeRF 360 Garden · photos → COLMAP points → our splat core")
    fig.savefig(path, dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--image-dir", type=Path, default=DATA/"images_8",
                        help="directory of JPGs named exactly as in the COLMAP model")
    parser.add_argument("--device", choices=CHOICES, default="cuda")
    parser.add_argument("--gaussians", type=int, default=1000)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--tile-size", type=int, default=None,
                        help="screen tile width/height; enables memory-bounded rendering")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--train-views", type=int, default=18)
    parser.add_argument("--holdout-views", type=int, default=3)
    parser.add_argument("--colmap-model", type=Path, default=DATA/"sparse"/"0",
                        help="directory containing locally generated cameras.bin/images.bin/points3D.bin")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs"/"garden-from-photos")
    args = parser.parse_args()
    if min(args.gaussians, args.width, args.steps, args.train_views, args.holdout_views) < 1: parser.error("all counts must be positive")
    if args.download: download_data()
    sparse = args.colmap_model
    if not (sparse/"cameras.bin").exists(): parser.error("missing data; add --download")
    device = select_device(args.device); info = describe_device(device, args.device); print(format_report(info), flush=True)
    samples = load_samples(load_poses(sparse/"images.bin"), load_cameras(sparse/"cameras.bin"),
                           args.image_dir, args.width, device)
    train, heldout = choose_views(samples, args.train_views, args.holdout_views)
    points, colors = load_points(sparse/"points3D.bin")
    scene, radius, initial_scale, initial_opacity = make_scene(points, colors, train, args.gaussians, args.seed, device)
    args.output.mkdir(parents=True, exist_ok=True)
    initial_train, initial_holdout = score(scene, train, args.tile_size), score(scene, heldout, args.tile_size)
    optimizer = torch.optim.Adam(([{"params": [scene.means], "lr": radius*.0008},
                                  {"params": [scene.log_scales, scene.quaternions], "lr": .005},
                                  {"params": [scene.color_logits, scene.opacity_logits], "lr": .03}]))
    synchronize(device); started = time.perf_counter(); history = []
    print(f"Training {args.gaussians:,} Gaussians from {len(train)} real photographs at {args.width}px", flush=True)
    for step in range(1, args.steps+1):
        _, camera, target = train[(step-1) % len(train)]
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = render(scene, camera, tile_size=args.tile_size)
        loss = (prediction-target).square().mean()
        if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward(); optimizer.step()
        with torch.no_grad():
            scene.log_scales.clamp_(math.log(initial_scale*.35), math.log(initial_scale*2.5))
            scene.opacity_logits.clamp_(math.log(.02/.98), math.log(.65/.35))
        if step == 1 or step % 25 == 0 or step == args.steps:
            history.append({"step": step, "current_view_mse": float(loss.item())})
            print(f"step {step:4d}/{args.steps}: current view PSNR {psnr(loss):.2f} dB", flush=True)
    synchronize(device); seconds = time.perf_counter()-started
    final_train, final_holdout = score(scene, train, args.tile_size), score(scene, heldout, args.tile_size)
    preview(args.output/"comparison.png", scene, heldout, args.tile_size)
    np.savez(args.output/"learned_scene.npz", **{key: value.detach().cpu().numpy() for key, value in scene.state_dict().items()})
    export_scene(scene, args.output/"learned_scene.json", "Garden reconstructed from photographs", y_up=False)
    report = {"description": "Real Garden JPGs + COLMAP poses/points; optimized only through splatlab.core/render", **info,
              "source": "https://huggingface.co/datasets/mileleap/mipnerf360/tree/main/garden", "input_photos": len(samples),
              "training_views": [name for name, _, _ in train], "heldout_views": [name for name, _, _ in heldout],
              "gaussians": args.gaussians, "training_width": args.width, "tile_size": args.tile_size,
              "steps": args.steps, "scene_radius": radius,
              "initial_scale": initial_scale, "initial_opacity": initial_opacity,
              "scale_bounds": [initial_scale*.35, initial_scale*2.5], "opacity_bounds": [.02, .65],
              "training_seconds": seconds, "initial_train_psnr_db": psnr(initial_train), "final_train_psnr_db": psnr(final_train),
              "initial_heldout_psnr_db": psnr(initial_holdout), "final_heldout_psnr_db": psnr(final_holdout),
              "loss_decreased": final_train < initial_train, "history": history}
    (args.output/"metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
