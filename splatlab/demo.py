"""Optimize nine 3D splats from five images, then render unseen cameras.

Run from the repository root: python -m splatlab.demo
No dataset, CUDA, pretrained model, or network download is needed at runtime.
"""

import argparse
import json
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch

from .camera import orbit_camera
from .device import CHOICES, describe_device, format_report, select_device, synchronize
from .scene import make_teacher, make_student
from .render import render


def psnr(mse):
    return -10 * math.log10(max(float(mse), 1e-12))  # Images have range [0,1].


@torch.no_grad()
def evaluate(scene, cameras, targets, core):
    images = [render(scene, camera, core)[0] for camera in cameras]
    mse = torch.stack([(image-target).square().mean()
                       for image, target in zip(images, targets)]).mean().item()
    return images, mse


def save_comparison(path, targets, before, after, train_count,
                    title="Nine learnable 3D Gaussians · synthetic experiment"):
    # A held-out row tests image agreement from a camera unused for optimization.
    rows = [0, train_count]
    fig, axes = plt.subplots(2, 3, figsize=(8, 5.7), constrained_layout=True)
    for row, index in enumerate(rows):
        for col, images in enumerate((targets, before, after)):
            axes[row, col].imshow(images[index].detach().cpu().numpy(), vmin=0, vmax=1)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel("Training view" if row == 0 else "Unseen view")
    for ax, column_title in zip(axes[0], ("Target", "Before learning", "After learning")):
        ax.set_title(column_title)
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def save_orbit(path, teacher, student, core, size):
    frames = []
    device = teacher.means.device
    # Include poses outside the training range to expose extrapolation limitations.
    for angle in np.linspace(-75, 75, 41):
        camera = orbit_camera(float(angle), size, device)
        # Pillow needs host memory; .cpu() is a no-op for a CPU run.
        left = render(teacher, camera)[0].cpu().numpy()
        right = render(student, camera, core)[0].cpu().numpy()
        pixels = (np.clip(np.concatenate((left, right), axis=1), 0, 1)*255).astype(np.uint8)
        content = Image.fromarray(pixels).resize((size*8, size*4), Image.Resampling.NEAREST)
        frame = Image.new("RGB", (size*8, size*4+28), (240, 240, 240))
        frame.paste(content, (0, 28))
        ImageDraw.Draw(frame).text((8, 7), f"Target | Learned     camera angle: {angle:.0f} deg", fill=(20, 20, 20))
        frames.append(frame)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=90, loop=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-views", type=int, choices=(1, 5), default=5)
    parser.add_argument("--device", default="auto", choices=CHOICES,
                        help="auto uses a GPU when one is usable, else the CPU")
    parser.add_argument("--student", action="store_true", help="Use YOUR exercises/core.py")
    parser.add_argument("--output", type=Path, default=Path("outputs/reference"))
    args = parser.parse_args()
    if args.steps < 1 or args.size < 8:
        parser.error("Use at least 1 step and an image size of at least 8.")
    from . import core
    if args.student:
        from exercises import core

    device = select_device(args.device)
    device_info = describe_device(device, args.device)
    print(format_report(device_info), flush=True)

    # Small tensors are often slower with a large pool of CPU worker threads.
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    teacher = make_teacher(device).requires_grad_(False)
    student = make_student(teacher, args.seed, device)
    train_angles = [-50., -25., 0., 25., 50.] if args.train_views == 5 else [0.]
    test_angles = [-37.5, 12.5, 37.5]  # Never contribute to the training loss.
    cameras = [orbit_camera(angle, args.size, device) for angle in train_angles + test_angles]
    with torch.no_grad():
        targets = [render(teacher, camera)[0] for camera in cameras]
    before, _ = evaluate(student, cameras, targets, core)
    n_train = len(train_angles)
    train_cameras, test_cameras = cameras[:n_train], cameras[n_train:]
    train_targets, test_targets = targets[:n_train], targets[n_train:]
    _, initial_train = evaluate(student, train_cameras, train_targets, core)
    _, initial_test = evaluate(student, test_cameras, test_targets, core)

    optimizer = torch.optim.Adam([
        {"params": [student.means], "lr": 0.012},
        {"params": [student.log_scales, student.quaternions], "lr": 0.015},
        {"params": [student.color_logits, student.opacity_logits], "lr": 0.04},
    ])
    history = [{"step": 0, "train_mse": initial_train, "heldout_mse": initial_test}]
    synchronize(device)  # GPU work is queued, so time it between two barriers.
    start = time.perf_counter()
    for step in range(1, args.steps+1):
        camera_index = (step-1) % n_train
        optimizer.zero_grad()  # PyTorch accumulates gradients unless we clear them.
        predicted, _ = render(student, train_cameras[camera_index], core)
        loss = (predicted - train_targets[camera_index]).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss: inspect scales, projection, and covariance.")
        loss.backward()  # Chain rule: loss -> image -> footprints -> 3D parameters.
        optimizer.step()
        if step % 25 == 0 or step == args.steps:
            _, train_mse = evaluate(student, train_cameras, train_targets, core)
            _, test_mse = evaluate(student, test_cameras, test_targets, core)
            history.append({"step": step, "train_mse": train_mse, "heldout_mse": test_mse})
            print(f"step {step:4d} | train {psnr(train_mse):5.2f} dB | unseen {psnr(test_mse):5.2f} dB", flush=True)
    synchronize(device)
    seconds = time.perf_counter() - start
    after, _ = evaluate(student, cameras, targets, core)
    final = history[-1]
    report = {
        "description": "Synthetic Gaussian targets, perturbed teacher initialization; no geometry supervision",
        **device_info, "seed": args.seed,
        "steps": args.steps, "size": args.size, "gaussians": len(student.means),
        "student_core": args.student, "train_angles_degrees": train_angles,
        "heldout_angles_degrees": test_angles, "training_seconds": seconds,
        "initial_train_psnr_db": psnr(initial_train), "final_train_psnr_db": psnr(final["train_mse"]),
        "initial_heldout_psnr_db": psnr(initial_test), "final_heldout_psnr_db": psnr(final["heldout_mse"]),
        "history": history,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez(args.output / "learned_scene.npz",  # NumPy checkpoints live on the host.
             **{key: value.detach().cpu().numpy() for key, value in student.state_dict().items()})
    save_comparison(args.output / "comparison.png", targets, before, after, n_train)
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    for key, label in (("train_mse", "Training views"), ("heldout_mse", "Unseen views")):
        ax.semilogy([h["step"] for h in history], [h[key] for h in history], label=label)
    ax.set(xlabel="Optimizer steps", ylabel="Mean squared RGB error", title="Image fitting; lower is better")
    ax.legend()
    fig.savefig(args.output / "loss.png", dpi=300)
    plt.close(fig)
    save_orbit(args.output / "orbit.gif", teacher, student, core, args.size)
    print(f"Saved results to {args.output.resolve()} ({seconds:.1f} s training)")


if __name__ == "__main__":
    main()
