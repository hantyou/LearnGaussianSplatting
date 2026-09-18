"""Device selection, footprint chunking, and opt-in GPU agreement checks.

Every test here passes on a CPU-only machine. The GPU tests skip themselves
when no accelerator is usable, so the suite never requires one.
"""

import subprocess
import sys

import pytest
import torch

from splatlab.camera import orbit_camera
from splatlab.core import gaussian_alpha
from splatlab.device import (CHOICES, available_backends, backend_name,
                             describe_device, select_device)
from splatlab.render import render
from splatlab.scene import make_student, make_teacher

GPU_BACKENDS = [name for name in ("cuda", "mps") if available_backends()[name]]
requires_gpu = pytest.mark.skipif(not GPU_BACKENDS,
                                  reason="CPU-only machine: no GPU backend available")

# Float32 on a GPU reassociates sums, so agreement is close, never bitwise.
# These bounds are loose enough for any vendor's kernels and tight enough that a
# genuinely different formula, layout, or coordinate convention still fails.
IMAGE_TOLERANCE = {"atol": 1e-4, "rtol": 1e-3}
GRADIENT_TOLERANCE = {"atol": 1e-4, "rtol": 5e-3}


def test_cpu_is_always_selectable():
    assert select_device("cpu") == torch.device("cpu")
    assert backend_name("cpu") == "cpu"


def test_auto_matches_what_the_machine_actually_offers():
    device = select_device("auto")
    assert (device.type == "cpu") == (not GPU_BACKENDS)


def test_unknown_device_name_is_rejected():
    # "rocm" is not a PyTorch device type; AMD GPUs answer to "cuda".
    with pytest.raises(ValueError, match="Unknown device"):
        select_device("rocm")


@pytest.mark.parametrize("name", ["cuda", "mps"])
def test_unavailable_backend_fails_loudly_instead_of_falling_back(name):
    if available_backends()[name]:
        pytest.skip(f"{name} is available here, so there is no failure to check")
    with pytest.raises(RuntimeError) as error:
        select_device(name)
    message = str(error.value)
    assert f"--device {name}" in message  # Says which request failed,
    assert "--device cpu" in message      # and what to do instead.
    assert torch.__version__ in message   # and which build was inspected.


def test_backend_name_never_calls_an_amd_gpu_cuda(monkeypatch):
    # A ROCm build answers to torch.device("cuda"); reporting must still say rocm.
    monkeypatch.setattr(torch.version, "hip", "6.2.0", raising=False)
    assert backend_name("cuda") == "rocm"
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert backend_name("cuda") == "cuda"


def test_description_carries_every_field_metrics_json_promises():
    info = describe_device(select_device("cpu"), requested="auto")
    assert set(info) == {"requested_device", "device", "backend",
                         "torch_version", "gpu_name"}
    assert info["requested_device"] == "auto"
    assert info["device"] == "cpu" and info["backend"] == "cpu"


def _footprint_inputs(n, width, dtype=torch.float32):
    torch.manual_seed(0)
    uv = torch.rand(n, 2, dtype=dtype) * width
    root = torch.rand(n, 2, 2, dtype=dtype)
    cov2 = root @ root.transpose(-1, -2) + 2*torch.eye(2, dtype=dtype)
    return uv, cov2, torch.rand(n, dtype=dtype)


@pytest.mark.parametrize("chunk", [1, 7, 40, 10**6])
def test_chunking_changes_memory_not_numbers(chunk):
    height, width, n = 12, 9, 40
    uv, cov2, opacities = _footprint_inputs(n, width)
    whole = gaussian_alpha(uv, cov2, opacities, height, width, chunk=n)
    chunked = gaussian_alpha(uv, cov2, opacities, height, width, chunk=chunk)
    assert torch.equal(chunked, whole)  # Bitwise: the rows are independent.


def test_chunking_of_an_empty_scene_keeps_the_shape_composite_expects():
    uv, cov2, opacities = _footprint_inputs(0, 9)
    assert gaussian_alpha(uv, cov2, opacities, 12, 9, chunk=4).shape == (0, 12, 9)


def test_chunking_leaves_gradients_intact():
    height, width, n = 5, 4, 6
    base_uv, cov2, opacities = _footprint_inputs(n, width, dtype=torch.double)

    def gradient(chunk):
        uv = base_uv.clone().requires_grad_(True)
        gaussian_alpha(uv, cov2, opacities, height, width, chunk=chunk).square().sum().backward()
        return uv.grad

    torch.testing.assert_close(gradient(2), gradient(n))


@pytest.mark.parametrize("module", ["splatlab.demo", "splatlab.fit_street"])
def test_both_entry_points_expose_the_device_flag(module):
    help_text = subprocess.run([sys.executable, "-m", module, "--help"],
                               capture_output=True, text=True, check=True).stdout
    assert "--device" in help_text
    assert all(name in help_text for name in CHOICES)


@requires_gpu
@pytest.mark.parametrize("name", GPU_BACKENDS)
def test_gpu_render_agrees_with_cpu(name):
    """The whole forward path on a GPU, checked against the CPU reference."""
    device = select_device(name)
    camera = orbit_camera(23., size=24)
    expected, expected_alpha = render(make_teacher().requires_grad_(False), camera)
    image, alpha = render(make_teacher(device).requires_grad_(False), camera.to(device))
    assert image.device.type == device.type  # It really ran there.
    torch.testing.assert_close(image.cpu(), expected, **IMAGE_TOLERANCE)
    torch.testing.assert_close(alpha.cpu(), expected_alpha, **IMAGE_TOLERANCE)


@requires_gpu
@pytest.mark.parametrize("name", GPU_BACKENDS)
def test_gpu_gradients_agree_with_cpu(name):
    """A backward pass too: a missing or wrong kernel need not break the forward."""
    def image_gradient(device):
        student = make_student(make_teacher(device), seed=7, device=device)
        loss = (render(student, orbit_camera(15., 20, device))[0]).square().mean()
        loss.backward()
        return loss.detach().cpu(), student.means.grad.cpu()

    expected_loss, expected_grad = image_gradient(torch.device("cpu"))
    loss, grad = image_gradient(select_device(name))
    torch.testing.assert_close(loss, expected_loss, **IMAGE_TOLERANCE)
    torch.testing.assert_close(grad, expected_grad, **GRADIENT_TOLERANCE)


@requires_gpu
@pytest.mark.parametrize("name", GPU_BACKENDS)
def test_gpu_optimization_actually_learns(name):
    """Twenty real Adam steps: exercises the optimizer, not only the renderer."""
    device = select_device(name)
    teacher = make_teacher(device).requires_grad_(False)
    student = make_student(teacher, seed=7, device=device)
    cameras = [orbit_camera(angle, 20, device) for angle in (-30., 0., 30.)]
    with torch.no_grad():
        targets = [render(teacher, camera)[0] for camera in cameras]

    def heldout_error():
        with torch.no_grad():
            return (render(student, cameras[1])[0] - targets[1]).square().mean().item()

    before = heldout_error()
    optimizer = torch.optim.Adam(student.parameters(), lr=0.02)
    for step in range(20):
        index = step % len(cameras)
        optimizer.zero_grad()
        loss = (render(student, cameras[index])[0] - targets[index]).square().mean()
        assert torch.isfinite(loss), f"{name} produced a nonfinite loss at step {step}"
        loss.backward()
        optimizer.step()
    assert heldout_error() < before
