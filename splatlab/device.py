"""Choose where the tensors live, without changing any of the mathematics.

PyTorch names AMD ROCm devices ``cuda`` as well: on a ROCm build
``torch.cuda.is_available()`` is true and tensors move to ``"cuda"`` even though
no NVIDIA hardware is present. That is an API name, not a hardware claim, so
:func:`backend_name` reports ``rocm`` separately and nothing here ever calls an
AMD GPU an NVIDIA CUDA device.

Run ``python -m splatlab.device`` to print what this machine offers.
"""

import torch

CHOICES = ("auto", "cpu", "cuda", "mps")


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def available_backends() -> dict:
    """Which --device values this interpreter can actually satisfy right now."""
    return {"cpu": True, "cuda": torch.cuda.is_available(), "mps": _mps_available()}


def backend_name(device) -> str:
    """cpu | cuda | rocm | mps -- the vendor stack behind a torch device type."""
    kind = torch.device(device).type
    if kind == "cuda":
        return "rocm" if getattr(torch.version, "hip", None) else "cuda"
    return kind


def _build_flavor() -> str:
    """Describe the installed wheel so an unavailable backend is explainable."""
    if getattr(torch.version, "hip", None):
        return f"a ROCm build (HIP {torch.version.hip})"
    if getattr(torch.version, "cuda", None):
        return f"a CUDA build (CUDA {torch.version.cuda})"
    return "a CPU-only build"


def _unavailable_message(request: str) -> str:
    build = _build_flavor()
    common = (f"Installed PyTorch {torch.__version__} is {build}. "
              "Use --device cpu here, or install a GPU build in a SEPARATE "
              "environment (see docs/08-hardware.md).")
    if request == "cuda":
        if getattr(torch.version, "hip", None):
            return ("Requested --device cuda (ROCm on this build) but no ROCm GPU is "
                    f"usable. {common}")
        if getattr(torch.version, "cuda", None):
            return ("Requested --device cuda but no NVIDIA GPU is visible; check the "
                    f"driver and that the GPU is not hidden by CUDA_VISIBLE_DEVICES. {common}")
        return f"Requested --device cuda but this PyTorch has no CUDA/ROCm support. {common}"
    return (f"Requested --device mps but Apple Metal is unavailable; it exists only on "
            f"Apple silicon macOS builds. {common}")


def select_device(request: str = "auto") -> torch.device:
    """Resolve "auto"|"cpu"|"cuda"|"mps" to a concrete torch.device.

    "auto" prefers a GPU and silently falls back to the CPU, which always works.
    A named backend that is unavailable raises instead of falling back: asking
    for a GPU and quietly getting CPU speed would be a misleading benchmark.
    """
    if request not in CHOICES:
        raise ValueError(f"Unknown device {request!r}; choose one of {', '.join(CHOICES)}.")
    available = available_backends()
    if request == "auto":
        for name in ("cuda", "mps"):
            if available[name]:
                request = name
                break
        else:
            return torch.device("cpu")
    elif not available[request]:
        raise RuntimeError(_unavailable_message(request))
    if request == "cuda":
        # Pin the index so reports and tensor .device values agree ("cuda:0").
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device(request)


def gpu_name(device) -> str:
    """Marketing name of the accelerator, or "" for the CPU."""
    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Silicon GPU (Metal)"
    return ""


def describe_device(device, requested: str = "auto") -> dict:
    """The device facts recorded in every metrics.json, so runs stay comparable."""
    device = torch.device(device)
    return {"requested_device": requested, "device": str(device),
            "backend": backend_name(device), "torch_version": torch.__version__,
            "gpu_name": gpu_name(device) or None}


def synchronize(device) -> None:
    """Wait for queued GPU work so elapsed time measures compute, not queueing."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def format_report(info: dict) -> str:
    """One console line: what was asked for, what was used, and on what stack."""
    name = f" [{info['gpu_name']}]" if info.get("gpu_name") else ""
    return (f"device: requested {info['requested_device']} -> using {info['device']} "
            f"(backend {info['backend']}, torch {info['torch_version']}){name}")


def main() -> None:
    available = available_backends()
    print(f"torch {torch.__version__}  ({_build_flavor()})")
    for name, ok in available.items():
        print(f"  --device {name:<4} {'available' if ok else 'unavailable'}")
    chosen = select_device("auto")
    print(format_report(describe_device(chosen, "auto")))


if __name__ == "__main__":
    main()
