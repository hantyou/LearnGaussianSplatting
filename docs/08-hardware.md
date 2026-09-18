# 8. Running the lab on CPU, AMD ROCm, and NVIDIA CUDA

The mathematics in `splatlab/core.py` never changes between backends. Only the
*placement* of the tensors changes, which is why the whole port is one small
module, `splatlab/device.py`, plus a `--device` flag on the two entry points.

```powershell
.\.venv\Scripts\python.exe -m splatlab.device
```

That prints which backends this interpreter can actually satisfy, and is the
first thing to run on any new machine.

## One flag, three backends

```powershell
# CPU (works everywhere, no GPU needed)
.\.venv\Scripts\python.exe -m splatlab.demo --device cpu

# AMD ROCm
.\.venv-rocm\Scripts\python.exe -m splatlab.demo --device cuda

# NVIDIA CUDA
.\.venv-cuda\Scripts\python.exe -m splatlab.demo --device cuda
```

On Linux and macOS replace `.\.venv-x\Scripts\python.exe` with
`.venv-x/bin/python`. `--device auto` picks a GPU when one is usable and falls
back to the CPU; a *named* backend that is unavailable raises instead of falling
back, because a "GPU" run that silently ran on the CPU is a misleading benchmark.

### Why AMD also uses `--device cuda`

`cuda` is the **PyTorch API name**, not a hardware claim. A ROCm build of PyTorch
reuses the entire `torch.cuda` namespace: `torch.cuda.is_available()` returns
true, tensors move to `"cuda"`, and `torch.cuda.get_device_name()` returns an AMD
part. There is no `torch.rocm`. So on a ROCm machine you ask for `--device cuda`
and you get your Radeon.

An AMD GPU is **not** an NVIDIA CUDA device, and this repository never says it is.
`splatlab/device.py` keeps the two apart by inspecting `torch.version.hip`, and
reports the vendor stack separately from the device name:

| Field in `metrics.json` | CPU | AMD ROCm | NVIDIA |
|---|---|---|---|
| `requested_device` | `cpu` | `cuda` | `cuda` |
| `device` | `cpu` | `cuda:0` | `cuda:0` |
| `backend` | `cpu` | **`rocm`** | **`cuda`** |
| `gpu_name` | `null` | `AMD Radeon ...` | `NVIDIA GeForce ...` |

`backend` is the field that tells you which vendor actually ran the job.

## One environment per backend, never mixed

The three wheel families install incompatible native libraries under the same
`torch` import name, so they get separate virtual environments and never share
one:

| Environment | Wheel index | Use on |
|---|---|---|
| `.venv` | `https://download.pytorch.org/whl/cpu` | anything |
| `.venv-rocm` | `https://download.pytorch.org/whl/rocm7.2` | supported AMD + Linux |
| `.venv-cuda` | `https://download.pytorch.org/whl/cu130` | NVIDIA |

Do not install ROCm wheels into `.venv-cuda` or CUDA wheels into `.venv-rocm`.
All three currently offer **torch 2.14.0** for CPython 3.12, so the three
environments can pin the identical PyTorch version and differ only in backend —
which is what makes CPU-vs-GPU output comparisons meaningful.

### NVIDIA (the RTX 5070 Ti)

Yes, it needs its own environment. The 5070 Ti is Blackwell (`sm_120`), which
requires a CUDA 12.8-or-newer build; `cu130` is the current index.

```powershell
python -m venv .venv-cuda
.\.venv-cuda\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.\.venv-cuda\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-cuda\Scripts\python.exe -m splatlab.device
```

### AMD

Checked against AMD's own matrices on 18 September 2026:

* **Windows: not supported for this laptop.** The Windows matrix lists exactly
  `gfx1201 gfx1200 gfx1100 gfx1101`. A Radeon 780M is `gfx1103` (Ryzen 7 7840HS,
  "Phoenix") and is absent. AMD also notes that on Windows "the entire ROCm stack
  is not yet supported".
* **WSL2: not supported for this laptop.** AMD's Ryzen WSL guide covers "Strix and
  Strix Halo SKUs" — `gfx1150`/`gfx1151` — not Phoenix.
* **Native Linux: listed.** The main ROCm compatibility matrix includes `gfx1103`
  among the Linux Ryzen APU targets. See [the Linux plan](../PLAN-linux-rocm.md)
  for the verification steps, because AMD's own pages are not fully consistent
  about `gfx1103` on Linux.

If ROCm turns out to be unavailable on a given machine, **keep using `.venv` and
`--device cpu`.** Do not force an unofficial wheel: the whole point of this lab is
that the numbers are trustworthy.

DirectML (`torch-directml`) is a possible Windows experiment for unsupported AMD
hardware. It is deliberately **not** a backend here: it is a separate device type
with its own operator coverage, and adding it would put vendor branching into
teaching code. Treat it as an experiment run outside this repository.

## What to expect from the performance

This is a **dense, educational renderer**: every Gaussian is evaluated at every
pixel, `O(N·H·W)`. Production systems (the original CUDA rasterizer, gsplat)
sort into tiles and touch only the pixels a splat actually covers. Keeping the
two separate is the point — you can read all of this one, and it is honest about
its own cost.

Measured on this machine (Ryzen 7 7840HS, CPU only, torch 2.14.0+cpu):

| Experiment | Gaussians | Images | Steps | CPU time |
|---|---|---|---|---|
| `splatlab.demo` | 9 | 5 × 32² | 20 | ~0.5 s |
| `splatlab.fit_street` | 755 | 6 × 48² | 350 | 260–340 s |

The street range is real measurement spread on eight cores, not uncertainty about
the work: an unloaded run recorded 257 s wall including output writing, and a run
sharing the machine with the test suite recorded 340 s of training alone. Quote a
GPU speedup only against a CPU number measured on an otherwise idle machine.

A GPU helps the 755-Gaussian case, where the `[N,H,W]` footprint tensor is large
enough to fill the machine. It does very little for the nine-Gaussian demo, where
the tensors are tiny and kernel-launch overhead dominates — expect a GPU to be
*slower* there. That is expected, not a bug.

**A Radeon 780M cannot train million-Gaussian real scenes**, and neither can this
renderer on any hardware. The 780M is an integrated GPU sharing system memory;
this lab's dense `[N,H,W]` evaluation would need tile-based rasterization, sorted
per-tile splat lists, and adaptive density control before scene scale mattered.
Use this to understand the equations, then read `docs/06-reading-code.md`.

### Memory: chunked footprints

`core.gaussian_alpha` evaluates at most `FOOTPRINT_CHUNK_ELEMENTS` worth of
Gaussians per pass instead of building the `[N,H,W,2]` pixel offsets for every
Gaussian at once. Gaussians are independent there, so this changes peak memory and
nothing else — the tests assert the chunked result is **bitwise** identical. The
returned `[N,H,W]` alpha is still built in full, because `composite()` needs every
sorted layer; what shrinks is the much larger scratch space around it.

## If something is slow or unsupported on a GPU

Report it and fall back — **do not change the mathematics to suit a backend.**
`--device cpu` always works and is the reference the GPU is checked against.
`tests/test_device.py` compares a GPU render and a GPU backward pass against CPU
within a stated float32 tolerance, and skips itself entirely on a CPU-only
machine, so the suite never requires a GPU.
