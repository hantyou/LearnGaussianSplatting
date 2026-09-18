# Plan: finish the GPU port from Linux on this laptop

**Status on 18 September 2026.** The code port is done and verified on CPU. What
is left needs hardware this Windows install cannot provide. Pick this file up
after booting Linux on the same machine (Ryzen 7 7840HS / Radeon 780M).

---

## 1. Why we stopped on Windows

The Radeon 780M is **`gfx1103`** ("Phoenix"). Checked against AMD's own matrices:

| Path | Verdict | Evidence |
|---|---|---|
| ROCm on **Windows** | **No** | Windows matrix lists exactly `gfx1201 gfx1200 gfx1100 gfx1101`. AMD adds that on Windows "the entire ROCm stack is not yet supported". |
| ROCm on **WSL2** | **No** | AMD's Ryzen WSL guide covers "Strix and Strix Halo SKUs" (`gfx1150`/`gfx1151`). Phoenix is neither. |
| ROCm on **native Linux** | **Listed - verify** | The main ROCm compatibility matrix includes `gfx1103` among Linux Ryzen APU targets. |

So there was no officially supported GPU backend to install here, and the rule was
not to force an unofficial wheel. `.venv-cuda` was likewise **not** created: these
are AMD-only rules, and putting CUDA wheels on this machine is pointless.

**One honest caveat to resolve first.** AMD's pages disagree. The *main* ROCm
compatibility matrix lists `gfx1103` for Linux; the *Radeon-and-Ryzen project's*
native-Linux page lists only discrete cards (RX 7700/7800/7900/9060/9070, AI PRO
R9600/R9700) and no APUs. Treat `gfx1103` on Linux as **plausible but unproven**
until step 2 passes. Community practice for Phoenix is
`HSA_OVERRIDE_GFX_VERSION=11.0.0`; that is a supported *environment variable* used
with official wheels, not an unofficial wheel, but it is a fallback - note it in
the run's metrics if it turns out to be load-bearing.

Sources, all fetched 18 September 2026:

- <https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html>
- <https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html>
- <https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/wsl/howto_wsl.html>
- <https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/windows/install-pytorch.html>

---

## 2. First thing to run on Linux: does the GPU answer at all?

Cheap, decisive, before installing anything into the project.

```bash
uname -r
lsb_release -a
rocminfo | grep -i gfx
ls /dev/kfd /dev/dri
groups
```

Expect `gfx1103` from `rocminfo`; `command not found` means the ROCm stack is not
installed yet. `/dev/kfd` and `/dev/dri` must exist, and your user must be in both
the `render` and `video` groups. AMD documents Ubuntu 22.04.5 (kernel 6.8) and
24.04.4 (kernel 6.17) as the supported distributions.

If ROCm is not installed, install it for the detected Ubuntu version from AMD's
Linux install guide, reboot, and re-run the above.

---

## 3. Create `.venv-rocm` (never touch `.venv`)

`.venv` here is CPU-only and must survive untouched - it is the reference the GPU
gets checked against.

```bash
python3.12 -m venv .venv-rocm
.venv-rocm/bin/python -m pip install --upgrade pip
.venv-rocm/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
.venv-rocm/bin/python -m pip install -r requirements.txt
.venv-rocm/bin/python -m splatlab.device
```

`rocm7.2` carries **torch 2.14.0** for cp312 - the same version `.venv` has on
CPU, so the two environments differ only in backend. That is what makes the
CPU-versus-GPU comparison meaningful rather than approximate.

Expected output of the last command:

```text
torch 2.14.0+rocm7.2  (a ROCm build (HIP ...))
  --device cpu  available
  --device cuda available
  --device mps  unavailable
device: requested auto -> using cuda:0 (backend rocm, torch 2.14.0+rocm7.2) [AMD Radeon 780M ...]
```

`backend rocm`, not `cuda`, is the line to check. If `--device cuda` reports
`unavailable`, retry once with `HSA_OVERRIDE_GFX_VERSION=11.0.0`; if it is still
unavailable, **stop and report it** - keep `--device cpu` and do not hunt for an
unofficial wheel.

---

## 4. Run the checks that are already written

```bash
.venv-rocm/bin/python -m pytest -q
.venv/bin/python -m pytest -q
.venv-rocm/bin/python -m splatlab.demo --device cuda --steps 20 --size 32
.venv/bin/python -m splatlab.demo --device cpu --steps 20 --size 32
```

The CPU suite must stay at 15 passed, 3 skipped. On the ROCm interpreter the three
skipped tests in `tests/test_device.py` activate by themselves:

- `test_gpu_render_agrees_with_cpu` - forward pass versus CPU, `atol=1e-4, rtol=1e-3`
- `test_gpu_gradients_agree_with_cpu` - backward pass versus CPU, `atol=1e-4, rtol=5e-3`
- `test_gpu_optimization_actually_learns` - 20 real Adam steps reduce held-out error

The CPU reference to match, already recorded: `demo --steps 20 --size 32` gives
**train 29.115 dB / unseen 29.095 dB**. The GPU run should land within a few
thousandths of a dB. A larger gap is a real difference, not float noise.

---

## 5. Benchmark honestly

```bash
.venv/bin/python -m splatlab.fit_street --device cpu --steps 350 --size 48
.venv-rocm/bin/python -m splatlab.fit_street --device cuda --steps 350 --size 48
```

Both write `requested_device`, `device`, `backend`, `torch_version`, `gpu_name`,
`size`, `gaussians` and `seconds` into `outputs/street/metrics.json`, and the timer
is bracketed by GPU barriers, so the numbers are comparable. Run on an otherwise
idle machine - CPU timings here spread 260-340 s purely from load.

Expect the 755-Gaussian case to gain and the nine-Gaussian demo to *lose*: its
tensors are too small to cover kernel-launch overhead. Record both. Do not
extrapolate either to million-Gaussian scenes - this is a dense `O(N*H*W)`
educational renderer, not a tiled rasterizer.

Then fill in the measured GPU row in `docs/08-hardware.md`, which currently has
CPU numbers only.

---

## 6. If ROCm misbehaves

Preserve the mathematics. In order: report the slow or missing operation, keep
`--device cpu` working, and only then consider a backend-specific workaround -
never a change to `core.py` that makes the numbers backend-dependent.
`splatlab/render.py` and `splatlab/core.py` stay pure PyTorch with no vendor
branching; `splatlab/device.py` is the only module that knows a vendor exists.

---

## 7. Separately: the RTX 5070 Ti

**Yes, it needs its own environment** - `.venv-cuda`, on that machine, never shared
with `.venv-rocm`. The three wheel families ship incompatible native libraries
under the same `torch` import name.

```powershell
python -m venv .venv-cuda
.\.venv-cuda\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.\.venv-cuda\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-cuda\Scripts\python.exe -m splatlab.device
```

The 5070 Ti is Blackwell (`sm_120`) and needs a CUDA 12.8-or-newer build; `cu130`
is current and also carries **torch 2.14.0** for cp312. No code change is needed -
`--device cuda` already works, and `backend` will read `cuda` instead of `rocm`.

---

## 8. What is already done (no need to redo)

Verified on CPU, Windows, torch 2.14.0+cpu:

- `splatlab/device.py` - `select_device("auto"|"cpu"|"cuda"|"mps")`, backend
  reporting that separates `rocm` from `cuda`, actionable errors when a requested
  backend is missing, `synchronize()` for honest GPU timing, and
  `python -m splatlab.device` as a machine report.
- `--device` on `splatlab.demo` and `splatlab.fit_street`.
- Device-aware scenes, cameras, pixel grids, identity matrices, backgrounds and
  targets. Random initialization is drawn on the **CPU** and then moved, so a GPU
  run starts from bit-identical parameters instead of a different RNG stream.
- Host transfers (`.cpu()`) only for NumPy checkpoints, matplotlib/Pillow and JSON.
- Chunked footprint evaluation in `core.gaussian_alpha`, asserted **bitwise**
  identical to the unchunked result at every chunk size, including the empty scene.
- `metrics.json` records requested device, actual device, backend, torch version,
  GPU name, image size, Gaussian count and elapsed time.
- `tests/test_device.py`: 15 passed, 3 skipped on CPU.

Proof the port changed nothing on CPU: after all edits, `demo --steps 20 --size 32`
and `fit_street --steps 350 --size 48` reproduce their pre-change checkpoints
**bit-for-bit**, PSNR identical to 16 significant digits, and
`viewer/data/street-*.json` byte-identical.

One deliberate behaviour change: `fit_street`'s `seconds` now measures training
only, barrier to barrier. It previously included PNG and JSON writing, which would
have made any CPU-versus-GPU comparison meaningless.
