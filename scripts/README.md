# GPU environment setup

Three scripts, run in this order. They use **conda environments**, not `venv`,
because this repository lives on the data partition shared with Windows and a
`venv` there is fragile; conda envs live under the conda root on the native
Linux filesystem instead.

```bash
bash scripts/preflight.sh          # 1. look, change nothing
bash scripts/setup-env.sh cpu      # 2. CPU reference environment
bash scripts/setup-env.sh rocm     # 3. AMD GPU environment (this laptop)
bash scripts/verify.sh             # 4. GPU vs CPU, same numbers or not
```

On the RTX 5070 Ti machine, step 3 becomes `bash scripts/setup-env.sh cuda`
and step 4 `bash scripts/verify.sh --gpu-env gslab-cuda`.

Run them with `bash script.sh` rather than `./script.sh` — the exec bit does not
survive on every filesystem, and `preflight.sh` reports whether it does here.

## The three environments

| Environment | Wheel index | Backend | Machine |
|---|---|---|---|
| `gslab-cpu` | `download.pytorch.org/whl/cpu` | `cpu` | any |
| `gslab-rocm` | `download.pytorch.org/whl/rocm7.2` | `rocm` | ThinkBook, Radeon 780M |
| `gslab-cuda` | `download.pytorch.org/whl/cu130` | `cuda` | the RTX 5070 Ti box |

All three pin **torch 2.14.0**, so they differ only in backend. That is what
makes a CPU-versus-GPU comparison meaningful rather than approximate. Never
install one backend's wheels into another's environment: they ship different
native libraries under the same `torch` import name.

The existing `.venv/` in this repository is a **Windows** virtualenv
(`Scripts/`, `Lib/`, built from `D:\Anaconda`). It cannot run on Linux. Leave it
alone — you presumably still boot Windows on this machine — and use `gslab-cpu`
as the Linux CPU reference instead.

## What each script does

### `preflight.sh`

Read-only. Reports the kernel, the distribution, the filesystem the repository
sits on, whether `/dev/kfd` and `/dev/dri` exist, whether you are in the `render`
and `video` groups, the detected `gfx` target, any installed ROCm packages,
`nvidia-smi`, and the conda installation. Ends with a verdict and the next
command to run. Writes `outputs/env-report-<host>.txt`.

The `gfx` target is read from `rocminfo` if it is installed, and otherwise from
`/sys/class/kfd/kfd/topology/nodes/*/properties`, which needs no ROCm packages at
all. A Radeon 780M should report `gfx1103`.

### `setup-env.sh <cpu|rocm|cuda>`

Creates the conda environment, installs the pinned torch from the backend's
index, installs `requirements.txt`, then **probes the backend with a real
matmul and a real backward pass** — not just `torch.cuda.is_available()`, which
can be true on a GPU whose kernels the wheel does not actually carry.

For `rocm`, if that probe fails it retries once with
`HSA_OVERRIDE_GFX_VERSION=11.0.0`. This makes the runtime load the `gfx1100`
code objects the wheel does ship, which is the documented route for Phoenix
(`gfx1103`). It is an official environment variable used with an official wheel,
not a third-party build. If the override turns out to be load-bearing the script
writes `.rocm-env` at the repository root and `verify.sh` sources it — and you
should record it in any benchmark you quote, because it is part of how the
number was produced.

It refuses to install the ROCm wheels at all if `/dev/kfd` is missing, since the
result would be a large download that can only ever run on the CPU.

### `verify.sh`

Runs `python -m splatlab.device`, the test suite, and `splatlab.demo` on both the
CPU and GPU environments, then prints them side by side. Add `--bench` to also
run the 755-Gaussian `fit_street` benchmark on both and report a speedup; that
takes minutes on the CPU.

The CPU reference already recorded for `demo --steps 20 --size 32` is
**train 29.115 dB / unseen 29.095 dB**. The GPU should land within a few
thousandths of a dB; the script flags a gap above 0.01 dB. A larger gap is a real
difference, not float noise.

On the ROCm environment the three GPU tests in `tests/test_device.py` stop
skipping themselves and start comparing forward pass, backward pass, and 20 real
Adam steps against the CPU.

## If ROCm does not work

The order from `PLAN-linux-rocm.md` still stands: report the missing or slow
operation, keep `--device cpu` working, and only then consider a
backend-specific workaround. Do not change `core.py` to suit a backend, and do
not chase an unofficial wheel — the value of this lab is that its numbers are
trustworthy.

## Notes on `gfx1103` specifically

AMD's own pages disagree, and it is worth knowing which is which:

- The **main ROCm compatibility matrix** (ROCm 10.0.0) lists `gfx1103` among the
  Linux Ryzen APU targets, alongside `gfx1150`/`gfx1151`/`gfx1152`/`gfx1153`,
  with Ubuntu 26.04 (GA 7.0 kernel) or Ubuntu 24.04.4 (OEM 6.17 kernel).
- The **Radeon-and-Ryzen project's** native-Linux matrix, which documents up to
  ROCm 7.2.1, lists only `gfx1150` and `gfx1151`.

So `gfx1103` on Linux is listed but thinly supported, and whether the *PyTorch
wheel* carries `gfx1103` code objects is a separate question from whether ROCm
supports the chip. That is exactly what `setup-env.sh`'s probe settles in about
ten seconds, and why the `HSA_OVERRIDE_GFX_VERSION` fallback is wired in.

Note also that MIOpen ships no precompiled convolution database for `gfx1103`.
It does not matter here — this renderer has no convolutions — but it will bite
any CNN work on the same environment.

## Generated files

`setup-env.sh` and `verify.sh` write `.rocm-env`, `outputs/env-report-*.txt` and
`outputs/verify/`. None of these belong in Git; add them to `.gitignore` when you
next touch it.
