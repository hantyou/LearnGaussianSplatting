#!/usr/bin/env bash
# verify.sh - check a GPU environment against the CPU reference.
#
#   bash scripts/verify.sh                          # tests + demo, a couple of minutes
#   bash scripts/verify.sh --bench                  # also the 755-Gaussian benchmark (slow)
#   bash scripts/verify.sh --gpu-env gslab-cuda     # the NVIDIA machine
#
# Options:
#   --gpu-env NAME   GPU conda environment (default: gslab-rocm)
#   --cpu-env NAME   CPU conda environment (default: gslab-cpu)
#   --bench          also run fit_street on both backends and time them
#   --steps N        benchmark steps (default: 350)
#   --size N         benchmark image size (default: 48)
#   --conda PATH     path to the conda executable, if it is not on PATH
#
# The CPU run is the reference, not a formality: the whole point of the port is
# that the mathematics is backend-independent, so the GPU numbers have to match
# the CPU numbers to float32 tolerance. A large gap is a real difference, not
# noise.
#
# Recorded CPU reference for demo --steps 20 --size 32:
#     train 29.115 dB / unseen 29.095 dB
# The GPU should land within a few thousandths of a dB.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ENV="gslab-rocm"
CPU_ENV="gslab-cpu"
BENCH=0
STEPS=350
SIZE=48
CONDA_BIN=""

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
note() { printf '\n========== %s\n' "$*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --gpu-env) GPU_ENV="${2:?}"; shift 2 ;;
    --cpu-env) CPU_ENV="${2:?}"; shift 2 ;;
    --bench) BENCH=1; shift ;;
    --steps) STEPS="${2:?}"; shift 2 ;;
    --size) SIZE="${2:?}"; shift 2 ;;
    --conda) CONDA_BIN="${2:?}"; shift 2 ;;
    -h|--help) sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option '$1'" ;;
  esac
done

if [ -z "$CONDA_BIN" ]; then
  if command -v conda >/dev/null 2>&1; then CONDA_BIN="$(command -v conda)"
  else
    for c in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" /opt/conda; do
      [ -x "$c/bin/conda" ] && { CONDA_BIN="$c/bin/conda"; break; }
    done
  fi
fi
[ -n "$CONDA_BIN" ] || die "conda not found; pass --conda /path/to/bin/conda"

env_python() {  # env_python NAME -> absolute interpreter path
  "$CONDA_BIN" run -n "$1" python -c 'import sys; print(sys.executable)' 2>/dev/null | tr -d '\r'
}

CPU_PY="$(env_python "$CPU_ENV")"
GPU_PY="$(env_python "$GPU_ENV")"
[ -x "${CPU_PY:-}" ] || die "CPU environment '$CPU_ENV' not found. Run: bash scripts/setup-env.sh cpu"
[ -x "${GPU_PY:-}" ] || die "GPU environment '$GPU_ENV' not found. Run: bash scripts/setup-env.sh rocm"

# If setup-env.sh found the gfx override load-bearing, it left this behind.
if [ -f "$REPO_ROOT/.rocm-env" ]; then
  # shellcheck disable=SC1091
  . "$REPO_ROOT/.rocm-env"
  printf 'using HSA_OVERRIDE_GFX_VERSION=%s\n' "${HSA_OVERRIDE_GFX_VERSION:-}"
fi

cd "$REPO_ROOT"
OUT="$REPO_ROOT/outputs/verify"
mkdir -p "$OUT"

printf 'cpu env: %s -> %s\n' "$CPU_ENV" "$CPU_PY"
printf 'gpu env: %s -> %s\n' "$GPU_ENV" "$GPU_PY"

note "What each environment thinks it has"
"$CPU_PY" -m splatlab.device 2>&1 | tee "$OUT/device-cpu.txt" || true
"$GPU_PY" -m splatlab.device 2>&1 | tee "$OUT/device-gpu.txt" || true

note "Test suite on CPU (expect 15 passed, 3 skipped for the device module)"
"$CPU_PY" -m pytest -q 2>&1 | tee "$OUT/pytest-cpu.txt" | tail -5 || true

note "Test suite on the GPU environment (the 3 skipped device tests should now run)"
"$GPU_PY" -m pytest -q 2>&1 | tee "$OUT/pytest-gpu.txt" | tail -5 || true

note "demo: 9 Gaussians, 20 steps, 32 px - CPU then GPU"
"$CPU_PY" -m splatlab.demo --device cpu  --steps 20 --size 32 --output "$OUT/demo-cpu" >"$OUT/demo-cpu.log" 2>&1 \
  || printf "  CPU demo FAILED - see %s\n" "$OUT/demo-cpu.log"
"$GPU_PY" -m splatlab.demo --device cuda --steps 20 --size 32 --output "$OUT/demo-gpu" >"$OUT/demo-gpu.log" 2>&1 \
  || printf "  GPU demo FAILED - see %s\n" "$OUT/demo-gpu.log"

if [ "$BENCH" -eq 1 ]; then
  # fit_street writes to a fixed outputs/street/, so each run is snapshotted
  # before the next one overwrites it.
  note "fit_street benchmark: $STEPS steps at ${SIZE}px - CPU (this takes minutes)"
  "$CPU_PY" -m splatlab.fit_street --device cpu --steps "$STEPS" --size "$SIZE" >"$OUT/street-cpu.log" 2>&1 \
    || printf "  CPU benchmark FAILED - see %s\n" "$OUT/street-cpu.log"
  cp "$REPO_ROOT/outputs/street/metrics.json" "$OUT/street-cpu.json" 2>/dev/null || true

  note "fit_street benchmark: $STEPS steps at ${SIZE}px - GPU"
  "$GPU_PY" -m splatlab.fit_street --device cuda --steps "$STEPS" --size "$SIZE" >"$OUT/street-gpu.log" 2>&1 \
    || printf "  GPU benchmark FAILED - see %s\n" "$OUT/street-gpu.log"
  cp "$REPO_ROOT/outputs/street/metrics.json" "$OUT/street-gpu.json" 2>/dev/null || true
fi

note "Comparison"
"$CPU_PY" - "$OUT" <<'PY'
import json, pathlib, sys

out = pathlib.Path(sys.argv[1])

def load(p):
    try:
        return json.loads(pathlib.Path(p).read_text())
    except Exception:
        return None

def row(label, a, b, fmt="{:.3f}"):
    fa = fmt.format(a) if isinstance(a, (int, float)) else str(a)
    fb = fmt.format(b) if isinstance(b, (int, float)) else str(b)
    print(f"  {label:<26} {fa:>18} {fb:>18}")

demo_cpu = load(out / "demo-cpu" / "metrics.json")
demo_gpu = load(out / "demo-gpu" / "metrics.json")

print("\ndemo (9 Gaussians, 20 steps, 32 px)")
print(f"  {'':<26} {'CPU':>18} {'GPU':>18}")
if demo_cpu and demo_gpu:
    for key in ("backend", "gpu_name", "torch_version"):
        row(key, demo_cpu.get(key), demo_gpu.get(key))
    for key in ("final_train_psnr_db", "final_heldout_psnr_db"):
        row(key, demo_cpu.get(key), demo_gpu.get(key))
        a, b = demo_cpu.get(key), demo_gpu.get(key)
        if isinstance(a, float) and isinstance(b, float):
            gap = abs(a - b)
            verdict = "OK" if gap < 0.01 else "LOOK AT THIS"
            print(f"  {'  -> gap':<26} {gap:>18.6f} dB   {verdict}")
else:
    print("  (missing metrics.json - check demo-*.log)")

street_cpu = load(out / "street-cpu.json")
street_gpu = load(out / "street-gpu.json")
if street_cpu and street_gpu:
    print("\nfit_street benchmark")
    print(f"  {'':<26} {'CPU':>18} {'GPU':>18}")
    for key in ("backend", "gpu_name", "gaussians", "size", "steps"):
        row(key, street_cpu.get(key), street_gpu.get(key), "{}")
    for key in ("final_train_psnr", "final_heldout_psnr"):
        row(key, street_cpu.get(key), street_gpu.get(key))
    row("seconds", street_cpu.get("seconds"), street_gpu.get("seconds"), "{:.1f}")
    cs, gs = street_cpu.get("seconds"), street_gpu.get("seconds")
    if isinstance(cs, float) and isinstance(gs, float) and gs > 0:
        print(f"  {'  -> speedup':<26} {cs / gs:>18.2f}x")
    print("\n  Quote this speedup only if the CPU run had the machine to itself.")
PY

printf '\nLogs and metrics: %s\n' "$OUT"
printf 'If the GPU numbers match, fill the measured GPU row into docs/08-hardware.md.\n'
