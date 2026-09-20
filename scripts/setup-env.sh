#!/usr/bin/env bash
# setup-env.sh - create one conda environment per PyTorch backend.
#
#   bash scripts/setup-env.sh cpu       # CPU reference, works anywhere
#   bash scripts/setup-env.sh rocm      # AMD, this laptop (Radeon 780M / gfx1103)
#   bash scripts/setup-env.sh cuda      # NVIDIA, the RTX 5070 Ti machine
#
# Options:
#   --name NAME        environment name (default: gslab-<backend>)
#   --python VERSION   Python version (default: 3.12, which is what the wheels target)
#   --torch VERSION    torch version to pin (default: 2.14.0)
#   --conda PATH       path to the conda executable, if it is not on PATH
#   --force            recreate the environment even if it already exists
#   --skip-check       do not run the post-install device probe
#
# Why one environment per backend: the CPU, ROCm and CUDA wheels ship different
# native libraries under the same "torch" import name. Mixing them produces
# import errors or, worse, a silently wrong library. All three pin the same
# torch version, so the environments differ only in backend - which is what
# makes a CPU-vs-GPU comparison meaningful instead of approximate.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

BACKEND=""
ENV_NAME=""
PY_VERSION="3.12"
TORCH_VERSION="2.14.0"
CONDA_BIN=""
FORCE=0
SKIP_CHECK=0

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
note() { printf '\n== %s\n' "$*"; }

usage() { sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

[ $# -gt 0 ] || { usage; exit 1; }

case "$1" in
  cpu|rocm|cuda) BACKEND="$1"; shift ;;
  -h|--help) usage; exit 0 ;;
  *) die "first argument must be cpu, rocm or cuda (got '$1')" ;;
esac

while [ $# -gt 0 ]; do
  case "$1" in
    --name) ENV_NAME="${2:?--name needs a value}"; shift 2 ;;
    --python) PY_VERSION="${2:?--python needs a value}"; shift 2 ;;
    --torch) TORCH_VERSION="${2:?--torch needs a value}"; shift 2 ;;
    --conda) CONDA_BIN="${2:?--conda needs a value}"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --skip-check) SKIP_CHECK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option '$1'" ;;
  esac
done

: "${ENV_NAME:=gslab-$BACKEND}"

# The wheel index decides the backend. Nothing else in this script does.
case "$BACKEND" in
  cpu)  INDEX_URL="https://download.pytorch.org/whl/cpu" ;;
  rocm) INDEX_URL="https://download.pytorch.org/whl/rocm7.2" ;;
  cuda) INDEX_URL="https://download.pytorch.org/whl/cu130" ;;
esac

# ---------------------------------------------------------------- locate conda
if [ -z "$CONDA_BIN" ]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
  else
    for c in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" /opt/conda; do
      if [ -x "$c/bin/conda" ]; then CONDA_BIN="$c/bin/conda"; break; fi
    done
  fi
fi
[ -n "$CONDA_BIN" ] || die "conda not found; pass --conda /path/to/bin/conda"
printf 'conda:   %s (%s)\n' "$CONDA_BIN" "$("$CONDA_BIN" --version 2>&1)"
printf 'backend: %s\n' "$BACKEND"
printf 'env:     %s (python %s)\n' "$ENV_NAME" "$PY_VERSION"
printf 'wheels:  %s\n' "$INDEX_URL"

# ------------------------------------------------------- hardware sanity checks
case "$BACKEND" in
  rocm)
    if [ ! -e /dev/kfd ]; then
      die "/dev/kfd does not exist, so there is no AMD compute device to target.
       Run 'bash scripts/preflight.sh' first and fix what it reports.
       Installing ROCm wheels now would only produce a slow, broken CPU run."
    fi
    if ! id -nG | tr ' ' '\n' | grep -qx render; then
      printf '\nwarning: you are not in the "render" group. PyTorch will probably fail\n'
      printf '         to open /dev/kfd. Fix with:\n'
      printf '           sudo usermod -aG render,video "$USER"\n'
      printf '         then log out and back in, and re-run this script.\n\n'
    fi
    ;;
  cuda)
    if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
      printf '\nwarning: nvidia-smi reports no GPU. Continuing, because the wheels install\n'
      printf '         fine without one - but --device cuda will refuse to run.\n\n'
    fi
    ;;
esac

# ------------------------------------------------------------- create the env
env_exists() { "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; }

if env_exists && [ "$FORCE" -eq 1 ]; then
  note "Removing existing environment $ENV_NAME (--force)"
  "$CONDA_BIN" env remove -y -n "$ENV_NAME"
fi

if env_exists; then
  note "Environment $ENV_NAME already exists; reusing it"
else
  note "Creating conda environment $ENV_NAME"
  # Anaconda's default channels now require an accepted Terms of Service. If the
  # default channels refuse, fall back to conda-forge, which does not.
  if ! "$CONDA_BIN" create -y -n "$ENV_NAME" "python=$PY_VERSION"; then
    printf '\ndefault channels failed; retrying with conda-forge only\n\n'
    "$CONDA_BIN" create -y -n "$ENV_NAME" -c conda-forge --override-channels "python=$PY_VERSION"
  fi
fi

# Resolve the interpreter path once and use it directly from here on. This
# avoids depending on "conda activate" working in a non-interactive shell.
ENV_PY="$("$CONDA_BIN" run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)' | tr -d '\r')"
[ -x "$ENV_PY" ] || die "could not resolve the python interpreter for $ENV_NAME"
printf 'python:  %s\n' "$ENV_PY"

# ------------------------------------------------------------------- install
note "Upgrading pip"
"$ENV_PY" -m pip install --upgrade pip

note "Installing torch==$TORCH_VERSION from $INDEX_URL"
# --index-url, not --extra-index-url: the backend-specific index must be the
# only source for torch, or pip may resolve a generic PyPI build instead.
"$ENV_PY" -m pip install "torch==$TORCH_VERSION" --index-url "$INDEX_URL"

note "Installing the project requirements"
# torch is already satisfied here, so this pulls only numpy/matplotlib/Pillow/pytest
# from PyPI and leaves the backend-specific torch untouched.
"$ENV_PY" -m pip install -r "$REPO_ROOT/requirements.txt"

# -------------------------------------------------------------------- verify
if [ "$SKIP_CHECK" -eq 1 ]; then
  note "Skipping the device probe (--skip-check)"
  printf '\nDone. Use it with:\n  conda activate %s\n' "$ENV_NAME"
  exit 0
fi

# A GPU that reports "available" can still fail on the first real kernel launch
# when the wheel carries no code object for this gfx target. So the probe runs
# an actual matmul and an actual backward pass, not just torch.cuda.is_available().
PROBE=$(cat <<'PY'
import json, sys, torch
info = {
    "torch": torch.__version__,
    "hip": getattr(torch.version, "hip", None),
    "cuda": getattr(torch.version, "cuda", None),
    "cuda_available": torch.cuda.is_available(),
    "gpu_name": None,
    "kernel_ok": False,
    "error": None,
}
if torch.cuda.is_available():
    try:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        a = torch.randn(256, 256, device="cuda", requires_grad=True)
        b = torch.randn(256, 256, device="cuda")
        (a @ b).sum().backward()
        torch.cuda.synchronize()
        info["kernel_ok"] = bool(a.grad is not None and torch.isfinite(a.grad).all())
    except Exception as exc:                      # noqa: BLE001 - report, do not hide
        info["error"] = f"{type(exc).__name__}: {exc}"
print("PROBE " + json.dumps(info))
PY
)

probe_run() {  # probe_run [env assignments...] -> prints the JSON line
  env "$@" "$ENV_PY" -c "$PROBE" 2>&1 | sed -n 's/^PROBE //p'
}

note "Probing the installed backend"
cd "$REPO_ROOT"
RESULT="$(probe_run || true)"
OVERRIDE_USED=""

if [ "$BACKEND" = rocm ]; then
  KERNEL_OK="$(printf '%s' "$RESULT" | "$ENV_PY" -c 'import json,sys; d=json.load(sys.stdin); print(d.get("kernel_ok"))' 2>/dev/null || echo False)"
  if [ "$KERNEL_OK" != "True" ]; then
    # Documented fallback for Phoenix/gfx1103: run the gfx1100 code objects that
    # the wheel does ship. This is an official environment variable used with an
    # official wheel - not a third-party build.
    note "GPU not usable as-is; retrying with HSA_OVERRIDE_GFX_VERSION=11.0.0"
    RETRY="$(probe_run HSA_OVERRIDE_GFX_VERSION=11.0.0 || true)"
    RETRY_OK="$(printf '%s' "$RETRY" | "$ENV_PY" -c 'import json,sys; d=json.load(sys.stdin); print(d.get("kernel_ok"))' 2>/dev/null || echo False)"
    if [ "$RETRY_OK" = "True" ]; then
      RESULT="$RETRY"
      OVERRIDE_USED="11.0.0"
      printf 'export HSA_OVERRIDE_GFX_VERSION=11.0.0\n' > "$REPO_ROOT/.rocm-env"
      printf '\nThe override is load-bearing on this machine. Wrote %s/.rocm-env;\n' "$REPO_ROOT"
      printf 'scripts/verify.sh sources it automatically. Record it in any benchmark.\n'
    fi
  else
    rm -f "$REPO_ROOT/.rocm-env"
  fi
fi

printf '\n--- probe result\n'
printf '%s\n' "$RESULT" | "$ENV_PY" -m json.tool 2>/dev/null || printf '%s\n' "$RESULT"

printf '\n--- splatlab.device report\n'
if [ -n "$OVERRIDE_USED" ]; then
  env HSA_OVERRIDE_GFX_VERSION="$OVERRIDE_USED" "$ENV_PY" -m splatlab.device || true
else
  "$ENV_PY" -m splatlab.device || true
fi

cat <<EOF

Done.

  conda activate $ENV_NAME
  cd $REPO_ROOT
EOF
if [ "$BACKEND" = cpu ]; then
  printf '  python -m splatlab.demo --device cpu --steps 20 --size 32\n'
else
  [ -n "$OVERRIDE_USED" ] && printf '  source .rocm-env\n'
  printf '  python -m splatlab.demo --device cuda --steps 20 --size 32\n'
  printf '\nThen compare against the CPU reference:\n'
  printf '  bash scripts/verify.sh --gpu-env %s\n' "$ENV_NAME"
fi
