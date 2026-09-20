#!/usr/bin/env bash
# preflight.sh - read-only survey of this machine's GPU stack.
#
# Changes nothing except a report file under outputs/. Run this FIRST, on any
# machine, before creating an environment. It answers the only question that
# matters up front: can PyTorch see a GPU here at all, and if not, what is
# missing?
#
#   bash scripts/preflight.sh
#
# The report is written to outputs/env-report-<host>.txt so it can be pasted
# back into a chat or attached to a bug report.

set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="$REPO_ROOT/outputs"
mkdir -p "$REPORT_DIR"
HOST="$(hostname -s 2>/dev/null || echo unknown)"
REPORT="$REPORT_DIR/env-report-$HOST.txt"

have() { command -v "$1" >/dev/null 2>&1; }

section() { printf '\n===== %s =====\n' "$1"; }

# run <binary> [args...] - run a command, or say plainly that it is absent.
run() {
  local bin="$1"
  printf -- '--- %s\n' "$*"
  if have "$bin"; then
    "$@" 2>&1 | sed 's/^/    /'
  else
    printf '    (%s: not installed)\n' "$bin"
  fi
}

survey() {
printf 'GPU preflight report\n'
printf 'generated: %s\n' "$(date -Is)"
printf 'host:      %s\n' "$HOST"
printf 'repo:      %s\n' "$REPO_ROOT"

section "Operating system"
run uname -a
printf -- '--- /etc/os-release\n'
sed 's/^/    /' /etc/os-release 2>/dev/null || printf '    (absent)\n'

section "Where the repository lives"
# A venv on an NTFS or noexec mount breaks in confusing ways. Conda envs live
# under the conda root instead, which is why this matters less here - but the
# repo's own outputs still get written to this filesystem.
if have findmnt; then
  findmnt -n -o SOURCE,TARGET,FSTYPE,OPTIONS -T "$REPO_ROOT" 2>&1 | sed 's/^/    /'
else
  df -T "$REPO_ROOT" 2>&1 | sed 's/^/    /'
fi
probe="$REPORT_DIR/.exec-probe.$$"
printf '#!/bin/sh\nexit 0\n' > "$probe" 2>/dev/null
if chmod +x "$probe" 2>/dev/null && "$probe" 2>/dev/null; then
  printf '    exec bit on this filesystem: yes\n'
else
  printf '    exec bit on this filesystem: NO (scripts cannot run from here; use "bash script.sh")\n'
fi
rm -f "$probe"

section "AMD GPU / ROCm"
printf -- '--- PCI display devices\n'
if have lspci; then
  lspci -nn 2>/dev/null | grep -Ei 'vga|3d|display' | sed 's/^/    /'
else
  printf '    (lspci: not installed - apt install pciutils)\n'
fi
printf -- '--- amdgpu kernel module\n'
if lsmod 2>/dev/null | grep -q '^amdgpu'; then
  lsmod | grep '^amdgpu' | sed 's/^/    /'
else
  printf '    amdgpu NOT loaded\n'
fi
printf -- '--- compute device nodes\n'
ls -l /dev/kfd 2>&1 | sed 's/^/    /'
ls -l /dev/dri 2>&1 | sed 's/^/    /'
printf -- '--- group membership (need: render, video)\n'
printf '    %s\n' "$(id -nG 2>/dev/null)"
printf -- '--- detected gfx target\n'
if have rocminfo; then
  rocminfo 2>/dev/null | grep -Ei 'Name:.*gfx|Marketing Name|Compute Unit' | sed 's/^/    /'
else
  printf '    (rocminfo: not installed - "sudo apt install rocminfo" gives it)\n'
  # Fall back to the kernel's own view, which needs no ROCm packages at all.
  if [ -r /sys/class/kfd/kfd/topology/nodes ]; then
    for n in /sys/class/kfd/kfd/topology/nodes/*/properties; do
      gfx=$(awk '/gfx_target_version/ {print $2}' "$n" 2>/dev/null)
      [ -n "${gfx:-}" ] && [ "$gfx" != "0" ] && printf '    kfd node %s gfx_target_version=%s\n' \
        "$(basename "$(dirname "$n")")" "$gfx"
    done
  fi
fi
run rocm-smi --showproductname
printf -- '--- installed ROCm packages\n'
if have dpkg; then
  dpkg -l 2>/dev/null | awk '/ (rocm|hip|hsa)/ {print $2, $3}' | head -30 | sed 's/^/    /'
  dpkg -l 2>/dev/null | awk '/ (rocm|hip|hsa)/' | grep -q . || printf '    (none)\n'
else
  printf '    (dpkg: not a Debian/Ubuntu system)\n'
fi

section "NVIDIA GPU / CUDA"
run nvidia-smi
run nvcc --version

section "Python and conda"
run conda --version
printf -- '--- conda base\n'
if have conda; then
  conda info --base 2>&1 | sed 's/^/    /'
  printf -- '--- existing envs\n'
  conda env list 2>&1 | sed 's/^/    /'
else
  for c in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" /opt/conda; do
    [ -x "$c/bin/conda" ] && printf '    found but not on PATH: %s/bin/conda\n' "$c"
  done
fi
run python3 --version

section "Memory (an iGPU shares this)"
run free -h

section "Verdict"
kfd_ok=no;   [ -e /dev/kfd ] && kfd_ok=yes
dri_ok=no;   [ -e /dev/dri ] && dri_ok=yes
grp_ok=no
if id -nG 2>/dev/null | tr ' ' '\n' | grep -qx render && \
   id -nG 2>/dev/null | tr ' ' '\n' | grep -qx video; then grp_ok=yes; fi
nv_ok=no;    have nvidia-smi && nvidia-smi -L >/dev/null 2>&1 && nv_ok=yes
conda_ok=no; have conda && conda_ok=yes

printf '    /dev/kfd present ............ %s\n' "$kfd_ok"
printf '    /dev/dri present ............ %s\n' "$dri_ok"
printf '    in render AND video groups .. %s\n' "$grp_ok"
printf '    NVIDIA GPU visible .......... %s\n' "$nv_ok"
printf '    conda on PATH ............... %s\n' "$conda_ok"
printf '\n'

if [ "$kfd_ok" = yes ] && [ "$grp_ok" = yes ]; then
  printf '    -> AMD compute stack looks reachable. Next:\n'
  printf '         bash scripts/setup-env.sh rocm\n'
elif [ "$kfd_ok" = yes ] && [ "$grp_ok" = no ]; then
  printf '    -> /dev/kfd exists but you are not in the render/video groups. Run:\n'
  printf '         sudo usermod -aG render,video "$USER"\n'
  printf '       then log out and back in (a new terminal is not enough).\n'
elif [ "$kfd_ok" = no ]; then
  printf '    -> No /dev/kfd, so no AMD compute device. The amdgpu kernel driver is\n'
  printf '       either not loaded or was built without KFD. Check "dmesg | grep -i amdgpu",\n'
  printf '       and if needed install the AMD driver:\n'
  printf '         https://rocm.docs.amd.com/projects/install-on-linux/en/latest/\n'
fi
if [ "$nv_ok" = yes ]; then
  printf '    -> NVIDIA GPU visible. Next:  bash scripts/setup-env.sh cuda\n'
fi
if [ "$conda_ok" = no ]; then
  printf '    -> conda is not on PATH. Either activate it, or pass --conda /path/to/conda\n'
  printf '       to setup-env.sh.\n'
fi
printf '    CPU baseline (always works):  bash scripts/setup-env.sh cpu\n'
}

survey 2>&1 | tee "$REPORT"
printf '\nReport saved to %s\n' "$REPORT"
