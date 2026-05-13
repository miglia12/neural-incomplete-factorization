#!/bin/bash
# Install all deps for this project into the currently-active venv on Alex
# The script reads $VIRTUAL_ENV to find the venv.
#
# Requirments:
#   - Modules loaded: cuda/12.9.0, gcc/11.2.0, python/3.12-conda
#   - A venv created and activated (anywhere on the filesystem)
#   - Run from the project root (where requirements.txt lives)
#
# Usage:
#   salloc --gres=gpu:a40:1 --time=01:00:00
#   module load cuda/12.9.0
#   module load gcc/11.2.0
#   module load python/3.12-conda
#   python3 -m venv <env folder>
#   source <env folder>/bin/activate
#   bash scripts/install_alex.sh

set -euo pipefail

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "ERROR: No virtual environment active." >&2
  echo "Create + activate one first, for example:" >&2
  echo '  python3 -m venv $WORK/venvs/nif' >&2
  echo '  source $WORK/venvs/nif/bin/activate' >&2
  exit 1
fi
echo "  venv:    $VIRTUAL_ENV"
echo "  python:  $(which python) ($(python --version))"

python - << 'PY'
import sys
v = sys.version_info
if not (v.major == 3 and v.minor in (11, 12)):
    sys.exit(f"ERROR: Need Python 3.11 or 3.12, got {v.major}.{v.minor} — load python/3.12-conda")
PY

if ! command -v nvcc &> /dev/null; then
  echo "ERROR: nvcc not found. Run: module load cuda/12.9.0" >&2
  exit 1
fi
echo "  nvcc:    $(nvcc --version | tail -1 | sed 's/^[ \t]*//')"

gcc_major=$(gcc -dumpversion | cut -d. -f1)
if [ "$gcc_major" -lt 9 ]; then
  echo "ERROR: gcc $gcc_major too old (need >=9). Run: module load gcc/11.2.0" >&2
  exit 1
fi
echo "  gcc:     $(gcc --version | head -1)"

if [ ! -f requirements.txt ]; then
  echo "ERROR: requirements.txt not found in cwd ($PWD)." >&2
  echo "Run this script from the project root." >&2
  exit 1
fi

export http_proxy="${http_proxy:-http://proxy.nhr.fau.de:80}"
export https_proxy="${https_proxy:-http://proxy.nhr.fau.de:80}"
echo "  proxy:   $http_proxy"
echo

log="install_alex_$(date +%Y%m%d_%H%M%S).log"
echo "Installing... Log in $log ==="

{
  pip install --upgrade pip
  pip install setuptools wheel
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  pip install pybind11 numpy
  pip install -r requirements.txt --no-build-isolation
} 2>&1 | tee "$log"

echo
python - << 'PY'
import torch
import numml.sparse as sp

print(f"  torch:       {torch.__version__}")
print(f"  torch.cuda:  {torch.version.cuda}")
assert torch.cuda.is_available(), "FAIL: CUDA not available at runtime"
print(f"  device:      {torch.cuda.get_device_name(0)}")

A = torch.sparse_coo_tensor(
    torch.tensor([[0, 1, 2], [0, 1, 2]], device='cuda'),
    torch.tensor([1., 2., 3.], device='cuda'),
    (3, 3), device='cuda',
)
L = sp.SparseCSRTensor(A)
assert L.data.device.type == 'cuda', f"FAIL: numml on {L.data.device}"
print(f"  numml CSR:   {L.shape} nnz={L.nnz} on {L.data.device}")
print()
print("=== ALL CHECKS PASSED ===")
PY
