#!/usr/bin/env bash
set -euo pipefail

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "CONDA_PREFIX is not set. Activate the target conda environment first." >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"

cp "$script_dir/activate.d/ddtw_cuda_clean.sh" \
  "$CONDA_PREFIX/etc/conda/activate.d/zz_ddtw_cuda_clean.sh"
cp "$script_dir/deactivate.d/ddtw_cuda_clean.sh" \
  "$CONDA_PREFIX/etc/conda/deactivate.d/zz_ddtw_cuda_clean.sh"

echo "Installed dDTW CUDA activation hooks into $CONDA_PREFIX"
echo "Reactivate the environment before building the CUDA extension."
