# dDTW CUDA Conda Environments

Create an environment from the repository root:

```bash
conda env create -f environments/ddtw_cu128.yml
conda activate ddtw_cu128
```

Install the activation hooks once per environment:

```bash
bash environments/install_activation_hooks.sh
conda deactivate
conda activate ddtw_cu128
```

The hooks clear noisy build variables such as `NVCC_PREPEND_FLAGS`, `CFLAGS`,
and `CXXFLAGS`, then set `CC`, `CXX`, and `CUDAHOSTCXX` to the conda compiler
wrappers when they are available. This avoids duplicate nvcc host-compiler flags
without falling back to an old system compiler.

Use `TORCH_CUDA_ARCH_LIST` only when PyTorch's auto-detection chooses the wrong
architecture:

```bash
export TORCH_CUDA_ARCH_LIST="8.9"   # RTX 4090
export TORCH_CUDA_ARCH_LIST="12.0"  # RTX PRO 6000 Blackwell
export TORCH_CUDA_ARCH_LIST="6.1"   # GTX 1080 Ti
```

Then build/test from a clean extension directory:

```bash
rm -rf ddtw/backend/_cpp_build
python -m pytest test
```
