# Clean compiler flags that can confuse torch.utils.cpp_extension/nvcc, then
# select the conda compiler wrappers when they are available.
#
# Some conda CUDA/compiler packages set NVCC_PREPEND_FLAGS, CC, CXX, CFLAGS, or
# CXXFLAGS during activation. The inherited compiler-bindir and search flags can
# produce duplicate -ccbin arguments or malformed nvcc commands. At the same
# time, some servers have an old system c++ first on PATH, so we set CC/CXX back
# to the clean conda compiler wrappers after removing the problematic flags.

unset CC
unset CXX
unset CUDAHOSTCXX
unset NVCC_PREPEND_FLAGS
unset NVCC_APPEND_FLAGS
unset CUDAFLAGS
unset CFLAGS
unset CXXFLAGS

if [ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc" ]; then
  export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
elif [ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" ]; then
  export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
fi

if [ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" ]; then
  export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
  export CUDAHOSTCXX="$CXX"
elif [ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" ]; then
  export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
  export CUDAHOSTCXX="$CXX"
fi

export CUDA_HOME="$CONDA_PREFIX"

if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH"
else
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64"
fi
