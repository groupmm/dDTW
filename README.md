<p align="right">
  <img src="https://raw.githubusercontent.com/groupmm/dDTW/master/docs/_static/figures/logo_ddtw.png" alt="dDTW logo" width="300">
</p>

# *d*DTW: A Unified and Efficient Toolbox for Differentiable Sequence Alignment

© [Johannes Zeitler](https://audiolabs-erlangen.de/fau/assistant/zeitler) and [Meinard Müller](https://www.audiolabs-erlangen.de/fau/professor/mueller), 2026

This repository contains the *d*DTW toolbox, a modular PyTorch toolbox with efficient CPU and GPU implementations that unifies DTW, soft/smooth/sparse DTW, subsequence DTW, partial matching, and CTC in a common framework.

If you use the *d*DTW toolbox, please cite the corresponding paper:
```
@article{ZeitlerM26_dDTW_toolbox,
    author = {Johannes Zeitler and Meinard M{\"u}ller},
    title  = {{dDTW}: A Unified and Efficient Toolbox for Differentiable Sequence Alignment},
    year   = {2026},
    note   = {Under Review}
    }
```

## Installation and Usage

To install the *d*DTW toolbox locally, you can clone this repository or use pip:
``` bash
pip install ddtw
```

To use a loss function from the *d*DTW toolbox, such as SDTW, simply import the module and use it like a normal PyTorch loss:
```python
from ddtw import SDTW

loss_fn = SDTW()
loss = loss_fn(X,Y) # assuming X and Y are, e.g., predictions and targets
loss.backward()
```

## System Requirements

The toolbox is implemented in Python and PyTorch. A basic CPU setup requires:

- Python with a recent PyTorch installation
- NumPy
- Numba when using the `cpu_numba` backend

The `torch` backend runs without a custom compiler. The optional but recommended `cuda_cpp` backend
requires a CUDA-capable PyTorch installation, an NVIDIA GPU, the CUDA toolkit
including `nvcc`, and a working C++ compiler toolchain because the extension is
compiled locally through `torch.utils.cpp_extension`.

### CUDA, PyTorch, and `nvcc` Version Matching

The `cuda_cpp` backend is compiled lazily the first time it is used. This means
`pip install ddtw` installs the Python package and CUDA/C++ source files,
but the native extension is built later by PyTorch's extension loader. For this
build to work, three CUDA-related components must be compatible:

- The NVIDIA driver must support the CUDA runtime used by PyTorch.
- The installed PyTorch wheel must be built for the intended CUDA version, shown
  by `torch.version.cuda`.
- The active CUDA compiler `nvcc` must be from a matching CUDA toolkit and must
  support the GPU architecture being compiled, for example Blackwell GPUs require
  a recent CUDA toolkit.

You can check the active environment with:

```bash
which nvcc
nvcc --version

python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch CUDA", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
print("GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("capability", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
PY
```

The examples below use conda for the CUDA compiler/toolkit and pip for PyTorch.
They install the full CUDA toolkit in the environment so `nvcc` and development
headers such as `cusparse.h` are available. For up-to-date PyTorch wheel
commands, also check the official PyTorch install selector at
https://pytorch.org/get-started/locally/.

Ready-made environment files are available in `environments/` for CUDA 11.8,
12.8, and 13.2. Create one from the repository root and install the activation
hooks once:

```bash
conda env create -f environments/ddtw_cu128.yml
conda activate ddtw_cu128
bash environments/install_activation_hooks.sh
conda deactivate
conda activate ddtw_cu128
```

The hooks clear inherited compiler and CUDA flags that can otherwise make
PyTorch's JIT extension builder pick the wrong host compiler or CUDA toolkit.

#### Example: CUDA 11.8

```bash
conda create -n ddtw_cu118 python=3.11 
conda activate ddtw_cu118

conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11 
conda install -c nvidia/label/cuda-11.8.0 cuda 

pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -e ".[test,benchmark]"
```

#### Example: CUDA 12.8

```bash
conda create -n ddtw_cu128 python=3.12 
conda activate ddtw_cu128

conda install -c conda-forge gcc_linux-64=14 gxx_linux-64=14 
conda install -c nvidia/label/cuda-12.8.0 cuda 

pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[test,benchmark]"
```

#### Example: CUDA 13.2
```bash
conda create -n ddtw_cu132 python=3.12 
conda activate ddtw_cu132

conda install -c conda-forge gcc_linux-64 gxx_linux-64 
conda install -c nvidia/label/cuda-13.2 cuda 

pip install torch --index-url https://download.pytorch.org/whl/cu132
pip install -e ".[test,benchmark]"
```

After creating any CUDA environment, rebuild the extension from a clean state:

```bash
rm -rf ddtw/backend/_cpp_build
python -m pytest test
python test/benchmark_CTC.py
python test/benchmark_SDTW.py #--implementation ddtw
```

If the CUDA binaries don't compile, make sure the activation hooks are installed.
They remove confusing linker/search flags while keeping the conda compiler
wrappers selected:

```bash
bash environments/install_activation_hooks.sh
conda deactivate
conda activate ddtw_cu128
```

The `benchmark_SDTW.py` baseline by Maghoumi uses an older Numba CUDA
implementation, which we tested for CUDA 11.8 and 12.8. In newer CUDA environments, it may fail with Numba PTX or CUDA context errors. 
Use `--implementation ddtw` to benchmark only the toolbox CUDA backend.

Additional development and example dependencies are only needed for specific tasks:

- `pytest` for the test suite
- `librosa` for the DTW and subsequence DTW reference tests
- Jupyter and Matplotlib for the demo notebook
- Sphinx and the packages in `docs/requirements.txt` for documentation builds

## Demo Notebook

The notebook `demo_SDTW.ipynb` provides a compact walkthrough of the `SDTW` loss.
It constructs example input sequences, runs forward and backward passes, and shows
how to inspect the intermediate tensors stored on `loss_fn.core`, including the
pairwise cost matrix `C_matrix`, accumulated cost matrix `D_matrix`, soft alignment
matrix `E_matrix`, and cost-gradient matrix `H_matrix`.

The final part of the notebook visualizes how different softmin temperatures
`gamma` change the SDTW alignment. The notebook automatically selects CUDA when
available and otherwise runs on CPU.

## Tests

The tests are located in `test/` and can be run from the repository root with:

```bash
python -m pytest test
```

The tests compare the toolbox implementations against reference implementations
for DTW, SDTW, subsequence DTW, partial matching, and CTC. They test the
`torch`, `cpu_numba`, and `cuda_cpp` backends where available. CUDA/Numba tests are
skipped automatically when no CUDA device is available or if the corresponding optional dependency or extension cannot be loaded.

## Benchmarks

Two CUDA benchmark scripts are provided under `test/`:

```bash
python test/benchmark_SDTW.py
python test/benchmark_CTC.py
```

`benchmark_SDTW.py` compares `ddtw.SDTW` with [Maghoumi's Soft-DTW](https://github.com/Maghoumi/pytorch-softdtw-cuda)
reference implementation. `benchmark_CTC.py` compares `ddtw.CTC` with PyTorch's
`torch.nn.functional.ctc_loss`. Both scripts run forward and backward passes,
report median timing and peak CUDA memory allocation.

## Documentation

HTML documentation is maintained with Sphinx under `docs/`.

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -M html docs docs/_build
```

Open `docs/_build/html/index.html` after the build finishes.

## License
This project is licensed under the [MIT License](LICENSE).

## Authors
[Johannes Zeitler](https://audiolabs-erlangen.de/fau/assistant/zeitler)

[Meinard Müller](https://www.audiolabs-erlangen.de/fau/professor/mueller)

## Acknowledgements
This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Grant No. 500643750 (MU 2686/15-1) and Grant No. 521420645 (MU 2686/17-1). The authors are with the International Audio Laboratories Erlangen, a joint institution of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) and Fraunhofer Institute for Integrated Circuits IIS.

The software architecture of the *d*DTW toolbox is inspired by Mehran Maghoumi's [Soft DTW for PyTorch in CUDA](https://github.com/Maghoumi/pytorch-softdtw-cuda).
