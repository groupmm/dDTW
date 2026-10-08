<p align="right">
  <img src="https://raw.githubusercontent.com/groupmm/dDTW/master/docs/_static/figures/logo_ddtw.png" alt="dDTW logo" width="300">
</p>

# *d*DTW: A Unified and Efficient Toolbox for Differentiable Sequence Alignment

© [Johannes Zeitler](https://audiolabs-erlangen.de/fau/assistant/zeitler) and [Meinard Müller](https://www.audiolabs-erlangen.de/fau/professor/mueller), 2026

Extended documentation available [HERE](https://groupmm.github.io/dDTW).

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

## Installation

### CPU

Create and activate a Conda environment, then install dDTW:

```bash
conda create -n ddtw python=3.12 pip
conda activate ddtw
python -m pip install ddtw
```

Use `backend="torch"` in the example below.
For faster CPU execution, install `python -m pip install "ddtw[numba]"`
and use `backend="cpu_numba"`.

### NVIDIA GPU (Linux)

You need Linux x86_64, [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/),
and an NVIDIA GPU with a working driver supporting CUDA 12.8 or newer.
Create and activate a dedicated environment once:

```bash
conda create -n ddtw python=3.12 pip
conda activate ddtw
```

Then install and configure dDTW with two commands:

```bash
python -m pip install ddtw
ddtw setup-cuda
```

No repository download is needed. The setup command installs CUDA 12.8,
GCC/G++ 13, PyTorch 2.11.0+cu128, Ninja and NumPy into the active Conda environment,
then compiles dDTW and checks GPU forward/backward execution. It replaces other
versions of these dependencies in that environment. Allow several minutes;
wait for **GPU check passed** and **Setup complete**.

Dependencies, dDTW build settings and compiled extensions stay inside the
environment. No `sudo`, system CUDA installation or manual activation hooks
are needed. In future sessions, only `conda activate ddtw` is needed.

Automatic GPU setup requires a dedicated Conda environment. The setup accepts
standard Python 3.10–3.13; Python 3.12 is the validated default.

To inspect the installation commands or repeat the GPU check:

```bash
ddtw setup-cuda --dry-run
ddtw check-cuda
```

`python -m ddtw` can replace `ddtw` in these commands. The `ddtw[cuda]` pip extra
installs Ninja only; it does not perform CUDA setup automatically.
See [environments/README.md](environments/README.md) for troubleshooting.

## Usage

```python
import torch
from ddtw import SDTW

X = torch.randn(2, 20, 8, requires_grad=True)
Y = torch.randn(2, 15, 8)

loss_fn = SDTW(backend="torch")
loss = loss_fn(X, Y)
loss.backward()
```

After GPU setup, use `SDTW(backend="cuda_cpp")` to run the loss on your NVIDIA
GPU. Inputs are moved to the selected backend's device automatically.

For development from a repository clone, install your checkout in editable mode:

```bash
python -m pip install -e ".[test,benchmark]"
```

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
