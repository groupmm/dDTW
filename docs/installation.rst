Installation
============

Install the package from PyPI:

.. code-block:: bash

   python -m pip install ddtw

For local development, clone the repository and install it in editable mode:

.. code-block:: bash

   python -m pip install -e ".[test,benchmark]"

Runtime Requirements
--------------------

The implemented losses require:

* Python
* PyTorch
* NumPy
* Numba for the ``cpu_numba`` backend
* A CUDA-capable PyTorch setup and a working compiler toolchain for the
  ``cuda_cpp`` backend

CUDA Backend
------------

The ``cuda_cpp`` backend uses PyTorch's JIT extension loader, which invokes ``ninja``,
``c++``, and ``nvcc`` to compile the native extension lazily on first use.
[#paszke2019]_
For this build to work, the CUDA-related components must be compatible:

* the NVIDIA driver must support the CUDA runtime used by PyTorch;
* the installed PyTorch wheel must match the intended CUDA version, visible as
  ``torch.version.cuda``;
* the active ``nvcc`` must come from a compatible CUDA toolkit and must see the
  CUDA development headers;
* the host C++ compiler must be recent enough for PyTorch's extension build.

``nvidia-smi`` reports the maximum CUDA version supported by the driver. It does
not show which CUDA toolkit or ``nvcc`` is active inside the Python environment.
Check the active setup with:

.. code-block:: bash

   which nvcc
   nvcc --version

   python - <<'PY'
   import torch
   print(torch.__version__)
   print(torch.version.cuda)
   print(torch.cuda.is_available())
   PY

Conda CUDA Environments
-----------------------

The repository contains tested conda environment files for CUDA 11.8, 12.8,
and 13.2 in ``environments/``. They install PyTorch with pip and the CUDA
toolkit, ``nvcc``, and host compilers with conda. For example:

.. code-block:: bash

   conda env create -f environments/ddtw_cu128.yml
   conda activate ddtw_cu128
   bash environments/install_activation_hooks.sh
   conda deactivate
   conda activate ddtw_cu128

The activation hooks clear inherited CUDA/compiler flags such as
``NVCC_PREPEND_FLAGS``, ``CFLAGS``, and ``CXXFLAGS``, then select the conda
compiler wrappers through ``CC``, ``CXX``, and ``CUDAHOSTCXX``. This avoids two
common failure modes: duplicate ``nvcc`` host-compiler flags and accidental use
of an old system ``c++``.

We tested and verified the ddtw-cuda environments for the following architectures:

.. list-table::
   :header-rows: 1

   * - GPU
     - CUDA 11.8
     - CUDA 12.8
     - CUDA 13.2
   * - RTX 1080 TI
     - ✓
     - ✗
     - ✗
   * - RTX 2080 TI
     - ✗
     - ✓
     - ✓
   * - RTX 4090 
     - ✗
     - ✓
     - ✓
   * - RTX A5500
     - ✗
     - ✓
     - ✓
   * - RTX Pro 6000
     - ✗
     - ✓
     - ✓



If PyTorch auto-detects the wrong GPU architecture, set
``TORCH_CUDA_ARCH_LIST`` before rebuilding the extension. Typical values are
``6.1`` for GTX 1080 Ti, ``7.5`` for RTX 2080, ``8.9`` for RTX 4090, and
``12.0`` for RTX PRO 6000 Blackwell.

After changing CUDA, compiler, or architecture settings, remove the cached
extension build and run the tests again:

.. code-block:: bash

   rm -rf ddtw/backend/_cpp_build
   python -m pytest test

The repository ``README.md`` and ``environments/README.md`` contain more
concrete setup examples and troubleshooting notes.

Documentation Requirements
--------------------------

The documentation dependencies are listed in ``docs/requirements.txt``:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt

Build the HTML documentation with:

.. code-block:: bash

   sphinx-build -M html docs docs/_build

or:

.. code-block:: bash

   make -C docs html

The Sphinx configuration mocks heavy runtime imports such as ``torch`` and
``numba``. This allows API documentation to build on machines that are not
configured for GPU training.

.. rubric:: References


.. [#paszke2019] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury,
   G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison,
   A. Kopf, E. Z. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy,
   B. Steiner, L. Fang, J. Bai, and S. Chintala, "PyTorch: An Imperative Style,
   High-Performance Deep Learning Library," in *Advances in Neural Information
   Processing Systems (NeurIPS)*, Vancouver, BC, Canada, 2019, pp. 8024-8035.
