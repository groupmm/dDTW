Installation
============

CPU
---

Create and activate a Conda environment, then install the published package:

.. code-block:: bash

   conda create -n ddtw python=3.12 pip
   conda activate ddtw
   python -m pip install ddtw

Use ``SDTW(backend="torch")`` for CPU execution.
For faster CPU execution, install ``ddtw[numba]`` and select
``backend="cpu_numba"``.

NVIDIA GPU (Linux)
------------------

You need Linux x86_64, Conda, and an NVIDIA GPU with a working driver reporting
CUDA support of 12.8 or newer in ``nvidia-smi``. Create an environment once:

.. code-block:: bash

   conda create -n ddtw python=3.12 pip
   conda activate ddtw

Then install and configure dDTW:

.. code-block:: bash

   python -m pip install ddtw
   ddtw setup-cuda

No repository download is needed. Setup installs CUDA 12.8, GCC/G++ 13,
PyTorch 2.11.0+cu128, Ninja and NumPy in the active environment. It replaces other
versions of these dependencies, compiles the extension and tests GPU
forward/backward execution. Allow several minutes and wait for **GPU check
passed** and **Setup complete**.

Use ``SDTW(backend="cuda_cpp")`` afterward. In future sessions, only
``conda activate ddtw`` is needed. Dependencies, build settings and compiled
extensions stay inside the environment. No system CUDA installation, ``sudo``
or manual activation hooks are needed. Package managers can use their normal
download caches.

Automatic GPU setup requires a dedicated Conda environment and refuses to modify
Conda base.
The setup accepts standard CPython 3.10–3.13; Python 3.12 is the validated default.

Checking the Installation
-------------------------

.. code-block:: bash

   ddtw setup-cuda --dry-run
   ddtw check-cuda

The dry run prints installation commands without modifying the environment;
it does not resolve dependencies or test GPU access. The check compiles/loads
the extension and tests a loss and gradient on the first visible GPU without
installing packages. ``python -m ddtw`` can replace ``ddtw`` in either command.
The repository's ``environments/README.md`` contains troubleshooting details.

Advanced Configuration
----------------------

The CUDA backend uses PyTorch's JIT extension loader. [#paszke2019]_
Setup saves the toolkit selection in ``$CONDA_PREFIX/etc/ddtw/``. dDTW uses it
when building and restores the process's compiler settings afterward. Compiled
extensions are cached in ``$CONDA_PREFIX/var/cache/ddtw/``, separated by Python
and PyTorch version. Managed builds use two compiler workers by default; set
``MAX_JOBS=1`` to reduce memory usage.

``nvidia-smi`` reports the driver's supported CUDA version, not the installed
toolkit. Setup selects the tested CUDA 12.8 stack even if the driver supports a
newer version. Both the environment's ``nvcc --version`` and
``torch.version.cuda`` should report 12.8 afterward.

Rerun ``ddtw setup-cuda`` after an interrupted installation or a PyTorch upgrade.
It reapplies the pinned dependencies and repeats the check. Failed setup steps
return a nonzero status; packages installed by previous steps remain available.
The ``ddtw[cuda]`` extra installs Ninja only and does not execute setup.

Local Development
-----------------

To work on a repository checkout, install it in editable mode:

.. code-block:: bash

   python -m pip install -e ".[test,benchmark]"
   ddtw setup-cuda
   python -m pytest test

To validate the published package, run the tests from a directory containing
``test/`` but no local ``ddtw/`` package directory, so that the checkout does
not shadow the installed package.

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
