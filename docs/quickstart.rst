Quickstart
==========

Soft-DTW Loss
-------------

Use ``ddtw.SDTW`` when both sequences are represented by feature vectors and should
be aligned with the standard DTW step pattern. [#cuturi2017]_

.. code-block:: python

   import torch
   from ddtw import SDTW

   X = torch.randn(8, 120, 16, requires_grad=True)
   Y = torch.randn(8, 80, 16)

   loss_fn = SDTW(cost_function="MSE",
                  gamma=0.1,
                  normalization="N")

   loss = loss_fn(X=X, Y=Y)
   loss.backward()

CTC-Style Loss
--------------

``ddtw.CTC`` implements a CTC parameterization inside the dDTW framework.
[#graves2006]_ [#zeitler2026ctc]_ Pass
log-probabilities as predictions, so the induced CTC local costs are 
negative log-probabilities. The target sequence can be passed as integer label
indices with shape ``(B, M)``.

.. code-block:: python

   import torch
   from ddtw import CTC

   logits = torch.randn(4, 200, 13, requires_grad=True)
   log_probs = torch.log_softmax(logits, dim=-1)
   targets = torch.tensor([[1, 4, 7, 2],
                           [3, 3, 5, 9],
                           [2, 8, 1, 6],
                           [4, 2, 2, 5],])

   loss_fn = CTC(blank_index=0, 
                 gamma=1.0)

   loss = loss_fn(X=log_probs, Y=targets)
   loss.backward()

Precomputed Cost Matrices
-------------------------

``dDTW``, ``DTW``, ``SDTW``, ``smoothDTW``, ``sparseDTW``, ``subSDTW``, and
``partial_matching`` can consume a precomputed cost matrix ``C`` with shape
``(B, N, M)``. This is useful when the local cost is computed by another model
or algorithmic component. ``CTC`` instead requires ``X`` and ``Y`` so it can
construct the blank-expanded target sequence and allowed transitions.

.. code-block:: python

   import torch
   from ddtw import SDTW

   C = torch.rand(4, 60, 45, requires_grad=True)

   loss_fn = SDTW(cost_function="MSE",
                  gamma=0.1,
                  normalization="N")

   loss = loss_fn(C=C)
   loss.backward()

Variable-Length Batches
-----------------------

Pass ``list_N`` and ``list_M`` to ignore padded regions in a batch. Each length
vector must contain one entry per batch item.

.. code-block:: python

   import torch
   from ddtw import SDTW

   X = torch.randn(4, 120, 16, requires_grad=True)
   Y = torch.randn(4, 80, 16)
   list_N = torch.tensor([120, 96, 88, 117])
   list_M = torch.tensor([80, 72, 60, 79])

   loss_fn = SDTW(cost_function="MSE",
                  gamma=0.1,
                  normalization="N")

   loss = loss_fn(X=X, Y=Y, list_N=list_N, list_M=list_M)
   loss.backward()



Backend Choice
--------------

The default ``auto`` backend selects the fastest practical implementation
available in this order: ``cuda_cpp``, ``cpu_numba``, then ``torch``. For
development and tests, use ``torch`` when readability matters or ``cpu_numba``
for faster CPU execution. For production training on a configured CUDA machine,
use ``cuda_cpp`` explicitly. The CUDA C++ backend compiles the extension lazily
on first use.

The backend string maps to a concrete implementation module:

.. list-table::
   :header-rows: 1

   * - Backend
     - Module
     - Typical use
   * - ``auto``
     - dispatcher
     - highest-priority available backend
   * - ``torch``
     - ``backend/backend_torch.py``
     - readable PyTorch reference for CPU or CUDA
   * - ``cpu_numba``
     - ``backend/backend_cpu_numba.py``
     - practical CPU execution
   * - ``cuda_cpp``
     - ``backend/backend_cuda_cpp.py``
     - optimized CUDA extension backend

.. rubric:: References


.. [#cuturi2017] M. Cuturi and M. Blondel, "Soft-DTW: a Differentiable Loss
   Function for Time-Series," in *Proceedings of the International Conference on
   Machine Learning (ICML)*, Sydney, NSW, Australia, 2017, pp. 894-903.
.. [#graves2006] A. Graves, S. Fernandez, F. J. Gomez, and J. Schmidhuber,
   "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data
   with Recurrent Neural Networks," in *Proceedings of the International
   Conference on Machine Learning (ICML)*, Pittsburgh, Pennsylvania, USA, 2006,
   pp. 369-376.
.. [#zeitler2026ctc] J. Zeitler and M. Müller, "A Unified Perspective on CTC
   and SDTW Using Differentiable DTW," *IEEE Transactions on Audio, Speech and
   Language Processing*, vol. 34, pp. 936-951, 2026.
