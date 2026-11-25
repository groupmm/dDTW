Benchmark
=========

These experiments compare the CUDA implementation of ``SDTW`` in dDTW with
Maghoumi's Soft-DTW implementation, and ``CTC`` in dDTW with PyTorch CTC.
They vary sequence lengths, batch size, and feature dimensionality or number
of classes. Fixed parameters are shown in each plot.

Runtime plots show forward, backward, and total runtime in milliseconds.
Lines represent medians; shaded regions show one standard deviation across
trials. Solid lines denote dDTW and dashed lines denote the baseline.
Both axes use logarithmic scales. Each curve ends at its last recorded
measurement.

Results for the GPU discussed in the paper (NVIDIA RTX A5500) are shown below. Expand another GPU model
to view its results. Memory usage and comparisons across GPUs are grouped
at the bottom of the page.

NVIDIA RTX A5500
----------------

SDTW
~~~~

.. figure:: _static/figures/benchmarks/runtime_SDTW_longSeg_rtx_a5500.png
   :width: 100%
   :align: center
   :alt: SDTW sequence length runtimes on NVIDIA RTX A5500.

.. figure:: _static/figures/benchmarks/runtime_SDTW_batch_rtx_a5500.png
   :width: 100%
   :align: center
   :alt: SDTW batch size runtimes on NVIDIA RTX A5500.

.. figure:: _static/figures/benchmarks/runtime_SDTW_dim_rtx_a5500.png
   :width: 100%
   :align: center
   :alt: SDTW feature dimensionality runtimes on NVIDIA RTX A5500.

CTC
~~~

.. figure:: _static/figures/benchmarks/runtime_CTC_inputLen_rtx_a5500.png
   :width: 100%
   :align: center
   :alt: CTC input sequence length runtimes on NVIDIA RTX A5500.

.. figure:: _static/figures/benchmarks/runtime_CTC_targetLen_rtx_a5500.png
   :width: 100%
   :align: center
   :alt: CTC target sequence length runtimes on NVIDIA RTX A5500.

.. figure:: _static/figures/benchmarks/runtime_CTC_batch_rtx_a5500.png
   :width: 100%
   :align: center
   :alt: CTC batch size runtimes on NVIDIA RTX A5500.

.. figure:: _static/figures/benchmarks/runtime_CTC_dim_rtx_a5500.png
   :width: 100%
   :align: center
   :alt: CTC number of classes runtimes on NVIDIA RTX A5500.

Other GPU Models
----------------

.. raw:: html

   <details class="benchmark-gpu">
   <summary>NVIDIA GeForce RTX 3090</summary>

.. rubric:: SDTW

.. figure:: _static/figures/benchmarks/runtime_SDTW_longSeg_rtx_3090.png
   :width: 100%
   :align: center
   :alt: SDTW sequence length runtimes on NVIDIA GeForce RTX 3090.

.. figure:: _static/figures/benchmarks/runtime_SDTW_batch_rtx_3090.png
   :width: 100%
   :align: center
   :alt: SDTW batch size runtimes on NVIDIA GeForce RTX 3090.

.. figure:: _static/figures/benchmarks/runtime_SDTW_dim_rtx_3090.png
   :width: 100%
   :align: center
   :alt: SDTW feature dimensionality runtimes on NVIDIA GeForce RTX 3090.

.. rubric:: CTC

.. figure:: _static/figures/benchmarks/runtime_CTC_inputLen_rtx_3090.png
   :width: 100%
   :align: center
   :alt: CTC input sequence length runtimes on NVIDIA GeForce RTX 3090.

.. figure:: _static/figures/benchmarks/runtime_CTC_targetLen_rtx_3090.png
   :width: 100%
   :align: center
   :alt: CTC target sequence length runtimes on NVIDIA GeForce RTX 3090.

.. figure:: _static/figures/benchmarks/runtime_CTC_batch_rtx_3090.png
   :width: 100%
   :align: center
   :alt: CTC batch size runtimes on NVIDIA GeForce RTX 3090.

.. figure:: _static/figures/benchmarks/runtime_CTC_dim_rtx_3090.png
   :width: 100%
   :align: center
   :alt: CTC number of classes runtimes on NVIDIA GeForce RTX 3090.

.. raw:: html

   </details>

.. raw:: html

   <details class="benchmark-gpu">
   <summary>NVIDIA GeForce RTX 4090</summary>

.. rubric:: SDTW

.. figure:: _static/figures/benchmarks/runtime_SDTW_longSeg_rtx_4090.png
   :width: 100%
   :align: center
   :alt: SDTW sequence length runtimes on NVIDIA GeForce RTX 4090.

.. figure:: _static/figures/benchmarks/runtime_SDTW_batch_rtx_4090.png
   :width: 100%
   :align: center
   :alt: SDTW batch size runtimes on NVIDIA GeForce RTX 4090.

.. figure:: _static/figures/benchmarks/runtime_SDTW_dim_rtx_4090.png
   :width: 100%
   :align: center
   :alt: SDTW feature dimensionality runtimes on NVIDIA GeForce RTX 4090.

.. rubric:: CTC

.. figure:: _static/figures/benchmarks/runtime_CTC_inputLen_rtx_4090.png
   :width: 100%
   :align: center
   :alt: CTC input sequence length runtimes on NVIDIA GeForce RTX 4090.

.. figure:: _static/figures/benchmarks/runtime_CTC_targetLen_rtx_4090.png
   :width: 100%
   :align: center
   :alt: CTC target sequence length runtimes on NVIDIA GeForce RTX 4090.

.. figure:: _static/figures/benchmarks/runtime_CTC_batch_rtx_4090.png
   :width: 100%
   :align: center
   :alt: CTC batch size runtimes on NVIDIA GeForce RTX 4090.

.. figure:: _static/figures/benchmarks/runtime_CTC_dim_rtx_4090.png
   :width: 100%
   :align: center
   :alt: CTC number of classes runtimes on NVIDIA GeForce RTX 4090.

.. raw:: html

   </details>

.. raw:: html

   <details class="benchmark-gpu">
   <summary>NVIDIA GeForce RTX 5090</summary>

.. rubric:: SDTW

.. figure:: _static/figures/benchmarks/runtime_SDTW_longSeg_rtx_5090.png
   :width: 100%
   :align: center
   :alt: SDTW sequence length runtimes on NVIDIA GeForce RTX 5090.

.. figure:: _static/figures/benchmarks/runtime_SDTW_batch_rtx_5090.png
   :width: 100%
   :align: center
   :alt: SDTW batch size runtimes on NVIDIA GeForce RTX 5090.

.. figure:: _static/figures/benchmarks/runtime_SDTW_dim_rtx_5090.png
   :width: 100%
   :align: center
   :alt: SDTW feature dimensionality runtimes on NVIDIA GeForce RTX 5090.

.. rubric:: CTC

.. figure:: _static/figures/benchmarks/runtime_CTC_inputLen_rtx_5090.png
   :width: 100%
   :align: center
   :alt: CTC input sequence length runtimes on NVIDIA GeForce RTX 5090.

.. figure:: _static/figures/benchmarks/runtime_CTC_targetLen_rtx_5090.png
   :width: 100%
   :align: center
   :alt: CTC target sequence length runtimes on NVIDIA GeForce RTX 5090.

.. figure:: _static/figures/benchmarks/runtime_CTC_batch_rtx_5090.png
   :width: 100%
   :align: center
   :alt: CTC batch size runtimes on NVIDIA GeForce RTX 5090.

.. figure:: _static/figures/benchmarks/runtime_CTC_dim_rtx_5090.png
   :width: 100%
   :align: center
   :alt: CTC number of classes runtimes on NVIDIA GeForce RTX 5090.

.. raw:: html

   </details>

.. raw:: html

   <details class="benchmark-gpu">
   <summary>NVIDIA RTX PRO 6000 Blackwell</summary>

.. rubric:: SDTW

.. figure:: _static/figures/benchmarks/runtime_SDTW_longSeg_rtx_pro_6000_blackwell.png
   :width: 100%
   :align: center
   :alt: SDTW sequence length runtimes on NVIDIA RTX PRO 6000 Blackwell.

.. figure:: _static/figures/benchmarks/runtime_SDTW_batch_rtx_pro_6000_blackwell.png
   :width: 100%
   :align: center
   :alt: SDTW batch size runtimes on NVIDIA RTX PRO 6000 Blackwell.

.. figure:: _static/figures/benchmarks/runtime_SDTW_dim_rtx_pro_6000_blackwell.png
   :width: 100%
   :align: center
   :alt: SDTW feature dimensionality runtimes on NVIDIA RTX PRO 6000 Blackwell.

.. rubric:: CTC

.. figure:: _static/figures/benchmarks/runtime_CTC_inputLen_rtx_pro_6000_blackwell.png
   :width: 100%
   :align: center
   :alt: CTC input sequence length runtimes on NVIDIA RTX PRO 6000 Blackwell.

.. figure:: _static/figures/benchmarks/runtime_CTC_targetLen_rtx_pro_6000_blackwell.png
   :width: 100%
   :align: center
   :alt: CTC target sequence length runtimes on NVIDIA RTX PRO 6000 Blackwell.

.. figure:: _static/figures/benchmarks/runtime_CTC_batch_rtx_pro_6000_blackwell.png
   :width: 100%
   :align: center
   :alt: CTC batch size runtimes on NVIDIA RTX PRO 6000 Blackwell.

.. figure:: _static/figures/benchmarks/runtime_CTC_dim_rtx_pro_6000_blackwell.png
   :width: 100%
   :align: center
   :alt: CTC number of classes runtimes on NVIDIA RTX PRO 6000 Blackwell.

.. raw:: html

   </details>

Grouped Comparisons
-------------------

Memory Usage
~~~~~~~~~~~~

Peak allocated CUDA memory is measured in MiB and includes live input
tensors. Colors identify GPU models; solid lines denote dDTW and dashed
lines denote the baseline. Curves overlap when memory usage is the same.

.. figure:: _static/figures/benchmarks/memory_by_experiment.png
   :width: 100%
   :align: center
   :alt: Grouped SDTW and CTC memory usage by experiment and GPU model.

Total Runtime Across GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~

The following comparison shows median total runtime for the same parameter
sweeps on each GPU model. Colors identify GPUs and line styles distinguish
dDTW from the baseline. Differences reflect the recorded systems, including
their software configurations.

.. figure:: _static/figures/benchmarks/total_runtime_by_gpu.png
   :width: 100%
   :align: center
   :alt: Grouped SDTW and CTC total runtimes across GPU models.
