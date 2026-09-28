*d*\ DTW Toolbox
================

.. figure:: _static/figures/teaser.png
   :width: 70%
   :align: center

|
The *d*\ DTW Toolbox provides PyTorch losses for differentiable sequence alignment.
It implements a unified graph-based dynamic-programming framework for weakly
supervised sequence learning problems where target order is known but frame-level timing
is not. The toolbox is based on the accompanying paper [#zeitler2026toolbox]_ and a prior article. [#zeitler2026ctc]_

.. admonition:: References

  [1] Johannes Zeitler and Meinard Müller. *dDTW: A Unified and Efficient Toolbox for Differentiable Sequence Alignment.* Submitted 2026.

  [2] Johannes Zeitler and Meinard Müller. *A Unified Perspective on CTC and SDTW Using Differentiable DTW.* IEEE Transactions on Audio, Speech and Language Processing, 34:936-951, 2026.

In the paper, alignment losses
are represented as path-cost aggregation on a weighted directed acyclic graph
(DAG). Vertices represent possible correspondences between sequence elements,
while step sizes, edge weights, boundary conditions, local costs, and aggregation
operators determine the actual objective. Classical DTW [#muller2021]_, Soft-DTW [#cuturi2017]_, smoothDTW [#hadji2021]_,
sparseDTW [#mensch2018]_, subsequence alignment [#zeitler2026subseq]_, partial matching [#pevzner2000]_, and CTC [#graves2006]_ are recovered as
particular graph configurations. 





.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   concepts
   variants
   architecture

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   citation

.. rubric:: References


.. [#zeitler2026toolbox] J. Zeitler and M. Müller, "dDTW: A Unified and
   Efficient Toolbox for Differentiable Sequence Alignment," submitted, 2026.
.. [#zeitler2026ctc] J. Zeitler and M. Müller, "A Unified Perspective on CTC and SDTW
   Using Differentiable DTW", *IEEE Transactions on Audio, Speech and Language Processing"*,
   34:936-951, 2026.
.. [#muller2021] M. Müller, *Fundamentals of Music Processing: Using Python and
   Jupyter Notebooks*, 2nd ed. Springer, 2021.
.. [#cuturi2017] M. Cuturi and M. Blondel, "Soft-DTW: a Differentiable Loss
   Function for Time-Series," in *Proceedings of the International Conference on
   Machine Learning (ICML)*, Sydney, NSW, Australia, 2017, pp. 894-903.
.. [#hadji2021] I. Hadji, K. G. Derpanis, and A. D. Jepson, "Representation
   Learning via Global Temporal Alignment and Cycle-Consistency," in
   *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
   Recognition (CVPR)*, Virtual, 2021, pp. 11068-11077.
.. [#mensch2018] A. Mensch and M. Blondel, "Differentiable Dynamic Programming
   for Structured Prediction and Attention," in *Proceedings of the
   International Conference on Machine Learning (ICML)*, Stockholm, Sweden,
   2018, pp. 3459-3468.
.. [#zeitler2026subseq] J. Zeitler and M. Muller, "Subsequence SDTW:
   Differentiable Alignment with Flexible Boundary Conditions," in
   *Proceedings of the IEEE International Conference on Acoustics, Speech, and
   Signal Processing (ICASSP)*, Barcelona, Spain, 2026.
.. [#pevzner2000] P. A. Pevzner, *Computational Molecular Biology: An
   Algorithmic Approach*. MIT Press, 2000.
.. [#graves2006] A. Graves, S. Fernandez, F. J. Gomez, and J. Schmidhuber,
   "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data
   with Recurrent Neural Networks," in *Proceedings of the International
   Conference on Machine Learning (ICML)*, Pittsburgh, Pennsylvania, USA, 2006,
   pp. 369-376.
