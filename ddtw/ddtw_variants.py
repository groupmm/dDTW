##################################################################################
#                                dDTW Toolbox                                    #
##################################################################################
#                                                                                #
# Authors: Johannes Zeitler and Meinard Müller, 2026                             #
#                                                                                #
# If you use this toolbox, please cite the accompanying paper:                   #
# Johannes Zeitler and Meinard Müller. dDTW: A Unified and Efficient Toolbox for #
#  Differentiable Sequence Alignment. Submitted 2026.                            #
##################################################################################


##################################################################################
# MIT License                                                                    #
#                                                                                #
# Copyright 2026 Johannes Zeitler and Meinard Müller                             #
#                                                                                #
# Permission is hereby granted, free of charge, to any person obtaining a copy   #
# of this software and associated documentation files (the "Software"), to deal  #
# in the Software without restriction, including without limitation the rights   #
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      #
# copies of the Software, and to permit persons to whom the Software is          #
# furnished to do so, subject to the following conditions:                       #
#                                                                                #
# The above copyright notice and this permission notice shall be included in all #
# copies or substantial portions of the Software.                                #
#                                                                                #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  #
# SOFTWARE.                                                                      #
##################################################################################

import torch
from .ddtw import dDTW
from .backend.backend_cuda_cpp import compute_ctc_initialization, compute_subseq_initialization
from itertools import product

####################### classical softDTW (SDTW) #################################
class SDTW(dDTW):
    def __init__(self, 
                 cost_function = "MSE",
                 gamma = 1.0,
                 step_sizes = [[1,0], [0,1], [1,1]],
                 global_step_weights = [1.0, 1.0, 1.0],
                 normalization = "N",
                 backend="auto",
                 dtype_float = torch.float32,
                 cuda_device=None,
                 store_debug=False
                ):
        """Initialize SDTW loss function, see [1, 2].

        [1] Marco Cuturi and Mathieu Blondel. Soft-DTW: A Differentiable Loss
        Function for Time-Series. In Proceedings of the International
        Conference on Neural Information Processing Systems (NIPS), vol. 2,
        pages 2292-2300, 2013.

        [2] Johannes Zeitler and Meinard Müller. A Unified Perspective on CTC
        and Soft-DTW Using Differentiable DTW. IEEE Transactions on Audio,
        Speech and Language Processing, vol. 34, pages 936-951, 2026.

        Parameters
        ----------
        cost_function : str
            Local cost function for pair-wise comparison of sequence elements.
            Choose among ("MSE", "BCE", "CTC"). Default: "MSE".
        
        gamma : float
            Softmin temperature hyperparameter. Default: 1.0

        step_sizes : list
            Alignment step sizes in [n,m] direction, given as list of tuples.
            Default: [[1,0], [0,1], [1,1]]

        global_step_weights: list
            Step weights associated to the step sizes. Default:
            [1.0, 1.0, 1.0]

        normalization : str
            Normalization of SDTW cost. Choose among ("M", "N", "NM", "ctc",
            "none"). "N": divide by N. "M": divide by M. "NM": divide by
            (N*M). "ctc": divide by (M-1)/2. "none": no normalization.
            Default: "N"

        backend : str
            Backend to use. Choose among ("auto", "torch", "cpu_numba",
            "cuda_cpp"). Default: "auto"

        dtype_float: torch.dtype
            Number format for internal computations. Default: torch.float32

        cuda_device : str or torch.device, optional
            CUDA device to use, for example ``"cuda:0"``. Default: ``None``.

        store_debug : bool
            Whether to retain intermediate backend matrices for inspection.
            Default: False.
        """        
        super().__init__(cost_function=cost_function,
                         min_function="softmin",
                         gamma=gamma,
                         step_sizes=step_sizes,
                         global_step_weights=global_step_weights,
                         normalization=normalization,
                         backend=backend,
                         dtype_float=dtype_float,
                         cuda_device=cuda_device,
                         store_debug=store_debug)

    def forward(self, X=None, Y=None, C=None, list_N=None, list_M=None):
        """
            Compute the SDTW loss.

            Parameters
            ----------
            X : torch.tensor [shape=(B, N, D)]
                Input sequence, usually the DNN predictions.

            Y : torch.tensor [shape=(B, M, D)]
                Input sequence, usually the weak targets.

            C : torch.tensor [shape=(B, N, M)]
                Pre-computed local cost matrix C.

            list_N : torch.tensor [shape=(B)]
                Sequence lengths of X (<= N) of the individual batch elements. If None, defaults to [N, N, ..., N]

            list_M : torch.tensor [shape=(B)]
                Sequence lengths of Y (<= M) of the individual batch elements. If None, defaults to [M, M, ..., M]            

            Returns
            -------
            torch.Tensor
                Scalar batch-mean SDTW loss.
        """
        return super().forward(X=X,
                               Y=Y,
                               C=C,
                               list_N=list_N,
                               list_M=list_M)
###################################################################################

############################ hard DTW #############################################
class DTW(dDTW): 
    def __init__(self, 
                 cost_function = "MSE",
                 step_sizes = [[1,0], [0,1], [1,1]],
                 global_step_weights = [1.0, 1.0, 1.0],
                 normalization = "N",
                 backend="auto",
                 dtype_float = torch.float32,
                 cuda_device=None,
                 store_debug=False
                ):
        """ Initialize DTW loss function, see [3].

            [3]. Meinard Müller. Fundamentals of Music Processing - Using Python and Jupyter Notebooks. Springer Verlag, 2nd edition, 2021.

            Parameters
            ----------
            cost_function : str
                Local cost function for pair-wise comparison of sequence elements. Choose among ("MSE", "BCE", "CTC"). Default: "MSE".

            step_sizes : list
                Alignment step sizes in [n,m] direction, given as list of tuples. Default: [[1,0], [0,1], [1,1]]

            global_step_weights: list
                Step weights associated to the step sizes. Default: [1.0, 1.0, 1.0]

            normalization : str
                Normalization of SDTW cost. Choose among ("M", "N", "NM", "ctc", "none"). "N": divide by N. "M": divide by M. "NM": divide by (N*M). "ctc": divide by (M-1)/2. "none": no normalization. Default: "N"

            backend : str
                Backend to use. Choose among ("auto", "torch", "cpu_numba", "cuda_cpp"). Default: "auto"

            dtype_float: torch.dtype
                Number format for internal computations. Default: torch.float32

            cuda_device : str or torch.device, optional
                CUDA device to use, for example ``"cuda:0"``. Default: ``None``.

            store_debug : bool
                Whether to retain intermediate backend matrices for inspection.
                Default: False.
        """  
        super().__init__(cost_function=cost_function,
                         min_function="hardmin",
                         step_sizes=step_sizes,
                         global_step_weights=global_step_weights,
                         normalization=normalization,
                         backend=backend,
                         dtype_float=dtype_float,
                         cuda_device=cuda_device,
                         store_debug=store_debug)

    def forward(self, X=None, Y=None, C=None, list_N=None, list_M=None):
        """
            Compute the DTW loss.

            Parameters
            ----------
            X : torch.tensor [shape=(B, N, D)]
                Input sequence, usually the DNN predictions.

            Y : torch.tensor [shape=(B, M, D)]
                Input sequence, usually the weak targets.

            C : torch.tensor [shape=(B, N, M)]
                Pre-computed local cost matrix C.

            list_N : torch.tensor [shape=(B)]
                Sequence lengths of X (<= N) of the individual batch elements. If None, defaults to [N, N, ..., N]

            list_M : torch.tensor [shape=(B)]
                Sequence lengths of Y (<= M) of the individual batch elements. If None, defaults to [M, M, ..., M]            

            Returns
            -------
            torch.Tensor
                Scalar batch-mean DTW loss. With a precomputed ``C``, gradients
                with respect to ``C`` mark the selected hard warping path.
        """
        return super().forward(X=X,
                               Y=Y,
                               C=C,
                               list_N=list_N,
                               list_M=list_M)
###################################################################################

################################# smooth DTW ######################################
class smoothDTW(dDTW):
    def __init__(self, 
                 cost_function = "MSE",
                 gamma = 1.0,
                 step_sizes = [[1,0], [0,1], [1,1]],
                 global_step_weights = [1.0, 1.0, 1.0],
                 normalization = "N",
                 backend="auto",
                 dtype_float = torch.float32,
                 cuda_device=None,
                 store_debug=False
                ):
        """ Initialize smoothDTW loss function, see [4].

            [4]. Isma Hadji, K. Derpanis, and A. Jepson. Representation learning via global temporal alignment and cycle-consistency. In IEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11068-11077, 2021.

            Parameters
            ----------
            cost_function : str
                Local cost function for pair-wise comparison of sequence elements. Choose among ("MSE", "BCE", "CTC"). Default: "MSE".

            gamma : float
                Softmin temperature hyperparameter. Default: 1.0

            step_sizes : list
                Alignment step sizes in [n,m] direction, given as list of tuples. Default: [[1,0], [0,1], [1,1]]

            global_step_weights: list
                Step weights associated to the step sizes. Default: [1.0, 1.0, 1.0]

            normalization : str
                Normalization of SDTW cost. Choose among ("M", "N", "NM", "ctc", "none"). "N": divide by N. "M": divide by M. "NM": divide by (N*M). "ctc": divide by (M-1)/2. "none": no normalization. Default: "N"

            backend : str
                Backend to use. Choose among ("auto", "torch", "cpu_numba", "cuda_cpp"). Default: "auto"

            dtype_float: torch.dtype
                Number format for internal computations. Default: torch.float32

            cuda_device : str or torch.device, optional
                CUDA device to use, for example ``"cuda:0"``. Default: ``None``.

            store_debug : bool
                Whether to retain intermediate backend matrices for inspection.
                Default: False.
        """  
        super().__init__(cost_function=cost_function,
                         min_function="smoothmin",
                         gamma=gamma,
                         step_sizes=step_sizes,
                         global_step_weights=global_step_weights,
                         normalization=normalization,
                         backend=backend,
                         dtype_float=dtype_float,
                         cuda_device=cuda_device,
                         store_debug=store_debug)

    def forward(self, X=None, Y=None, C=None, list_N=None, list_M=None):
        """
            Compute the smoothDTW loss.

            Parameters
            ----------
            X : torch.tensor [shape=(B, N, D)]
                Input sequence, usually the DNN predictions.

            Y : torch.tensor [shape=(B, M, D)]
                Input sequence, usually the weak targets.

            C : torch.tensor [shape=(B, N, M)]
                Pre-computed local cost matrix C.

            list_N : torch.tensor [shape=(B)]
                Sequence lengths of X (<= N) of the individual batch elements. If None, defaults to [N, N, ..., N]

            list_M : torch.tensor [shape=(B)]
                Sequence lengths of Y (<= M) of the individual batch elements. If None, defaults to [M, M, ..., M]            

            Returns
            -------
            torch.Tensor
                Scalar batch-mean smoothDTW loss.
        """
        return super().forward(X=X,
                               Y=Y,
                               C=C,
                               list_N=list_N,
                               list_M=list_M)
###################################################################################

########################## sparse DTW (sparseDTW) #################################
class sparseDTW(dDTW):
    def __init__(self, 
                 cost_function = "MSE",
                 gamma = 1.0,
                 step_sizes = [[1,0], [0,1], [1,1]],
                 global_step_weights = [1.0, 1.0, 1.0],
                 normalization = "N",
                 backend="auto",
                 dtype_float = torch.float32,
                 cuda_device=None,
                 store_debug=False
                ):
        """ Initialize sparseDTW loss function, see [5].

            [5] Arthur Mensch and Mathieu Blondel. Differentiable Dynamic Programming for Structured Prediction and Attention. In Proceedings of the International Converence on Machine Learning (ICML), pages 3459-3468, Stockholm, Sweden, 2018.

            Parameters
            ----------
            cost_function : str
                Local cost function for pair-wise comparison of sequence elements. Choose among ("MSE", "BCE", "CTC"). Default: "MSE".

            gamma : float
                Sparsemin temperature hyperparameter. Default: 1.0

            step_sizes : list
                Alignment step sizes in [n,m] direction, given as list of tuples. Default: [[1,0], [0,1], [1,1]]

            global_step_weights: list
                Step weights associated to the step sizes. Default: [1.0, 1.0, 1.0]

            normalization : str
                Normalization of SDTW cost. Choose among ("M", "N", "NM", "ctc", "none"). "N": divide by N. "M": divide by M. "NM": divide by (N*M). "ctc": divide by (M-1)/2. "none": no normalization. Default: "N"

            backend : str
                Backend to use. Choose among ("auto", "torch", "cpu_numba", "cuda_cpp"). Default: "auto"

            dtype_float: torch.dtype
                Number format for internal computations. Default: torch.float32

            cuda_device : str or torch.device, optional
                CUDA device to use, for example ``"cuda:0"``. Default: ``None``.

            store_debug : bool
                Whether to retain intermediate backend matrices for inspection.
                Default: False.
        """
        super().__init__(cost_function=cost_function,
                         min_function="sparsemin",
                         gamma=gamma,
                         step_sizes=step_sizes,
                         global_step_weights=global_step_weights,
                         normalization=normalization,
                         backend=backend,
                         dtype_float=dtype_float,
                         cuda_device=cuda_device,
                         store_debug=store_debug)

    def forward(self, X=None, Y=None, C=None, list_N=None, list_M=None):
        """
            Compute the sparseDTW loss.

            Parameters
            ----------
            X : torch.tensor [shape=(B, N, D)]
                Input sequence, usually the DNN predictions.

            Y : torch.tensor [shape=(B, M, D)]
                Input sequence, usually the weak targets.

            C : torch.tensor [shape=(B, N, M)]
                Pre-computed local cost matrix C.

            list_N : torch.tensor [shape=(B)]
                Sequence lengths of X (<= N) of the individual batch elements. If None, defaults to [N, N, ..., N]

            list_M : torch.tensor [shape=(B)]
                Sequence lengths of Y (<= M) of the individual batch elements. If None, defaults to [M, M, ..., M]            

            Returns
            -------
            torch.Tensor
                Scalar batch-mean sparseDTW loss.
        """
        return super().forward(X=X,
                               Y=Y,
                               C=C,
                               list_N=list_N,
                               list_M=list_M)
###################################################################################

########################### subsequence SDTW ######################################
class subSDTW(dDTW):
    def __init__(self, 
                 cost_function = "MSE",
                 min_function="softmin",
                 gamma = 1.0,
                 step_sizes = [[1,0], [0,1], [1,1]],
                 global_step_weights = [1.0, 1.0, 1.0],
                 normalization = "N",
                 backend="auto",
                 dtype_float = torch.float32,
                 cuda_device=None,
                 store_debug=False,
                 sub_X=True, # whether to do sub-sequence along the X-direction (crop predictions)
                 sub_Y=True, # whether to do sub-sequence along the Y-direction (crop targets)
                 compensate_subseq=True # whether to compensate for shorter sequences
                ):
        """ Initialize subsequence SDTW loss function, see [7].

            [7] Johannes Zeitler and Meinard Müller. Subsequence Soft Dynamic Time Warping. In Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (CASSP), Barcelona, Spain, 2026.

            Parameters
            ----------
            cost_function : str
                Local cost function for pair-wise comparison of sequence elements. Choose among ("MSE", "BCE", "CTC"). Default: "MSE".

            min_function : str
                Minimum function or approximation thereof. Choose among ("softmin", "sparsemin", "smoothmin", "hardmin"). Default: "softmin".

            gamma : float
                Min. function temperature hyperparameter. Default: 1.0

            step_sizes : list
                Alignment step sizes in [n,m] direction, given as list of tuples. Default: [[1,0], [0,1], [1,1]]

            global_step_weights: list
                Step weights associated to the step sizes. Default: [1.0, 1.0, 1.0]

            normalization : str
                Normalization of SDTW cost. Choose among ("M", "N", "NM", "ctc", "none"). "N": divide by N. "M": divide by M. "NM": divide by (N*M). "ctc": divide by (M-1)/2. "none": no normalization. Default: "N"

            backend : str
                Backend to use. Choose among ("auto", "torch", "cpu_numba", "cuda_cpp"). Default: "auto"

            dtype_float: torch.dtype
                Number format for internal computations. Default: torch.float32

            cuda_device : str or torch.device, optional
                CUDA device to use, for example ``"cuda:0"``. Default: ``None``.

            store_debug : bool
                Whether to retain intermediate backend matrices for inspection.
                Default: False.

            sub_X : bool
                Whether to allow subsequence starts and ends along the
                X-direction (row axis). Default: True.

            sub_Y : bool
                Whether to allow subsequence starts and ends along the
                Y-direction (column axis). Default: True.

            compensate_subseq : bool
                Whether to add start/end penalties for skipped prefixes or
                suffixes. Default: True.
        """
        super().__init__(cost_function=cost_function,
                         min_function=min_function,
                         gamma=gamma,
                         step_sizes=step_sizes,
                         global_step_weights=global_step_weights,
                         normalization=normalization,
                         backend=backend,
                         dtype_float=dtype_float,
                         cuda_device=cuda_device,
                         store_debug=store_debug)

        self.sub_X = sub_X
        self.sub_Y = sub_Y
        self.compensate_subseq = compensate_subseq

    def forward(self, X=None, Y=None, C=None, list_N=None, list_M=None):
        """
            Compute the subsequence SDTW loss.

            Parameters
            ----------
            X : torch.tensor [shape=(B, N, D)]
                Input sequence, usually the DNN predictions.

            Y : torch.tensor [shape=(B, M, D)]
                Input sequence, usually the weak targets.

            C : torch.tensor [shape=(B, N, M)]
                Pre-computed local cost matrix C.

            list_N : torch.tensor [shape=(B)]
                Sequence lengths of X (<= N) of the individual batch elements. If None, defaults to [N, N, ..., N]

            list_M : torch.tensor [shape=(B)]
                Sequence lengths of Y (<= M) of the individual batch elements. If None, defaults to [M, M, ..., M]            

            Returns
            -------
            torch.Tensor
                Scalar batch-mean subsequence SDTW loss.
        """
        if (X is not None) and (Y is not None):
            B = X.shape[0]
            N_max = X.shape[1]
            M_max = Y.shape[1]
        else:
            B = C.shape[0]
            N_max = C.shape[1]
            M_max = C.shape[2]

        if self.backend == "cuda_cpp":
            list_N = (torch.full((B,), N_max, device=self.device, dtype=torch.int64)
                      if list_N is None else
                      torch.as_tensor(list_N, device=self.device, dtype=torch.int64))
            list_M = (torch.full((B,), M_max, device=self.device, dtype=torch.int64)
                      if list_M is None else
                      torch.as_tensor(list_M, device=self.device, dtype=torch.int64))
            B_start, B_end, start_penalty, end_penalty, num_conditions = compute_subseq_initialization(
                list_N,
                list_M,
                self.global_step_weights,
                N_max,
                M_max,
                self.sub_X,
                self.sub_Y,
                self.compensate_subseq,
            )
        else:
            B_start = [[] for _ in range(B)]
            B_end = [[] for _ in range(B)]
            start_penalty = [[] for _ in range(B)]
            end_penalty = [[] for _ in range(B)]

            if list_N is None:
                list_N = [N_max for _ in range(B)]
            if list_M is None:
                list_M = [M_max for _ in range(B)]

            for b in range(B):
                N = list_N[b]
                M = list_M[b]

                B_start[b].append([0,0])
                start_penalty[b].append(1)

                B_end[b].append([N-1,M-1])
                end_penalty[b].append(0)

                if self.sub_X:
                    for n in range(1,N-1):
                        B_start[b].append([n, 0])
                        B_end[b].append([n, M-1])

                        if self.compensate_subseq:
                            start_penalty[b].append(1 + n*self.global_step_weights[0])
                            end_penalty[b].append( (N-1 - n)*self.global_step_weights[0])
                        else:
                            start_penalty[b].append(1)
                            end_penalty[b].append(0)

                if self.sub_Y:
                    for m in range(1,M-1):
                        B_start[b].append([0, m])
                        B_end[b].append([N-1,m])

                        if self.compensate_subseq:
                            start_penalty[b].append(1 + m*self.global_step_weights[1])
                            end_penalty[b].append( (M-1-m)*self.global_step_weights[1])
                        else:
                            start_penalty[b].append(1)
                            end_penalty[b].append(0)

            num_conditions = None
        
        return super().forward(X=X,
                               Y=Y,
                               C=C,
                               list_N=list_N,
                               list_M=list_M,
                               B_start=B_start,
                               B_end=B_end,
                               start_penalty=start_penalty,
                               end_penalty=end_penalty,
                               num_start_conditions=num_conditions,
                               num_end_conditions=num_conditions)
###################################################################################

################################## CTC ############################################
class CTC(dDTW):
    # the algorithm assumes step sizes [(1,0), (1,1), (1,2)]
    def __init__(self, 
                 blank_penalty_weight = 1.0,
                 blank_index=0,
                 gamma = 1.0,
                 global_step_weights = [1.0, 1.0, 1.0],
                 backend="auto",
                 dtype_float = torch.float32,
                 cuda_device=None,
                 store_debug=False
                ):
        """ Initialize CTC loss function, parameterized within the dDTW framework, see [8, 2].

            [2] Johannes Zeitler and Meinard Müller. A Unified Perspective on CTC and Soft-DTW Using Differentiable DTW. IEEE Transactions on Audio, Speech and Language Processing, vol. 34, pages 936-951, 2026.
            
            [8] Alex Graves et al. Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks. In Proceedings of the International Conference on Machine Learning (ICML), pages 369-376, Pittsburgh, Pennsylvania, USA, 2006.

            Parameters
            ----------
            blank_penalty_weight : float
                Penalty for alignment of the blank symbol. Default: 1.0 (no penalty).

            blank_index : int
                Class index of the blank symbol. Default: 0.

            gamma : float
                Softmin temperature hyperparameter. Default: 1.0

            global_step_weights: list
                Step weights associated to the CTC step sizes ``[[1, 0],
                [1, 1], [1, 2]]``. Default: [1.0, 1.0, 1.0].

            backend : str
                Backend to use. Choose among ("auto", "torch", "cpu_numba", "cuda_cpp"). Default: "auto"

            dtype_float: torch.dtype
                Number format for internal computations. Default: torch.float32

            cuda_device : str or torch.device, optional
                CUDA device to use, for example ``"cuda:0"``. Default: ``None``.

            store_debug : bool
                Whether to retain intermediate backend matrices for inspection.
                Default: False.

            Notes
            -----
            The CTC variant fixes ``cost_function="CTC"``,
            ``min_function="softmin"``, ``step_sizes=[[1, 0], [1, 1],
            [1, 2]]``, and ``normalization="ctc"``. The normalization divides
            each batch item by its unexpanded target length before averaging.
        """
        super().__init__(cost_function="CTC",
                         min_function="softmin",
                         gamma=gamma,
                         step_sizes=[[1,0], [1,1], [1,2]],
                         global_step_weights=global_step_weights,
                         normalization="ctc",
                         backend=backend,
                         dtype_float=dtype_float,
                         cuda_device=cuda_device,
                         store_debug=store_debug)

        self.blankP = blank_penalty_weight
        self.blank_index = blank_index

    def forward(self, X=None, Y=None, list_N=None, list_M=None):
        """
            Compute the CTC loss in the dDTW graph formulation.

            Parameters
            ----------
            X : torch.tensor [shape=(B, N, D)]
                Input log-probabilities. The CTC local cost is the 
                negative log-probability of the active target or blank state.

            Y : torch.tensor [shape=(B, M)]
                Integer target-label indices before blank expansion. Labels
                should use ``blank_index`` only for padding beyond ``list_M``.

            list_N : torch.tensor [shape=(B)]
                Sequence lengths of X (<= N) of the individual batch elements. If None, defaults to [N, N, ..., N]

            list_M : torch.tensor [shape=(B)]
                Target-label lengths before CTC blank expansion. If None,
                defaults to [M, M, ..., M].

            Returns
            -------
            torch.Tensor
                Scalar batch-mean CTC loss.
        """
        X = X.to(device=self.device)
        Y = Y.to(device=self.device)

        if Y.dim() == 2:
            CTC_targets = True
        else:
            CTC_targets = False

        B = X.shape[0]
        N_max = X.shape[1]
        M_max = Y.shape[1]
        M_e = 2*M_max + 1
        device = torch.device(self.device)
        dtype = X.dtype
        
        D = X.shape[2]

        if self.backend == "cuda_cpp":
            list_N_tensor = (torch.full((B,), N_max, device=device, dtype=torch.int64)
                             if list_N is None else
                             torch.as_tensor(list_N, device=device, dtype=torch.int64))
            list_M_tensor = (torch.full((B,), M_max, device=device, dtype=torch.int64)
                             if list_M is None else
                             torch.as_tensor(list_M, device=device, dtype=torch.int64))
            W, Y_e, list_M_e, B_start, B_end = compute_ctc_initialization(
                X,
                Y.to(device=device, dtype=torch.int64),
                list_N_tensor,
                list_M_tensor,
                self.global_step_weights,
                self.blankP,
                self.blank_index,
            )
            list_N = list_N_tensor
        else:
            if list_N is None:
                list_N = [N_max for _ in range(B)]
            if list_M is None:
                list_M = [M_max for _ in range(B)]

            Y_e = torch.zeros((B, M_e, D), device=device, dtype=dtype)
            W = torch.zeros((B, N_max, M_e, 3), device=device, dtype=dtype)
            for i_w, w in enumerate(self.global_step_weights):
                W[:,:,1::2,i_w] = w # transition into an acutal target
                W[:,:,0::2,i_w] = self.blankP # transition into blank

            B_start = []
            B_end = []
            list_M_e = []
            for b in range(B):
                list_M_e.append(2*list_M[b]+1)
                B_start.append([[0,0], [0,1]])
                B_end.append([[list_N[b]-1, 2*list_M[b]+1-1], [list_N[b]-1, 2*list_M[b]+1-2]])

                last_tgt = None
                for m_c in range(list_M[b]):
                    Y_e[b,2*m_c,self.blank_index] = 1
                    Y_e[b,2*m_c+1, Y[b,m_c]] = 1
                    # skipping blank is never allowed
                    W[b,:,2*m_c,-1] = 1e20
                    tgt = Y[b,m_c]
                    if tgt == last_tgt:
                        # skipping identical targets is not allowed
                        W[b,:,2*m_c+1,-1] = 1e20
                        # but we must allow to go through a blank symbol with a (1,1) step
                        W[b,:,2*m_c, 1] = self.global_step_weights[1]
                    last_tgt = tgt

                # skipping the last blank is also not allowed
                W[b,:,2*list_M[b],-1] = 1e20
                Y_e[b,2*list_M[b],self.blank_index] = 1
        
        return super().forward(X=X,
                               Y=Y_e,
                               list_N=list_N,
                               list_M=list_M_e,
                               B_start=B_start,
                               B_end=B_end,
                               local_step_weights=W)
###################################################################################

############################ partial matching #####################################
class partial_matching(dDTW):
    def __init__(self, 
                 cost_function = "CTC",
                 min_function="hardmin",
                 gamma = 1.0,
                 normalization = "none",
                 backend="auto",
                 dtype_float = torch.float32,
                 cuda_device=None,
                 store_debug=False
                ):
        """ Initialize partial matching loss function, parameterized within the dDTW framework, see [9, 2].

            [2] Johannes Zeitler and Meinard Müller. A Unified Perspective on CTC and Soft-DTW Using Differentiable DTW. IEEE Transactions on Audio, Speech and Language Processing, vol. 34, pages 936-951, 2026.
            
            [9] Pavel A. Pevzner. Computational Molecular Biology: An Algorithmic Approach. MIT Press, 2000.

            Parameters
            ----------
            cost_function : str
                Local cost function used when ``X`` and ``Y`` are supplied.
                Default: "CTC".

            min_function : str
                Minimum function or differentiable approximation thereof. Default: "hardmin".

            gamma : float
                Temperature parameter for differentiable minimum functions.
                Default: 1.0.

            normalization : str
                Normalization of PM cost. Choose among ("M", "N", "NM",
                "ctc", "none"). "N": divide by N. "M": divide by M. "NM":
                divide by (N*M). "ctc": divide by (M-1)/2. "none": no
                normalization. Default: "none".

            backend : str
                Backend to use. Choose among ("auto", "torch", "cpu_numba", "cuda_cpp"). Default: "auto"

            dtype_float: torch.dtype
                Number format for internal computations. Default: torch.float32

            cuda_device : str or torch.device, optional
                CUDA device to use, for example ``"cuda:0"``. Default: ``None``.

            store_debug : bool
                Whether to retain intermediate backend matrices for inspection.
                Default: False.

            Notes
            -----
            This variant fixes ``step_sizes=[[1, 0], [0, 1], [1, 1]]`` and
            ``global_step_weights=[0.0, 0.0, 1.0]``. Horizontal and vertical
            moves therefore do not accumulate local cost; only diagonal matches
            do.
        """
        super().__init__(cost_function=cost_function,
                         min_function=min_function,
                         gamma=gamma,
                         step_sizes=[[1,0], [0,1], [1,1]],
                         global_step_weights=[0., 0., 1.],
                         normalization=normalization,
                         backend=backend,
                         dtype_float=dtype_float,
                         cuda_device=cuda_device,
                         store_debug=store_debug)

    def forward(self, X=None, Y=None, C=None, list_N=None, list_M=None):
        """
            Compute the partial matching loss.

            Parameters
            ----------
            X : torch.tensor [shape=(B, N, D)]
                Input sequence, usually the DNN predictions.

            Y : torch.tensor [shape=(B, M, D)]
                Input sequence, usually the weak targets.

            C : torch.tensor [shape=(B, N, M)]
                Pre-computed local cost matrix C. To compare against a
                score-maximizing partial matching reference, pass ``C=-S`` for
                score matrix ``S``.

            list_N : torch.tensor [shape=(B)]
                Sequence lengths of X (<= N) of the individual batch elements. If None, defaults to [N, N, ..., N]

            list_M : torch.tensor [shape=(B)]
                Sequence lengths of Y (<= M) of the individual batch elements. If None, defaults to [M, M, ..., M]            

            Returns
            -------
            torch.Tensor
                Scalar batch-mean partial matching loss.
        """
        if (X is not None) and (Y is not None):
            B = X.shape[0]
            N_max = X.shape[1]
            M_max = Y.shape[1]
        else:
            B = C.shape[0]
            N_max = C.shape[1]
            M_max = C.shape[2]

        if list_N is None:
            list_N = [N_max for _ in range(B)]
        if list_M is None:
            list_M = [M_max for _ in range(B)]

        # all cells of the cost matrix
        I = []
        for b in range(B):
            I.append([[n,m] for n,m in product(range(list_N[b]), range(list_M[b]))])          
        
        return super().forward(X=X,
                               Y=Y,
                               C=C,
                               list_N=list_N,
                               list_M=list_M,
                               B_start=I,
                               B_end=I)
###################################################################################
