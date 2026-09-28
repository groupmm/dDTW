##################################################################################
#                                dDTW Toolbox                                    #
##################################################################################
#                                                                                #
# Authors: Johannes Zeitler and Meinard Müller, 2026                             #
#                                                                                #
# If you use this toolbox, please cite the accompanying paper:                   #
# Johannes Zeitler and Meinard Müller. dDTW: A Unified and Efficient Toolbox for #
#  Differentiable Sequence Alignment. Submitted 2026.                            #
#                                                                                #
# Code based on:                                                                 #
# Mehran Maghoumi et al. "DeepNAG: Deep Non-Adversarial Gesture Generation".     #
#  International Conference on Intelligent User Interfaces, 2021.                #
#  https://github.com/Maghoumi/pytorch-softdtw-cuda/                             #
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
import torch.cuda

from .cost_function import get_cost_function
from .backend.backend_torch import _backend_torch
from .backend.backend_cuda_cpp import _backend_CUDA_CPP

try:
    from .backend.backend_cpu_numba import _backend_CPU_Numba
except ImportError:
    _backend_CPU_Numba = None


SUPPORTED_BACKENDS = ("auto", "torch", "cpu_numba", "cuda_cpp")
MIN_FUNCTION_IDS = {
    "softmin": 1,
    "sparsemin": 2,
    "smoothmin": 3,
    "hardmin": 4,
}


def _cuda_device_available(cuda_device):
    if not torch.cuda.is_available():
        return False
    if cuda_device is None:
        return True

    try:
        device = torch.device(cuda_device)
    except (TypeError, RuntimeError):
        return False

    if device.type != "cuda":
        return False
    if device.index is not None and device.index >= torch.cuda.device_count():
        return False
    return True


def _resolve_backend(backend, cuda_device):
    if backend == "auto":
        if _cuda_device_available(cuda_device):
            return "cuda_cpp"
        if _backend_CPU_Numba is not None:
            return "cpu_numba"
        return "torch"

    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            "Unsupported backend: %s. Choose among %s."
            % (backend, SUPPORTED_BACKENDS)
        )
    if backend == "cpu_numba" and _backend_CPU_Numba is None:
        raise ImportError("backend='cpu_numba' requires numba to be installed")
    if backend == "cuda_cpp" and not _cuda_device_available(cuda_device):
        raise RuntimeError("backend='cuda_cpp' requires an available CUDA device")
    if backend == "torch" and cuda_device is not None:
        device = torch.device(cuda_device)
        if device.type == "cuda" and not _cuda_device_available(cuda_device):
            raise RuntimeError("backend='torch' with a CUDA device requires CUDA")
    return backend


def _backend_device(backend, cuda_device):
    if backend == "cuda_cpp":
        return "cuda" if cuda_device is None else cuda_device
    if backend == "torch" and cuda_device is not None:
        return cuda_device
    return "cpu"


# dDTW loss class
class dDTW(torch.nn.Module):
    def __init__(self, 
                 cost_function = "MSE",
                 min_function = "softmin",
                 gamma = 1.0,
                 step_sizes = [[1,0], [0,1], [1,1]],
                 global_step_weights = [1.0, 1.0, 1.0],
                 normalization = "N",
                 backend="auto",
                 dtype_float = torch.float32,
                 cuda_device=None,
                 store_debug=False
                 ):
        """Initialize the general dDTW loss function.

        See [1] for the graph formulation.

        [1] Johannes Zeitler and Meinard Müller. A Unified Perspective on CTC
        and Soft-DTW Using Differentiable DTW. IEEE Transactions on Audio,
        Speech and Language Processing, vol. 34, pages 936-951, 2026.

        Parameters
        ----------
        cost_function : str or callable, optional
            Local cost function used when ``X`` and ``Y`` are passed to
            :meth:`forward`. Built-in strings are ``"MSE"``, ``"BCE"``, and
            ``"CTC"``. A callable must return a cost tensor with shape
            ``(B, N, M)``. Default: ``"MSE"``.
        min_function : str, optional
            Recursive minimum or differentiable approximation. Choose among
            ``"softmin"``, ``"sparsemin"``, ``"smoothmin"``, and
            ``"hardmin"``. Default: ``"softmin"``.
        gamma : float, optional
            Temperature parameter used by differentiable minimum functions.
            Default: ``1.0``.
        step_sizes : list of list of int, optional
            Alignment step sizes ``[dn, dm]``. Each step points from the
            current cell ``(n, m)`` to predecessor ``(n-dn, m-dm)``.
            Default: ``[[1, 0], [0, 1], [1, 1]]``.
        global_step_weights : list of float, optional
            Local cost weights associated with ``step_sizes``. Must contain one
            scalar per step. Default: ``[1.0, 1.0, 1.0]``.
        normalization : str, optional
            Normalization applied to each batch loss before averaging. Choose
            among ``"N"``, ``"M"``, ``"NM"``, ``"ctc"``, and ``"none"``.
            Default: ``"N"``.
        backend : str, optional
            Backend to use. Choose among ``"auto"``, ``"torch"``,
            ``"cpu_numba"``, and ``"cuda_cpp"``. ``"auto"`` tries CUDA first,
            then Numba CPU, then pure PyTorch. Default: ``"auto"``.
        dtype_float : torch.dtype, optional
            Floating-point dtype for internal tensors. The CUDA C++ backend
            currently requires ``torch.float32``. Default: ``torch.float32``.
        cuda_device : str or torch.device, optional
            CUDA device to use, for example ``"cuda:0"``. Default: ``None``.
        store_debug : bool, optional
            If ``True``, retain intermediate matrices on the backend class for
            inspection after forward/backward. Default: ``False``.
        """

        super(dDTW, self).__init__()

        self.dtype_float=dtype_float
        self.store_debug = store_debug

        self.backend = _resolve_backend(backend, cuda_device)
        self.requested_backend = backend
        self.device = _backend_device(self.backend, cuda_device)

        if self.backend == "torch":
            self.core = _backend_torch
            self.compute_dDTW = _backend_torch.apply
        elif self.backend == "cpu_numba":
            self.core = _backend_CPU_Numba
            self.compute_dDTW = _backend_CPU_Numba.apply
        else:
            self.core = _backend_CUDA_CPP
            self.compute_dDTW = _backend_CUDA_CPP.apply

        if normalization not in ["M", "N", "NM", "ctc", "none"]:
            raise ValueError(
                "Unsupported normalization: %s. Choose among %s."
                % (normalization, ("M", "N", "NM", "ctc", "none"))
            )
        self.normalization = normalization

        self.gamma = torch.tensor([gamma], device=self.device, dtype=self.dtype_float, requires_grad=False)

        self.step_sizes = torch.tensor(step_sizes, device=self.device, dtype=torch.int16, requires_grad=False)
        if self.step_sizes.dim() != 2 or self.step_sizes.shape[1] != 2:
            raise ValueError("step_sizes must have shape (num_steps, 2)")

        self.global_step_weights = torch.tensor(global_step_weights, device=self.device, dtype=self.dtype_float, requires_grad=False)
        if self.global_step_weights.dim() != 1 or self.global_step_weights.shape[0] != self.step_sizes.shape[0]:
            raise ValueError("global_step_weights must contain one scalar weight per step size")

        if min_function not in MIN_FUNCTION_IDS:
            raise ValueError(
                "Unsupported min_function: %s. Choose among %s."
                % (min_function, tuple(MIN_FUNCTION_IDS))
            )
        self.min_function = MIN_FUNCTION_IDS[min_function]


        if isinstance(cost_function, str):
            self.cost_function = get_cost_function(cost_function, self.backend == "cuda_cpp")
        else:
            self.cost_function = cost_function        
        return None

    def _as_fixed_tensor(self, data, dtype):
        if isinstance(data, torch.Tensor):
            return data.detach().to(device=self.device, dtype=dtype)
        return torch.as_tensor(data, device=self.device, dtype=dtype)

    def _prepare_boundary_conditions(self, boundary_conditions, penalties, num_conditions, default_boundary, default_penalty, B):
        if num_conditions is not None:
            num_conditions = self._as_fixed_tensor(num_conditions, torch.int32)
            if num_conditions.shape != torch.Size([B]):
                raise ValueError("num_conditions must have shape (B,)")

        if boundary_conditions is None:
            boundary_tensor = default_boundary.to(device=self.device, dtype=torch.int16)
            if num_conditions is None:
                num_conditions = torch.ones(B, device=self.device, dtype=torch.int32, requires_grad=False)
        elif isinstance(boundary_conditions, torch.Tensor):
            if boundary_conditions.dim() != 3:
                raise ValueError("boundary_conditions tensor must have shape (B, num_conditions, 2)")
            if boundary_conditions.shape[0] != B:
                raise ValueError("boundary_conditions batch dimension must match the cost matrix batch size")
            if boundary_conditions.shape[2] != 2:
                raise ValueError("boundary_conditions must store two indices per condition")
            boundary_tensor = boundary_conditions.detach().to(device=self.device, dtype=torch.int16)
            if num_conditions is None:
                if penalties is not None and not isinstance(penalties, torch.Tensor):
                    num_conditions = torch.as_tensor([len(batch_penalties) for batch_penalties in penalties],
                                                     device=self.device, dtype=torch.int32)
                else:
                    num_conditions = torch.full((B,), boundary_tensor.shape[1], device=self.device, dtype=torch.int32, requires_grad=False)
        else:
            if len(boundary_conditions) != B:
                raise ValueError("boundary_conditions must contain one condition list per batch item")
            num_conditions_list = [len(batch_conditions) for batch_conditions in boundary_conditions]
            if num_conditions is not None:
                expected_num_conditions = torch.as_tensor(num_conditions_list, dtype=torch.int32)
                if not torch.equal(num_conditions.cpu(), expected_num_conditions):
                    raise ValueError("num_conditions does not match the provided boundary_conditions")
            max_conditions = max(num_conditions_list)
            if max_conditions <= 0:
                raise ValueError("Each batch must provide at least one boundary condition")

            padded_conditions = []
            for batch_conditions in boundary_conditions:
                if isinstance(batch_conditions, torch.Tensor):
                    batch_conditions = batch_conditions.detach().cpu().tolist()
                batch_conditions = [list(condition) for condition in batch_conditions]
                batch_conditions.extend([[0, 0] for _ in range(max_conditions - len(batch_conditions))])
                padded_conditions.append(batch_conditions)

            boundary_tensor = torch.as_tensor(padded_conditions, device=self.device, dtype=torch.int16)
            num_conditions = torch.as_tensor(num_conditions_list, device=self.device, dtype=torch.int32)

        max_conditions = boundary_tensor.shape[1]
        if int(torch.max(num_conditions)) > max_conditions:
            raise ValueError("num_conditions cannot exceed the padded number of boundary conditions")
        if penalties is None:
            weights = torch.full((B, max_conditions), default_penalty, device=self.device, dtype=self.dtype_float, requires_grad=False)
        elif isinstance(penalties, torch.Tensor):
            if penalties.shape != torch.Size([B, max_conditions]):
                raise ValueError("penalties must have shape (B, max_conditions)")
            weights = penalties.detach().to(device=self.device, dtype=self.dtype_float)
        else:
            if len(penalties) != B:
                raise ValueError("penalties must contain one penalty list per batch item")
            padded_penalties = []
            for batch_penalties, count in zip(penalties, num_conditions.detach().cpu().tolist()):
                if isinstance(batch_penalties, torch.Tensor):
                    batch_penalties = batch_penalties.detach().cpu().tolist()
                if len(batch_penalties) != count:
                    raise ValueError("Each penalty list must match the corresponding number of boundary conditions")
                padded_batch = list(batch_penalties)
                padded_batch.extend([default_penalty for _ in range(max_conditions - len(padded_batch))])
                padded_penalties.append(padded_batch)
            weights = torch.as_tensor(padded_penalties, device=self.device, dtype=self.dtype_float)

        return boundary_tensor, num_conditions, weights

    
    def forward(self, X=None, Y=None, C=None, B_start=None, B_end=None, list_N=None, list_M=None, local_step_weights=None, 
                start_penalty=None, end_penalty=None, num_start_conditions=None, num_end_conditions=None):
        """Compute the dDTW loss.

        Pass either ``X`` and ``Y`` or a precomputed cost matrix ``C``. If
        ``X`` and ``Y`` are provided, ``self.cost_function`` computes ``C``.

        Parameters
        ----------
        X : torch.Tensor, optional
            First input sequence with shape ``(B, N, D)``. Usually the model
            predictions.
        Y : torch.Tensor, optional
            Second input sequence with shape ``(B, M, D)``. Usually the target
            or reference sequence.
        C : torch.Tensor, optional
            Precomputed local cost matrix with shape ``(B, N, M)``.
        B_start : list or torch.Tensor, optional
            Start boundary conditions. For each batch item, stores one or more
            zero-based ``[n, m]`` cells. Tensor form must have shape
            ``(B, max_start_conditions, 2)``. If ``None``, defaults to
            ``[[0, 0]]`` for each batch item.
        B_end : list or torch.Tensor, optional
            End boundary conditions. For each batch item, stores one or more
            zero-based ``[n, m]`` cells. Tensor form must have shape
            ``(B, max_end_conditions, 2)``. If ``None``, defaults to
            ``[[list_N[b] - 1, list_M[b] - 1]]``.
        list_N : list or torch.Tensor, optional
            Active lengths along the ``X``/row axis with shape ``(B,)``. If
            ``None``, all batch items use the full padded length ``N``.
        list_M : list or torch.Tensor, optional
            Active lengths along the ``Y``/column axis with shape ``(B,)``. If
            ``None``, all batch items use the full padded length ``M``.
        local_step_weights : torch.Tensor, optional
            Cell-wise step weights with shape ``(B, N, M, S)``, where ``S`` is
            the number of configured steps. If ``None``,
            ``global_step_weights`` are broadcast to all cells.
        start_penalty : list or torch.Tensor, optional
            Multiplicative local-cost weights for start boundary conditions.
            Tensor form must have shape ``(B, max_start_conditions)``. If
            ``None``, defaults to ``1`` for every start condition.
        end_penalty : list or torch.Tensor, optional
            Multiplicative local-cost weights for end boundary conditions.
            Tensor form must have shape ``(B, max_end_conditions)``. If
            ``None``, defaults to ``0`` for every end condition.
        num_start_conditions : list or torch.Tensor, optional
            Number of valid start conditions for each batch item when
            ``B_start`` is padded. Shape ``(B,)``.
        num_end_conditions : list or torch.Tensor, optional
            Number of valid end conditions for each batch item when ``B_end``
            is padded. Shape ``(B,)``.

        Returns
        -------
        torch.Tensor
            Scalar batch-mean dDTW loss after the configured normalization.
        """

        # cost matrix #####################################################################################
        if X is not None:
            X = X.to(device=self.device)
        if Y is not None:
            Y = Y.to(device=self.device)
        if C is not None:
            C = C.to(device=self.device)

        if (X is not None) and (Y is not None):
            if (len(X.shape) == 3) and (len(Y.shape) == 3):
                X_ = X[:,:,:,None]
                Y_ = Y[:,:,:,None]
                #print("expanded dim.")
                C = self.cost_function(X_, Y_)

            else:
                C = self.cost_function(X, Y)
        else:
            if C is None:
                raise ValueError("Pass either X and Y, or a precomputed cost matrix C.")
            
        B = C.shape[0]
        N = C.shape[1]
        M = C.shape[2]

        ##################################################################################################

        # sequence lengths ################################################################################
        if list_N is None:
            self.list_N = torch.full((B,), N, device=self.device, dtype=torch.int16, requires_grad=False)
        else:
            self.list_N = self._as_fixed_tensor(list_N, torch.int16)

        if list_M is None:
            self.list_M = torch.full((B,), M, device=self.device, dtype=torch.int16, requires_grad=False)
        else:
            self.list_M = self._as_fixed_tensor(list_M, torch.int16)
        ####################################################################################################

        
        # boundary conditions ##############################################################################
        default_B_start = torch.zeros((B, 1, 2), device=self.device, dtype=torch.int16, requires_grad=False)
        default_B_end = torch.stack((self.list_N - 1, self.list_M - 1), dim=-1).unsqueeze(1)

        self.B_start, self.num_start_conditions, self.W_start = self._prepare_boundary_conditions(
            B_start, start_penalty, num_start_conditions, default_B_start, 1.0, B)
        self.B_end, self.num_end_conditions, self.W_end = self._prepare_boundary_conditions(
            B_end, end_penalty, num_end_conditions, default_B_end, 0.0, B)
        

        ########################################################################################################


        # step weights #########################################################################################
        if local_step_weights is not None:
            self.step_weights = self._as_fixed_tensor(local_step_weights, self.dtype_float)
        else:
            self.step_weights = self.global_step_weights.expand(B,N,M,-1)
        ########################################################################################################

        dDTW_cost = self.compute_dDTW(C, self.min_function, self.gamma, self.step_sizes, self.step_weights, 
                                          self.list_N, self.list_M, self.B_start, self.B_end, self.num_start_conditions, self.num_end_conditions,
                                         self.W_start, self.W_end, self.store_debug)

        if self.normalization == "N":
            dDTW_cost_mean = torch.mean(dDTW_cost/self.list_N)
        elif self.normalization == "M":
            dDTW_cost_mean = torch.mean(dDTW_cost/self.list_M)
        elif self.normalization == "NM":
            dDTW_cost_mean = torch.mean(dDTW_cost/self.list_N/self.list_M)
        elif self.normalization == "ctc":
            dDTW_cost_mean = torch.mean(dDTW_cost/( (self.list_M-1)/2))
        else:            
            dDTW_cost_mean = torch.mean(dDTW_cost)

        return dDTW_cost_mean        
