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

"""Numba-compiled CPU backend for differentiable DTW.

The dynamic-programming recurrences operate on zero-copy NumPy views of CPU
PyTorch tensors.  Each batch item is independent and is processed by one Numba
worker; cells within an item retain their dependency-preserving order.
"""

import math

import torch
from numba import njit, prange
from torch.autograd import Function


MINFUNC_SOFTMIN = 1
MINFUNC_SPARSEMIN = 2
MINFUNC_SMOOTHMIN = 3
MINFUNC_HARDMIN = 4


@njit(inline="always")
def _all_inf(values):
    for i in range(values.shape[0]):
        if not math.isinf(values[i]):
            return False
    return True


@njit(inline="always")
def _softmin(values, gamma, grads):
    size = values.shape[0]
    if _all_inf(values):
        value = math.inf
        uniform = 1.0 / size
        for i in range(size):
            grads[i] = uniform
        return value

    minimum = values[0]
    for i in range(1, size):
        if values[i] < minimum:
            minimum = values[i]

    exp_sum = 0.0
    for i in range(size):
        weight = math.exp(-(values[i] - minimum) / gamma)
        grads[i] = weight
        exp_sum += weight

    for i in range(size):
        grads[i] /= exp_sum
    return minimum - gamma * math.log(exp_sum)


@njit(inline="always")
def _smoothmin(values, gamma, grads):
    size = values.shape[0]
    for i in range(size):
        if values[i] > 1.0e20:
            values[i] = 1.0e20

    _softmin(values, gamma, grads)
    value = 0.0
    for i in range(size):
        value += values[i] * grads[i]
    for i in range(size):
        grads[i] *= 1.0 - (values[i] - value) / gamma
    return value


@njit(inline="always")
def _sparsemin(values, gamma, grads, sparsemin_sort_buffer):
    size = values.shape[0]
    if _all_inf(values):
        value = math.inf
        uniform = 1.0 / size
        for i in range(size):
            grads[i] = uniform
        return value

    for i in range(size):
        capped = min(values[i], 1.0e20)
        projected = -capped / gamma
        grads[i] = projected
        sparsemin_sort_buffer[i] = projected

    # Insertion sort in descending order.  The number of steps is normally
    # very small, so this avoids allocating or invoking a general sorter.
    for i in range(1, size):
        key = sparsemin_sort_buffer[i]
        j = i - 1
        while j >= 0 and sparsemin_sort_buffer[j] < key:
            sparsemin_sort_buffer[j + 1] = sparsemin_sort_buffer[j]
            j -= 1
        sparsemin_sort_buffer[j + 1] = key

    cumulative = sparsemin_sort_buffer[0] - 1.0
    cmax = cumulative
    rho = 1
    for i in range(1, size):
        cumulative += sparsemin_sort_buffer[i]
        if sparsemin_sort_buffer[i] - cumulative / (i + 1) > 0.0:
            rho = i + 1
            cmax = cumulative
        else:
            break
    theta = cmax / rho

    value = -gamma / 2.0
    for i in range(size):
        projected = max(grads[i] - theta, 0.0)
        grads[i] = projected
        value -= projected * (projected + theta - 0.5 * projected) * gamma
    return value


@njit(inline="always")
def _hardmin(values, grads):
    size = values.shape[0]
    if _all_inf(values):
        value = math.inf
        uniform = 1.0 / size
        for i in range(size):
            grads[i] = uniform
        return value

    minimum_index = 0
    minimum = values[0]
    for i in range(size):
        grads[i] = 0.0
        if values[i] < minimum:
            minimum = values[i]
            minimum_index = i
    grads[minimum_index] = 1.0
    return minimum


@njit(inline="always")
def _minimum(values, gamma, min_func_id, grads, sparsemin_sort_buffer):
    if min_func_id == MINFUNC_SOFTMIN:
        return _softmin(values, gamma, grads)
    if min_func_id == MINFUNC_SPARSEMIN:
        return _sparsemin(values, gamma, grads, sparsemin_sort_buffer)
    if min_func_id == MINFUNC_SMOOTHMIN:
        return _smoothmin(values, gamma, grads)
    return _hardmin(values, grads)


@njit(parallel=True, nogil=True, cache=True)
def compute_dDTW_forward_numba(
    C,
    D,
    G,
    GE,
    K,
    W,
    C_start,
    B_end,
    W_end,
    cost_end,
    grad_end,
    cost_out,
    gamma,
    min_func_id,
    list_N,
    list_M,
    step_sizes,
    num_end_conditions,
    candidate_costs_scratch,
    sparsemin_scratch,
):
    batch_size = C.shape[0]
    num_steps = step_sizes.shape[0]

    for b in prange(batch_size):
        n_limit = int(list_N[b])
        m_limit = int(list_M[b])
        candidate_costs = candidate_costs_scratch[b]
        sparsemin_sort_buffer = sparsemin_scratch[b]

        for n in range(n_limit):
            for m in range(m_limit):
                for s in range(num_steps):
                    predecessor_n = n - int(step_sizes[s, 0])
                    predecessor_m = m - int(step_sizes[s, 1])
                    if predecessor_n >= 0 and predecessor_m >= 0:
                        candidate_costs[s] = (
                            W[b, n, m, s] * C[b, n, m]
                            + D[b, predecessor_n, predecessor_m]
                        )
                    else:
                        candidate_costs[s] = math.inf
                candidate_costs[num_steps] = C_start[b, n, m]

                D[b, n, m] = _minimum(
                    candidate_costs,
                    gamma,
                    min_func_id,
                    K[b, n, m],
                    sparsemin_sort_buffer,
                )

                local_gradient = K[b, n, m, num_steps]
                for s in range(num_steps):
                    local_gradient += K[b, n, m, s] * W[b, n, m, s]
                G[b, n, m] = local_gradient

        end_count = int(num_end_conditions[b])
        for i in range(end_count):
            n = int(B_end[b, i, 0])
            m = int(B_end[b, i, 1])
            cost_end[b, i] = D[b, n, m] + W_end[b, i] * C[b, n, m]

        cost_out[b] = _minimum(
            cost_end[b], gamma, min_func_id, grad_end[b], sparsemin_sort_buffer
        )
        for i in range(end_count):
            n = int(B_end[b, i, 0])
            m = int(B_end[b, i, 1])
            GE[b, n, m] = grad_end[b, i]


@njit(parallel=True, nogil=True, cache=True)
def compute_dDTW_backward_numba(E, GE, K, list_N, list_M, step_sizes):
    batch_size = E.shape[0]
    num_steps = step_sizes.shape[0]

    for b in prange(batch_size):
        n_limit = int(list_N[b])
        m_limit = int(list_M[b])
        for n in range(n_limit - 1, -1, -1):
            for m in range(m_limit - 1, -1, -1):
                value = GE[b, n, m]
                for s in range(num_steps):
                    successor_n = n + int(step_sizes[s, 0])
                    successor_m = m + int(step_sizes[s, 1])
                    if successor_n < n_limit and successor_m < m_limit:
                        value += (
                            E[b, successor_n, successor_m]
                            * K[b, successor_n, successor_m, s]
                        )
                E[b, n, m] = value


def _numpy_view(tensor):
    if tensor.device.type != "cpu":
        raise ValueError("backend='cpu_numba' requires CPU tensors")
    return tensor.detach().numpy()


def _clear_debug_matrices(cls):
    cls.C_matrix = None
    cls.D_matrix = None
    cls.E_matrix = None
    cls.G_matrix = None
    cls.H_matrix = None
    cls.GE_matrix = None
    cls.C_start_matrix = None


class _backend_CPU_Numba(Function):
    C_matrix = None
    D_matrix = None
    E_matrix = None
    G_matrix = None
    H_matrix = None
    GE_matrix = None
    C_start_matrix = None

    @staticmethod
    def forward(
        ctx,
        C,
        min_function,
        gamma,
        step_sizes,
        W,
        list_N,
        list_M,
        B_start,
        B_end,
        num_start_conditions,
        num_end_conditions,
        W_start,
        W_end,
        store_debug,
    ):
        dtype = C.dtype
        if dtype not in (torch.float32, torch.float64):
            raise TypeError(
                "backend='cpu_numba' supports torch.float32 and torch.float64 tensors; "
                f"got {dtype}"
            )
        batch_size, max_n, max_m = C.shape
        num_directions = step_sizes.shape[0] + 1

        C_start = torch.full_like(C, torch.inf)
        for b in range(B_start.shape[0]):
            for i in range(int(num_start_conditions[b])):
                n = int(B_start[b, i, 0])
                m = int(B_start[b, i, 1])
                if n < list_N[b] and m < list_M[b]:
                    C_start[b, n, m] = C[b, n, m] * W_start[b, i]

        GE = torch.zeros_like(C)
        cost_end = torch.full(
            (batch_size, B_end.shape[1]), torch.inf, device=C.device, dtype=dtype
        )
        grad_end = torch.zeros_like(cost_end)
        D = torch.zeros_like(C)
        G = torch.zeros_like(C)
        K = torch.zeros(
            (batch_size, max_n, max_m, num_directions),
            device=C.device,
            dtype=dtype,
        )
        cost_out = torch.zeros(batch_size, device=C.device, dtype=dtype)
        candidate_costs_scratch = torch.empty(
            (batch_size, num_directions), device=C.device, dtype=dtype
        )
        sparsemin_scratch = torch.empty(
            (batch_size, max(num_directions, B_end.shape[1])),
            device=C.device,
            dtype=dtype,
        )

        with torch.no_grad():
            compute_dDTW_forward_numba(
                _numpy_view(C),
                _numpy_view(D),
                _numpy_view(G),
                _numpy_view(GE),
                _numpy_view(K),
                _numpy_view(W),
                _numpy_view(C_start),
                _numpy_view(B_end),
                _numpy_view(W_end),
                _numpy_view(cost_end),
                _numpy_view(grad_end),
                _numpy_view(cost_out),
                float(gamma.item()),
                int(min_function),
                _numpy_view(list_N),
                _numpy_view(list_M),
                _numpy_view(step_sizes),
                _numpy_view(num_end_conditions),
                _numpy_view(candidate_costs_scratch),
                _numpy_view(sparsemin_scratch),
            )

        ctx.list_N = list_N
        ctx.list_M = list_M
        ctx.step_sizes = step_sizes
        ctx.store_debug = store_debug
        ctx.save_for_backward(G.detach(), K.detach(), GE.detach())

        if store_debug:
            _backend_CPU_Numba.C_matrix = C.detach()
            _backend_CPU_Numba.D_matrix = D.detach()
            _backend_CPU_Numba.G_matrix = G.detach()
            _backend_CPU_Numba.GE_matrix = GE.detach()
            _backend_CPU_Numba.C_start_matrix = C_start.detach()
            _backend_CPU_Numba.E_matrix = None
            _backend_CPU_Numba.H_matrix = None
        else:
            _clear_debug_matrices(_backend_CPU_Numba)

        return cost_out

    @staticmethod
    def backward(ctx, grad_output):
        G, K, GE = ctx.saved_tensors
        E = torch.zeros_like(G)

        with torch.no_grad():
            compute_dDTW_backward_numba(
                _numpy_view(E),
                _numpy_view(GE),
                _numpy_view(K),
                _numpy_view(ctx.list_N),
                _numpy_view(ctx.list_M),
                _numpy_view(ctx.step_sizes),
            )

        grad_C = grad_output.view(-1, 1, 1) * E * G

        if ctx.store_debug:
            _backend_CPU_Numba.E_matrix = E.detach()
            _backend_CPU_Numba.H_matrix = (E * G).detach()

        return (
            grad_C,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
