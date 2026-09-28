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

"""Pure PyTorch reference backend for differentiable DTW.

This backend keeps the dynamic-programming recurrences in plain Python/Torch.
It is primarily useful as a readable reference implementation and test oracle.
It follows the device of the input tensors, so it can run on CPU or CUDA.
"""

import torch
from torch.autograd import Function


MINFUNC_SOFTMIN = 1
MINFUNC_SPARSEMIN = 2
MINFUNC_SMOOTHMIN = 3
MINFUNC_HARDMIN = 4


def _clear_debug_matrices(cls):
    cls.C_matrix = None
    cls.D_matrix = None
    cls.E_matrix = None
    cls.G_matrix = None
    cls.H_matrix = None
    cls.GE_matrix = None
    cls.C_start_matrix = None


def _softmin(values, gamma):
    size = values.shape[0]
    if torch.prod(torch.isinf(values)):
        value = torch.inf
        grads = torch.ones_like(values) / size
    else:
        minimum = torch.min(values)
        exp_sum = torch.sum(torch.exp(-(values - minimum) / gamma))
        value = minimum - gamma * torch.log(exp_sum)
        grads = torch.exp(-(values - minimum) / gamma) / exp_sum
    return value, grads


def _smoothmin(values, gamma):
    values = torch.clamp(values, max=1e20)
    _, soft_grads = _softmin(values, gamma)
    value = torch.sum(values * soft_grads)
    grads = soft_grads * (1 - (values - value) / gamma)
    return value, grads


def _sparsemin(values, gamma):
    size = values.shape[0]
    if torch.prod(torch.isinf(values)):
        value = torch.inf
        grads = torch.ones_like(values) / size
    else:
        projected_values = -torch.clamp(values, max=1e20) / gamma
        sorted_values = projected_values.clone()

        # The candidate vector is small, so insertion sort keeps the reference
        # implementation close to the Numba/CUDA backends.
        for i in range(1, size):
            key = sorted_values[i].clone()
            j = i - 1
            while j >= 0 and sorted_values[j] < key:
                sorted_values[j + 1] = sorted_values[j]
                j -= 1
            sorted_values[j + 1] = key

        cumulative = sorted_values[0] - 1.0
        cmax = cumulative
        rho = 1
        for i in range(1, size):
            cumulative = cumulative + sorted_values[i]
            if sorted_values[i] - cumulative / (i + 1) > 0:
                rho = i + 1
                cmax = cumulative
            else:
                break

        theta = cmax / rho
        grads = torch.clamp(projected_values - theta, min=0)
        value = -torch.sum(grads * (projected_values - 0.5 * grads) * gamma)
        value = value - gamma / 2
    return value, grads


def _hardmin(values, gamma):
    size = values.shape[0]
    if torch.prod(torch.isinf(values)):
        value = torch.inf
        grads = torch.ones_like(values) / size
    else:
        minimum_index = torch.argmin(values)
        value = values[minimum_index]
        grads = torch.zeros_like(values)
        grads[minimum_index] = 1
    return value, grads


def _minimum(values, gamma, min_func_id):
    if min_func_id == MINFUNC_SOFTMIN:
        return _softmin(values, gamma)
    if min_func_id == MINFUNC_SPARSEMIN:
        return _sparsemin(values, gamma)
    if min_func_id == MINFUNC_SMOOTHMIN:
        return _smoothmin(values, gamma)
    if min_func_id == MINFUNC_HARDMIN:
        return _hardmin(values, gamma)
    raise ValueError(f"Unsupported min function id: {min_func_id}")


def compute_dDTW_forward_torch(
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
):
    batch_size = C.shape[0]
    max_n = C.shape[1]
    max_m = C.shape[2]
    num_steps = step_sizes.shape[0]

    for b in range(batch_size):
        for n in range(max_n):
            for m in range(max_m):
                if n >= list_N[b] or m >= list_M[b]:
                    continue

                candidate_costs = torch.empty(
                    (num_steps + 1), dtype=C.dtype, device=C.device
                )
                for s in range(num_steps):
                    predecessor_n = n - step_sizes[s, 0]
                    predecessor_m = m - step_sizes[s, 1]
                    if predecessor_n >= 0 and predecessor_m >= 0:
                        candidate_costs[s] = (
                            W[b, n, m, s] * C[b, n, m]
                            + D[b, predecessor_n, predecessor_m]
                        )
                    else:
                        candidate_costs[s] = torch.inf
                candidate_costs[num_steps] = C_start[b, n, m]

                value, gradients = _minimum(candidate_costs, gamma, min_func_id)
                D[b, n, m] = value
                K[b, n, m] = gradients

                local_gradient = K[b, n, m, num_steps]
                for s in range(num_steps):
                    local_gradient = local_gradient + K[b, n, m, s] * W[b, n, m, s]
                G[b, n, m] = local_gradient

        for i in range(num_end_conditions[b]):
            n = B_end[b, i, 0]
            m = B_end[b, i, 1]
            cost_end[b, i] = D[b, n, m] + W_end[b, i] * C[b, n, m]

        value, gradients = _minimum(cost_end[b], gamma, min_func_id)
        cost_out[b] = value
        grad_end[b] = gradients

        for i in range(num_end_conditions[b]):
            n = B_end[b, i, 0]
            m = B_end[b, i, 1]
            GE[b, n, m] = grad_end[b, i]

    return cost_out, D, G, GE, K


def compute_dDTW_backward_torch(E, GE, K, list_N, list_M, step_sizes):
    batch_size = E.shape[0]
    max_n = E.shape[1]
    max_m = E.shape[2]
    num_steps = step_sizes.shape[0]

    for b in range(batch_size):
        for n in range(max_n - 1, -1, -1):
            for m in range(max_m - 1, -1, -1):
                if n >= list_N[b] or m >= list_M[b]:
                    continue

                for s in range(num_steps):
                    successor_n = n + step_sizes[s, 0]
                    successor_m = m + step_sizes[s, 1]
                    if successor_n < list_N[b] and successor_m < list_M[b]:
                        E[b, n, m] += (
                            E[b, successor_n, successor_m]
                            * K[b, successor_n, successor_m, s]
                        )

                E[b, n, m] += GE[b, n, m]
    return E


class _backend_torch(Function):
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
        batch_size, max_n, max_m = C.shape

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
            (batch_size, max_n, max_m, step_sizes.shape[0] + 1),
            device=C.device,
            dtype=dtype,
        )
        cost_out = torch.zeros(batch_size, device=C.device, dtype=dtype)

        with torch.no_grad():
            cost_out, D, G, GE, K = compute_dDTW_forward_torch(
                C.detach(),
                D.detach(),
                G.detach(),
                GE.detach(),
                K.detach(),
                W.detach(),
                C_start.detach(),
                B_end.detach(),
                W_end.detach(),
                cost_end.detach(),
                grad_end.detach(),
                cost_out.detach(),
                gamma.item(),
                int(min_function),
                list_N,
                list_M,
                step_sizes,
                num_end_conditions,
            )

        ctx.list_N = list_N
        ctx.list_M = list_M
        ctx.step_sizes = step_sizes
        ctx.store_debug = store_debug
        ctx.save_for_backward(G.detach(), K.detach(), GE.detach())

        if store_debug:
            _backend_torch.C_matrix = C.detach()
            _backend_torch.D_matrix = D.detach()
            _backend_torch.G_matrix = G.detach()
            _backend_torch.GE_matrix = GE.detach()
            _backend_torch.C_start_matrix = C_start.detach()
            _backend_torch.E_matrix = None
            _backend_torch.H_matrix = None
        else:
            _clear_debug_matrices(_backend_torch)

        return cost_out

    @staticmethod
    def backward(ctx, grad_output):
        G, K, GE = ctx.saved_tensors
        E = torch.zeros_like(G)

        with torch.no_grad():
            E = compute_dDTW_backward_torch(
                E,
                GE,
                K,
                ctx.list_N,
                ctx.list_M,
                ctx.step_sizes,
            )

        grad_C = grad_output.view(-1, 1, 1) * E * G

        if ctx.store_debug:
            _backend_torch.E_matrix = E.detach()
            _backend_torch.H_matrix = (E * G).detach()

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
