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

"""CUDA C++ extension backend for differentiable DTW.

The minimum-function implementations live in the CUDA extension sources under
``ddtw/backend/csrc``. This Python wrapper forwards the configured minimum
function id to the extension.
"""

import torch
from torch.autograd import Function

from .cpp_extension import get_extension


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


def _min_func_id(min_function):
    return int(min_function)


def compute_ctc_initialization(
    X,
    targets,
    list_N,
    list_M,
    global_step_weights,
    blank_penalty,
    blank_index,
):
    return get_extension().ctc_initialize(
        X,
        targets,
        list_N,
        list_M,
        global_step_weights,
        blank_penalty,
        blank_index,
    )


def compute_subseq_initialization(
    list_N,
    list_M,
    global_step_weights,
    N_max,
    M_max,
    sub_X,
    sub_Y,
    compensate_subseq,
):
    return get_extension().subseq_initialize(
        list_N,
        list_M,
        global_step_weights,
        N_max,
        M_max,
        sub_X,
        sub_Y,
        compensate_subseq,
    )


class _backend_CUDA_CPP(Function):
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
        ext = get_extension()
        gamma_value = float(gamma.item())

        cost_out, G, K, GE, D, C_start = ext.ddtw_forward(
            C,
            _min_func_id(min_function),
            gamma_value,
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
        )

        ctx.list_N = list_N
        ctx.list_M = list_M
        ctx.step_sizes = step_sizes
        ctx.store_debug = store_debug
        ctx.save_for_backward(G.detach(), K.detach(), GE.detach())

        if store_debug:
            _backend_CUDA_CPP.C_matrix = C.detach()
            _backend_CUDA_CPP.D_matrix = D.detach()
            _backend_CUDA_CPP.G_matrix = G.detach()
            _backend_CUDA_CPP.GE_matrix = GE.detach()
            _backend_CUDA_CPP.C_start_matrix = C_start.detach()
            _backend_CUDA_CPP.E_matrix = None
            _backend_CUDA_CPP.H_matrix = None
        else:
            _clear_debug_matrices(_backend_CUDA_CPP)

        return cost_out

    @staticmethod
    def backward(ctx, grad_output):
        ext = get_extension()
        G, K, GE = ctx.saved_tensors
        grad_C, E = ext.ddtw_backward(
            G,
            K,
            GE,
            grad_output,
            ctx.list_N,
            ctx.list_M,
            ctx.step_sizes,
        )

        if ctx.store_debug:
            _backend_CUDA_CPP.E_matrix = E.detach()
            _backend_CUDA_CPP.H_matrix = (E * G).detach()

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
