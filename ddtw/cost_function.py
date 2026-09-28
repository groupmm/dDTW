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

"""Local cost functions for dDTW frontends.

CPU-oriented backends use PyTorch implementations. The ``cuda_cpp`` backend
uses the compiled CUDA extension for the same built-in local costs.
"""

import torch
from torch.autograd import Function

from .backend.cpp_extension import get_extension


class CUDACost(Function):
    @staticmethod
    def forward(ctx, X, Y, local_cost_function="MSE"):
        ext = get_extension()
        ctx.local_cost_function = local_cost_function

        if local_cost_function == "MSE":
            C = ext.mse_cost_forward(X, Y)
            ctx.save_for_backward(X.detach(), Y.detach())
        elif local_cost_function == "CTC":
            C = ext.ctc_cost_forward(X, Y)
            ctx.save_for_backward(X.detach(), Y.detach())
        elif local_cost_function == "BCE":
            X_ = torch.clamp(X, min=1e-20, max=1 - 1e-20)
            logX = torch.log(X_)
            log1minusX = torch.log(1 - X_)
            C = ext.bce_cost_forward(logX, log1minusX, Y)
            ctx.save_for_backward(X_.detach(), logX.detach(), log1minusX.detach(), Y.detach())
        else:
            raise ValueError("Unsupported local cost function: %s" % local_cost_function)

        return C

    @staticmethod
    def backward(ctx, grad_output):
        ext = get_extension()

        if ctx.local_cost_function == "MSE":
            X, Y = ctx.saved_tensors
            grad_X, grad_Y = ext.mse_cost_backward(X, Y, grad_output)
        elif ctx.local_cost_function == "CTC":
            X, Y = ctx.saved_tensors
            grad_X, grad_Y = ext.ctc_cost_backward(X, Y, grad_output)
        elif ctx.local_cost_function == "BCE":
            X, logX, log1minusX, Y = ctx.saved_tensors
            grad_X, grad_Y = ext.bce_cost_backward(X, logX, log1minusX, Y, grad_output)
        else:
            raise ValueError("Unsupported local cost function: %s" % ctx.local_cost_function)

        return grad_X, grad_Y, None


def mse_cost_torch(X, Y):
    N = X.size(1)
    M = Y.size(1)
    D = X.size(2)
    Q = X.size(3)
    X_ = X.unsqueeze(2).expand(-1, N, M, D, Q)
    Y_ = Y.unsqueeze(1).expand(-1, N, M, D, Q)
    return torch.pow(X_ - Y_, 2).sum((3, 4))


def bce_cost_torch(X, Y):
    n = X.size(1)
    m = Y.size(1)
    d = X.size(2)
    q = X.size(3)
    X = X.unsqueeze(2).expand(-1, n, m, d, q)
    Y = Y.unsqueeze(1).expand(-1, n, m, d, q)
    return torch.nn.functional.binary_cross_entropy(X.double(), Y.double(), reduction="none").sum((3, 4)).to(X.dtype)


def get_cost_function(local_cost="MSE", use_cuda=True):
    if use_cuda:
        cost = CUDACost.apply
        return lambda x, y: cost(x, y, local_cost)

    if local_cost == "MSE":
        return mse_cost_torch
    if local_cost == "BCE":
        return bce_cost_torch
    if local_cost == "CTC":
        return lambda x, y: -(x.unsqueeze(2) * y.unsqueeze(1)).sum((3, 4))
    raise ValueError("Unsupported local cost function: %s" % local_cost)
