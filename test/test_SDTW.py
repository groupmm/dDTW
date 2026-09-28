import importlib.util
import warnings
from pathlib import Path

import pytest
import torch

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"torch\.utils\.cpp_extension",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"pkg_resources(\.|$)",
)

from ddtw import SDTW


def _import_reference_soft_dtw():
    reference_path = (
        Path(__file__).resolve().parent
        / "reference_implementations"
        / "soft_dtw_cuda.py"
    )
    spec = importlib.util.spec_from_file_location(
        "soft_dtw_cuda_reference",
        reference_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SoftDTW


def _random_cost_matrix(seed, batch_size, rows, cols):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(
        batch_size,
        rows,
        cols,
        generator=generator,
        dtype=torch.float32,
        requires_grad=True,
    )


@pytest.mark.parametrize("backend", ["torch", "cpu_numba", "cuda_cpp"])
def test_sdtw_matches_reference_soft_dtw_loss_and_gradient(backend):
    if backend == "cuda_cpp" and not torch.cuda.is_available():
        pytest.skip("backend='cuda_cpp' requires an available CUDA device")

    print("testing SDTW for %s backend" % (backend))

    batch_size = 2
    rows = 6
    cols = 5
    gamma = 0.7

    cost = _random_cost_matrix(
        seed=31,
        batch_size=batch_size,
        rows=rows,
        cols=cols,
    )
    reference_cost = cost.detach().clone().requires_grad_()

    try:
        sdtw = SDTW(
            gamma=gamma,
            backend=backend,
            normalization="none",
            dtype_float=torch.float32,
        )
    except ImportError as exc:
        pytest.skip(str(exc))

    ddtw_loss = sdtw(C=cost)

    SoftDTW = _import_reference_soft_dtw()
    reference_sdtw = SoftDTW(
        use_cuda=False,
        gamma=gamma,
        normalize=False,
        dist_func=lambda X, Y: reference_cost,
    )
    X = torch.empty(batch_size, rows, 1)
    Y = torch.empty(batch_size, cols, 1)
    reference_loss = reference_sdtw(X, Y).mean()

    assert torch.allclose(
        ddtw_loss.cpu(),
        reference_loss.cpu(),
        atol=1e-5,
        rtol=1e-5,
    )

    ddtw_loss.backward()
    reference_loss.backward()

    assert torch.allclose(
        cost.grad,
        reference_cost.grad,
        atol=1e-5,
        rtol=1e-5,
    )
