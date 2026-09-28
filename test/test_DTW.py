import importlib
import sys
import warnings

import numpy as np
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

from ddtw import DTW


def _import_librosa_sequence():
    try:
        return importlib.import_module("librosa.sequence")
    except ImportError as exc:
        pytest.skip(f"librosa is required for this test: {exc}")
    except RuntimeError as exc:
        if "cannot cache function" not in str(exc):
            raise

    for module_name in list(sys.modules):
        if module_name == "librosa" or module_name.startswith("librosa."):
            del sys.modules[module_name]

    try:
        import numba
    except ImportError as exc:
        pytest.skip(f"librosa requires numba in this environment: {exc}")

    original_jit = numba.jit

    def jit_without_cache(*args, **kwargs):
        kwargs["cache"] = False
        return original_jit(*args, **kwargs)

    numba.jit = jit_without_cache
    try:
        return importlib.import_module("librosa.sequence")
    finally:
        numba.jit = original_jit


def _random_cost_matrix(seed, max_rows, max_cols):
    generator = torch.Generator().manual_seed(seed)
    rows = int(torch.randint(3, max_rows + 1, (1,), generator=generator))
    cols = int(torch.randint(3, max_cols + 1, (1,), generator=generator))
    cost = torch.rand(
        1,
        max_rows,
        max_cols,
        generator=generator,
        dtype=torch.float32,
        requires_grad=True,
    )
    return cost, rows, cols



@pytest.mark.parametrize("backend", ["torch", "cpu_numba", "cuda_cpp"])
def test_dtw_gradient_matches_librosa_optimal_path(backend):
    if backend == "cuda_cpp" and not torch.cuda.is_available():
        pytest.skip("backend='cuda_cpp' requires an available CUDA device")

    print("testing DTW for %s backend"%(backend))

    cost, rows, cols = _random_cost_matrix(seed=23, max_rows=8, max_cols=7)
    sequence = _import_librosa_sequence()
    accumulated_cost, warping_path = sequence.dtw(
        C=cost.detach().numpy()[0, :rows, :cols],
        step_sizes_sigma=np.array([[1, 0], [0, 1], [1, 1]]),
        weights_add=np.zeros(3),
        weights_mul=np.ones(3),
        subseq=False,
        backtrack=True,
    )


    try:
        dtw = DTW(
            backend=backend,
            normalization="none",
            dtype_float=torch.float32,
        )
    except ImportError as exc:
        pytest.skip(str(exc))

    loss = dtw(
        C=cost,
        list_N=torch.tensor([rows], dtype=torch.long),
        list_M=torch.tensor([cols], dtype=torch.long),
    )
    loss.backward()

    expected_path_gradient = torch.zeros_like(cost)
    for row, col in warping_path:
        expected_path_gradient[0, row, col] = 1

    assert torch.allclose(
        loss.cpu(),
        torch.tensor(accumulated_cost[-1, -1], dtype=loss.cpu().dtype),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.equal(cost.grad, expected_path_gradient)
