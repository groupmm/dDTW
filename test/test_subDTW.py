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

from ddtw import subSDTW


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


def _subsequence_cost_matrix():
    cost = torch.full((1, 5, 8), 8.0, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        for row, col in enumerate(range(2, 7)):
            cost[0, row, col] = 0.1 + 0.01 * row
        cost[0, 1, 3] = 0.05
        cost[0, 3, 5] = 0.05
    return cost


@pytest.mark.parametrize("backend", ["torch", "cpu_numba", "cuda_cpp"])
def test_subdtw_gradient_matches_librosa_subsequence_path(backend):
    if backend == "cuda_cpp" and not torch.cuda.is_available():
        pytest.skip("backend='cuda_cpp' requires an available CUDA device")

    print("testing subSDTW for %s backend" % (backend))

    cost = _subsequence_cost_matrix()
    sequence = _import_librosa_sequence()
    accumulated_cost, warping_path = sequence.dtw(
        C=cost.detach().numpy()[0],
        step_sizes_sigma=np.array([[1, 0], [0, 1], [1, 1]]),
        weights_add=np.zeros(3),
        weights_mul=np.ones(3),
        subseq=True,
        backtrack=True,
    )

    try:
        subdtw = subSDTW(
            min_function="hardmin",
            backend=backend,
            normalization="none",
            dtype_float=torch.float32,
            sub_X=False,
            sub_Y=True,
            compensate_subseq=False,
        )
    except ImportError as exc:
        pytest.skip(str(exc))

    loss = subdtw(C=cost)
    loss.backward()

    expected_path_gradient = torch.zeros_like(cost)
    for row, col in warping_path:
        expected_path_gradient[0, row, col] = 1

    assert torch.allclose(
        loss.cpu(),
        torch.tensor(accumulated_cost[-1].min(), dtype=loss.cpu().dtype),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.equal(cost.grad, expected_path_gradient)
