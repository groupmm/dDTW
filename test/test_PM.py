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

from ddtw import partial_matching


def _import_reference_partial_matching():
    reference_path = (
        Path(__file__).resolve().parent
        / "reference_implementations"
        / "partial_matching.py"
    )
    spec = importlib.util.spec_from_file_location(
        "partial_matching_reference",
        reference_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_partial_matching


def _random_score_batch(seed, batch_size, max_rows, max_cols):
    generator = torch.Generator().manual_seed(seed)
    scores = torch.rand(
        batch_size,
        max_rows,
        max_cols,
        generator=generator,
        dtype=torch.float32,
    )
    rows = torch.randint(3, max_rows + 1, (batch_size,), generator=generator)
    cols = torch.randint(3, max_cols + 1, (batch_size,), generator=generator)
    return scores, rows, cols


def _assert_gradient_is_monotone_path(gradient):
    path = torch.nonzero(gradient > 0, as_tuple=False)
    if path.numel() == 0:
        return
    rows = path[:, 0]
    cols = path[:, 1]
    order = torch.argsort(rows)
    rows = rows[order]
    cols = cols[order]
    assert torch.all(rows[1:] > rows[:-1])
    assert torch.all(cols[1:] > cols[:-1])


@pytest.mark.parametrize("backend", ["torch", "cpu_numba", "cuda_cpp"])
def test_partial_matching_matches_reference_score_and_path_gradient(backend):
    if backend == "cuda_cpp" and not torch.cuda.is_available():
        pytest.skip("backend='cuda_cpp' requires an available CUDA device")

    print("testing partial_matching for %s backend" % (backend))

    batch_size = 2
    scores, rows, cols = _random_score_batch(
        seed=47,
        batch_size=batch_size,
        max_rows=6,
        max_cols=7,
    )
    cost = (-scores).detach().clone().requires_grad_()

    compute_partial_matching = _import_reference_partial_matching()
    expected_loss = 0.0
    expected_gradient = torch.zeros_like(cost)
    for batch_index in range(batch_size):
        active_scores = scores[
            batch_index,
            : rows[batch_index],
            : cols[batch_index],
        ].numpy()
        accumulated_score, path = compute_partial_matching(active_scores)
        expected_loss -= accumulated_score[-1, -1] / batch_size
        for row, col in path:
            expected_gradient[batch_index, row, col] = 1.0 / batch_size

    try:
        pm = partial_matching(
            backend=backend,
            normalization="none",
            dtype_float=torch.float32,
        )
    except ImportError as exc:
        pytest.skip(str(exc))

    loss = pm(C=cost, list_N=rows, list_M=cols)
    loss.backward()

    for batch_index in range(batch_size):
        _assert_gradient_is_monotone_path(cost.grad[batch_index])

    assert torch.allclose(
        loss.cpu(),
        torch.tensor(expected_loss, dtype=loss.cpu().dtype),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        cost.grad,
        expected_gradient,
        atol=1e-5,
        rtol=1e-5,
    )
