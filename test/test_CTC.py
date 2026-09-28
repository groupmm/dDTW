import warnings

import torch
import torch.nn.functional as F
import pytest

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

from ddtw import CTC


def _random_valid_ctc_batch(
    *,
    seed,
    batch_size,
    max_input_length,
    max_target_length,
    num_classes,
    blank_index,
):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(
        batch_size,
        max_input_length,
        num_classes,
        generator=generator,
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.full(
        (batch_size, max_target_length),
        blank_index,
        dtype=torch.long,
    )
    input_lengths = torch.empty(batch_size, dtype=torch.long)
    target_lengths = torch.empty(batch_size, dtype=torch.long)

    for batch_index in range(batch_size):
        while True:
            target_length = int(
                torch.randint(
                    1,
                    max_target_length + 1,
                    (1,),
                    generator=generator,
                )
            )
            target = torch.randint(
                1,
                num_classes,
                (target_length,),
                generator=generator,
                dtype=torch.long,
            )
            repeated_labels = int((target[1:] == target[:-1]).sum())
            min_input_length = target_length + repeated_labels
            if min_input_length <= max_input_length:
                break

        input_length = int(
            torch.randint(
                min_input_length,
                max_input_length + 1,
                (1,),
                generator=generator,
            )
        )
        targets[batch_index, :target_length] = target
        input_lengths[batch_index] = input_length
        target_lengths[batch_index] = target_length

    return logits, targets, input_lengths, target_lengths


@pytest.mark.parametrize("backend", ["torch", "cpu_numba", "cuda_cpp"])
def test_ctc_matches_pytorch_loss_and_gradient(backend):
    if backend == "cuda_cpp" and not torch.cuda.is_available():
        pytest.skip("backend='cuda_cpp' requires an available CUDA device")

    print("testing CTC for %s backend"%(backend))

    batch_size = 2
    max_input_length = 32
    max_target_length = 8
    num_classes = 5
    blank_index = 0

    logits, targets, input_lengths, target_lengths = _random_valid_ctc_batch(
        seed=41,
        batch_size=batch_size,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        num_classes=num_classes,
        blank_index=blank_index,
    )
    logits_reference = logits.detach().clone().requires_grad_()
    log_probs = torch.log_softmax(logits, dim=-1)
    log_probs_reference = torch.log_softmax(logits_reference, dim=-1)

    try:
        ctc = CTC(
            blank_index=blank_index,
            gamma=1.0,
            backend=backend,
            dtype_float=torch.float32,
        )
    except ImportError as exc:
        pytest.skip(str(exc))

    ddtw_loss = ctc(
        X=log_probs,
        Y=targets,
        list_N=input_lengths,
        list_M=target_lengths,
    )
    pytorch_loss = F.ctc_loss(
        log_probs_reference.transpose(0, 1),
        targets,
        input_lengths,
        target_lengths,
        blank=blank_index,
        reduction="mean",
        zero_infinity=False,
    )

    assert torch.allclose(ddtw_loss.cpu(), pytorch_loss.cpu(), atol=1e-5, rtol=1e-5)

    ddtw_loss.backward()
    pytorch_loss.backward()

    assert torch.allclose(
        logits.grad,
        logits_reference.grad,
        atol=1e-5,
        rtol=1e-5,
    )
