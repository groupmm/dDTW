#!/usr/bin/env python3
"""Benchmark PyTorch CTC against ddtw.CTC.

Both implementations run on CUDA and use the same random logits, targets, and
lengths.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
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


def make_lengths(args):
    if not args.variable_lengths:
        input_lengths = torch.full((args.batch_size,), args.input_len, dtype=torch.long)
        target_lengths = torch.full((args.batch_size,), args.target_len, dtype=torch.long)
        return input_lengths, target_lengths

    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    min_target_len = max(1, args.target_len // 2)
    target_lengths = torch.randint(
        min_target_len,
        args.target_len + 1,
        (args.batch_size,),
        generator=generator,
        dtype=torch.long,
    )
    min_input_lengths = torch.maximum(
        target_lengths,
        torch.full_like(target_lengths, max(1, args.input_len // 2)),
    )
    input_lengths = torch.empty(args.batch_size, dtype=torch.long)
    for batch_index, min_input_len in enumerate(min_input_lengths.tolist()):
        input_lengths[batch_index] = torch.randint(
            min_input_len,
            args.input_len + 1,
            (1,),
            generator=generator,
            dtype=torch.long,
        )
    return input_lengths, target_lengths


def make_targets(args, target_lengths):
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 2)
    targets = torch.full(
        (args.batch_size, args.target_len),
        args.blank_index,
        dtype=torch.long,
    )
    for batch_index, target_len in enumerate(target_lengths.tolist()):
        previous = args.blank_index
        for target_index in range(target_len):
            label = int(
                torch.randint(
                    1,
                    args.num_classes,
                    (1,),
                    generator=generator,
                    dtype=torch.long,
                )
            )
            if args.num_classes > 2:
                while label == previous:
                    label = int(
                        torch.randint(
                            1,
                            args.num_classes,
                            (1,),
                            generator=generator,
                            dtype=torch.long,
                        )
                    )
            targets[batch_index, target_index] = label
            previous = label
    return targets


def make_logits(args):
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    return torch.randn(
        args.batch_size,
        args.input_len,
        args.num_classes,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).cuda(args.device)


def time_forward_backward(forward_fn, base_logits):
    logits = base_logits.detach().clone().requires_grad_(True)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    loss = forward_fn(logits)
    torch.cuda.synchronize()
    forward_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    backward_ms = (time.perf_counter() - start) * 1000.0

    return {
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "total_ms": forward_ms + backward_ms,
        "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024.0**2),
        "loss": float(loss.detach().cpu()),
    }


def benchmark(label, forward_fn, logits, warmups, trials):
    for _ in range(warmups):
        time_forward_backward(forward_fn, logits)
    torch.cuda.empty_cache()

    measurements = [time_forward_backward(forward_fn, logits) for _ in range(trials)]
    return {
        "label": label,
        "forward_ms": statistics.median(m["forward_ms"] for m in measurements),
        "backward_ms": statistics.median(m["backward_ms"] for m in measurements),
        "total_ms": statistics.median(m["total_ms"] for m in measurements),
        "cuda_peak_allocated_mb": max(m["cuda_peak_allocated_mb"] for m in measurements),
        "loss": measurements[-1]["loss"],
    }


def print_table(results):
    print()
    print(
        f"{'implementation':18} {'fwd ms':>10} {'bwd ms':>10} "
        f"{'total ms':>10} {'CUDA MiB':>10} {'loss':>14}"
    )
    print("-" * 78)
    for row in results:
        print(
            f"{row['label']:18} {row['forward_ms']:10.3f} "
            f"{row['backward_ms']:10.3f} {row['total_ms']:10.3f} "
            f"{row['cuda_peak_allocated_mb']:10.1f} {row['loss']:14.6f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch CTC and ddtw.CTC on CUDA."
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--target-len", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=88)
    parser.add_argument("--blank-index", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--variable-lengths", action="store_true")
    parser.add_argument("--zero-infinity", action="store_true")
    args = parser.parse_args()
    if args.trials < 1 or args.warmups < 0:
        parser.error("--trials must be >= 1 and --warmups must be >= 0")
    if min(args.batch_size, args.input_len, args.target_len, args.num_classes) < 1:
        parser.error("input dimensions must all be >= 1")
    if args.num_classes < 2:
        parser.error("--num-classes must be >= 2 because class 0 is blank")
    if not 0 <= args.blank_index < args.num_classes:
        parser.error("--blank-index must be in [0, num_classes)")
    if args.target_len > args.input_len:
        parser.error("--target-len must be <= --input-len")
    if args.gamma <= 0:
        parser.error("--gamma must be > 0")
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.cuda.set_device(args.device)
    logits = make_logits(args)
    input_lengths, target_lengths = make_lengths(args)
    targets = make_targets(args, target_lengths)

    targets_cuda = targets.cuda(args.device)
    input_lengths_cuda = input_lengths.cuda(args.device)
    target_lengths_cuda = target_lengths.cuda(args.device)

    ddtw_ctc = CTC(
        blank_index=args.blank_index,
        gamma=args.gamma,
        backend="cuda_cpp",
        dtype_float=torch.float32,
        cuda_device=f"cuda:{args.device}",
    )

    def pytorch_forward(batch_logits):
        log_probs = F.log_softmax(batch_logits, dim=-1).transpose(0, 1)
        return F.ctc_loss(
            log_probs,
            targets_cuda,
            input_lengths_cuda,
            target_lengths_cuda,
            blank=args.blank_index,
            reduction="mean",
            zero_infinity=args.zero_infinity,
        )

    def ddtw_forward(batch_logits):
        log_probs = F.log_softmax(batch_logits, dim=-1)
        return ddtw_ctc(
            X=log_probs,
            Y=targets_cuda,
            list_N=input_lengths_cuda,
            list_M=target_lengths_cuda,
        )

    results = [
        benchmark("PyTorch", pytorch_forward, logits, args.warmups, args.trials),
        benchmark("dDTW", ddtw_forward, logits, args.warmups, args.trials),
    ]
    print_table(results)


if __name__ == "__main__":
    main()
