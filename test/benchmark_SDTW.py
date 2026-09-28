#!/usr/bin/env python3
"""Benchmark Maghoumi Soft-DTW against ddtw.SDTW.

Both implementations run on CUDA and use the same random input tensors.
"""

from __future__ import annotations

import argparse
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.\*DtypeTensor constructors.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*GPU under-utilization due to low occupancy.*",
)

from ddtw import SDTW


def parse_version_tuple(version_string):
    parts = []
    for token in version_string.replace("-", ".").split("."):
        digits = ""
        for character in token:
            if not character.isdigit():
                break
            digits += character
        if digits:
            parts.append(int(digits))
    return tuple(parts)


def get_package_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Maghoumi Soft-DTW CUDA benchmark requires %s to be installed. "
            "Install the benchmark extras or run with --implementation ddtw."
            % package_name
        ) from exc


def check_maghoumi_dependencies():
    numba_version = get_package_version("numba")
    llvmlite_version = get_package_version("llvmlite")
    numba_tuple = parse_version_tuple(numba_version)

    if not (numba_tuple >= (0, 65) and numba_tuple < (0, 66)):
        raise RuntimeError(
            "Maghoumi Soft-DTW CUDA benchmark is known to work with "
            "numba>=0.65,<0.66. Found numba==%s and llvmlite==%s. "
            "Install a compatible benchmark stack or run with "
            "--implementation ddtw."
            % (numba_version, llvmlite_version)
        )


def load_maghoumi_softdtw():
    check_maghoumi_dependencies()
    path = ROOT / "reference_implementations" / "soft_dtw_cuda.py"
    spec = importlib.util.spec_from_file_location("maghoumi_soft_dtw_cuda", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Maghoumi SoftDTW from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SoftDTW


def make_inputs(args):
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    x = torch.randn(
        args.batch_size,
        args.len_x,
        args.dims,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).cuda(args.device)
    y = torch.randn(
        args.batch_size,
        args.len_y,
        args.dims,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).cuda(args.device)
    return x, y


def time_forward_backward(module, base_x, base_y):
    x = base_x.detach().clone().requires_grad_(True)
    y = base_y.detach()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    loss = module(x, y)
    if loss.ndim > 0:
        loss = loss.mean()
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


def benchmark(label, module, x, y, warmups, trials):
    for _ in range(warmups):
        time_forward_backward(module, x, y)
    torch.cuda.empty_cache()

    measurements = [time_forward_backward(module, x, y) for _ in range(trials)]
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
        description="Benchmark Maghoumi Soft-DTW and ddtw.SDTW on CUDA."
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--len-x", type=int, default=512)
    parser.add_argument("--len-y", type=int, default=128)
    parser.add_argument("--dims", type=int, default=88)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--implementation",
        choices=("both", "ddtw", "maghoumi"),
        default="both",
        help="Implementation to benchmark. Default: both.",
    )
    args = parser.parse_args()
    if args.trials < 1 or args.warmups < 0:
        parser.error("--trials must be >= 1 and --warmups must be >= 0")
    if min(args.batch_size, args.len_x, args.len_y, args.dims) < 1:
        parser.error("input dimensions must all be >= 1")
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.cuda.set_device(args.device)
    x, y = make_inputs(args)

    implementations = []
    if args.implementation in ("both", "maghoumi"):
        SoftDTW = load_maghoumi_softdtw()
        implementations.append(
            (
                "Maghoumi",
                SoftDTW(
                    use_cuda=True,
                    gamma=args.gamma,
                    normalize=False,
                ).cuda(args.device),
            )
        )
    if args.implementation in ("both", "ddtw"):
        implementations.append(
            (
                "dDTW",
                SDTW(
                    gamma=args.gamma,
                    backend="cuda_cpp",
                    normalization="none",
                    dtype_float=torch.float32,
                    cuda_device=f"cuda:{args.device}",
                ),
            )
        )

    results = [
        benchmark(label, module, x, y, args.warmups, args.trials)
        for label, module in implementations
    ]
    print_table(results)


if __name__ == "__main__":
    main()
