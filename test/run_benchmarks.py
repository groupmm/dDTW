# %% [markdown]
# # SDTW and CTC benchmarks
# 
# Run the setup and function cells, then edit and run the examples below. All benchmark arguments are keyword parameters with the same defaults as the original scripts. CUDA and the repository dependencies must be available in this notebook's kernel; the Maghoumi comparison also requires the benchmark extras described in `benchmark_SDTW.py`.
# 
# Each call returns `{"parameters": {...}, "results": {implementation: metrics}}`. Metrics include forward, backward and total time in milliseconds, peak CUDA allocation in MiB, and loss. Timing values are medians across trials; peak memory is the maximum. The original scripts provide input generation and timing and are left unchanged.
# Each PNG has a companion NPZ containing plotted metrics, per-implementation
# x values, fixed dimensions, and plot labels/limits. Load with np.load(path):
# e.g. data["ddtw_x"], data["ddtw_total"], data["ddtw_total_std"].
# 

# %%
import argparse

parser = argparse.ArgumentParser(description="Run SDTW and CTC benchmark sweeps.")
parser.add_argument(
    "--machine", required=True,
    help="Machine name used in benchmark output filenames (e.g. holst).",
)
machine = parser.parse_args().machine


# %%
import matplotlib.pyplot as plt
import torch
import numpy as np
import gc


# %%
from pathlib import Path
from types import SimpleNamespace
import importlib.util


def _save_benchmark_plot(figure, filename, *, sweep, series, parameters):
    """Save a PNG and a pickle-free NPZ with the same stem.

    series maps an implementation name to (x_values, metrics). Metric keys
    match the plotting dictionaries: fwd/bwd/total and their std values are
    in milliseconds; mem is in MiB. Each implementation has its own x array
    so truncated or empty Maghoumi runs can be reproduced without guessing.
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "machine": np.asarray(machine),
        "gpu_name": np.asarray(torch.cuda.get_device_name(0)),
        "sweep": np.asarray(sweep),
        "implementations": np.asarray(list(series)),
        "time_unit": np.asarray("ms"),
        "memory_unit": np.asarray("MiB"),
        "figure_title": np.asarray(figure._suptitle.get_text() if figure._suptitle else ""),
        "figure_size_inches": np.asarray(figure.get_size_inches()),
        "dpi": np.asarray(200),
    }
    data.update({name: np.asarray(value) for name, value in parameters.items()})
    for implementation, (x_values, metrics) in series.items():
        data[f"{implementation}_x"] = np.asarray(x_values, dtype=np.int64)
        for metric, values in metrics.items():
            data[f"{implementation}_{metric}"] = np.asarray(values, dtype=float)
    for index, axis in enumerate(figure.axes):
        for name, value in {
            "title": axis.get_title(),
            "xlabel": axis.get_xlabel(), "ylabel": axis.get_ylabel(),
            "xlim": axis.get_xlim(), "ylim": axis.get_ylim(),
            "xscale": axis.get_xscale(), "yscale": axis.get_yscale(),
        }.items():
            data[f"axis_{index}_{name}"] = np.asarray(value)
    np.savez_compressed(path.with_suffix(".npz"), **data)
    figure.savefig(path, dpi=200)


def _load_benchmark(name):
    # Locate the scripts whether the kernel starts in test/ or the repository root.
    for directory in (Path.cwd(), *Path.cwd().parents):
        for candidate in (directory / f"{name}.py", directory / "test" / f"{name}.py"):
            if candidate.is_file():
                spec = importlib.util.spec_from_file_location(f"_notebook_{name}", candidate)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load {candidate}")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
    raise FileNotFoundError(f"Cannot locate {name}.py; start the kernel in the repository.")


sdtw_benchmark = _load_benchmark("benchmark_SDTW")
ctc_benchmark = _load_benchmark("benchmark_CTC")
torch = sdtw_benchmark.torch


def _validate_common(args, dimensions):
    if args.trials < 1 or args.warmups < 0:
        raise ValueError("trials must be >= 1 and warmups must be >= 0")
    if min(getattr(args, name) for name in dimensions) < 1:
        raise ValueError("input dimensions must all be >= 1")
    if args.gamma <= 0:
        raise ValueError("gamma must be > 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")


# %%
def _benchmark_maghoumi(args, x, y):
    """Return metrics, or None if Maghoumi runs out of host/CUDA memory."""
    try:
        SoftDTW = sdtw_benchmark.load_maghoumi_softdtw()
        module = SoftDTW(
            use_cuda=True, gamma=args.gamma, normalize=False,
        ).cuda(args.device)
        return sdtw_benchmark.benchmark(
            "Maghoumi", module, x, y, args.warmups, args.trials,
        )
    except Exception as error:
        
        # Numba can report CUDA_ERROR_OUT_OF_MEMORY using its own exception
        # type. Match that message without adding a mandatory Numba import.
        message = str(error).lower().replace("_", " ")
        print("caught exception during Maghoumi benchmark:", message)
        #if not isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)) and not any(
        #    text in message
        #    for text in ("out of memory", "can't allocate memory", "cannot allocate memory")
        #):
        #    raise
    # Return after leaving the handler so the traceback releases its tensors
    # before the caller collects garbage and clears the CUDA cache.
    return None


def _set_maghoumi_plot_range(axis, values):
    """Use the successful sweep values, including empty/single-point runs."""
    if not values:
        axis.text(0.5, 0.5, "No successful Maghoumi measurements",
                  transform=axis.transAxes, ha="center", va="center")
        return
    lower, upper = min(values), max(values)
    padding = max(abs(lower) * 0.05, 0.5) if lower == upper else 0
    axis.set_xlim(lower - padding, upper + padding)


def run_sdtw_benchmark(
    *, batch_size=16, len_x=512, len_y=128, dims=88, gamma=0.1,
    seed=42, warmups=5, trials=20, device=0, implementation="both",
):
    """Benchmark SDTW and return parameters plus metrics keyed by implementation.

    implementation: "both", "ddtw", or "maghoumi". Maghoumi is omitted
    from results on memory failure or for sequence lengths above 1024.
    """
    args = SimpleNamespace(
        batch_size=batch_size, len_x=len_x, len_y=len_y, dims=dims,
        gamma=gamma, seed=seed, warmups=warmups, trials=trials,
        device=device, implementation=implementation,
    )
    if implementation not in ("both", "ddtw", "maghoumi"):
        raise ValueError('implementation must be "both", "ddtw", or "maghoumi"')
    _validate_common(args, ("batch_size", "len_x", "len_y", "dims"))

    with torch.cuda.device(device):
        x, y = sdtw_benchmark.make_inputs(args)
        results = {}
        if implementation in ("both", "maghoumi") and max(len_x, len_y) <= 1024:
            metrics = _benchmark_maghoumi(args, x, y)
            if metrics is not None:
                results["Maghoumi"] = metrics
            else:
                gc.collect()
                torch.cuda.empty_cache()
        if implementation in ("both", "ddtw"):
            module = sdtw_benchmark.SDTW(
                gamma=gamma, backend="cuda_cpp", normalization="none",
                dtype_float=torch.float32, cuda_device=f"cuda:{device}",
            )
            results["dDTW"] = sdtw_benchmark.benchmark(
                "dDTW", module, x, y, warmups, trials,
            )
    return {"parameters": vars(args).copy(), "results": results}


# %%
def run_ctc_benchmark(
    *, batch_size=16, input_len=512, target_len=128, num_classes=88,
    blank_index=0, gamma=1.0, seed=42, warmups=5, trials=20,
    device=0, variable_lengths=False, zero_infinity=False,
):
    """Benchmark PyTorch CTC and dDTW CTC and return a results dictionary.

    As in the original script, zero_infinity applies to PyTorch only, and
    gamma applies to dDTW only. Use gamma=1.0 for matching CTC objectives.
    """
    args = SimpleNamespace(
        batch_size=batch_size, input_len=input_len, target_len=target_len,
        num_classes=num_classes, blank_index=blank_index, gamma=gamma,
        seed=seed, warmups=warmups, trials=trials, device=device,
        variable_lengths=variable_lengths, zero_infinity=zero_infinity,
    )
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    if not 0 <= blank_index < num_classes:
        raise ValueError("blank_index must be in [0, num_classes)")
    if target_len > input_len:
        raise ValueError("target_len must be <= input_len")
    _validate_common(args, ("batch_size", "input_len", "target_len", "num_classes"))

    with torch.cuda.device(device):
        logits = ctc_benchmark.make_logits(args)
        input_lengths, target_lengths = ctc_benchmark.make_lengths(args)
        # The script's target generator assumes blank=0. Generate in that
        # convention, then permute labels to support a custom blank index.
        target_args = SimpleNamespace(**vars(args))
        target_args.blank_index = 0
        targets = ctc_benchmark.make_targets(target_args, target_lengths)
        if blank_index != 0:
            targets = torch.where(
                targets == 0, blank_index,
                torch.where(targets <= blank_index, targets - 1, targets),
            )
        targets_cuda = targets.cuda(device)
        input_lengths_cuda = input_lengths.cuda(device)
        target_lengths_cuda = target_lengths.cuda(device)
        ddtw_ctc = ctc_benchmark.CTC(
            blank_index=blank_index, gamma=gamma, backend="cuda_cpp",
            dtype_float=torch.float32, cuda_device=f"cuda:{device}",
        )

        def pytorch_forward(batch_logits):
            log_probs = ctc_benchmark.F.log_softmax(batch_logits, dim=-1).transpose(0, 1)
            return ctc_benchmark.F.ctc_loss(
                log_probs, targets_cuda, input_lengths_cuda, target_lengths_cuda,
                blank=blank_index, reduction="mean", zero_infinity=zero_infinity,
            )

        def ddtw_forward(batch_logits):
            log_probs = ctc_benchmark.F.log_softmax(batch_logits, dim=-1)
            return ddtw_ctc(
                X=log_probs, Y=targets_cuda,
                list_N=input_lengths_cuda, list_M=target_lengths_cuda,
            )

        results = {
            label: ctc_benchmark.benchmark(label, forward, logits, warmups, trials)
            for label, forward in (("PyTorch", pytorch_forward), ("dDTW", ddtw_forward))
        }
    return {"parameters": vars(args).copy(), "results": results}


# %% [markdown]
# ## Custom runs
# 
# Edit these parameters as needed. Each example cell starts a benchmark when you run it. For SDTW, set `implementation="ddtw"` to run without the Maghoumi dependencies. Use separate calls in a loop to collect a parameter sweep.
# 

# %% [markdown]
# # SDTW

# %% [markdown]
# ## vary one seq. length - long seg.

# %%
N=512
M_list = np.concatenate([np.arange(10, 100, 10), 
                         np.arange(100, 1000, 100), 
                         np.arange(1000, 10000+1, 1000)])
D = 88
B = 16

# %%
gc.collect()
torch.cuda.empty_cache()


results_ddtw = {"fwd": [], "bwd": [], "total": [], "fwd_std": [], "bwd_std": [], "total_std": [], "mem": []}
results_magh = {"fwd": [], "bwd": [], "total": [], "fwd_std": [], "bwd_std": [], "total_std": [], "mem": []}
M_list_magh = []
maghoumi_active = True
for M in M_list:
    print(f"Benchmarking SDTW with N={N}, M={M}, D={D}, B={B}...")
    
    benchmark_results = run_sdtw_benchmark(len_x=N, len_y=M, dims=D, batch_size=B,implementation="both" if maghoumi_active else "ddtw")
    maghoumi_active = "Maghoumi" in benchmark_results["results"]
    for label, metrics in benchmark_results["results"].items():
        if label == "dDTW":
            results_ddtw["fwd"].append(metrics["forward_ms"])
            results_ddtw["bwd"].append(metrics["backward_ms"])
            results_ddtw["total"].append(metrics["total_ms"])
            results_ddtw["fwd_std"].append(metrics["forward_ms_std"])
            results_ddtw["bwd_std"].append(metrics["backward_ms_std"])
            results_ddtw["total_std"].append(metrics["total_ms_std"])
            results_ddtw["mem"].append(metrics["cuda_peak_allocated_mb"])
        elif label == "Maghoumi":
            M_list_magh.append(M)
            results_magh["fwd"].append(metrics["forward_ms"])
            results_magh["bwd"].append(metrics["backward_ms"])
            results_magh["total"].append(metrics["total_ms"])
            results_magh["fwd_std"].append(metrics["forward_ms_std"])
            results_magh["bwd_std"].append(metrics["backward_ms_std"])
            results_magh["total_std"].append(metrics["total_ms_std"])
            results_magh["mem"].append(metrics["cuda_peak_allocated_mb"])
            


# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4))

ax[0].fill_between(M_list, 
                [t - s for t, s in zip(results_ddtw["total"], results_ddtw["total_std"])],
                [t + s for t, s in zip(results_ddtw["total"], results_ddtw["total_std"])],
                color='tab:blue', alpha=0.2)
ax[0].plot(M_list, results_ddtw["total"], label="total", marker=None, color="tab:blue", linewidth=2)

ax[0].fill_between(M_list, 
                [t - s for t, s in zip(results_ddtw["fwd"], results_ddtw["fwd_std"])],
                [t + s for t, s in zip(results_ddtw["fwd"], results_ddtw["fwd_std"])],
                color='tab:orange', alpha=0.2)
ax[0].plot(M_list, results_ddtw["fwd"], label="fwd", marker=None, color="tab:orange", linewidth=2)

ax[0].fill_between(M_list, 
                [t - s for t, s in zip(results_ddtw["bwd"], results_ddtw["bwd_std"])],
                [t + s for t, s in zip(results_ddtw["bwd"], results_ddtw["bwd_std"])],
                color='tab:green', alpha=0.2)
ax[0].plot(M_list, results_ddtw["bwd"], label="bwd", marker=None, color="tab:green", linewidth=2)



ax[1].fill_between(M_list_magh, 
                [t - s for t, s in zip(results_magh["total"], results_magh["total_std"])],
                [t + s for t, s in zip(results_magh["total"], results_magh["total_std"])],
                color='tab:blue', alpha=0.2)
ax[1].plot(M_list_magh, results_magh["total"], label="total", marker="." if len(M_list_magh) == 1 else None, color="tab:blue", linestyle="--", linewidth=2)

ax[1].fill_between(M_list_magh, 
                [t - s for t, s in zip(results_magh["fwd"], results_magh["fwd_std"])],
                [t + s for t, s in zip(results_magh["fwd"], results_magh["fwd_std"])],
                color='tab:orange', alpha=0.2)
ax[1].plot(M_list_magh, results_magh["fwd"], label="fwd", marker="." if len(M_list_magh) == 1 else None, color="tab:orange", linestyle="--", linewidth=2)

ax[1].fill_between(M_list_magh, 
                [t - s for t, s in zip(results_magh["bwd"], results_magh["bwd_std"])],
                [t + s for t, s in zip(results_magh["bwd"], results_magh["bwd_std"])],
                color='tab:green', alpha=0.2)
ax[1].plot(M_list_magh, results_magh["bwd"], label="bwd", marker="." if len(M_list_magh) == 1 else None, color="tab:green", linestyle="--", linewidth=2)

ax[2].plot(M_list, results_ddtw["mem"], label="dDTW", marker=None, color="k", linewidth=2)
ax[2].plot(M_list_magh, results_magh["mem"], label="Maghoumi", marker="." if len(M_list_magh) == 1 else None, color="k", linestyle="--", linewidth=2)


ax[0].set_xlabel("M")
ax[0].set_ylabel("Time (s)")
ax[0].set_title("dDTW-SDTW")
ax[0].legend()

ax[1].set_xlabel("M")
ax[1].set_ylabel("Time (s)")
ax[1].set_title("Maghoumi-SDTW")
ax[1].legend()

ax[2].set_xlabel("M")
ax[2].set_ylabel("Memory (MB)")
ax[2].set_title("Memory Usage")
ax[2].legend()

ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)
ax[0].set_xlim([min(M_list), max(M_list)])
_set_maghoumi_plot_range(ax[1], M_list_magh)
ax[2].set_xlim([min(M_list), max(M_list)])

ax[0].set_ylim(bottom=0)
ax[1].set_ylim(bottom=0)
ax[2].set_ylim(bottom=0)

fig.suptitle("N=%i, D=%i, B=%i @ %s" % (N, D, B, torch.cuda.get_device_name(0)))

fig.tight_layout()
_save_benchmark_plot(
    fig, "results/benchmark_SDTW_longSeg_%s.png" % machine,
    sweep="M", parameters=dict(N=N, D=D, B=B),
    series={
        "ddtw": (M_list, results_ddtw),
        "maghoumi": (M_list_magh, results_magh),
    },
)

# %% [markdown]
# ## vary one seq. length - short seg.

# %%
N=512
M_list = np.concatenate([np.arange(10, 100, 10), 
                         np.arange(100, 1000+1, 100)])
D = 88
B = 16

# %%
gc.collect()
torch.cuda.empty_cache()


results_ddtw = {"fwd": [], "bwd": [], "total": [], "fwd_std": [], "bwd_std": [], "total_std": [], "mem": []}
results_magh = {"fwd": [], "bwd": [], "total": [], "fwd_std": [], "bwd_std": [], "total_std": [], "mem": []}
M_list_magh = []
maghoumi_active = True
for M in M_list:
    print(f"Benchmarking SDTW with N={N}, M={M}, D={D}, B={B}...")
    
    benchmark_results = run_sdtw_benchmark(len_x=N, len_y=M, dims=D, batch_size=B,implementation="both" if maghoumi_active else "ddtw")
    maghoumi_active = "Maghoumi" in benchmark_results["results"]
    for label, metrics in benchmark_results["results"].items():
        if label == "dDTW":
            results_ddtw["fwd"].append(metrics["forward_ms"])
            results_ddtw["bwd"].append(metrics["backward_ms"])
            results_ddtw["total"].append(metrics["total_ms"])
            results_ddtw["fwd_std"].append(metrics["forward_ms_std"])
            results_ddtw["bwd_std"].append(metrics["backward_ms_std"])
            results_ddtw["total_std"].append(metrics["total_ms_std"])
            results_ddtw["mem"].append(metrics["cuda_peak_allocated_mb"])
        elif label == "Maghoumi":
            M_list_magh.append(M)
            results_magh["fwd"].append(metrics["forward_ms"])
            results_magh["bwd"].append(metrics["backward_ms"])
            results_magh["total"].append(metrics["total_ms"])
            results_magh["fwd_std"].append(metrics["forward_ms_std"])
            results_magh["bwd_std"].append(metrics["backward_ms_std"])
            results_magh["total_std"].append(metrics["total_ms_std"])
            results_magh["mem"].append(metrics["cuda_peak_allocated_mb"])
            


# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4))

ax[0].fill_between(M_list, 
                [t - s for t, s in zip(results_ddtw["total"], results_ddtw["total_std"])],
                [t + s for t, s in zip(results_ddtw["total"], results_ddtw["total_std"])],
                color='tab:blue', alpha=0.2)
ax[0].plot(M_list, results_ddtw["total"], label="total", marker=None, color="tab:blue", linewidth=2)

ax[0].fill_between(M_list, 
                [t - s for t, s in zip(results_ddtw["fwd"], results_ddtw["fwd_std"])],
                [t + s for t, s in zip(results_ddtw["fwd"], results_ddtw["fwd_std"])],
                color='tab:orange', alpha=0.2)
ax[0].plot(M_list, results_ddtw["fwd"], label="fwd", marker=None, color="tab:orange", linewidth=2)

ax[0].fill_between(M_list, 
                [t - s for t, s in zip(results_ddtw["bwd"], results_ddtw["bwd_std"])],
                [t + s for t, s in zip(results_ddtw["bwd"], results_ddtw["bwd_std"])],
                color='tab:green', alpha=0.2)
ax[0].plot(M_list, results_ddtw["bwd"], label="bwd", marker=None, color="tab:green", linewidth=2)



ax[1].fill_between(M_list_magh, 
                [t - s for t, s in zip(results_magh["total"], results_magh["total_std"])],
                [t + s for t, s in zip(results_magh["total"], results_magh["total_std"])],
                color='tab:blue', alpha=0.2)
ax[1].plot(M_list_magh, results_magh["total"], label="total", marker="." if len(M_list_magh) == 1 else None, color="tab:blue", linestyle="--", linewidth=2)

ax[1].fill_between(M_list_magh, 
                [t - s for t, s in zip(results_magh["fwd"], results_magh["fwd_std"])],
                [t + s for t, s in zip(results_magh["fwd"], results_magh["fwd_std"])],
                color='tab:orange', alpha=0.2)
ax[1].plot(M_list_magh, results_magh["fwd"], label="fwd", marker="." if len(M_list_magh) == 1 else None, color="tab:orange", linestyle="--", linewidth=2)

ax[1].fill_between(M_list_magh, 
                [t - s for t, s in zip(results_magh["bwd"], results_magh["bwd_std"])],
                [t + s for t, s in zip(results_magh["bwd"], results_magh["bwd_std"])],
                color='tab:green', alpha=0.2)
ax[1].plot(M_list_magh, results_magh["bwd"], label="bwd", marker="." if len(M_list_magh) == 1 else None, color="tab:green", linestyle="--", linewidth=2)

ax[2].plot(M_list, results_ddtw["mem"], label="dDTW", marker=None, color="k", linewidth=2)
ax[2].plot(M_list_magh, results_magh["mem"], label="Maghoumi", marker="." if len(M_list_magh) == 1 else None, color="k", linestyle="--", linewidth=2)


ax[0].set_xlabel("M")
ax[0].set_ylabel("Time (s)")
ax[0].set_title("dDTW-SDTW")
ax[0].legend()

ax[1].set_xlabel("M")
ax[1].set_ylabel("Time (s)")
ax[1].set_title("Maghoumi-SDTW")
ax[1].legend()

ax[2].set_xlabel("M")
ax[2].set_ylabel("Memory (MB)")
ax[2].set_title("Memory Usage")
ax[2].legend()

ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)
ax[0].set_xlim([min(M_list), max(M_list)])
_set_maghoumi_plot_range(ax[1], M_list_magh)
ax[2].set_xlim([min(M_list), max(M_list)])

ax[0].set_ylim(bottom=0)
ax[1].set_ylim(bottom=0)
ax[2].set_ylim(bottom=0)

fig.suptitle("N=%i, D=%i, B=%i @ %s" % (N, D, B, torch.cuda.get_device_name(0)))

fig.tight_layout()
_save_benchmark_plot(
    fig, "results/benchmark_SDTW_shortSeg_%s.png" % machine,
    sweep="M", parameters=dict(N=N, D=D, B=B),
    series={
        "ddtw": (M_list, results_ddtw),
        "maghoumi": (M_list_magh, results_magh),
    },
)

# %% [markdown]
# ## vary batch size

# %%
N=512
M = 128
D = 88
B_list = np.concatenate([np.arange(1, 10, 1), np.arange(10, 100, 10), np.arange(100, 1000+1, 100)])

# %%
gc.collect()
torch.cuda.empty_cache()

results_ddtw = {"fwd": [], "bwd": [], "total": [], "fwd_std": [], "bwd_std": [], "total_std": [], "mem": []}
results_magh = {"fwd": [], "bwd": [], "total": [], "fwd_std": [], "bwd_std": [], "total_std": [], "mem": []}
B_list_magh = []
maghoumi_active = True
for B in B_list:
    print(f"Benchmarking SDTW with N={N}, M={M}, D={D}, B={B}...")
    
    benchmark_results = run_sdtw_benchmark(len_x=N, len_y=M, dims=D, batch_size=B,implementation="both" if maghoumi_active else "ddtw")
    maghoumi_active = "Maghoumi" in benchmark_results["results"]
    for label, metrics in benchmark_results["results"].items():
        if label == "dDTW":
            results_ddtw["fwd"].append(metrics["forward_ms"])
            results_ddtw["bwd"].append(metrics["backward_ms"])
            results_ddtw["total"].append(metrics["total_ms"])
            results_ddtw["fwd_std"].append(metrics["forward_ms_std"])
            results_ddtw["bwd_std"].append(metrics["backward_ms_std"])
            results_ddtw["total_std"].append(metrics["total_ms_std"])
            results_ddtw["mem"].append(metrics["cuda_peak_allocated_mb"])
        elif label == "Maghoumi":
            B_list_magh.append(B)
            results_magh["fwd"].append(metrics["forward_ms"])
            results_magh["bwd"].append(metrics["backward_ms"])
            results_magh["total"].append(metrics["total_ms"])
            results_magh["fwd_std"].append(metrics["forward_ms_std"])
            results_magh["bwd_std"].append(metrics["backward_ms_std"])
            results_magh["total_std"].append(metrics["total_ms_std"])
            results_magh["mem"].append(metrics["cuda_peak_allocated_mb"])
            


# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4))

ax[0].fill_between(B_list, 
                [t - s for t, s in zip(results_ddtw["total"], results_ddtw["total_std"])],
                [t + s for t, s in zip(results_ddtw["total"], results_ddtw["total_std"])],
                color='tab:blue', alpha=0.2)
ax[0].plot(B_list, results_ddtw["total"], label="total", marker=None, color="tab:blue", linewidth=2)

ax[0].fill_between(B_list, 
                [t - s for t, s in zip(results_ddtw["fwd"], results_ddtw["fwd_std"])],
                [t + s for t, s in zip(results_ddtw["fwd"], results_ddtw["fwd_std"])],
                color='tab:orange', alpha=0.2)
ax[0].plot(B_list, results_ddtw["fwd"], label="fwd", marker=None, color="tab:orange", linewidth=2)

ax[0].fill_between(B_list, 
                [t - s for t, s in zip(results_ddtw["bwd"], results_ddtw["bwd_std"])],
                [t + s for t, s in zip(results_ddtw["bwd"], results_ddtw["bwd_std"])],
                color='tab:green', alpha=0.2)
ax[0].plot(B_list, results_ddtw["bwd"], label="bwd", marker=None, color="tab:green", linewidth=2)




ax[1].fill_between(B_list_magh, 
                [t - s for t, s in zip(results_magh["total"], results_magh["total_std"])],
                [t + s for t, s in zip(results_magh["total"], results_magh["total_std"])],
                color='tab:blue', alpha=0.2)
ax[1].plot(B_list_magh, results_magh["total"], label="total", marker="." if len(B_list_magh) == 1 else None, color="tab:blue", linestyle="--", linewidth=2)

ax[1].fill_between(B_list_magh, 
                [t - s for t, s in zip(results_magh["fwd"], results_magh["fwd_std"])],
                [t + s for t, s in zip(results_magh["fwd"], results_magh["fwd_std"])],
                color='tab:orange', alpha=0.2)
ax[1].plot(B_list_magh, results_magh["fwd"], label="fwd", marker="." if len(B_list_magh) == 1 else None, color="tab:orange", linestyle="--", linewidth=2)

ax[1].fill_between(B_list_magh, 
                [t - s for t, s in zip(results_magh["bwd"], results_magh["bwd_std"])],
                [t + s for t, s in zip(results_magh["bwd"], results_magh["bwd_std"])],
                color='tab:green', alpha=0.2)
ax[1].plot(B_list_magh, results_magh["bwd"], label="bwd", marker="." if len(B_list_magh) == 1 else None, color="tab:green", linestyle="--", linewidth=2)

ax[2].plot(B_list, results_ddtw["mem"], label="dDTW", marker=None, color="k", linewidth=2)
ax[2].plot(B_list_magh, results_magh["mem"], label="Maghoumi", marker="." if len(B_list_magh) == 1 else None, color="k", linestyle="--", linewidth=2)


ax[0].set_xlabel("B")
ax[0].set_ylabel("Time (s)")
ax[0].set_title("dDTW-SDTW")
ax[0].legend()

ax[1].set_xlabel("B")
ax[1].set_ylabel("Time (s)")
ax[1].set_title("Maghoumi-SDTW")
ax[1].legend()

ax[2].set_xlabel("B")
ax[2].set_ylabel("Memory (MB)")
ax[2].set_title("Memory Usage")
ax[2].legend()

ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)
ax[0].set_xlim([min(B_list), max(B_list)])
_set_maghoumi_plot_range(ax[1], B_list_magh)
ax[2].set_xlim([min(B_list), max(B_list)])

ax[0].set_ylim(bottom=0)
ax[1].set_ylim(bottom=0)
ax[2].set_ylim(bottom=0)

fig.suptitle("N=%i, M=%i, D=%i, @ %s" % (N, M, D, torch.cuda.get_device_name(0)))

fig.tight_layout()
_save_benchmark_plot(
    fig, "results/benchmark_SDTW_batch_%s.png" % machine,
    sweep="B", parameters=dict(N=N, M=M, D=D),
    series={
        "ddtw": (B_list, results_ddtw),
        "maghoumi": (B_list_magh, results_magh),
    },
)

# %%


# %% [markdown]
# ## vary dimensions

# %%
N=512
M = 128
D_list = np.concatenate([np.arange(2, 10, 1), np.arange(10, 100, 10), np.arange(100, 1000+1, 100)])
B = 16

# %%
gc.collect()
torch.cuda.empty_cache()

results_ddtw = {"fwd": [], "bwd": [], "total": [], "fwd_std": [], "bwd_std": [], "total_std": [], "mem": []}
results_magh = {"fwd": [], "bwd": [], "total": [], "fwd_std": [], "bwd_std": [], "total_std": [], "mem": []}
D_list_magh = []
maghoumi_active = True
for D in D_list:
    print(f"Benchmarking SDTW with N={N}, M={M}, D={D}, B={B}...")
    
    benchmark_results = run_sdtw_benchmark(len_x=N, len_y=M, dims=D, batch_size=B,implementation="both" if maghoumi_active else "ddtw")
    maghoumi_active = "Maghoumi" in benchmark_results["results"]
    for label, metrics in benchmark_results["results"].items():
        if label == "dDTW":
            results_ddtw["fwd"].append(metrics["forward_ms"])
            results_ddtw["bwd"].append(metrics["backward_ms"])
            results_ddtw["total"].append(metrics["total_ms"])
            results_ddtw["fwd_std"].append(metrics["forward_ms_std"])
            results_ddtw["bwd_std"].append(metrics["backward_ms_std"])
            results_ddtw["total_std"].append(metrics["total_ms_std"])
            results_ddtw["mem"].append(metrics["cuda_peak_allocated_mb"])
        elif label == "Maghoumi":
            D_list_magh.append(D)
            results_magh["fwd"].append(metrics["forward_ms"])
            results_magh["bwd"].append(metrics["backward_ms"])
            results_magh["total"].append(metrics["total_ms"])
            results_magh["fwd_std"].append(metrics["forward_ms_std"])
            results_magh["bwd_std"].append(metrics["backward_ms_std"])
            results_magh["total_std"].append(metrics["total_ms_std"])
            results_magh["mem"].append(metrics["cuda_peak_allocated_mb"])
            


# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4))

ax[0].fill_between(D_list, 
                [t - s for t, s in zip(results_ddtw["total"], results_ddtw["total_std"])],
                [t + s for t, s in zip(results_ddtw["total"], results_ddtw["total_std"])],
                color='tab:blue', alpha=0.2)
ax[0].plot(D_list, results_ddtw["total"], label="total", marker=None, color="tab:blue", linewidth=2)

ax[0].fill_between(D_list, 
                [t - s for t, s in zip(results_ddtw["fwd"], results_ddtw["fwd_std"])],
                [t + s for t, s in zip(results_ddtw["fwd"], results_ddtw["fwd_std"])],
                color='tab:orange', alpha=0.2)
ax[0].plot(D_list, results_ddtw["fwd"], label="fwd", marker=None, color="tab:orange", linewidth=2)

ax[0].fill_between(D_list, 
                [t - s for t, s in zip(results_ddtw["bwd"], results_ddtw["bwd_std"])],
                [t + s for t, s in zip(results_ddtw["bwd"], results_ddtw["bwd_std"])],
                color='tab:green', alpha=0.2)
ax[0].plot(D_list, results_ddtw["bwd"], label="bwd", marker=None, color="tab:green", linewidth=2)





ax[1].fill_between(D_list_magh, 
                [t - s for t, s in zip(results_magh["total"], results_magh["total_std"])],
                [t + s for t, s in zip(results_magh["total"], results_magh["total_std"])],
                color='tab:blue', alpha=0.2)
ax[1].plot(D_list_magh, results_magh["total"], label="total", marker="." if len(D_list_magh) == 1 else None, color="tab:blue", linestyle="--", linewidth=2)

ax[1].fill_between(D_list_magh, 
                [t - s for t, s in zip(results_magh["fwd"], results_magh["fwd_std"])],
                [t + s for t, s in zip(results_magh["fwd"], results_magh["fwd_std"])],
                color='tab:orange', alpha=0.2)
ax[1].plot(D_list_magh, results_magh["fwd"], label="fwd", marker="." if len(D_list_magh) == 1 else None, color="tab:orange", linestyle="--", linewidth=2)

ax[1].fill_between(D_list_magh, 
                [t - s for t, s in zip(results_magh["bwd"], results_magh["bwd_std"])],
                [t + s for t, s in zip(results_magh["bwd"], results_magh["bwd_std"])],
                color='tab:green', alpha=0.2)
ax[1].plot(D_list_magh, results_magh["bwd"], label="bwd", marker="." if len(D_list_magh) == 1 else None, color="tab:green", linestyle="--", linewidth=2)

ax[2].plot(D_list, results_ddtw["mem"], label="dDTW", marker=None, color="k", linewidth=2)
ax[2].plot(D_list_magh, results_magh["mem"], label="Maghoumi", marker="." if len(D_list_magh) == 1 else None, color="k", linestyle="--", linewidth=2)


ax[0].set_xlabel("D")
ax[0].set_ylabel("Time (s)")
ax[0].set_title("dDTW-SDTW")
ax[0].legend()

ax[1].set_xlabel("D")
ax[1].set_ylabel("Time (s)")
ax[1].set_title("Maghoumi-SDTW")
ax[1].legend()

ax[2].set_xlabel("D")
ax[2].set_ylabel("Memory (MB)")
ax[2].set_title("Memory Usage")
ax[2].legend()

ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)
ax[0].set_xlim([min(D_list), max(D_list)])
_set_maghoumi_plot_range(ax[1], D_list_magh)
ax[2].set_xlim([min(D_list), max(D_list)])

ax[0].set_ylim(bottom=0)
ax[1].set_ylim(bottom=0)
ax[2].set_ylim(bottom=0)

fig.suptitle("N=%i, M=%i, B=%i, @ %s" % (N, M, B, torch.cuda.get_device_name(0)))

fig.tight_layout()
_save_benchmark_plot(
    fig, "results/benchmark_SDTW_dim_%s.png" % machine,
    sweep="D", parameters=dict(N=N, M=M, B=B),
    series={
        "ddtw": (D_list, results_ddtw),
        "maghoumi": (D_list_magh, results_magh),
    },
)

# %% [markdown]
# ## =================================

# %% [markdown]
# # CTC

# %% [markdown]
# ## vary one seq. length - target seg.

# %%
N=512
M_list = np.concatenate([np.arange(10, 100, 10), 
                         np.arange(100, 500+1, 100)])#, 
                         #np.arange(1000, 10000+1, 1000)])
D = 88
B = 16

# %%
gc.collect()
torch.cuda.empty_cache()


results_ddtw = {"fwd": [], "bwd": [], "total": [], "fwd_std": [], "bwd_std": [], "total_std": [], "mem": []}
results_torch = {"fwd": [], "bwd": [], "total": [], "fwd_std": [], "bwd_std": [], "total_std": [], "mem": []}
for M in M_list:
    print(f"Benchmarking CTC with N={N}, M={M}, D={D}, B={B}...")
    
    benchmark_results = run_ctc_benchmark(input_len=N, target_len=M, num_classes=D, batch_size=B)
    for label, metrics in benchmark_results["results"].items():
        if label == "dDTW":
            results_ddtw["fwd"].append(metrics["forward_ms"])
            results_ddtw["bwd"].append(metrics["backward_ms"])
            results_ddtw["total"].append(metrics["total_ms"])
            results_ddtw["fwd_std"].append(metrics["forward_ms_std"])
            results_ddtw["bwd_std"].append(metrics["backward_ms_std"])
            results_ddtw["total_std"].append(metrics["total_ms_std"])
            results_ddtw["mem"].append(metrics["cuda_peak_allocated_mb"])
        elif label == "PyTorch":
            results_torch["fwd"].append(metrics["forward_ms"])
            results_torch["bwd"].append(metrics["backward_ms"])
            results_torch["total"].append(metrics["total_ms"])
            results_torch["fwd_std"].append(metrics["forward_ms_std"])
            results_torch["bwd_std"].append(metrics["backward_ms_std"])
            results_torch["total_std"].append(metrics["total_ms_std"])
            results_torch["mem"].append(metrics["cuda_peak_allocated_mb"])
            


# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4))

ax[0].fill_between(M_list, 
                [t - s for t, s in zip(results_ddtw["total"], results_ddtw["total_std"])],
                [t + s for t, s in zip(results_ddtw["total"], results_ddtw["total_std"])],
                color='tab:blue', alpha=0.2)
ax[0].plot(M_list, results_ddtw["total"], label="total", marker=None, color="tab:blue", linewidth=2)

ax[0].fill_between(M_list, 
                [t - s for t, s in zip(results_ddtw["fwd"], results_ddtw["fwd_std"])],
                [t + s for t, s in zip(results_ddtw["fwd"], results_ddtw["fwd_std"])],
                color='tab:orange', alpha=0.2)
ax[0].plot(M_list, results_ddtw["fwd"], label="fwd", marker=None, color="tab:orange", linewidth=2)

ax[0].fill_between(M_list, 
                [t - s for t, s in zip(results_ddtw["bwd"], results_ddtw["bwd_std"])],
                [t + s for t, s in zip(results_ddtw["bwd"], results_ddtw["bwd_std"])],
                color='tab:green', alpha=0.2)
ax[0].plot(M_list, results_ddtw["bwd"], label="bwd", marker=None, color="tab:green", linewidth=2)



ax[1].fill_between(M_list, 
                [t - s for t, s in zip(results_torch["total"], results_torch["total_std"])],
                [t + s for t, s in zip(results_torch["total"], results_torch["total_std"])],
                color='tab:blue', alpha=0.2)
ax[1].plot(M_list, results_torch["total"], label="total", marker=None, color="tab:blue", linestyle="--", linewidth=2)

ax[1].fill_between(M_list, 
                [t - s for t, s in zip(results_torch["fwd"], results_torch["fwd_std"])],
                [t + s for t, s in zip(results_torch["fwd"], results_torch["fwd_std"])],
                color='tab:orange', alpha=0.2)
ax[1].plot(M_list, results_torch["fwd"], label="fwd", marker=None, color="tab:orange", linestyle="--", linewidth=2)

ax[1].fill_between(M_list, 
                [t - s for t, s in zip(results_torch["bwd"], results_torch["bwd_std"])],
                [t + s for t, s in zip(results_torch["bwd"], results_torch["bwd_std"])],
                color='tab:green', alpha=0.2)
ax[1].plot(M_list, results_torch["bwd"], label="bwd", marker=None, color="tab:green", linestyle="--", linewidth=2)

ax[2].plot(M_list, results_ddtw["mem"], label="dDTW", marker=None, color="k", linewidth=2)
ax[2].plot(M_list, results_torch["mem"], label="PyTorch", marker=None, color="k", linestyle="--", linewidth=2)


ax[0].set_xlabel("M")
ax[0].set_ylabel("Time (s)")
ax[0].set_title("dDTW-CTC")
ax[0].legend()

ax[1].set_xlabel("M")
ax[1].set_ylabel("Time (s)")
ax[1].set_title("Torch-CTC")
ax[1].legend()

ax[2].set_xlabel("M")
ax[2].set_ylabel("Memory (MB)")
ax[2].set_title("Memory Usage")
ax[2].legend()

ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)
ax[0].set_xlim([min(M_list), max(M_list)])
ax[1].set_xlim([min(M_list), max(M_list)])
ax[2].set_xlim([min(M_list), max(M_list)])

ax[0].set_ylim(bottom=0)
ax[1].set_ylim(bottom=0)
ax[2].set_ylim(bottom=0)

fig.suptitle("N=%i, D=%i, B=%i @ %s" % (N, D, B, torch.cuda.get_device_name(0)))

fig.tight_layout()
_save_benchmark_plot(
    fig, "results/benchmark_CTC_targetLen_%s.png" % machine,
    sweep="M", parameters=dict(N=N, D=D, B=B),
    series={
        "ddtw": (M_list, results_ddtw),
        "pytorch": (M_list, results_torch),
    },
)

# %% [markdown]
# ## vary one seq. length - input seg.

# %%
N_list=np.concatenate([[128],
                        np.arange(200, 1000, 100), 
                         np.arange(1000, 10000+1, 1000)])
M  = 128 
D = 88
B = 16

# %%
gc.collect()
torch.cuda.empty_cache()


results_ddtw = {"fwd": [], "bwd": [], "total": [], "fwd_std": [], "bwd_std": [], "total_std": [], "mem": []}
results_torch = {"fwd": [], "bwd": [], "total": [], "fwd_std": [], "bwd_std": [], "total_std": [], "mem": []}
for N in N_list:
    print(f"Benchmarking CTC with N={N}, M={M}, D={D}, B={B}...")
    
    benchmark_results = run_ctc_benchmark(input_len=N, target_len=M, num_classes=D, batch_size=B)
    for label, metrics in benchmark_results["results"].items():
        if label == "dDTW":
            results_ddtw["fwd"].append(metrics["forward_ms"])
            results_ddtw["bwd"].append(metrics["backward_ms"])
            results_ddtw["total"].append(metrics["total_ms"])
            results_ddtw["fwd_std"].append(metrics["forward_ms_std"])
            results_ddtw["bwd_std"].append(metrics["backward_ms_std"])
            results_ddtw["total_std"].append(metrics["total_ms_std"])
            results_ddtw["mem"].append(metrics["cuda_peak_allocated_mb"])
        elif label == "PyTorch":
            results_torch["fwd"].append(metrics["forward_ms"])
            results_torch["bwd"].append(metrics["backward_ms"])
            results_torch["total"].append(metrics["total_ms"])
            results_torch["fwd_std"].append(metrics["forward_ms_std"])
            results_torch["bwd_std"].append(metrics["backward_ms_std"])
            results_torch["total_std"].append(metrics["total_ms_std"])
            results_torch["mem"].append(metrics["cuda_peak_allocated_mb"])
            


# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4))

ax[0].fill_between(N_list, 
                [t - s for t, s in zip(results_ddtw["total"], results_ddtw["total_std"])],
                [t + s for t, s in zip(results_ddtw["total"], results_ddtw["total_std"])],
                color='tab:blue', alpha=0.2)
ax[0].plot(N_list, results_ddtw["total"], label="total", marker=None, color="tab:blue", linewidth=2)

ax[0].fill_between(N_list, 
                [t - s for t, s in zip(results_ddtw["fwd"], results_ddtw["fwd_std"])],
                [t + s for t, s in zip(results_ddtw["fwd"], results_ddtw["fwd_std"])],
                color='tab:orange', alpha=0.2)
ax[0].plot(N_list, results_ddtw["fwd"], label="fwd", marker=None, color="tab:orange", linewidth=2)

ax[0].fill_between(N_list, 
                [t - s for t, s in zip(results_ddtw["bwd"], results_ddtw["bwd_std"])],
                [t + s for t, s in zip(results_ddtw["bwd"], results_ddtw["bwd_std"])],
                color='tab:green', alpha=0.2)
ax[0].plot(N_list, results_ddtw["bwd"], label="bwd", marker=None, color="tab:green", linewidth=2)



ax[1].fill_between(N_list, 
                [t - s for t, s in zip(results_torch["total"], results_torch["total_std"])],
                [t + s for t, s in zip(results_torch["total"], results_torch["total_std"])],
                color='tab:blue', alpha=0.2)
ax[1].plot(N_list, results_torch["total"], label="total", marker=None, color="tab:blue", linestyle="--", linewidth=2)

ax[1].fill_between(N_list, 
                [t - s for t, s in zip(results_torch["fwd"], results_torch["fwd_std"])],
                [t + s for t, s in zip(results_torch["fwd"], results_torch["fwd_std"])],
                color='tab:orange', alpha=0.2)
ax[1].plot(N_list, results_torch["fwd"], label="fwd", marker=None, color="tab:orange", linestyle="--", linewidth=2)

ax[1].fill_between(N_list, 
                [t - s for t, s in zip(results_torch["bwd"], results_torch["bwd_std"])],
                [t + s for t, s in zip(results_torch["bwd"], results_torch["bwd_std"])],
                color='tab:green', alpha=0.2)
ax[1].plot(N_list, results_torch["bwd"], label="bwd", marker=None, color="tab:green", linestyle="--", linewidth=2)

ax[2].plot(N_list, results_ddtw["mem"], label="dDTW", marker=None, color="k", linewidth=2)
ax[2].plot(N_list, results_torch["mem"], label="PyTorch", marker=None, color="k", linestyle="--", linewidth=2)


ax[0].set_xlabel("N")
ax[0].set_ylabel("Time (s)")
ax[0].set_title("dDTW-CTC")
ax[0].legend()

ax[1].set_xlabel("N")
ax[1].set_ylabel("Time (s)")
ax[1].set_title("Torch-CTC")
ax[1].legend()

ax[2].set_xlabel("N")
ax[2].set_ylabel("Memory (MB)")
ax[2].set_title("Memory Usage")
ax[2].legend()

ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)
ax[0].set_xlim([min(N_list), max(N_list)])
ax[1].set_xlim([min(N_list), max(N_list)])
ax[2].set_xlim([min(N_list), max(N_list)])

ax[0].set_ylim(bottom=0)
ax[1].set_ylim(bottom=0)
ax[2].set_ylim(bottom=0)

fig.suptitle("M=%i, D=%i, B=%i @ %s" % (M, D, B, torch.cuda.get_device_name(0)))

fig.tight_layout()
_save_benchmark_plot(
    fig, "results/benchmark_CTC_inputLen_%s.png" % machine,
    sweep="N", parameters=dict(M=M, D=D, B=B),
    series={
        "ddtw": (N_list, results_ddtw),
        "pytorch": (N_list, results_torch),
    },
)

# %% [markdown]
# ## vary batch size

# %%
N = 512
M  = 128 
D = 88
B_list = np.concatenate([np.arange(1, 10, 1), np.arange(10, 100, 10), np.arange(100, 1000+1, 100)])

# %%
gc.collect()
torch.cuda.empty_cache()


results_ddtw = {"fwd": [], "bwd": [], "total": [], "fwd_std": [], "bwd_std": [], "total_std": [], "mem": []}
results_torch = {"fwd": [], "bwd": [], "total": [], "fwd_std": [], "bwd_std": [], "total_std": [], "mem": []}
for B in B_list:
    print(f"Benchmarking CTC with N={N}, M={M}, D={D}, B={B}...")
    
    benchmark_results = run_ctc_benchmark(input_len=N, target_len=M, num_classes=D, batch_size=B)
    for label, metrics in benchmark_results["results"].items():
        if label == "dDTW":
            results_ddtw["fwd"].append(metrics["forward_ms"])
            results_ddtw["bwd"].append(metrics["backward_ms"])
            results_ddtw["total"].append(metrics["total_ms"])
            results_ddtw["fwd_std"].append(metrics["forward_ms_std"])
            results_ddtw["bwd_std"].append(metrics["backward_ms_std"])
            results_ddtw["total_std"].append(metrics["total_ms_std"])
            results_ddtw["mem"].append(metrics["cuda_peak_allocated_mb"])
        elif label == "PyTorch":
            results_torch["fwd"].append(metrics["forward_ms"])
            results_torch["bwd"].append(metrics["backward_ms"])
            results_torch["total"].append(metrics["total_ms"])
            results_torch["fwd_std"].append(metrics["forward_ms_std"])
            results_torch["bwd_std"].append(metrics["backward_ms_std"])
            results_torch["total_std"].append(metrics["total_ms_std"])
            results_torch["mem"].append(metrics["cuda_peak_allocated_mb"])
            


# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4))

ax[0].fill_between(B_list, 
                [t - s for t, s in zip(results_ddtw["total"], results_ddtw["total_std"])],
                [t + s for t, s in zip(results_ddtw["total"], results_ddtw["total_std"])],
                color='tab:blue', alpha=0.2)
ax[0].plot(B_list, results_ddtw["total"], label="total", marker=None, color="tab:blue", linewidth=2)

ax[0].fill_between(B_list, 
                [t - s for t, s in zip(results_ddtw["fwd"], results_ddtw["fwd_std"])],
                [t + s for t, s in zip(results_ddtw["fwd"], results_ddtw["fwd_std"])],
                color='tab:orange', alpha=0.2)
ax[0].plot(B_list, results_ddtw["fwd"], label="fwd", marker=None, color="tab:orange", linewidth=2)

ax[0].fill_between(B_list, 
                [t - s for t, s in zip(results_ddtw["bwd"], results_ddtw["bwd_std"])],
                [t + s for t, s in zip(results_ddtw["bwd"], results_ddtw["bwd_std"])],
                color='tab:green', alpha=0.2)
ax[0].plot(B_list, results_ddtw["bwd"], label="bwd", marker=None, color="tab:green", linewidth=2)



ax[1].fill_between(B_list, 
                [t - s for t, s in zip(results_torch["total"], results_torch["total_std"])],
                [t + s for t, s in zip(results_torch["total"], results_torch["total_std"])],
                color='tab:blue', alpha=0.2)
ax[1].plot(B_list, results_torch["total"], label="total", marker=None, color="tab:blue", linestyle="--", linewidth=2)

ax[1].fill_between(B_list, 
                [t - s for t, s in zip(results_torch["fwd"], results_torch["fwd_std"])],
                [t + s for t, s in zip(results_torch["fwd"], results_torch["fwd_std"])],
                color='tab:orange', alpha=0.2)
ax[1].plot(B_list, results_torch["fwd"], label="fwd", marker=None, color="tab:orange", linestyle="--", linewidth=2)

ax[1].fill_between(B_list, 
                [t - s for t, s in zip(results_torch["bwd"], results_torch["bwd_std"])],
                [t + s for t, s in zip(results_torch["bwd"], results_torch["bwd_std"])],
                color='tab:green', alpha=0.2)
ax[1].plot(B_list, results_torch["bwd"], label="bwd", marker=None, color="tab:green", linestyle="--", linewidth=2)

ax[2].plot(B_list, results_ddtw["mem"], label="dDTW", marker=None, color="k", linewidth=2)
ax[2].plot(B_list, results_torch["mem"], label="PyTorch", marker=None, color="k", linestyle="--", linewidth=2)


ax[0].set_xlabel("B")
ax[0].set_ylabel("Time (s)")
ax[0].set_title("dDTW-CTC")
ax[0].legend()

ax[1].set_xlabel("B")
ax[1].set_ylabel("Time (s)")
ax[1].set_title("Torch-CTC")
ax[1].legend()

ax[2].set_xlabel("B")
ax[2].set_ylabel("Memory (MB)")
ax[2].set_title("Memory Usage")
ax[2].legend()

ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)
ax[0].set_xlim([min(B_list), max(B_list)])
ax[1].set_xlim([min(B_list), max(B_list)])
ax[2].set_xlim([min(B_list), max(B_list)])

ax[0].set_ylim(bottom=0)
ax[1].set_ylim(bottom=0)
ax[2].set_ylim(bottom=0)

fig.suptitle("N=%i, M=%i D=%i @ %s" % (N, M, D, torch.cuda.get_device_name(0)))

fig.tight_layout()
_save_benchmark_plot(
    fig, "results/benchmark_CTC_batch_%s.png" % machine,
    sweep="B", parameters=dict(N=N, M=M, D=D),
    series={
        "ddtw": (B_list, results_ddtw),
        "pytorch": (B_list, results_torch),
    },
)

# %% [markdown]
# ## vary dimensions

# %%
N = 512
M  = 128 
D_list = np.concatenate([np.arange(2, 10, 1), np.arange(10, 100, 10), np.arange(100, 1000+1, 100)])
B = 16

# %%
gc.collect()
torch.cuda.empty_cache()


results_ddtw = {"fwd": [], "bwd": [], "total": [], "fwd_std": [], "bwd_std": [], "total_std": [], "mem": []}
results_torch = {"fwd": [], "bwd": [], "total": [], "fwd_std": [], "bwd_std": [], "total_std": [], "mem": []}
for D in D_list:
    print(f"Benchmarking CTC with N={N}, M={M}, D={D}, B={B}...")
    
    benchmark_results = run_ctc_benchmark(input_len=N, target_len=M, num_classes=D, batch_size=B)
    for label, metrics in benchmark_results["results"].items():
        if label == "dDTW":
            results_ddtw["fwd"].append(metrics["forward_ms"])
            results_ddtw["bwd"].append(metrics["backward_ms"])
            results_ddtw["total"].append(metrics["total_ms"])
            results_ddtw["fwd_std"].append(metrics["forward_ms_std"])
            results_ddtw["bwd_std"].append(metrics["backward_ms_std"])
            results_ddtw["total_std"].append(metrics["total_ms_std"])
            results_ddtw["mem"].append(metrics["cuda_peak_allocated_mb"])
        elif label == "PyTorch":
            results_torch["fwd"].append(metrics["forward_ms"])
            results_torch["bwd"].append(metrics["backward_ms"])
            results_torch["total"].append(metrics["total_ms"])
            results_torch["fwd_std"].append(metrics["forward_ms_std"])
            results_torch["bwd_std"].append(metrics["backward_ms_std"])
            results_torch["total_std"].append(metrics["total_ms_std"])
            results_torch["mem"].append(metrics["cuda_peak_allocated_mb"])
            


# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4))

ax[0].fill_between(D_list, 
                [t - s for t, s in zip(results_ddtw["total"], results_ddtw["total_std"])],
                [t + s for t, s in zip(results_ddtw["total"], results_ddtw["total_std"])],
                color='tab:blue', alpha=0.2)
ax[0].plot(D_list, results_ddtw["total"], label="total", marker=None, color="tab:blue", linewidth=2)

ax[0].fill_between(D_list, 
                [t - s for t, s in zip(results_ddtw["fwd"], results_ddtw["fwd_std"])],
                [t + s for t, s in zip(results_ddtw["fwd"], results_ddtw["fwd_std"])],
                color='tab:orange', alpha=0.2)
ax[0].plot(D_list, results_ddtw["fwd"], label="fwd", marker=None, color="tab:orange", linewidth=2)

ax[0].fill_between(D_list, 
                [t - s for t, s in zip(results_ddtw["bwd"], results_ddtw["bwd_std"])],
                [t + s for t, s in zip(results_ddtw["bwd"], results_ddtw["bwd_std"])],
                color='tab:green', alpha=0.2)
ax[0].plot(D_list, results_ddtw["bwd"], label="bwd", marker=None, color="tab:green", linewidth=2)



ax[1].fill_between(D_list, 
                [t - s for t, s in zip(results_torch["total"], results_torch["total_std"])],
                [t + s for t, s in zip(results_torch["total"], results_torch["total_std"])],
                color='tab:blue', alpha=0.2)
ax[1].plot(D_list, results_torch["total"], label="total", marker=None, color="tab:blue", linestyle="--", linewidth=2)

ax[1].fill_between(D_list, 
                [t - s for t, s in zip(results_torch["fwd"], results_torch["fwd_std"])],
                [t + s for t, s in zip(results_torch["fwd"], results_torch["fwd_std"])],
                color='tab:orange', alpha=0.2)
ax[1].plot(D_list, results_torch["fwd"], label="fwd", marker=None, color="tab:orange", linestyle="--", linewidth=2)

ax[1].fill_between(D_list, 
                [t - s for t, s in zip(results_torch["bwd"], results_torch["bwd_std"])],
                [t + s for t, s in zip(results_torch["bwd"], results_torch["bwd_std"])],
                color='tab:green', alpha=0.2)
ax[1].plot(D_list, results_torch["bwd"], label="bwd", marker=None, color="tab:green", linestyle="--", linewidth=2)

ax[2].plot(D_list, results_ddtw["mem"], label="dDTW", marker=None, color="k", linewidth=2)
ax[2].plot(D_list, results_torch["mem"], label="PyTorch", marker=None, color="k", linestyle="--", linewidth=2)


ax[0].set_xlabel("D")
ax[0].set_ylabel("Time (s)")
ax[0].set_title("dDTW-CTC")
ax[0].legend()

ax[1].set_xlabel("D")
ax[1].set_ylabel("Time (s)")
ax[1].set_title("Torch-CTC")
ax[1].legend()

ax[2].set_xlabel("D")
ax[2].set_ylabel("Memory (MB)")
ax[2].set_title("Memory Usage")
ax[2].legend()

ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)
ax[0].set_xlim([min(D_list), max(D_list)])
ax[1].set_xlim([min(D_list), max(D_list)])
ax[2].set_xlim([min(D_list), max(D_list)])

ax[0].set_ylim(bottom=0)
ax[1].set_ylim(bottom=0)
ax[2].set_ylim(bottom=0)

fig.suptitle("N=%i, M=%i B=%i @ %s" % (N, M, B, torch.cuda.get_device_name(0)))

fig.tight_layout()
_save_benchmark_plot(
    fig, "results/benchmark_CTC_dim_%s.png" % machine,
    sweep="D", parameters=dict(N=N, M=M, B=B),
    series={
        "ddtw": (D_list, results_ddtw),
        "pytorch": (D_list, results_torch),
    },
)


