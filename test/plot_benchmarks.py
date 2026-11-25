#!/usr/bin/env python3
"""Plot saved benchmark archives as PNGs; never run the benchmarks.

Usage from the repository root:
    python test/plot_benchmarks.py
    python test/plot_benchmarks.py --data-dir test/results --output-dir plots

Outputs:
    runtime_<experiment>_<gpu>.png: forward/backward/total median +/- SD.
    memory_by_experiment.png: all experiments, implementations and GPUs.
    total_runtime_by_gpu.png: all experiments, implementations and GPUs.

Requires NumPy and Matplotlib. Only GPU metadata, numeric measurements and
fixed dimensions are read; private machine names never enter the plots.
PNG backgrounds are transparent; legend backgrounds remain opaque white.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
import numpy as np


# SDTW has three experiments; the free top-left panel holds title and legends.
EXPERIMENT_ROWS = (
    (None, "CTC_inputLen"),
    ("SDTW_longSeg", "CTC_targetLen"),
    ("SDTW_batch", "CTC_batch"),
    ("SDTW_dim", "CTC_dim"),
)
EXPERIMENTS = {
    "SDTW_longSeg": ("Sequence length", "Sequence length M", "M"),
    "SDTW_batch": ("Batch size", "Batch size B", "B"),
    "SDTW_dim": ("Feature dimensionality", "Feature dimensions D", "D"),
    "CTC_inputLen": ("Input sequence length", "Input length N", "N"),
    "CTC_targetLen": ("Target sequence length", "Target length M", "M"),
    "CTC_batch": ("Batch size", "Batch size B", "B"),
    "CTC_dim": ("Number of classes", "Classes C", "D"),
}
GPU_NAMES = {
    "NVIDIA GeForce RTX 3090": "RTX 3090",
    "NVIDIA GeForce RTX 4090": "RTX 4090",
    "NVIDIA GeForce RTX 5090": "RTX 5090",
    "NVIDIA RTX A5500": "RTX A5500",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": "RTX PRO 6000 Blackwell",
}
IMPLEMENTATIONS = {"ddtw": "dDTW", "maghoumi": "Maghoumi", "pytorch": "PyTorch"}
PHASES = {"fwd": ("Forward", "#2563eb"), "bwd": ("Backward", "#d97706"), "total": ("Total", "#059669")}
METRICS = ("fwd", "bwd", "total", "fwd_std", "bwd_std", "total_std", "mem")


@dataclass
class Run:
    experiment: str
    gpu: str
    fixed: dict[str, int]
    series: dict[str, dict[str, np.ndarray]]


def load_runs(directory: Path) -> list[Run]:
    """Retain each implementation's actual x values, including empty series."""
    pattern = re.compile(r"^benchmark_(" + "|".join(EXPERIMENTS) + r")_")
    runs = []
    seen = set()
    for path in sorted(directory.rglob("*.npz")):
        match = pattern.match(path.stem)
        if not match:
            continue
        experiment = match.group(1)  # The private suffix is deliberately unused.
        with np.load(path, allow_pickle=False) as data:
            gpu = str(data["gpu_name"].item())
            label = f"{experiment}, {GPU_NAMES.get(gpu, gpu)}"
            if (experiment, gpu) in seen:
                raise ValueError(f"Multiple archives for {label}; choose a narrower --data-dir")
            seen.add((experiment, gpu))
            if str(data["time_unit"].item()) != "ms" or str(data["memory_unit"].item()) != "MiB":
                raise ValueError(f"Unexpected measurement units for {label}")
            variable = EXPERIMENTS[experiment][2]
            if str(data["sweep"].item()) != variable:
                raise ValueError(f"Unexpected sweep variable for {label}")
            fixed = {key: int(data[key].item()) for key in "NMBD" if key in data}
            if set(fixed) != set("NMBD") - {variable}:
                raise ValueError(f"Incomplete fixed dimensions for {label}")
            reference = "maghoumi" if experiment.startswith("SDTW") else "pytorch"
            series = {}
            for impl in ("ddtw", reference):
                if impl + "_x" not in data:
                    continue
                measurements = {key: np.array(data[f"{impl}_{key}"], dtype=float)
                                for key in ("x", *METRICS)}
                x = measurements["x"]
                if x.ndim != 1 or any(values.shape != x.shape for values in measurements.values()):
                    raise ValueError(f"Misaligned arrays for {label}, {IMPLEMENTATIONS[impl]}")
                if np.any(x <= 0) or np.any(np.diff(x) <= 0):
                    raise ValueError(f"Nonpositive or unsorted x values for {label}")
                for key, values in measurements.items():
                    if not np.all(np.isfinite(values)) or np.any(values < 0):
                        raise ValueError(f"Invalid {key} values for {label}")
                    if key in ("fwd", "bwd", "total", "mem") and np.any(values <= 0):
                        raise ValueError(f"Nonpositive {key} values for {label}")
                series[impl] = measurements
            if not series or not any(len(values["x"]) for values in series.values()):
                raise ValueError(f"No measurements for {label}")
            runs.append(Run(experiment, gpu, fixed, series))
    if not runs:
        raise ValueError("No benchmark NPZ measurements found in --data-dir")
    for experiment in EXPERIMENTS:
        conditions = {tuple(sorted(run.fixed.items())) for run in runs if run.experiment == experiment}
        if len(conditions) > 1:
            raise ValueError(f"Fixed dimensions differ across GPUs for {experiment}")
    return runs


def fixed_label(run: Run) -> str:
    return " · ".join(f"{'C' if key == 'D' and run.experiment.startswith('CTC') else key}={value:,}"
                      for key, value in run.fixed.items())


def style_axis(axis, experiment: str, ylabel: str, linear: bool):
    axis.set_xscale("log")
    if linear:
        axis.set_ylim(bottom=0)
    else:
        axis.set_yscale("log")
    axis.set_xlabel(EXPERIMENTS[experiment][1])
    axis.set_ylabel(ylabel)
    axis.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=1000))
    axis.yaxis.set_minor_locator(AutoMinorLocator(4) if linear else
                                 LogLocator(base=10, subs=np.arange(2, 10), numticks=1000))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.grid(True, which="major", color="#dbe3ec", linewidth=0.7)
    axis.grid(True, which="minor", color="#edf1f5", linewidth=0.45)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=9)


def runtime_plot(run: Run, output: Path, dpi: int, linear: bool):
    figure, axis = plt.subplots(figsize=(9, 5.7))
    for impl, values in run.series.items():
        x = values["x"]
        if not len(x):
            continue
        for phase, (name, color) in PHASES.items():
            median, std = values[phase], values[phase + "_std"]
            # A negative lower bound cannot appear on a logarithmic axis.
            lower = np.maximum(median - std, 0 if linear else median.min() * 0.01)
            axis.fill_between(x, lower, median + std, color=color, alpha=0.12, linewidth=0)
            axis.plot(x, median, color=color, linestyle="-" if impl == "ddtw" else "--",
                      linewidth=2, marker="." if len(x) == 1 else None,
                      label=f"{IMPLEMENTATIONS[impl]} · {name}")
    style_axis(axis, run.experiment, "Runtime (ms)", linear)
    title = EXPERIMENTS[run.experiment][0]
    gpu = GPU_NAMES.get(run.gpu, run.gpu.removeprefix("NVIDIA "))
    axis.set_title(f"{run.experiment.split('_')[0]} · {title}\n{gpu} · {fixed_label(run)}",
                   loc="left", fontsize=13, pad=14)
    axis.legend(ncol=2, loc="best", fontsize=9, frameon=True,
                facecolor="white", framealpha=1)
    figure.text(0.11, 0.025, "Lines: medians · Shading: ±1 trial standard deviation · Solid: dDTW · Dashed: baseline",
                fontsize=9, color="#526575")
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    slug = re.sub(r"[^a-z0-9]+", "_", gpu.lower()).strip("_")
    figure.savefig(output / f"runtime_{run.experiment}_{slug}.png", dpi=dpi,
                   transparent=True, facecolor="none", edgecolor="none")
    plt.close(figure)


def grouped_plot(runs: list[Run], gpus: list[str], metric: str, output: Path, dpi: int, linear: bool):
    """Seven experiments plus a title/legend panel: SDTW left, CTC right."""
    figure, axes = plt.subplots(4, 2, figsize=(14, 15))
    palette = plt.get_cmap("tab10")
    colors = {gpu: palette(index % 10) for index, gpu in enumerate(gpus)}
    for row, experiments in enumerate(EXPERIMENT_ROWS):
        for column, experiment in enumerate(experiments):
            axis = axes[row, column]
            if experiment is None:
                axis.set_axis_off()
                continue
            selected = [run for run in runs if run.experiment == experiment]
            ylabel = "Peak allocated CUDA memory (MiB)" if metric == "mem" else "Median total runtime (ms)"
            style_axis(axis, experiment, ylabel, linear)
            title = EXPERIMENTS[experiment][0]
            axis.set_title(f"{experiment.split('_')[0]} · {title}\n" + (fixed_label(selected[0]) if selected else "No measurements"),
                           loc="left", fontsize=11, pad=10)
            for run in selected:
                for impl, values in run.series.items():
                    if len(values["x"]):
                        axis.plot(values["x"], values[metric], color=colors[run.gpu],
                                  linestyle="-" if impl == "ddtw" else "--", linewidth=1.8,
                                  marker="." if len(values["x"]) == 1 else None)
    gpu_handles = [Line2D([], [], color=colors[gpu], linewidth=2,
                          label=GPU_NAMES.get(gpu, gpu.removeprefix("NVIDIA "))) for gpu in gpus]
    style_handles = [Line2D([], [], color="#334155", linewidth=2, linestyle=style, label=label)
                     for style, label in (("-", "dDTW"), ("--", "Baseline: Maghoumi (SDTW) / PyTorch (CTC)"))]
    title = "Peak allocated memory\nby experiment" if metric == "mem" else "Total runtime across\nGPU models"
    legend_axis = axes[0, 0]
    legend_axis.text(0, 1.0, title, transform=legend_axis.transAxes, va="top",
                     fontsize=19, fontweight="bold", color="#162b36")
    gpu_legend = legend_axis.legend(handles=gpu_handles, loc="upper left", bbox_to_anchor=(0, 0.69),
                                    ncol=2, frameon=True, facecolor="white", framealpha=1,
                                    edgecolor="none", fontsize=10, borderaxespad=0)
    legend_axis.add_artist(gpu_legend)
    legend_axis.legend(handles=style_handles, loc="upper left", bbox_to_anchor=(0, 0.27),
                       ncol=1, frameon=True, facecolor="white", framealpha=1,
                       edgecolor="none", fontsize=10, borderaxespad=0)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.085, top=0.95, wspace=0.24, hspace=0.50)
    note = ("Peak live allocation includes input tensors; memory curves can coincide across GPU models."
            if metric == "mem" else "Total = forward + backward. Curves show recorded system performance, including software/configuration effects.")
    figure.text(0.08, 0.018, note + "\nEach series ends at its last recorded measurement; no extrapolation.",
                fontsize=10, color="#526575")
    name = "memory_by_experiment.png" if metric == "mem" else "total_runtime_by_gpu.png"
    figure.savefig(output / name, dpi=dpi, transparent=True, facecolor="none", edgecolor="none")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Search recursively for NPZ archives (default: test/).")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "benchmark_plots")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--linear", action="store_true", help="Use linear y axes instead of logarithmic y axes; x stays logarithmic.")
    args = parser.parse_args()
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    try:
        runs = load_runs(args.data_dir)
    except ValueError as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Remove the excluded experiment's PNGs when reusing an earlier output folder.
    for obsolete in args.output_dir.glob("runtime_SDTW_shortSeg_*.png"):
        obsolete.unlink()
    known = list(GPU_NAMES)
    gpus = sorted({run.gpu for run in runs}, key=lambda gpu: (known.index(gpu) if gpu in known else len(known), gpu))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    for run in runs:
        runtime_plot(run, args.output_dir, args.dpi, args.linear)
    for metric in ("mem", "total"):
        grouped_plot(runs, gpus, metric, args.output_dir, args.dpi, args.linear)
    print(f"Saved {len(runs)} runtime PNGs and 2 grouped PNGs to {args.output_dir}")


if __name__ == "__main__":
    main()
