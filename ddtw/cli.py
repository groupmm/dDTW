"""Install and check the tested Linux CUDA stack inside a Conda environment."""

import argparse
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig

from ._cuda_config import CUDA_VERSION, TORCH_VERSION, config_path, toolchain


def _run(command, **kwargs):
    print("+ " + shlex.join([str(arg) for arg in command]), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def _conda_environment():
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Automatic CUDA setup supports Linux x86_64 only.")
    if (platform.python_implementation() != "CPython"
            or not (3, 10) <= sys.version_info[:2] <= (3, 13)
            or sysconfig.get_config_var("Py_GIL_DISABLED")):
        raise RuntimeError("CUDA setup supports CPython 3.10–3.13 (standard builds). Use Python 3.12 for the tested default.")
    prefix = Path(sys.prefix).resolve()
    active = os.environ.get("CONDA_PREFIX")
    if not active or Path(active).resolve() != prefix or not (prefix / "conda-meta").is_dir():
        raise RuntimeError(
            "Activate a dedicated Conda environment first:\n"
            "  conda create -n ddtw python=3.12 pip\n"
            "  conda activate ddtw\n"
            "  python -m pip install ddtw\n"
            "  ddtw setup-cuda\n"
            "A plain Python venv cannot supply the Conda compiler packages."
        )
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda or not shutil.which(conda):
        raise RuntimeError("Conda was not found. Initialize Conda and activate your environment again.")
    base = subprocess.check_output([conda, "info", "--base"], text=True).strip()
    if prefix == Path(base).resolve():
        raise RuntimeError("Create and activate a dedicated Conda environment; setup does not modify base.")
    return conda, prefix


def _check_driver():
    smi = shutil.which("nvidia-smi")
    if not smi:
        raise RuntimeError("nvidia-smi was not found. A working NVIDIA driver must already be installed.")
    info = subprocess.check_output([smi], text=True, stderr=subprocess.STDOUT)
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", info)
    if not match or tuple(map(int, match.groups())) < (12, 8):
        raise RuntimeError("The default setup requires an NVIDIA driver reporting CUDA support of 12.8 or newer in nvidia-smi.")
    devices = subprocess.check_output([smi, "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True)
    if not devices.strip():
        raise RuntimeError("No NVIDIA GPU was detected by nvidia-smi.")
    print("NVIDIA GPU / driver: " + devices.strip(), flush=True)


def setup_cuda(dry_run=False):
    conda, prefix = _conda_environment()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    commands = [
        [conda, "install", "--yes", "--prefix", str(prefix), "--solver=libmamba",
         "--satisfied-skip-solve",
         "--override-channels", "--strict-channel-priority",
         "-c", "nvidia/label/cuda-12.8.0", "-c", "conda-forge",
         "cuda=12.8", "cuda-version=12.8", "gcc_linux-64=13", "gxx_linux-64=13",
         f"python={python_version}", "pip"],
        [sys.executable, "-I", "-m", "pip", "--isolated", "install",
         "--no-user", "--prefix", str(prefix),
         "--index-url", "https://download.pytorch.org/whl/cu128", f"torch=={TORCH_VERSION}"],
        [sys.executable, "-I", "-m", "pip", "--isolated", "install",
         "--no-user", "--prefix", str(prefix),
         "--index-url", "https://pypi.org/simple", "ninja", "numpy"],
    ]
    print(f"Environment: {prefix}\nSetup: PyTorch {TORCH_VERSION}, CUDA {CUDA_VERSION}, GCC/G++ 13", flush=True)
    print("This installs the tested versions, replacing other PyTorch/CUDA/compiler versions in this environment.", flush=True)
    if dry_run:
        for command in commands:
            print("+ " + shlex.join(command))
        print("Then save environment-local build settings and run ddtw check-cuda. No changes made.")
        return
    _check_driver()
    for command in commands:
        _run(command)
    settings = toolchain()
    for executable in (str(prefix / "bin" / "nvcc"), settings["CC"], settings["CXX"]):
        if not os.access(executable, os.X_OK):
            raise RuntimeError(f"Installation did not provide the expected compiler: {executable}")
    config = config_path()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(CUDA_VERSION + "\n")
    # Start a new interpreter after pip/Conda changes. Use this installed module
    # directly so an unrelated checkout in the working directory cannot shadow it.
    _run([sys.executable, "-I", "-c",
          "import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); "
          "runpy.run_module('ddtw', run_name='__main__')",
          str(Path(__file__).resolve().parent.parent), "check-cuda"])
    print("Setup complete. Use SDTW(backend='cuda_cpp') in this environment.", flush=True)


def check_cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access an NVIDIA GPU. Check nvidia-smi and run ddtw setup-cuda.")
    managed = config_path().is_file()
    if managed and str(torch.__version__) != TORCH_VERSION:
        raise RuntimeError(f"This environment was configured for PyTorch {TORCH_VERSION}, but has {torch.__version__}. Run ddtw setup-cuda again.")
    nvcc = str(Path(sys.prefix) / "bin" / "nvcc") if managed else shutil.which("nvcc")
    if not nvcc or not os.access(nvcc, os.X_OK):
        raise RuntimeError("CUDA compiler nvcc was not found. Run ddtw setup-cuda.")
    compiler_info = subprocess.check_output([nvcc, "--version"], text=True)
    match = re.search(r"release (\d+\.\d+)", compiler_info)
    compiler_cuda = match.group(1) if match else "unknown"
    if compiler_cuda != torch.version.cuda:
        raise RuntimeError(f"CUDA mismatch: nvcc {compiler_cuda}, PyTorch {torch.version.cuda}. Run ddtw setup-cuda.")
    from . import SDTW

    try:
        package_version = version("ddtw")
    except PackageNotFoundError:
        package_version = "source checkout"
    print(f"dDTW {package_version}: {Path(__file__).parent}", flush=True)
    print(f"PyTorch {torch.__version__}; CUDA {compiler_cuda}; GPU {torch.cuda.get_device_name(0)}", flush=True)
    print("Checking CUDA forward/backward (the first run compiles the extension)...", flush=True)
    with torch.cuda.device(0):
        x = torch.rand(1, 8, 3, device="cuda:0", requires_grad=True)
        y = torch.rand(1, 5, 3, device="cuda:0")
        loss = SDTW(backend="cuda_cpp", cuda_device="cuda:0")(x, y)
        loss.backward()
        torch.cuda.synchronize()
    if not torch.isfinite(loss).all() or x.grad is None or not torch.isfinite(x.grad).all():
        raise RuntimeError("GPU loss or gradient contains non-finite values.")
    print("GPU check passed: forward and backward completed successfully.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ddtw", description="Set up and check dDTW CUDA support.")
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup-cuda", help="Install CUDA 12.8 in the active Linux Conda environment")
    setup.add_argument("--dry-run", action="store_true", help="Show installation commands without changing the environment")
    commands.add_parser("check-cuda", help="Compile/load the extension and test GPU forward/backward")
    args = parser.parse_args(argv)
    try:
        if args.command == "setup-cuda":
            setup_cuda(dry_run=args.dry_run)
        else:
            check_cuda()
    except (ImportError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"dDTW {args.command} failed: {exc}", file=sys.stderr)
        return 1
    return 0
