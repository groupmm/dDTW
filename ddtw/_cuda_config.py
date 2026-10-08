"""Environment-local configuration for the optional CUDA build."""

from contextlib import contextmanager
import os
from pathlib import Path
import sys


CUDA_VERSION = "12.8"
TORCH_VERSION = "2.11.0+cu128"


def config_path():
    return Path(sys.prefix) / "etc" / "ddtw" / "cuda-version"


def toolchain():
    prefix = Path(sys.prefix)
    return {
        "CUDA_HOME": str(prefix),
        "CC": str(prefix / "bin" / "x86_64-conda-linux-gnu-cc"),
        "CXX": str(prefix / "bin" / "x86_64-conda-linux-gnu-c++"),
    }


@contextmanager
def build_environment():
    """Use the configured compiler only during ddtw builds, then restore flags."""
    configured = config_path().is_file()
    if not configured:
        yield
        return
    if config_path().read_text().strip() != CUDA_VERSION:
        raise RuntimeError("Unknown dDTW CUDA configuration. Run ddtw setup-cuda again.")
    settings = toolchain()
    for executable in ("CC", "CXX"):
        if not os.access(settings[executable], os.X_OK):
            raise RuntimeError("The dDTW compiler is missing. Run ddtw setup-cuda again.")
    settings.update({
        "CUDAHOSTCXX": settings["CXX"],
        "PATH": str(Path(sys.prefix) / "bin") + os.pathsep + os.environ.get("PATH", ""),
        "MAX_JOBS": os.environ.get("MAX_JOBS", "2"),
    })
    cleared = ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "CUDAFLAGS", "CFLAGS", "CXXFLAGS")
    previous = {key: os.environ.get(key) for key in (*settings, *cleared)}
    try:
        for key in cleared:
            os.environ.pop(key, None)
        os.environ.update(settings)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
