##################################################################################
#                                dDTW Toolbox                                    #
##################################################################################
#                                                                                #
# Authors: Johannes Zeitler and Meinard Müller, 2026                             #
#                                                                                #
# If you use this toolbox, please cite the accompanying paper:                   #
# Johannes Zeitler and Meinard Müller. dDTW: A Unified and Efficient Toolbox for #
#  Differentiable Sequence Alignment. Submitted 2026.                            #
##################################################################################


##################################################################################
# MIT License                                                                    #
#                                                                                #
# Copyright 2026 Johannes Zeitler and Meinard Müller                             #
#                                                                                #
# Permission is hereby granted, free of charge, to any person obtaining a copy   #
# of this software and associated documentation files (the "Software"), to deal  #
# in the Software without restriction, including without limitation the rights   #
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      #
# copies of the Software, and to permit persons to whom the Software is          #
# furnished to do so, subject to the following conditions:                       #
#                                                                                #
# The above copyright notice and this permission notice shall be included in all #
# copies or substantial portions of the Software.                                #
#                                                                                #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  #
# SOFTWARE.                                                                      #
##################################################################################

import os
from pathlib import Path
import sys

import torch

from .._cuda_config import TORCH_VERSION, build_environment, config_path


_EXTENSION = None


def get_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    if config_path().is_file() and str(torch.__version__) != TORCH_VERSION:
        raise RuntimeError(f"dDTW CUDA setup expects PyTorch {TORCH_VERSION}. Run ddtw setup-cuda again.")
    source_dir = Path(__file__).resolve().parent / "csrc"
    build_dir = (Path(sys.prefix) / "var" / "cache" / "ddtw" /
                 f"py{sys.version_info.major}{sys.version_info.minor}-torch{torch.__version__}")
    build_dir.mkdir(parents=True, exist_ok=True)

    with build_environment():
        # Import after selecting the environment-local toolkit. PyTorch caches
        # CUDA_HOME at import time; also handle callers that imported it earlier.
        from torch.utils import cpp_extension
        previous_home = cpp_extension.CUDA_HOME
        if config_path().is_file():
            cpp_extension.CUDA_HOME = os.environ["CUDA_HOME"]
        try:
            _EXTENSION = cpp_extension.load(
                name="ddtw_cuda_ext",
                sources=[str(source_dir / "ddtw_extension.cpp"), str(source_dir / "ddtw_cuda.cu")],
                build_directory=str(build_dir),
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                verbose=False,
            )
        finally:
            cpp_extension.CUDA_HOME = previous_home
    return _EXTENSION
