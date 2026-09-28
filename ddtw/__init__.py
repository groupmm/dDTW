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

"""Public package interface for the dDTW toolbox."""

from .ddtw import dDTW
from .ddtw_variants import CTC
from .ddtw_variants import DTW
from .ddtw_variants import SDTW
from .ddtw_variants import partial_matching
from .ddtw_variants import smoothDTW
from .ddtw_variants import sparseDTW
from .ddtw_variants import subSDTW

__all__ = [
    "dDTW",
    "SDTW",
    "DTW",
    "smoothDTW",
    "sparseDTW",
    "subSDTW",
    "CTC",
    "partial_matching",
]
