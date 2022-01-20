from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def dropoutresidual(in_arr, matB_arr, dropout_rate, out_arr, seed, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(matB_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuDropoutResidual(in_arr.handle, matB_arr.handle, ctypes.c_float(
        dropout_rate), out_arr.handle, ctypes.byref(seed), stream.handle if stream else None)
