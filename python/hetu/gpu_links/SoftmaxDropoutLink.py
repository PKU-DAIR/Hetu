from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def softmaxdropout(in_arr, dropout_rate, out_arr, seed, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuSoftmaxDropout(in_arr.handle, ctypes.c_float(
        dropout_rate), out_arr.handle, ctypes.byref(seed), stream.handle if stream else None)


def softmaxdropout_gradient(grad_arr, softmax_input_arr, dropout_rate, out_arr, seed, stream=None):
    assert isinstance(grad_arr, _nd.NDArray)
    assert isinstance(softmax_input_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuSoftmaxDropoutGradient(grad_arr.handle, softmax_input_arr.handle, ctypes.c_float(
        dropout_rate), out_arr.handle, seed, stream.handle if stream else None)
