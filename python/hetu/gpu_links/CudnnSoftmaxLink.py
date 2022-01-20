from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def CuDNN_softmax(in_arr, out_arr, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.CuDNN_DLGpuSoftmax(in_arr.handle, out_arr.handle,
                            stream.handle if stream else None)


def CuDNN_softmax_gradient(y_arr, grad_arr, out_arr, stream=None):
    assert isinstance(y_arr, _nd.NDArray)
    assert isinstance(grad_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.CuDNN_DLGpuSoftmaxGradient(
        y_arr.handle, grad_arr.handle, out_arr.handle, stream.handle if stream else None)

def CuDNN_softmax_gradient_recompute(x_arr, grad_arr, out_arr, stream=None):
    assert isinstance(x_arr, _nd.NDArray)
    assert isinstance(grad_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.CuDNN_DLGpuSoftmaxGradientRecompute(
        x_arr.handle, grad_arr.handle, out_arr.handle, stream.handle if stream else None)