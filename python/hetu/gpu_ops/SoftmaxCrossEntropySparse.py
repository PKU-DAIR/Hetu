from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import softmax_cross_entropy_sparse
from ..gpu_links import softmax_cross_entropy_sparse_gradient


class SoftmaxCrossEntropySparseOp(Op):
    def __init__(self, node_A, node_B, ignored_index, grad_inplace = False, ctx=None):
        super().__init__(SoftmaxCrossEntropySparseOp, [node_A, node_B], ctx)
        self.ignored_index = ignored_index
        self.grad_inplace = grad_inplace

    def compute(self, input_vals, output_val, stream_handle=None):
        y = input_vals[0]
        y_ = input_vals[1]
        softmax_cross_entropy_sparse(
            y, y_, self.ignored_index, output_val, stream_handle)

    def gradient(self, output_grad):
        grad_A = softmaxcrossentropy_sparse_gradient_op(
            self.inputs[0], self.inputs[1], output_grad, self.ignored_index, self.grad_inplace, ctx=output_grad.ctx)
        return [grad_A, None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        assert len(input_shapes[0]) >= 2
        assert input_shapes[0][:-1] == input_shapes[1]
        return input_shapes[0][:-1]


class SoftmaxCrossEntropySparseGradientOp(Op):
    def __init__(self, node_A, node_B, node_C, ignored_index, inplace, ctx=None):
        super().__init__(SoftmaxCrossEntropySparseGradientOp,
                         [node_A, node_B, node_C], ctx)
        self.ignored_index = ignored_index
        self.inplace = inplace

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.inplace:
            softmax_cross_entropy_sparse_gradient(
                input_vals[0], input_vals[1], input_vals[2], self.ignored_index, input_vals[0], stream_handle)
            input_vals[0].inplace_copy(output_val)
        else:
            softmax_cross_entropy_sparse_gradient(
                input_vals[0], input_vals[1], input_vals[2], self.ignored_index, output_val, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 3
        return input_shapes[0]


def softmaxcrossentropy_sparse_op(node_A, node_B, ignored_index=-1, grad_inplace = False, ctx=None):
    return SoftmaxCrossEntropySparseOp(node_A, node_B, ignored_index, grad_inplace, ctx=ctx)


def softmaxcrossentropy_sparse_gradient_op(node_A, node_B, node_C, ignored_index, inplace = False, ctx=None):
    return SoftmaxCrossEntropySparseGradientOp(node_A, node_B, node_C, ignored_index, inplace, ctx=ctx)
