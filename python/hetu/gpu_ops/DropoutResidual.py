from __future__ import absolute_import
from .Node import Op
import ctypes
import numpy as np
from .._base import DNNL_LIB
from ..gpu_links import dropout_gradient
from ..gpu_links import dropoutresidual


class DropoutResidualOp(Op):
    def __init__(self, node_in, node_res, keep_prob, ctx=None):
        super().__init__(DropoutResidualOp, [node_in, node_res], ctx)
        self.seed = ctypes.c_ulonglong(0)
        self.mask = None
        self.keep_prob = keep_prob

    def compute(self, input_vals, output_val, stream_handle=None, inference=False):
        if inference == False:
            dropoutresidual(input_vals[0], input_vals[1], 1 - self.keep_prob,
                    output_val, self.seed, stream_handle)

    def gradient(self, output_grad):
        return [dropoutresidual_gradient_op(output_grad, self.keep_prob, self, ctx=self.raw_ctx), output_grad]

    def infer_shape(self, input_shapes):
        return input_shapes[0]


class DropoutResidual_GradientOp(Op):
    def __init__(self, node_in, keep_prob, forward_node, ctx=None):
        super().__init__(DropoutResidual_GradientOp, [node_in], ctx)
        self.forward_node = forward_node
        self.seed = forward_node.seed
        self.keep_prob = keep_prob

    def compute(self, input_vals, output_val, stream_handle=None):
        dropout_gradient(input_vals[0], 1 - self.keep_prob,
                            output_val, self.seed, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return input_shapes[0]


def dropoutresidual_op(node_in, node_res, keep_prob, ctx=None):
    return DropoutResidualOp(node_in, node_res, keep_prob, ctx=ctx)


def dropoutresidual_gradient_op(node_in, keep_prob, forward_node, ctx=None):
    return DropoutResidual_GradientOp(node_in, keep_prob, forward_node, ctx=ctx)