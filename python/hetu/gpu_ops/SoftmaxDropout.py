from __future__ import absolute_import
from .Node import Op
import ctypes
import numpy as np
from .._base import DNNL_LIB
from ..gpu_links import softmaxdropout_gradient
from ..gpu_links import softmaxdropout


class SoftmaxDropoutOp(Op):
    def __init__(self, node_in, keep_prob, ctx=None):
        super().__init__(SoftmaxDropoutOp, [node_in], ctx)
        self.seed = ctypes.c_ulonglong(0)
        self.mask = None
        self.keep_prob = keep_prob

    def compute(self, input_vals, output_val, stream_handle=None, inference=False):
        if inference == False:
            softmaxdropout(input_vals[0], 1 - self.keep_prob,
                    output_val, self.seed, stream_handle)

    def gradient(self, output_grad):
        return [softmaxdropout_gradient_op(output_grad, self.inputs[0], self.keep_prob, self, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        return input_shapes[0]


class SoftmaxDropout_GradientOp(Op):
    def __init__(self, node_grad, node_softmax_input, keep_prob, forward_node, ctx=None):
        super().__init__(SoftmaxDropout_GradientOp, [node_grad, node_softmax_input], ctx)
        self.forward_node = forward_node
        self.seed = forward_node.seed
        self.keep_prob = keep_prob

    def compute(self, input_vals, output_val, stream_handle=None):
        softmaxdropout_gradient(input_vals[0], input_vals[1], 1 - self.keep_prob,
                            output_val, self.seed, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return input_shapes[0]


def softmaxdropout_op(node_in, keep_prob, ctx=None):
    """Drops elements of input variable randomly.
    Parameters:
    ----
    node_in : Node
        Input variable.
    keep_prob : float
        Probability of the results to be kept.
    Returns:
    ----
    A new Node instance created by Op.
    """
    return SoftmaxDropoutOp(node_in, keep_prob, ctx=ctx)


def softmaxdropout_gradient_op(node_grad, node_softmax_input, keep_prob, forward_node, ctx=None):
    """Gradient node of dropout operation.
    Parameters:
    ----
    node_in : Node
        Input variable.
    keep_prob : float
        Probability of the results to be kept.
    Returns:
    ----
    A new Node instance created by Op.
    """
    return SoftmaxDropout_GradientOp(node_grad, node_softmax_input, keep_prob, forward_node, ctx=ctx)