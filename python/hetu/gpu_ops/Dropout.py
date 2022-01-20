from __future__ import absolute_import
from .Node import Op
import ctypes
import numpy as np
from .._base import DNNL_LIB
from ..cpu_links import dropout as cpu_dropout
from ..cpu_links import dropout_gradient as cpu_dropout_gradient
from ..gpu_links import dropout_gradient, dropout_gradient_recompute
from ..gpu_links import dropout


class DropoutOp(Op):
    def __init__(self, node_in, keep_prob, recompute = True, inplace = False, ctx=None):
        super().__init__(DropoutOp, [node_in], ctx)
        self.seed = ctypes.c_ulonglong(0)
        self.mask = None
        self.keep_prob = keep_prob
        self.recompute = recompute
        self.inplace = inplace
        if inplace and not recompute:
            assert(False, 'Inplace Dropout requires recomputing during Backward!')

    def compute(self, input_vals, output_val, stream_handle=None, inference=False):
        if inference == False:
            if self.on_cpu:
                if DNNL_LIB['cpu_Dropout']:
                    cpu_dropout(input_vals[0], self.keep_prob, output_val)
                else:
                    np.random.seed(self.seed.value)
                    if self.mask is None:
                        self.mask = np.random.uniform(
                            0, 1.0, input_vals[0].shape) >= (1-self.keep_prob)
                    output_val[:] = dropout_np(
                        input_vals[0].asnumpy(), self.keep_prob, output_val, self.mask)
            else:
                if self.inplace:
                    dropout(input_vals[0], 1 - self.keep_prob,
                            input_vals[0], self.seed, stream_handle)
                    input_vals[0].inplace_copy(output_val)
                else:
                    dropout(input_vals[0], 1 - self.keep_prob,
                            output_val, self.seed, stream_handle)

    def gradient(self, output_grad):
        if self.recompute:
            return [dropout_gradient_recompute_op(output_grad, self.keep_prob, self, ctx=self.raw_ctx)]
        else:
            return [dropout_gradient_op(output_grad, self.keep_prob, self, ctx=self.raw_ctx)]


    def infer_shape(self, input_shapes):
        return input_shapes[0]


class Dropout_Gradient_recomputeOp(Op):
    def __init__(self, node_in, keep_prob, forward_node, ctx=None):
        super().__init__(Dropout_Gradient_recomputeOp, [node_in], ctx)
        self.forward_node = forward_node
        self.seed = forward_node.seed
        self.keep_prob = keep_prob

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['cpu_Dropout_Gradient']:
                cpu_dropout_gradient(input_vals[0], self.keep_prob, output_val)
            else:
                output_val[:] = dropout_np_gradient(
                    input_vals[0].asnumpy(), self.keep_prob, self.forward_node.mask)
        else:
            dropout_gradient_recompute(input_vals[0], 1 - self.keep_prob,
                            output_val, self.seed, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return input_shapes[0]


class Dropout_GradientOp(Op):
    def __init__(self, node_in, keep_prob, forward_node, ctx=None):
        super().__init__(Dropout_GradientOp, [node_in, forward_node], ctx)
        self.forward_node = forward_node
        self.keep_prob = keep_prob

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['cpu_Dropout_Gradient']:
                cpu_dropout_gradient(input_vals[0], self.keep_prob, output_val)
            else:
                output_val[:] = dropout_np_gradient(
                    input_vals[0].asnumpy(), self.keep_prob, self.forward_node.mask)
        else:
            dropout_gradient(input_vals[0], input_vals[1], 1 - self.keep_prob,
                             output_val, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return input_shapes[0]


def dropout_op(node_in, keep_prob, recompute = True, inplace = False, ctx=None):
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
    return DropoutOp(node_in, keep_prob, recompute = recompute, inplace = inplace, ctx=ctx)


def dropout_gradient_op(node_in, keep_prob, forward_node, ctx=None):
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
    return Dropout_GradientOp(node_in, keep_prob, forward_node, ctx=ctx)

def dropout_gradient_recompute_op(node_in, keep_prob, forward_node, ctx=None):
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
    return Dropout_Gradient_recomputeOp(node_in, keep_prob, forward_node, ctx=ctx)

def dropout_np(inputs, keep_prob, out_arr, mask):
    return mask*inputs*(1/keep_prob)


def dropout_np_gradient(in_gradient_y, keep_prob, mask):
    out_grads = in_gradient_y
    out_grads *= mask * (1 / keep_prob)
    return out_grads
