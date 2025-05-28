from __future__ import absolute_import
from .Node import Op
import numpy as np
from .._base import DNNL_LIB
from ..cpu_links import concat as cpu_concat
from ..cpu_links import concat_gradient as cpu_concat_gradient
from ..gpu_links import concat
from ..gpu_links import concat_gradient


class ConcatOp(Op):
    def __init__(self, node_A, node_B, axis=0, ctx=None):
        super().__init__(ConcatOp, [node_A, node_B], ctx)
        self.axis = axis

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['DnnlConcat']:
                cpu_concat(input_vals[0], input_vals[1], output_val, self.axis)
            else:
                output_val[:] = np.concatenate(
                    (input_vals[0].asnumpy(), input_vals[1].asnumpy()), self.axis)
        else:
            concat(input_vals[0], input_vals[1],
                   output_val, self.axis, stream_handle)

    def gradient(self, output_grad):
        return [concat_gradient_op(output_grad, self.inputs[0], self.axis, idx=0, ctx=self.raw_ctx),
                concat_gradient_op(output_grad, self.inputs[1], self.axis, idx=1, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        assert len(input_shapes[0]) == len(input_shapes[1])
        for i in range(self.axis):
            assert input_shapes[0][i] == input_shapes[1][i]
        for i in range(self.axis+1, len(input_shapes[0])):
            assert input_shapes[0][i] == input_shapes[1][i]
        out_shape = list(input_shapes[0])
        out_shape[self.axis] = out_shape[self.axis] + \
            input_shapes[1][self.axis]

        return tuple(out_shape)


class Concat_gradientOP(Op):
    def __init__(self, grad_node, input_node, axis, idx, ctx=None):
        super().__init__(Concat_gradientOP, [grad_node, input_node], ctx)
        self.axis = axis
        self.idx = idx

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if 'cpu_Concat_Gradient' in DNNL_LIB:
                cpu_concat_gradient(
                    input_vals[0], output_val, self.axis, self.idx)
            else:
                output_val[:] = concat_backward(
                    input_vals[0].asnumpy(), self.idx, self.axis)
        else:
            concat_gradient(input_vals[0], output_val,
                            self.axis, self.idx, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[1]


def concat_op(node_A, node_B, axis=0, ctx=None):
    """Concatenates given variables along an axis.

    Parameters:
    ----
    node_A : Node
        The first node to be concated.
    node_B : Node
        The second node to be concated.
    axis :
        The axis along which two nodes are concated.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return ConcatOp(node_A, node_B, axis, ctx=ctx)


def concat_gradient_op(grad_node, input_node, axis, idx, ctx=None):
    """Gradient node of concat operation.

    Parameters:
    ----
    grad_node : Node
        Previous gradient node.
    input_node : Node
    axis :
        Axis along which to be concatenated.
    idx :
        The index of concatenation.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Concat_gradientOP(grad_node, input_node, axis, idx, ctx=ctx)


def concat_backward(grad, idx, axis=0):
    if axis == 0:
        gradient_x1 = grad[:idx]
        gradient_x2 = grad[idx:]
    elif axis == 1:
        gradient_x1 = grad[:, :idx]
        gradient_x2 = grad[:, idx:]
    elif axis == 2:
        gradient_x1 = grad[:, :, :idx]
        gradient_x2 = grad[:, :, idx:]
    else:
        gradient_x1 = grad[:, :, :, :idx]
        gradient_x2 = grad[:, :, :, idx:]
    return [gradient_x1, gradient_x2]
