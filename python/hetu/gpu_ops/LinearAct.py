from __future__ import absolute_import
import numpy as np

from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import matmul_with_bias
from ..gpu_links import gelu
from ..gpu_links import gelu_gradient
from ..gpu_links import relu
from ..gpu_links import relu_gradient
from .. import ndarray


class LinearActGradient(Op):
    def __init__(self, output_grad, node_A, node_B, bias, trans_A=False, trans_B=False, act = "gelu", ctx=None):
        super().__init__(LinearActGradient, [output_grad, node_A, node_B, bias], ctx)
        self.inplace = True
        self.matmul_attr_trans_A = trans_A
        self.matmul_attr_trans_B = trans_B
        self.act = act
        if self.act not in ["relu", "gelu"]:
            print("Activation should be relu or gelu!")
            assert(False)
    def compute(self, input_vals, output_val, stream_handle=None):
        act_input = ndarray.empty(shape=input_vals[0].shape, ctx=input_vals[0].ctx)
        matmul_with_bias(
            input_vals[1], self.matmul_attr_trans_A,
            input_vals[2], self.matmul_attr_trans_B, input_vals[3],
            act_input, stream_handle)
        if self.act is "relu":
            relu_gradient(act_input, input_vals[0],
                          input_vals[0], stream_handle)
        elif self.act is "gelu":
            gelu_gradient(act_input, input_vals[0],
                          input_vals[0], stream_handle)
        input_vals[0].inplace_copy(output_val)
        del act_input

    def infer_shape(self, input_shapes):
        return input_shapes[0]

class LinearActOp(Op):
    def __init__(self, node_A, node_B, bias, trans_A=False, trans_B=False, act = "gelu", ctx=None):
        super().__init__(LinearActOp, [node_A, node_B, bias], ctx)
        self.matmul_attr_trans_A = trans_A
        self.matmul_attr_trans_B = trans_B
        self.act = act
        if self.act not in ["relu", "gelu"]:
            print("Activation should be relu or gelu!")
            assert(False)

    def compute(self, input_vals, output_val, stream_handle=None):
        matmul_with_bias(
            input_vals[0], self.matmul_attr_trans_A,
            input_vals[1], self.matmul_attr_trans_B, input_vals[2],
            output_val, stream_handle)
        if self.act is "relu":
            relu(output_val, output_val, stream_handle)
        elif self.act is "gelu":
            gelu(output_val, output_val, stream_handle)


    def gradient(self, output_grad):
        from .MatrixMult import matmul_op
        from .ReduceSum import reduce_sum_op

        output_grad = linearActGradient_op(output_grad, self.inputs[0], self.inputs[1], self.inputs[2], self.matmul_attr_trans_A, self.matmul_attr_trans_B, self.act, ctx=self.raw_ctx)

        if ((self.matmul_attr_trans_A is False) and
                (self.matmul_attr_trans_B is False)):
            # if Y=AB, then dA=dY B^T, dB=A^T dY
            lhs_grad = matmul_op(
                output_grad, self.inputs[1], trans_A=False, trans_B=True, ctx=self.raw_ctx)
            rhs_grad = matmul_op(
                self.inputs[0], output_grad, trans_A=True, trans_B=False, ctx=self.raw_ctx)
        elif ((self.matmul_attr_trans_A is True) and
                (self.matmul_attr_trans_B is False)):
            # if Y=A^T B, then dA=(dY B^T)^T=B dY^T, dB=A dY
            lhs_grad = matmul_op(
                self.inputs[1], output_grad, trans_A=False, trans_B=True, ctx=self.raw_ctx)
            rhs_grad = matmul_op(
                self.inputs[0], output_grad, trans_A=False, trans_B=False, ctx=self.raw_ctx)
        elif ((self.matmul_attr_trans_A is False) and
                (self.matmul_attr_trans_B is True)):
            # if Y=A B^T, then dA=dY B, dB=(A^T dY)^T=dY^T A
            lhs_grad = matmul_op(
                output_grad, self.inputs[1], trans_A=False, trans_B=False, ctx=self.raw_ctx)
            rhs_grad = matmul_op(
                output_grad, self.inputs[0], trans_A=True, trans_B=False, ctx=self.raw_ctx)
        elif ((self.matmul_attr_trans_A is True) and
                (self.matmul_attr_trans_B is True)):
            # if Y=A^T B^T, then dA=(dY B)^T=B^T dY^T, dB=(A dY)^T=dY^T A^T
            lhs_grad = matmul_op(
                self.inputs[1], output_grad, trans_A=True, trans_B=True, ctx=self.raw_ctx)
            rhs_grad = matmul_op(
                output_grad, self.inputs[0], trans_A=True, trans_B=True, ctx=self.raw_ctx)
        bias_grad = reduce_sum_op(
            output_grad, [0], keepdims=False, ctx=self.raw_ctx)
        return [lhs_grad, rhs_grad, bias_grad]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 3
        assert all([len(shape) == 2 for shape in input_shapes[:2]])
        assert len(input_shapes[2]) == 1
        A = input_shapes[0]
        B = input_shapes[1]
        bias_shape = input_shapes[2]
        shape_A = A[0]
        shape_B = B[1]
        if self.matmul_attr_trans_A == True:
            shape_A = A[1]
        if self.matmul_attr_trans_B == True:
            shape_B = B[0]
        assert bias_shape == (shape_B,)
        return (shape_A, shape_B)

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        assert len(input_statuses) == 3
        l2res_map = [
            {-1: 1, 0: 0, 1: -1},  # no trans
            {-1: 1, 1: 0, 0: -1},  # trans A
        ][self.matmul_attr_trans_A]
        r2res_map = [
            {-1: 0, 0: -1, 1: 1},  # no trans
            {-1: 0, 0: 1, 1: -1},  # trans B
        ][self.matmul_attr_trans_B]
        if deduce_order:
            if input_statuses[0].valid_all():
                order = input_statuses[0].order
                status.set_order(tuple(l2res_map[x] for x in order))
            elif input_statuses[1].valid_all():
                order = input_statuses[1].order
                status.set_order(tuple(r2res_map[x] for x in order))
        else:
            if input_statuses[0].valid_state():
                state, duplicate = input_statuses[0].get()
                res_state = (
                    state.get(int(self.matmul_attr_trans_A), 1), duplicate)
                res_duplicate = state.get(1-self.matmul_attr_trans_A, 1)
                status.set_state(res_state, res_duplicate)
            elif input_statuses[1].valid_state():
                state, duplicate = input_statuses[1].get()
                res_state = (duplicate, state.get(
                    1-self.matmul_attr_trans_B, 1))
                res_duplicate = state.get(int(self.matmul_attr_trans_B), 1)
                status.set_state(res_state, res_duplicate)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        def revert(x, whether=True):
            return (x[1], x[0]) if whether else x
        assert len(input_statuses) == 3
        res2l_map = [
            {-1: 1, 0: 0, 1: -1},  # no trans
            {-1: 0, 0: 1, 1: -1},  # trans A
        ][self.matmul_attr_trans_A]
        res2r_map = [
            {-1: 0, 0: -1, 1: 1},  # no trans
            {-1: 1, 0: -1, 1: 0},  # trans B
        ][self.matmul_attr_trans_B]
        if deduce_order:
            if status.valid_all():
                res_order = tuple(res2l_map[x] for x in status.order)
                input_statuses[0].set_order(res_order)
                res_order = tuple(res2r_map[x] for x in status.order)
                input_statuses[1].set_order(res_order)
                new_order = list(status.order)
                if 0 in new_order:
                    new_order[new_order.index(0)] = -1
                if 1 in new_order:
                    new_order[new_order.index(1)] = 0
                appeared = False
                for o in new_order:
                    if o == -1:
                        assert not appeared
                        appeared = True
                new_order = tuple(new_order)
                input_statuses[2].set_order(new_order)
        else:
            if status.valid_state():
                state, duplicate = status.get()
                res_state = revert(
                    (state.get(0, 1), duplicate), self.matmul_attr_trans_A)
                res_duplicate = state.get(1, 1)
                input_statuses[0].set_state(res_state, res_duplicate)
                res_state = revert(
                    (duplicate, state.get(1, 1)), self.matmul_attr_trans_B)
                res_duplicate = state.get(0, 1)
                input_statuses[1].set_state(res_state, res_duplicate)
                new_state = state.copy()
                if 0 in new_state:
                    duplicate *= new_state[0]
                if 1 in new_state:
                    new_state[0] = new_state.pop(1)
                else:
                    new_state = {}
                input_statuses[2].set_state(new_state, duplicate)
            else:
                if input_statuses[0].state is not None:
                    input_statuses[1].set_state(
                        None, input_statuses[0].state.get(int(self.matmul_attr_trans_A), 1))
                if input_statuses[1].state is not None:
                    input_statuses[0].set_state(
                        None, input_statuses[1].state.get(1 - self.matmul_attr_trans_B, 1))


def linearAct_op(node_A, node_B, bias, trans_A=False, trans_B=False, act="gelu", ctx=None):
    return LinearActOp(node_A, node_B, bias, trans_A, trans_B, act, ctx=ctx)

def linearActGradient_op(output_grad, node_A, node_B, bias, trans_A=False, trans_B=False, act="gelu", ctx=None):
    return LinearActGradient(output_grad, node_A, node_B, bias, trans_A, trans_B, act, ctx=ctx)
