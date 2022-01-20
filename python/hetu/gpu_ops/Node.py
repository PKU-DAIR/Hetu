from __future__ import absolute_import
import numpy as np
from .. import ndarray
from .. import stream
from ..context import get_current_context, DeviceGroup
from copy import copy, deepcopy
G_NODE_ID = 0


class Op(object):
    """Node in a computation graph."""

    def __init__(self, op_type, inputs, ctx=None):
        """Constructor
            Instance variables
            ------------------
            self.inputs: the list of input nodes.
            self.const_attr: the add or multiply constant.
                e.g. self.const_attr=5 if this node is created by x+5.
            self.name: node name for debugging.
        """
        self.inputs = inputs
        self.raw_ctx = get_current_context() if ctx is None else DeviceGroup(ctx)
        self.ctx = ctx
        self.const_attr = None
        self.dtype = None
        self.inplace = False
        self.lazy_execution = False
        self.event = None
        self.op_type = op_type.__name__
        global G_NODE_ID
        self.id = G_NODE_ID
        G_NODE_ID = G_NODE_ID + 1
        self.name = self.op_type + str(self.id)
        self.desc = self.name + \
            '(' + ', '.join([inp.name for inp in inputs]) + ')'

    def __add__(self, other):
        """Adding two nodes return a new node."""
        from .AddElewise import add_op
        from .AddConst import addbyconst_op

        # here the operator does NOT specify context
        # please explicitly specify the context in gradients!!
        if isinstance(other, Op):
            new_node = add_op(self, other)
        else:
            # Add by a constant stores the constant in new node's const_attr
            # 'other' argument is a constant
            new_node = addbyconst_op(self, other)
        return new_node

    def __mul__(self, other):
        """Multiplying two nodes return a new node."""
        from .MultiplyElewise import mul_op
        from .MultiplyConst import mul_byconst_op

        if isinstance(other, Op):
            new_node = mul_op(self, other)
        else:
            # Mul by a constant stores the constant in new node's const_attr
            # 'other' argument is a constant
            new_node = mul_byconst_op(self, other)
        return new_node

    # Allow left-hand-side add and multiply.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow print to display node name."""
        return self.name

    def __repr__(self):
        """Allow representation to display node name."""
        return self.name

    def __deepcopy__(self, memo):
        # this function should be used before graph splits and hooks (in distributed strategies)
        # use copy for DeviceGroup and NDArray (shared data); use deepcopy for nodes and optimizer
        if id(self) not in memo:
            new_op = copy(self)
            memo[id(self)] = new_op
            for k, v in self.__dict__.items():
                if k in ('inputs', 'grad_nodes'):
                    new_op.__dict__[k] = [
                        deepcopy(n, memo) for n in v]
                elif k in ('grad_node', 'forward_node', 'optimizer'):
                    new_op.__dict__[k] = deepcopy(v, memo)
        return memo[id(self)]

    def compute(self, input_vals, output_val, stream_handle=None):
        """Given values of input nodes, compute the output value.
        Parameters
        ----------
        node: node that performs the compute.
        input_vals: values of input nodes.
        output_val: output value of the node, modified in-place.
        """
        raise NotImplementedError

    def gradient(self, output_grad):
        """Given output gradient, compute partial gradient to each input node.
        Parameters
        ----------
        node: node that performs the gradient.
        output_grad: output gradient summed from children nodes' contributions
        Returns
        -------
        A list of gradient contributions to each input node respectively.
        """
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        """Given shapes of input nodes, compute shape of output node.
        Implementation note:
        It's simpler to treat shape of constants as (1,), so that constants can
        be stored as a numpy array too and you would need fewer special case
        handling.
        Parameters
        ----------
        node: node whose shape is being inferred.
        input_vals: shapes of input nodes.
        Returns
        -------
        A tuple representing the shape of output node.
        """
        raise NotImplementedError

    def naive_infer_shape(self, input_shapes):
        return self.infer_shape(input_shapes)

    def add_transfer_op(self, src_node, dst_ctx, h2d_ops, d2h_ops):
        from .DataTransfer import datah2d_op, datad2h_op, datad2h_sparse_op

        def add_h2d(prev_node, cur_ctx):
            if prev_node not in h2d_ops:
                h2d_ops[prev_node] = datah2d_op(prev_node, cur_ctx)
            return h2d_ops[prev_node]

        def add_d2h(prev_node):
            from .EmbeddingLookUp import EmbeddingLookUp_Gradient
            if prev_node not in d2h_ops:
                if isinstance(prev_node, EmbeddingLookUp_Gradient):
                    d2h_ops[prev_node] = datad2h_sparse_op(prev_node)
                else:
                    d2h_ops[prev_node] = datad2h_op(prev_node)
                if prev_node.event is None:
                    # here we should ensure the computation complete before d2h
                    prev_node.event = stream.create_event_handle(prev_node.ctx)
            return d2h_ops[prev_node]
        src_ctx = src_node.ctx
        result = src_node
        if src_ctx != dst_ctx:
            if ndarray.is_gpu_ctx(dst_ctx):
                if ndarray.is_gpu_ctx(src_ctx):
                    assert False, 'Please use NCCL to P2P communicate!'
                else:
                    result = add_h2d(result, dst_ctx)
            else:
                result = add_d2h(result)
        return result

    def forward_hook(self, config):
        # disable inplace if not lazy execution
        # previously we use array reshape lazy callback to do this, which is deprecated (not efficient)
        if not self.lazy_execution:
            for node in self.inputs:
                if node.op_type not in ["Array_ReshapeOp", "Array_Reshape_GradientOp", "DropoutOp", "BroadcastToOp", "BroadcastShapeOp", "SoftmaxCrossEntropySparseGradientOp", "LinearActGradient"]:
                    node.inplace = False

        # insert data transfer op if needed
        input_ctxs = set([n.ctx for n in self.inputs])
        assert None not in input_ctxs, 'Inputs contexts should already be determined.'
        if self.ctx is None:
            self.ctx = config.context
        for i in range(len(self.inputs)):
            self.inputs[i] = self.add_transfer_op(
                self.inputs[i], self.ctx, config.h2d_ops, config.d2h_ops)
        self.on_gpu = ndarray.is_gpu_ctx(self.ctx)
        self.on_cpu = not self.on_gpu
        if self in config.eval_node_list and self.on_gpu and self.event is None:
            self.event = stream.create_event_handle(self.ctx)

    def backward_hook(self, config):
        pass

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        for nst in input_statuses:
            status.copy_from(nst, deduce_order)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        for nst in input_statuses:
            nst.copy_from(status, deduce_order)

    def get_default_state(self, status, enforce_order):
        if status.valid_state() and not status.valid_all():
            splits = len(status.state)
            has_dup = status.duplicate > 1
            if splits == 1 and not has_dup:
                status.set_order(tuple(status.state.keys()))
            elif splits == 0 and has_dup:
                status.set_order((-1,))
        if enforce_order:
            order = tuple(sorted(status.state.keys()))
            if status.duplicate > 1:
                order = (-1,) + order
            status.set_order(order)
