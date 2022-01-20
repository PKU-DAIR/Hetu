from .ndarray import cpu, gpu, rcpu, rgpu, DLContext, is_gpu_ctx
import contextlib
import re
import numpy as np
from collections import defaultdict


class DeviceGroup(object):
    def __init__(self, ctxs):
        self._contexts = self.parse_contexts(ctxs)
        self.get_servers_n_workers()
        self._is_mp = False
        self._mp_device_num = None
        for c in self._contexts:
            if isinstance(c, tuple):
                self._is_mp = True
                assert self._mp_device_num in (None, len(c)), \
                    'Now only support same model parallel in data parallel.'
                self._mp_device_num = len(c)
        if self._mp_device_num is None:
            self._mp_device_num = 1

    @classmethod
    def parse_contexts(cls, ctxs):
        if isinstance(ctxs, DeviceGroup):
            return ctxs
        if isinstance(ctxs, str):
            ctxs = re.split(';|,| +', ctxs.lower())
        if not isinstance(ctxs, list):
            ctxs = [ctxs]
        new_ctxs = []
        for c in ctxs:
            if isinstance(c, tuple):
                c = tuple([ccc for ccc in [cls.str2ctx(cc)
                                           for cc in c] if ccc is not None])
            else:
                c = cls.str2ctx(c)
            if c is not None:
                new_ctxs.append(c)
        return new_ctxs

    @classmethod
    def str2ctx(cls, c):
        if isinstance(c, str):
            c = c.lower().split(':')
            assert c[-2] in ('cpu', 'gpu'), 'Context invalid: %s' % c
            hostname = 'localhost' if len(c) == 2 else c[0]
            idx = int(c[-1])
            c = rcpu(hostname, idx) if c[-2] == 'cpu' else rgpu(hostname, idx)
        assert isinstance(c, DLContext), 'Context invalid: %s' % c
        return c

    def index(self, ctx):
        return self._contexts.index(ctx)

    def is_mp(self):
        return self._is_mp

    def __getitem__(self, key):
        return self._contexts[key]

    def __iter__(self):
        return iter(self._contexts)

    def __len__(self):
        return len(self._contexts)

    @property
    def mp_device_num(self):
        return self._mp_device_num

    @property
    def worker_num(self):
        return len(self._workers)

    @property
    def server_num(self):
        return len(self._servers)

    @property
    def workers(self):
        return self._workers

    @property
    def servers(self):
        return self._servers

    def get_servers_n_workers(self):
        self._workers = []
        self._servers = []
        for ctx in self._contexts:
            if isinstance(ctx, tuple) or is_gpu_ctx(ctx):
                self._workers.append(ctx)
            else:
                self._servers.append(ctx)

    def __repr__(self):
        result = 'DeviceGroup('
        for c in self._contexts:
            result += ('(' + ', '.join([str(cc) for cc in c]) +
                       '), ') if isinstance(c, tuple) else '%s, ' % c
        result += ')'
        return result

    def __hash__(self):
        if not hasattr(self, 'hash'):
            self.hash = hash(tuple(self._contexts))
        return self.hash

    def __eq__(self, other):
        return hash(self) == hash(other)

    def get_sorted(self):
        return DeviceGroup(sorted(self._contexts, key=lambda x: hash(x.hostname) + hash(x.device_id)))


class ContextStack(object):
    def __init__(self):
        self._stack = []

    def peek(self):
        return self._stack[-1] if self._stack else None

    def push(self, ctx):
        return self._stack.append(ctx)

    def pop(self):
        self._stack.pop()


_default_ctx_stack = ContextStack()


class NodeStatus(object):
    def __init__(self, state=None, dev_num=None):
        if state is not None:
            if not isinstance(state, dict):
                state = {i: v for i, v in enumerate(state) if v != 1}
            else:
                state = {i: v for i, v in sorted(state.items()) if v != 1}
        self._state = state
        self._device_num = dev_num
        if self._device_num is not None and self._state is not None:
            temp = int(np.prod(list(self._state.values()), dtype=int))
            assert dev_num % temp == 0
            self._duplicate = dev_num // temp
        else:
            self._duplicate = None
        self._order = None
        self._valid_state = False
        self._valid_all = False

    def get(self):
        return self._state, self._duplicate

    def get_all(self):
        return self._state, self._duplicate, self._order

    def is_dist(self):
        return self._device_num > 1

    @ property
    def dev_num(self):
        return self._device_num

    @ property
    def state(self):
        return self._state

    @ property
    def duplicate(self):
        return self._duplicate

    @ property
    def order(self):
        return self._order

    def set_dev_num(self, dev_num):
        if self._device_num is not None:
            assert dev_num == self._device_num
        self._device_num = dev_num
        if self._state is not None:
            temp = int(np.prod(list(self._state.values()), dtype=int))
            assert dev_num % temp == 0
            temp_duplicate = dev_num // temp
            assert self._duplicate in (None, temp_duplicate)
            self._duplicate = temp_duplicate

    def set_state(self, state=None, duplicate=None):
        if state is not None:
            if not isinstance(state, dict):
                state = {i: v for i, v in enumerate(state) if v != 1}
            else:
                state = {i: v for i, v in sorted(state.items()) if v != 1}
            assert self._state in (state, None)
            self._state = state
            if self._device_num is not None:
                temp = int(np.prod(list(self._state.values()), dtype=int))
                assert self._device_num % temp == 0
                temp_duplicate = self._device_num // temp
                assert self._duplicate in (temp_duplicate, None)
                self._duplicate = temp_duplicate
        if duplicate is not None:
            assert self._duplicate in (duplicate, None)
            self._duplicate = duplicate

    def set_order(self, order=None):
        if order is not None:
            assert self._order in (order, None)
            self._order = order

    def set_one(self):
        assert self._state is None or all(
            [s == 1 for s in self._state.values()])
        assert self._duplicate in (None, 1)
        assert self._order is None
        self._state = {}
        self._duplicate = 1
        self._order = None
        self._device_num = 1
        self._valid_state = True
        self._valid_all = True

    def copy_state_from(self, other):
        self.set_state(other.state, other.duplicate)

    def copy_order_from(self, other):
        self.set_order(other.order)

    def copy_from(self, other, copy_order):
        if copy_order:
            self.copy_order_from(other)
        else:
            self.copy_state_from(other)

    def valid_state(self):
        if self._valid_state:
            return True
        if self._state is None or self._duplicate is None:
            return False
        else:
            self._valid_state = True
            if self._device_num is None:
                self._device_num = int(np.prod(
                    list(self._state.values()), dtype=int)) * self._duplicate
            return True

    def valid_all(self):
        if self._valid_all:
            # the one-device case has already set _valid_all to True
            return True
        if not self._valid_state or self._order is None:
            return False
        else:
            self._valid_all = True
            return True

    def valid(self, include_order):
        if include_order:
            return self.valid_all()
        else:
            return self.valid_state()

    def check_state(self, max_dim, check_order):
        # only allow dimensions lower than max_dim to split
        if check_order:
            assert all([o < max_dim for o in self.order])
        else:
            assert all([o < max_dim for o in self.state])

    def map_dev_to_index(self, global_index):
        cur_state_index = {}
        for cur_order in self._order[::-1]:
            if cur_order < 0:
                global_index //= self._duplicate
            else:
                ts = self._state[cur_order]
                cur_state_index[cur_order] = global_index % ts
                global_index //= ts
        return cur_state_index

    def get_loop_sizes(self):
        loop_sizes = [1]
        for rord in self._order[::-1]:
            temp_size = loop_sizes[0] * \
                self._duplicate if rord < 0 else loop_sizes[0] * \
                self._state[rord]
            loop_sizes.insert(0, temp_size)
        loop_sizes.pop(0)
        return loop_sizes

    def __repr__(self):
        return '(' + str(self.state) + ', ' + str(self.duplicate) + ', ' + str(self.order) + ')'

    def __eq__(self, other):
        if self is other:
            return True
        assert self.valid_all() and other.valid_all(), 'Cannot check equal if not valid.'
        return self.state == other.state \
            and self.duplicate == other.duplicate \
            and self.order == other.order

    def __hash__(self):
        return hash(id(self))


def get_current_context():
    return _default_ctx_stack.peek()


@ contextlib.contextmanager
def context(ctx):
    try:
        ctx = DeviceGroup(ctx)
        _default_ctx_stack.push(ctx)
        yield ctx
    finally:
        _default_ctx_stack.pop()


def get_launch_config_by_traverse_nodes(node_list, default_ctx):
    node_strategy = dict()
    devices = set()
    for ctx in default_ctx:
        if isinstance(ctx, tuple):
            devices.update(ctx)
        else:
            devices.add(ctx)
    launchPS = default_ctx.server_num > 0
    launchMPI = (not launchPS) and default_ctx.worker_num > 1
    nrank = default_ctx.worker_num
    for node in node_list:
        traverse_dfs(node, node_strategy, devices, nrank)
    launchPS = launchPS or any([x == 'PS' for x in node_strategy.values()])
    launchMPI = launchMPI or any(
        [x == 'AllReduce' for x in node_strategy.values()])
    return launchMPI, launchPS, node_strategy, devices


def traverse_dfs(node, node_strategy, devices, nrank):
    from .gpu_ops.Dispatch import DispatchOp, DispatchGradientOp
    from .optimizer import OptimizerOp
    if node in node_strategy:
        return
    strategy = None
    if node.raw_ctx is not None and node.raw_ctx.server_num > 0 and node.raw_ctx.worker_num > 0:
        strategy = 'PS'
    elif node.raw_ctx is not None and node.raw_ctx.worker_num > 1:
        strategy = 'AllReduce'
    node_strategy[node] = strategy
    if not isinstance(node, (DispatchOp, DispatchGradientOp, OptimizerOp)):
        for ctx in node.raw_ctx:
            if isinstance(ctx, tuple):
                devices.update(ctx)
            else:
                devices.add(ctx)
    local_nrank = nrank if node.raw_ctx is None else node.raw_ctx.worker_num
    assert local_nrank in (
        0, nrank), 'Number of workers not consist: (%d, %d).' % (local_nrank, nrank)
    for n in node.inputs:
        traverse_dfs(n, node_strategy, devices, nrank)


def parse_graph_with_dispatch(node_list):
    # return node-state map, state count
    # the dispatch ops will be removed
    from .dataloader import DataloaderOp
    from .optimizer import OptimizerOp
    from .gpu_ops.Dispatch import DispatchOp, DispatchGradientOp
    from .gpu_ops.Variable import PlaceholderOp

    def remove_dispatch(node):
        # TODO: handle consecutive dispatch ops
        if node in visited:
            return
        visited.add(node)
        single = False
        if not isinstance(node, (DataloaderOp, OptimizerOp, DispatchOp, DispatchGradientOp)):
            # placeholder op with only dispatch output will not get here
            node_cur_state_map[node] = NodeStatus(
                dev_num=node.raw_ctx.mp_device_num)
            node.get_default_state(
                node_cur_state_map[node], enforce_order=False)
            single = not node.raw_ctx.is_mp()
            if single:
                node_cur_state_map[node].set_one()
            if not node_cur_state_map[node].valid_all():
                invalid_states.append(node_cur_state_map[node])
        for i, n in enumerate(node.inputs):
            if isinstance(n, DispatchOp):
                real_node = n.inputs[0]
                if n not in dispatch_to_state_map:
                    dispatch_to_state_map[n] = NodeStatus(n.parts)
                    if single:
                        dispatch_to_state_map[n].set_one()
                    if not dispatch_to_state_map[n].valid_all():
                        invalid_states.append(dispatch_to_state_map[n])
                if isinstance(real_node, PlaceholderOp):
                    if real_node not in node_cur_state_map:
                        # placeholder op use the first met status as default status
                        # the raw ctx should be corresponding to current state
                        node_cur_state_map[real_node] = dispatch_to_state_map[n]
                        node_cur_state_map[real_node].set_dev_num(
                            node.raw_ctx.mp_device_num)
                        node.get_default_state(
                            node_cur_state_map[real_node], enforce_order=False)
                        if single:
                            node_cur_state_map[real_node].set_one()
                        visited.add(real_node)
                    elif dispatch_to_state_map[n] != node_cur_state_map[real_node]:
                        node_tar_state_map[real_node][node] = dispatch_to_state_map[n]
                else:
                    remove_dispatch(real_node)
                    node_tar_state_map[real_node][node] = dispatch_to_state_map[n]
                node.inputs[i] = real_node
            elif isinstance(n, DispatchGradientOp):
                real_node = n.inputs[0]
                assert not isinstance(
                    real_node, PlaceholderOp), 'Should not get here. Please debug!'
                remove_dispatch(real_node)
                remove_dispatch(n.inputs[1])
                if n not in dispatch_to_state_map:
                    dispatch_to_state_map[n] = node_cur_state_map[n.inputs[1]]
                if not isinstance(node, OptimizerOp):
                    # if DispatchGradientOp appears before OptimizerOp,
                    # the parameter should set the current state instead of target state,
                    # so we ignore the DispatchGradientOp before OptimizerOp.
                    node_tar_state_map[real_node][node] = dispatch_to_state_map[n]
                node.inputs[i] = real_node
            else:
                remove_dispatch(n)

    visited = set()
    invalid_states = []
    node_cur_state_map = {}  # save nodes' current state
    node_tar_state_map = defaultdict(dict)  # save nodes' target state
    dispatch_to_state_map = {}
    for node in node_list:
        remove_dispatch(node)

    node_cur_state_map, node_tar_state_map = infer_states(
        node_list, node_cur_state_map, node_tar_state_map, invalid_states)

    return node_cur_state_map, node_tar_state_map


def complete_state_map_with_partial_information(forward_node_list, node_list, node_cur_state_map, opt):
    # given current state of the node,
    # add target state to the forward graph
    from .dataloader import DataloaderOp
    from .optimizer import OptimizerOp
    from .gpu_ops.Variable import PlaceholderOp

    def determine_state(node):
        if node in visited:
            return
        visited.add(node)
        if not isinstance(node, DataloaderOp):
            single = not node.raw_ctx.is_mp()
            if node not in node_cur_state_map:
                node_cur_state_map[node] = NodeStatus(
                    dev_num=node.raw_ctx.mp_device_num)
                node.get_default_state(
                    node_cur_state_map[node], enforce_order=False)
            else:
                # already specified
                for n in node.inputs:
                    new_state = NodeStatus(dev_num=node.raw_ctx.mp_device_num)
                    if single:
                        new_state.set_one()
                    if not new_state.valid_all():
                        invalid_states.append(new_state)
                    if isinstance(n, PlaceholderOp) and n not in node_cur_state_map:
                        assert n.raw_ctx == node.raw_ctx
                        node_cur_state_map[n] = new_state
                    else:
                        node_tar_state_map[n][node] = new_state
            if single:
                node_cur_state_map[node].set_one()
            if not node_cur_state_map[node].valid_all():
                invalid_states.append(node_cur_state_map[node])
        for n in node.inputs:
            determine_state(n)

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        if not isinstance(node, OptimizerOp):
            single = not node.raw_ctx.is_mp()
            node_cur_state_map[node] = NodeStatus(
                dev_num=node.raw_ctx.mp_device_num)
            node.get_default_state(
                node_cur_state_map[node], enforce_order=False)
            if single:
                node_cur_state_map[node].set_one()
            if not node_cur_state_map[node].valid_all():
                invalid_states.append(node_cur_state_map[node])
        if node in bf_map:
            fnode, _ = bf_map[node]
        for n in node.inputs:
            dfs(n)
            if n in bf_map:
                forward_node, forward_ns = bf_map[n]
                for local_fn, need_target in forward_ns:
                    if (local_fn not in fb_map or node in fb_map[local_fn]) and (need_target or forward_node in node_tar_state_map[local_fn]):
                        node_tar_state_map[n][node] = node_cur_state_map[local_fn]
                        break
            elif node in bf_map and fnode in node_tar_state_map[n]:
                node_tar_state_map[n][node] = node_tar_state_map[n][fnode]

    visited = set()
    invalid_states = []
    assert isinstance(opt, OptimizerOp)
    bf_map = opt.optimizer.backward2forward
    fb_map = opt.optimizer.forward2backward
    node_tar_state_map = defaultdict(dict)
    for node in forward_node_list:
        determine_state(node)
    for node in node_list:
        dfs(node)

    node_cur_state_map, node_tar_state_map = infer_states(
        node_list, node_cur_state_map, node_tar_state_map, invalid_states)
    return node_cur_state_map, node_tar_state_map


def infer_states(node_list, node_cur_state_map, node_tar_state_map, invalid_states):
    # propagate states forward and backward
    from .dataloader import DataloaderOp
    from .optimizer import OptimizerOp

    def infer_node_states(node, infer_order):
        if node in visited:
            return
        nonlocal chance
        visited.add(node)
        if isinstance(node, DataloaderOp):
            pass
        elif isinstance(node, OptimizerOp):
            for n in node.inputs:
                infer_node_states(n, infer_order)
        else:
            input_statuses = []
            if node.raw_ctx.is_mp():
                if infer_order and chance and not node_cur_state_map[node].valid_all():
                    node.get_default_state(
                        node_cur_state_map[node], enforce_order=True)
                    chance = False
                input_statuses = []
                for n in node.inputs:
                    node_status = node_tar_state_map[n].get(
                        node, node_cur_state_map[n])
                    input_statuses.append(node_status)
                node.backward_deduce_states(
                    node_cur_state_map[node], input_statuses, deduce_order=infer_order)
                for n in node.inputs:
                    infer_node_states(n, infer_order)
                node.forward_deduce_states(
                    input_statuses, node_cur_state_map[node], deduce_order=infer_order)
            else:
                for n in node.inputs:
                    infer_node_states(n, infer_order)

    # first infer state and duplicate
    invalid_order = invalid_states.copy()
    while True:
        visited = set()
        for node in node_list:
            infer_node_states(node, infer_order=False)
        progress = False
        for i in range(len(invalid_states))[::-1]:
            if invalid_states[i].valid_state():
                invalid_states.pop(i)
                progress = True
        if invalid_states == []:
            break
        assert progress, "Not enough information for model parallel."
    chance = False
    # next infer order
    while True:
        visited = set()
        for node in node_list:
            infer_node_states(node, infer_order=True)
        progress = False
        for i in range(len(invalid_order))[::-1]:
            if invalid_order[i].valid_all():
                invalid_order.pop(i)
                progress = True
        if invalid_order == []:
            break
        if not progress:
            chance = True
    return node_cur_state_map, node_tar_state_map


def assign_context_by_traverse_nodes(node_list, ctx, mpi_comm, p2p_stream, node_cur_state_map, node_tar_state_map):
    from .dataloader import DataloaderOp
    from .optimizer import OptimizerOp, Optimizer_per_grad_Op
    from .gpu_ops.PipelineSend import pipeline_send_op
    from .gpu_ops.PipelineReceive import pipeline_receive_op
    from .gpu_ops.Variable import PlaceholderOp
    from .gpu_ops.Concatenate import concatenate_op
    from .gpu_ops.Split import split_op
    from .gpu_ops.Sum import sum_op
    from .gpu_ops.executor import new_group_comm

    def receiving(key, from_ctx):
        if from_ctx == ctx:
            return self_buffer[key]
        else:
            hostname = from_ctx.hostname
            target_id = from_ctx.device_id
            result = pipeline_receive_op(mpi_comm.getRankFromDevice(
                hostname, target_id), mpi_comm, stream=p2p_stream, ctx=ctx)
            layer_indices[result] = layer_id
            return result

    def sending(key, node, to_ctx):
        if ctx == to_ctx:
            self_buffer[key] = node
        else:
            hostname = to_ctx.hostname
            target_id = to_ctx.device_id
            target_rank = mpi_comm.getRankFromDevice(
                hostname, target_id)
            result = pipeline_send_op(
                node, target_rank, mpi_comm, stream=p2p_stream, ctx=ctx)
            layer_indices[result] = layer_id
            my_eval_nodes.append(result)

    def receive_model_parallel(prev_input, node):
        key = node_tar_state_map[prev_input][node]
        assert dp_index_map[node] >= 0, 'The receive node must be on local device.'
        dev_pos = dp_index_map[node]
        if not isinstance(prev_input.raw_ctx.workers[dev_pos], tuple):
            # here we receive from a node on one device dispatching to many
            # in this case current node MUST have mp_index, and the split will be handled in sending
            assert mp_index_map[node] >= 0, 'Here only support 1 to N.'
            if key not in recv_src[prev_input]:
                recv_src[prev_input][key] = receiving(
                    prev_input, prev_input.raw_ctx.workers[dev_pos])
        elif not isinstance(node.raw_ctx.workers[dev_pos], tuple):
            # here we receive from a node on multiple devices
            # in this case current node MUST NOT have mp_index, and handle the combination
            target = node_tar_state_map[prev_input][node]
            assert mp_index_map[node] < 0 and not target.is_dist(
            ), 'Here only support N to 1.'
            if key not in recv_src[prev_input]:
                device_index = 0

                def make_comb(depth):
                    if depth == len(cur_order):
                        nonlocal device_index
                        result = receiving(prev_input, devices[device_index])
                        device_index += 1
                    else:
                        cur_dim = cur_order[depth]
                        if cur_dim < 0:
                            if cur_duplicate == 1:
                                result = make_comb(depth + 1)
                            else:
                                result = sum_op([make_comb(depth + 1)
                                                 for _ in range(cur_duplicate)], ctx=ctx)
                            layer_indices[result] = layer_id
                        else:
                            if cur_state[cur_dim] == 1:
                                result = make_comb(depth + 1)
                            else:
                                result = concatenate_op(
                                    [make_comb(depth + 1) for _ in range(cur_state[cur_dim])], axis=cur_dim, ctx=ctx)
                            layer_indices[result] = layer_id
                    return result
                devices = prev_input.raw_ctx.workers[dev_pos]
                cur_state, cur_duplicate, cur_order = node_cur_state_map[prev_input].get_all(
                )
                recv_src[prev_input][key] = make_comb(0)
                assert device_index == len(prev_input.raw_ctx.workers[dev_pos])
        else:
            # here the prev input and the node are both in model parallel, with different states
            assert mp_index_map[node] >= 0, 'Here only support N to N.'
            if key not in recv_src[prev_input]:
                prev_ns = node_cur_state_map[prev_input]
                target_ns = node_tar_state_map[prev_input][node]
                prev_state, prev_duplicate, prev_order = prev_ns.get_all()
                target_state = target_ns.state
                loop_sizes = prev_ns.get_loop_sizes()
                cur_state_index = target_ns.map_dev_to_index(
                    mp_index_map[node])
                device_index = 0

                def cross_receive(depth):
                    nonlocal device_index
                    if depth == len(prev_order):
                        res = receiving(prev_input, devices[device_index])
                        device_index += 1
                    else:
                        cur_dim = prev_order[depth]
                        if cur_dim < 0:
                            if prev_duplicate == 1:
                                res = cross_receive(depth+1)
                            else:
                                res = sum_op([cross_receive(depth+1)
                                              for _ in range(prev_duplicate)], ctx=ctx)
                            layer_indices[res] = layer_id
                        else:
                            tar_st = target_state.get(cur_dim, 1)
                            cur_st = cur_state_index.get(cur_dim, 0)
                            if prev_state[cur_dim] % tar_st == 0:
                                # at `cur_dim` dimension we need to concat some inputs
                                multiple = prev_state[cur_dim] // tar_st
                                device_index += cur_st * \
                                    multiple * loop_sizes[depth]
                                if multiple == 1:
                                    res = cross_receive(depth+1)
                                else:
                                    res = concatenate_op(
                                        [cross_receive(depth+1) for _ in range(multiple)], axis=cur_dim, ctx=ctx)
                                layer_indices[res] = layer_id
                                device_index += (tar_st - 1 - cur_st) * \
                                    multiple * loop_sizes[depth]
                            elif tar_st % prev_state[cur_dim] == 0:
                                # at `cur_dim` dimension we need to specify one input
                                multiple = tar_st // prev_state[cur_dim]
                                device_index += cur_st // multiple * \
                                    loop_sizes[depth]
                                res = cross_receive(depth+1)
                                device_index += (tar_st - 1 -
                                                 cur_st) // multiple * loop_sizes[depth]
                            else:
                                assert False, 'The dispatch state (%d, %d) at dimension %d is invalid.' % (
                                    prev_state[cur_dim], tar_st, cur_dim)
                    return res
                devices = prev_input.raw_ctx.workers[dev_pos]
                recv_src[prev_input][key] = cross_receive(0)
                assert device_index == len(devices)
        return recv_src[prev_input][key]

    def send_model_parallel(prev_input, node):
        key = node_tar_state_map[prev_input][node]
        assert dp_index_map[prev_input] >= 0, 'The send node must be on local device.'
        dev_pos = dp_index_map[prev_input]
        if not isinstance(prev_input.raw_ctx.workers[dev_pos], tuple):
            # here we send from a node on one device dispatching to many nodes
            # in this case current node MUST have mp_index, and the split will be handled in sending
            assert mp_index_map[prev_input] < 0, 'Here only support 1 to N.'
            device_index = 0

            devices = node.raw_ctx.workers[dev_pos]
            if key not in send_dst[prev_input]:
                send_dst[prev_input][key] = True

                def make_split(cur_state, depth):
                    if len(target_order) == depth:
                        nonlocal device_index
                        if target_ns.is_dist():
                            keys = list(target_state.keys())
                            indices = [cur_state[k] for k in keys]
                            splits = [target_state[k] for k in keys]
                            cur_node = split_op(
                                prev_input, keys, indices, splits, ctx=ctx)
                            layer_indices[cur_node] = layer_id
                        else:
                            cur_node = prev_input
                        sending(prev_input, cur_node, devices[device_index])
                        device_index += 1
                    else:
                        cur_dim = target_order[depth]
                        if cur_dim < 0:
                            for _ in range(target_duplicate):
                                make_split(cur_state, depth + 1)
                        else:
                            for ts in range(target_state[cur_dim]):
                                cur_state[cur_dim] = ts
                                make_split(cur_state, depth + 1)
                target_ns = node_tar_state_map[prev_input][node]
                target_state, target_duplicate, target_order = target_ns.get_all()
                make_split({}, 0)
                assert device_index == len(
                    node.raw_ctx.workers[dev_pos])
        elif not isinstance(node.raw_ctx.workers[dev_pos], tuple):
            # here we send from a node on multiple devices to one node
            # in this case current node MUST NOT have mp_index, and the combination will be handled in receiving
            target = node_tar_state_map[prev_input][node]
            assert mp_index_map[prev_input] >= 0 and not target.is_dist(
            ), 'Here only support N to 1.'
            device = node.raw_ctx.workers[dev_pos]
            if key not in send_dst[prev_input]:
                send_dst[prev_input][key] = True
                sending(prev_input, prev_input, device)
        else:
            # here the prev input and the node are both in model parallel, with different states
            assert mp_index_map[prev_input] >= 0, 'Here only support N to N.'
            devices = node.raw_ctx.workers[dev_pos]
            if key not in send_dst[prev_input]:
                send_dst[prev_input][key] = True
                prev_ns = node_cur_state_map[prev_input]
                target_ns = node_tar_state_map[prev_input][node]
                prev_state = prev_ns.state
                target_state, target_duplicate, target_order = target_ns.get_all()
                cur_state_index = prev_ns.map_dev_to_index(
                    mp_index_map[prev_input])
                loop_sizes = target_ns.get_loop_sizes()
                device_index = 0

                def cross_send(split_cur_state, split_target_state, depth, need_split):
                    nonlocal device_index
                    if depth == len(target_order):
                        if need_split:
                            keys = list(split_target_state.keys())
                            indices = [split_cur_state[k] for k in keys]
                            splits = [split_target_state[k] for k in keys]
                            cur_node = split_op(
                                prev_input, keys, indices, splits, ctx=ctx)
                            layer_indices[cur_node] = layer_id
                        else:
                            cur_node = prev_input
                        sending(prev_input, cur_node, devices[device_index])
                        device_index += 1
                    else:
                        cur_dim = target_order[depth]
                        if cur_dim < 0:
                            for _ in range(target_duplicate):
                                cross_send(
                                    split_cur_state, split_target_state, depth+1, need_split)
                        else:
                            pre_st = prev_state.get(cur_dim, 1)
                            cur_st = cur_state_index.get(cur_dim, 0)
                            if pre_st % target_state[cur_dim] == 0:
                                # at `cur_dim` dimension we need to send one output
                                multiple = pre_st // target_state[cur_dim]
                                device_index += cur_st // multiple * \
                                    loop_sizes[depth]
                                split_cur_state[cur_dim] = 0
                                split_target_state[cur_dim] = 1
                                cross_send(split_cur_state,
                                           split_target_state, depth+1, need_split)
                                device_index += (pre_st - 1 -
                                                 cur_st) // multiple * loop_sizes[depth]
                            elif target_state[cur_dim] % pre_st == 0:
                                # at `cur_dim` dimension we need to split and send some outputs
                                multiple = target_state[cur_dim] // pre_st
                                device_index += cur_st * \
                                    multiple * loop_sizes[depth]
                                for index in range(multiple):
                                    split_cur_state[cur_dim] = index
                                    split_target_state[cur_dim] = multiple
                                    cross_send(split_cur_state,
                                               split_target_state, depth+1, True)
                                device_index += (pre_st - 1 -
                                                 cur_st) * multiple * loop_sizes[depth]
                            else:
                                assert False, 'The dispatch state (%d, %d) at dimension %d is invalid.' % (
                                    pre_st, target_state[cur_dim], cur_dim)
                cross_send({}, {}, 0, False)
                assert device_index == len(devices)

    def set_index(node):
        mp_index = -1
        dp_index = -1
        for i, c in enumerate(node.raw_ctx.workers):
            if isinstance(c, tuple):
                if ctx in c:
                    mp_index = c.index(ctx)
                    dp_index = i
            elif ctx == c:
                dp_index = i
        mp_index_map[node] = mp_index
        dp_index_map[node] = dp_index
        return mp_index, dp_index

    def assign_ctx(node):
        nonlocal layer_id
        if node in dp_index_map:
            return
        mp_index_map[node] = -1
        dp_index_map[node] = -1
        if isinstance(node, DataloaderOp):
            layer_indices[node] = layer_id
            layer_id += 1
            return
        elif isinstance(node, PlaceholderOp):
            assert node_cur_state_map[node].dev_num == node.raw_ctx.mp_device_num,\
                'The node status must conform with raw context.'
            mp_index, dp_index = set_index(node)
            if mp_index >= 0:
                node_status = node_cur_state_map[node]
                node.reshape_in_mp(node_status.map_dev_to_index(
                    mp_index), node_status.state)
            layer_indices[node] = layer_id
            layer_id += 1
            if dp_index >= 0:
                node.ctx = ctx
                if node in node_list:
                    my_eval_nodes.append(node)
                if node.trainable:
                    trainable_params.append(node)
        elif isinstance(node, OptimizerOp):
            nonlocal opt
            assert opt is None, 'Multiple optimizer is invalid.'
            opt = node
            for n in node.inputs:
                assign_ctx(n)
            grads = []
            original_params = node.optimizer.params
            #print(original_params)
            for ind, param in enumerate(original_params):
                ori_grad = node.inputs[ind]
                assert ori_grad.raw_ctx.worker_num == param.raw_ctx.worker_num, \
                    'Worker number of gradient and parameter should be equal!'
                assert mp_index_map[ori_grad] == mp_index_map[param], \
                    'Model parallel state of gradient and parameter should be the same!'
                if mp_index_map[param] < 0:
                    # this branch is pipeline + data parallel
                    ori_dp_index = dp_index_map[ori_grad]
                    par_dp_index = dp_index_map[param]
                    assert (par_dp_index >= 0) == (
                        param in trainable_params), 'Bug appears!'
                    # handle receiving
                    if par_dp_index >= 0 and ori_dp_index != par_dp_index:
                        my_pos = par_dp_index
                        if -1 not in recv_src[ori_grad]:
                            recv_src[ori_grad][-1] = receiving(
                                ori_grad, ori_grad.raw_ctx.workers[my_pos])
                        grads.append(recv_src[ori_grad][-1])
                    elif par_dp_index >= 0:
                        grads.append(ori_grad)
                    # handle sending
                    if ori_dp_index >= 0 and ori_dp_index != par_dp_index:
                        my_pos = ori_dp_index
                        device = param.raw_ctx.workers[my_pos]
                        if -1 not in send_dst[ori_grad]:
                            send_dst[ori_grad][-1] = True
                            sending(ori_grad, ori_grad, device)
                else:
                    # here in the same model parallel
                    assert ori_grad.raw_ctx == param.raw_ctx
                    grads.append(ori_grad)
                layer_id += 2
            #print(trainable_params)
            if trainable_params:
                #print(grads)
                #print(node.inputs)
                node.optimizer.params = trainable_params
                node.inputs = grads
                node.ctx = ctx
                my_eval_nodes.append(node)
                layer_indices[node] = layer_id
                for param in trainable_params:
                    # here handle the nodes that need allreduce
                    if param.raw_ctx.server_num == 0:
                        allreduce_devices = None
                        if mp_index_map[param] < 0 and param.raw_ctx.worker_num > 1:
                            allreduce_devices = param.raw_ctx
                        elif mp_index_map[param] >= 0:
                            target_state, target_duplicate, target_order = node_cur_state_map[param].get_all(
                            )
                            if -1 in target_order:
                                mp_index = mp_index_map[param]
                                interval = 1
                                allreduce_devices = []
                                dup_dim = target_order.index(-1)
                                for cur_order in target_order[dup_dim+1:]:
                                    interval *= target_state[cur_order]
                                macro_interval = interval * target_duplicate
                                start = mp_index - mp_index % macro_interval + mp_index % interval
                                for rc in range(param.raw_ctx.worker_num):
                                    for ind in range(start, start + interval * target_duplicate, interval):
                                        allreduce_devices.append(
                                            param.raw_ctx[rc][ind])
                                allreduce_devices = None if len(
                                    allreduce_devices) <= 1 else DeviceGroup(allreduce_devices)
                        if allreduce_devices is not None:
                            allreduce_devices = allreduce_devices.get_sorted()
                            if allreduce_devices not in comm_groups:
                                if len(allreduce_devices) == mpi_comm.nrank:
                                    comm_groups[allreduce_devices] = mpi_comm
                                else:
                                    comm_groups[allreduce_devices] = new_group_comm(
                                        allreduce_devices)
                            param_allreduce_group[param] = comm_groups[allreduce_devices]
            # here we establish a group comm for loss
            # in case users wants to reduce loss and accuracy to one worker
            loss_node = node.optimizer.loss
            assert mp_index_map[loss_node] < 0, 'Currently loss cannot occur in model parallel.'
            if loss_node.raw_ctx.worker_num > 1:
                allreduce_devices = loss_node.raw_ctx.get_sorted()
                if allreduce_devices not in comm_groups:
                    if len(allreduce_devices) == mpi_comm.nrank:
                        comm_groups[allreduce_devices] = mpi_comm
                    else:
                        comm_groups[allreduce_devices] = new_group_comm(
                            allreduce_devices)
                param_allreduce_group['loss'] = comm_groups[allreduce_devices]
        elif isinstance(node, Optimizer_per_grad_Op):
            # nonlocal opt
            # assert opt is None, 'Multiple optimizer is invalid.'
            opt = node
            for n in node.inputs:
                assign_ctx(n)
            grads = []
            original_params = node.optimizer.params
            # for ind, param in enumerate(original_params):
            param = original_params[node.idx]
            ori_grad = node.inputs[0]
            assert ori_grad.raw_ctx.worker_num == param.raw_ctx.worker_num, \
                'Worker number of gradient and parameter should be equal!'
            assert mp_index_map[ori_grad] == mp_index_map[param], \
                'Model parallel state of gradient and parameter should be the same!'
            if mp_index_map[param] < 0:
                # this branch is pipeline + data parallel
                ori_dp_index = dp_index_map[ori_grad]
                par_dp_index = dp_index_map[param]
                assert (par_dp_index >= 0) == (
                    param in trainable_params), 'Bug appears!'
                # handle receiving
                if par_dp_index >= 0 and ori_dp_index != par_dp_index:
                    print(par_dp_index, ori_dp_index)
                    my_pos = par_dp_index
                    if -1 not in recv_src[ori_grad]:
                        recv_src[ori_grad][-1] = receiving(
                            ori_grad, ori_grad.raw_ctx.workers[my_pos])
                    
                    grads.append(recv_src[ori_grad][-1])
                elif par_dp_index >= 0:
                    grads.append(ori_grad)
                # handle sending
                if ori_dp_index >= 0 and ori_dp_index != par_dp_index:
                    my_pos = ori_dp_index
                    device = param.raw_ctx.workers[my_pos]
                    if -1 not in send_dst[ori_grad]:
                        send_dst[ori_grad][-1] = True
                        sending(ori_grad, ori_grad, device)
            else:
                # here in the same model parallel
                assert ori_grad.raw_ctx == param.raw_ctx
                grads.append(ori_grad)
            layer_id += 2

            if trainable_params:
                if node.idx == len(original_params) - 1:
                    node.optimizer.params_ori = node.optimizer.params
                    node.optimizer.params = trainable_params
                # print(grads, node.inputs, node.name)
                
                # if len(grads)==0:
                #     print(node.name, ctx)
                if len(grads) > 0:
                    node.inputs = grads
                    node.ctx = ctx
                    my_eval_nodes.append(node)
                    layer_indices[node] = layer_id


        else:
            # now we only support SAME model parallel in data parallel
            # and 1 context can only appear once
            mp_index, dp_index = set_index(node)
            for i, n in enumerate(node.inputs):
                if isinstance(n, DataloaderOp):
                    if n not in dp_index_map:
                        layer_indices[n] = layer_id
                        layer_id += 1
                        dp_index_map[n] = -1
                    continue
                assign_ctx(n)
            for i, n in enumerate(node.inputs):
                if isinstance(n, DataloaderOp):
                    if dp_index >= 0 and n in node_list and n not in my_eval_nodes:
                        my_eval_nodes.append(n)
                    continue
                # we assume that in pipeline + data parallel mode,
                # devices number of each stage is equal
                # the device in correspondent place will communicate with each other

                if node in node_tar_state_map[n] \
                        and (node_cur_state_map[n] != node_tar_state_map[n][node] or n.raw_ctx != node.raw_ctx) \
                        and (node_cur_state_map[n].is_dist() or node_tar_state_map[n][node].is_dist()):
                    # here in every context each device appear only once
                    # TODO: consider whether or not release the constraint above?
                    if dp_index_map[n] >= 0:
                        send_model_parallel(n, node)
                    if dp_index >= 0:
                        node.inputs[i] = receive_model_parallel(n, node)
                else:
                    assert node.raw_ctx.worker_num == n.raw_ctx.worker_num, \
                        'In pipeline + data parallel, devices number of each stage should be equal!'
                    assert mp_index == mp_index_map[n]
                    if mp_index < 0:
                        # handle receiving
                        if dp_index >= 0 and dp_index != dp_index_map[n]:
                            my_pos = dp_index
                            if -1 not in recv_src[n]:
                                recv_src[n][-1] = receiving(
                                    n, n.raw_ctx.workers[my_pos])
                            node.inputs[i] = recv_src[n][-1]
                        # handle sending
                        if dp_index_map[n] >= 0 and dp_index != dp_index_map[n]:
                            my_pos = dp_index_map[n]
                            device = node.raw_ctx.workers[my_pos]
                            if -1 not in send_dst[n]:
                                send_dst[n][-1] = True
                                sending(n, n, device)
                    else:
                        # here in the same model parallel
                        assert node.raw_ctx == n.raw_ctx
                layer_id += 1

            layer_indices[node] = layer_id
            layer_id += 1

            if dp_index >= 0:
                node.ctx = ctx
                if node in node_list:
                    my_eval_nodes.append(node)

    opt = None
    trainable_params = []
    comm_groups = {}
    param_allreduce_group = {}
    # in send_dst and recv_src we use node state as part of the key
    # this ignore following cases: the node use the same node state, but into different devices;
    # or the node has two node state, but part of the split is the same on some devices.
    # TODO: solve the problem above
    send_dst = defaultdict(dict)
    recv_src = defaultdict(dict)
    self_buffer = {}  # send and receive from self device
    mp_index_map = {}  # model parallel index
    dp_index_map = {}  # data parallel index
    layer_indices = {}
    layer_id = 0
    my_eval_nodes = []

    for node in node_list:
        assign_ctx(node)
    return my_eval_nodes, param_allreduce_group, layer_indices
