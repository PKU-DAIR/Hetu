import numpy as np
import ctypes
from copy import copy, deepcopy
import hetu as ht
from . import ndarray
from . import gpu_links as gpu_op
from .lr_scheduler import FixedScheduler
from .gpu_ops.Node import Op
from .gpu_ops.EmbeddingLookUp import EmbeddingLookUp_Gradient
from .gpu_ops.ParameterServerCommunicate import ParameterServerCommunicateOp
from .gpu_ops.Variable import PlaceholderOp


class Optimizer(object):
    """Optimizers."""

    def __init__(self, learning_rate, l2reg=0):
        if isinstance(learning_rate, FixedScheduler):
            self.lr_sched = learning_rate
        else:
            assert learning_rate >= 0, \
                "learning rate must be non-negative"
            self.lr_sched = FixedScheduler(learning_rate)
        # now we don't support l2 regularizer for PS mode parameters
        # TODO: support l2 regularizer for PS mode parameters (after PS mode has optimizer on Servers)
        assert l2reg >= 0, 'L2 regularizer should be positive or 0.'
        self.l2reg = l2reg
        self.params = None
        self.tensors = None
        self.initiated = False

    @property
    def learning_rate(self):
        return self.lr_sched.get()

    @staticmethod
    def get_var_list(loss):
        def topo_sort_dfs(node, visited, var_list):
            if node in visited:
                return
            visited.add(node)
            if isinstance(node, PlaceholderOp) and node.trainable:
                var_list.append(node)
                return
            for n in node.inputs:
                topo_sort_dfs(n, visited, var_list)

        visited = set()
        trainable_vars = []
        if isinstance(loss, list):
            for l in loss:
                topo_sort_dfs(l, visited, trainable_vars)
        else:
            topo_sort_dfs(loss, visited, trainable_vars)
        return trainable_vars

    def initiate_states(self, config):
        assert not self.initiated, "Optimizer already initiated."
        self.tensors = [config.placeholder_to_arr_map[node]
                        for node in self.params]
        self.initiated = True

    def update_tensors_version(self, tensor_map):
        self.tensors = [tensor_map[node] for node in self.params]

    def minimize(self, loss, var_list=None):
        """Return an optimizer op to update parameters.

        Parameters
        ----------
        loss: loss node that we are minimizing.
        var_list: list of nodes that we are taking derivative wrt.

        Returns
        -------
        An optimizer node.

        """
        self.loss = loss
        if not var_list:
            var_list = self.get_var_list(loss)
        self.params = var_list
        grads, self.backward2forward, self.forward2backward = ht.gradients(
            loss, self.params, return_all=True)
        optimizer_node = OptimizerOp(grads, self)
        return optimizer_node

    def minimize_per_grad(self, loss, var_list=None):
        self.loss = loss
        if not var_list:
            var_list = self.get_var_list(loss)
        self.params = var_list
        grads, self.backward2forward, self.forward2backward = ht.gradients(
            loss, self.params, return_all=True)
        self.optimizer_nodes = []
        for idx in range(len(grads)):
            self.optimizer_nodes.append(Optimizer_per_grad_Op([grads[idx]], self, idx))
        return self.optimizer_nodes

    def __deepcopy__(self, memo):
        assert not self.initiated, 'Should not deep copy optimizer if already initiated!'
        new_opt = copy(self)
        new_opt.loss = deepcopy(self.loss, memo)
        new_opt.params = [deepcopy(node, memo) for node in self.params]
        new_opt.backward2forward = dict([(deepcopy(k, memo), (deepcopy(n, memo) for n in v))
                                         for k, v in self.backward2forward.items()])
        new_opt.forward2backward = dict([(deepcopy(k, memo), (deepcopy(n, memo) for n in v))
                                         for k, v in self.forward2backward.items()])
        return new_opt


class Optimizer_per_grad_Op(Op):
    def __init__(self, grad, optimizer, idx):
        super().__init__(Optimizer_per_grad_Op, grad, None)
        self.name = "Optimizer_%s_%s" % (optimizer.name, grad[0].name)
        self.optimizer = optimizer
        self.idx = idx

    def compute(self, input_vals, output_val, stream_handle=None, new_tensors_map=None):
        assert output_val is None
        # For PS op, this input_vals is None
        # PS mode doesn't need local update
        if new_tensors_map is not None:
            self.optimizer.update_tensors_version(new_tensors_map)
        if self.comm_mode != 'PS':
            self.optimizer.update_coefficients()
            self.optimizer.update_per_grad(input_vals[0], self.idx, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return None

    def forward_hook(self, config):
        # disable inplace if not lazy execution
        # previously we use array reshape lazy callback to do this, which is deprecated (not efficient)
        for node in self.inputs:
            node.inplace = False

        # if self.idx == len(self.optimizer.params_ori) - 1 or self.idx == 0:
        #     self.optimizer.initiate_states(config)
        if not self.optimizer.initiated:
            self.optimizer.initiate_states(config)
        self.on_cpu = self.on_gpu = None
        self.comm_mode = config.comm_mode
        # some things todo.
        if self.comm_mode != 'PS':
            for i in range(len(self.inputs)):
                # Though the gradients for transfer ops are well defined,
                # we called gradients in optimizer op before transfer ops are added.
                # So here we also add tranfer ops for gradients update.
                # Could be optimized later.
                if not isinstance(self.inputs[i], ParameterServerCommunicateOp):
                    paramctx = self.optimizer.params[i].ctx
                    self.inputs[i] = super().add_transfer_op(
                        self.inputs[i], paramctx, config.h2d_ops, config.d2h_ops)
        #print(self.optimizer.params_ori)
        param = self.optimizer.params_ori[self.idx]
        self.idx = self.optimizer.params.index(param)

    def backward_hook(self, config):
        self.comm_mode = config.comm_mode
        new_inputs = []
        for i, node in enumerate(self.inputs):
            current_strategy = config.node_strategy.get(
                self.optimizer.params[i], self.comm_mode)
            cur_node = node
            if current_strategy == 'AllReduce' or (current_strategy == 'Hybrid' and not isinstance(node, EmbeddingLookUp_Gradient)):
                cur_node = ht.allreduceCommunicate_op(
                    node, config.param_allreduce_group.get(self.optimizer.params[i], config.nccl_comm))
                if config.layer_indices is not None and node in config.layer_indices:
                    config.layer_indices[cur_node] = config.layer_indices[node] + 1
            elif current_strategy == 'PS' or (current_strategy == 'Hybrid' and isinstance(node, EmbeddingLookUp_Gradient)):
                cur_node = ht.parameterServerCommunicate_op(
                    node, self.optimizer.params[i], self.optimizer.get_config())
                if config.layer_indices is not None and node in config.layer_indices:
                    config.layer_indices[cur_node] = config.layer_indices[node] + 1
            new_inputs.append(cur_node)
        self.inputs = new_inputs

    def re_minimize(self):
        new_grads = ht.gradients(self.optimizer.loss, self.optimizer.params)
        self.inputs = new_grads

class OptimizerOp(Op):
    def __init__(self, grads, optimizer):
        super().__init__(OptimizerOp, grads, None)
        self.name = "Optimizer_%s" % (optimizer.name)
        self.optimizer = optimizer

    def compute(self, input_vals, output_val, stream_handle=None, new_tensors_map=None):
        assert output_val is None
        # For PS op, this input_vals is None
        # PS mode doesn't need local update
        if new_tensors_map is not None:
            self.optimizer.update_tensors_version(new_tensors_map)
        if self.comm_mode != 'PS':
            self.optimizer.update(input_vals, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return None

    def forward_hook(self, config):
        # disable inplace if not lazy execution
        # previously we use array reshape lazy callback to do this, which is deprecated (not efficient)
        for node in self.inputs:
            node.inplace = False

        self.optimizer.initiate_states(config)
        self.on_cpu = self.on_gpu = None
        self.comm_mode = config.comm_mode
        # some things todo.
        if self.comm_mode != 'PS':
            for i in range(len(self.inputs)):
                # Though the gradients for transfer ops are well defined,
                # we called gradients in optimizer op before transfer ops are added.
                # So here we also add tranfer ops for gradients update.
                # Could be optimized later.
                if not isinstance(self.inputs[i], ParameterServerCommunicateOp):
                    paramctx = self.optimizer.params[i].ctx
                    self.inputs[i] = super().add_transfer_op(
                        self.inputs[i], paramctx, config.h2d_ops, config.d2h_ops)

    def backward_hook(self, config):
        self.comm_mode = config.comm_mode
        new_inputs = []
        for i, node in enumerate(self.inputs):
            current_strategy = config.node_strategy.get(
                self.optimizer.params[i], self.comm_mode)
            cur_node = node
            if current_strategy == 'AllReduce' or (current_strategy == 'Hybrid' and not isinstance(node, EmbeddingLookUp_Gradient)):
                cur_node = ht.allreduceCommunicate_op(
                    node, config.param_allreduce_group.get(self.optimizer.params[i], config.nccl_comm))
                if config.layer_indices is not None and node in config.layer_indices:
                    config.layer_indices[cur_node] = config.layer_indices[node] + 1
            elif current_strategy == 'PS' or (current_strategy == 'Hybrid' and isinstance(node, EmbeddingLookUp_Gradient)):
                cur_node = ht.parameterServerCommunicate_op(
                    node, self.optimizer.params[i], self.optimizer.get_config())
                if config.layer_indices is not None and node in config.layer_indices:
                    config.layer_indices[cur_node] = config.layer_indices[node] + 1
            new_inputs.append(cur_node)
        self.inputs = new_inputs

    def re_minimize(self):
        new_grads = ht.gradients(self.optimizer.loss, self.optimizer.params)
        self.inputs = new_grads


class SGDOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, l2reg=0):
        super(SGDOptimizer, self).__init__(learning_rate, l2reg)
        self.name = 'SGD'

    def get_config(self):
        return (ctypes.c_int(0), (ctypes.c_float * 1)(self.learning_rate), ctypes.c_int(1))

    def initiate_states(self, config):
        super().initiate_states(config)

    def update(self, grads, stream_handle=None):
        assert self.initiated is True
        params_size = len(self.params)
        assert params_size == len(grads)
        for i in range(params_size):
            if grads[i] == None:
                continue
            if self.params[i].on_gpu:
                assert isinstance(self.tensors[i], ndarray.NDArray)
                assert isinstance(
                    grads[i], (ndarray.NDArray, ndarray.IndexedSlices))
                if self.l2reg > 0:
                    gpu_op.add_l2_regularization(
                        self.tensors[i], grads[i], self.l2reg, stream_handle)
                gpu_op.sgd_update(
                    self.tensors[i], grads[i], self.learning_rate, stream_handle)
            else:
                from ._base import DNNL_LIB
                if isinstance(grads[i], ndarray.IndexedSlices):
                    if DNNL_LIB['cpu_SGDOptimizerSparseUpdate']:
                        from .cpu_links import sgd_update_sparse as cpu_sgd_update_sparse
                        cpu_sgd_update_sparse(
                            self.tensors[i], grads[i].indices, grads[i].values, self.learning_rate)
                    else:
                        grads[i].cpu_deduplicate()
                        np_tensor = self.tensors[i].asnumpy()
                        np_tensor[grads[i].indices.asnumpy().astype(
                            np.int)] -= self.learning_rate * grads[i].values.asnumpy()
                        self.tensors[i][:] = np_tensor
                        grads[i].free_deduplicate()
                else:
                    if DNNL_LIB['cpu_SGDOptimizerUpdate']:
                        from .cpu_links import sgd_update as cpu_sgd_update
                        if self.l2reg > 0:
                            from .cpu_links import add_l2_regularization as cpu_add_l2_regularization
                            cpu_add_l2_regularization(
                                self.tensors[i], grads[i], self.l2reg)
                        cpu_sgd_update(
                            self.tensors[i], grads[i], self.learning_rate)
                    else:
                        prev_param = self.tensors[i].asnumpy()
                        grad = grads[i].asnumpy(
                        ) + self.l2reg * prev_param if self.l2reg > 0 else grads[i].asnumpy()
                        self.tensors[i][:] = prev_param - \
                            self.learning_rate * grad


class MomentumOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, nesterov=False, l2reg=0):
        super(MomentumOptimizer, self).__init__(learning_rate, l2reg)
        self.momentum = momentum
        self.nesterov = nesterov
        self.name = "Momentum"

    def get_config(self):
        return (ctypes.c_int(self.nesterov + 1), (ctypes.c_float * 2)(self.learning_rate, self.momentum), ctypes.c_int(2))

    def initiate_states(self, config):
        super().initiate_states(config)
        self.velocity = []
        for t in self.tensors:
            self.velocity.append(None if t is None else ndarray.array(
                np.zeros(t.shape, dtype=np.float32), t.ctx))

    def update(self, grads, stream_handle=None):
        assert self.initiated is True
        params_size = len(self.params)
        assert params_size == len(grads)
        for i in range(params_size):
            if grads[i] == None:
                continue
            if self.params[i].on_gpu:
                assert isinstance(self.tensors[i], ndarray.NDArray)
                assert isinstance(
                    grads[i], (ndarray.NDArray, ndarray.IndexedSlices))
                assert isinstance(self.velocity[i], ndarray.NDArray)
                if self.l2reg > 0:
                    gpu_op.add_l2_regularization(
                        self.tensors[i], grads[i], self.l2reg, stream_handle)
                gpu_op.momentum_update(self.tensors[i], grads[i], self.velocity[i], self.learning_rate, self.momentum,
                                       self.nesterov, stream_handle)
            else:
                if isinstance(grads[i], ndarray.IndexedSlices):
                    raise NotImplementedError
                else:
                    from ._base import DNNL_LIB
                    if DNNL_LIB['cpu_MomentumOptimizerUpdate']:
                        from .cpu_links import momentum_update as cpu_momentum_update
                        if self.l2reg > 0:
                            from .cpu_links import add_l2_regularization as cpu_add_l2_regularization
                            cpu_add_l2_regularization(
                                self.tensors[i], grads[i], self.l2reg)
                        cpu_momentum_update(self.tensors[i], grads[i], self.velocity[i], self.learning_rate, self.momentum,
                                            self.nesterov)
                    else:
                        prev_param = self.tensors[i].asnumpy()
                        grad = grads[i].asnumpy(
                        ) + self.l2reg * prev_param if self.l2reg > 0 else grads[i].asnumpy()
                        velo = self.velocity[i].asnumpy()
                        if self.nesterov:
                            lr_grads = -self.learning_rate * grad
                            self.velocity[i][:] = self.momentum * \
                                (velo + lr_grads)
                            self.tensors[i][:] = prev_param + velo + lr_grads
                        else:
                            self.velocity[i][:] = self.momentum * \
                                velo - self.learning_rate * grad
                            self.tensors[i][:] = prev_param + velo


class AdaGradOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, initial_accumulator_value=0.0, eps=1e-7, l2reg=0):
        assert initial_accumulator_value >= 0.0, \
            "initial accumulator value must be non-negative"
        assert eps > 0.0, \
            "epsilon must be positive"
        super(AdaGradOptimizer, self).__init__(learning_rate, l2reg)
        self.initial_accumulator_value = initial_accumulator_value
        self.eps = eps
        self.name = "AdaGrad"

    def get_config(self):
        return (ctypes.c_int(3), (ctypes.c_float * 3)(self.learning_rate, self.initial_accumulator_value, self.eps), ctypes.c_int(3))

    def initiate_states(self, config):
        super().initiate_states(config)
        self.accumulator_value = []
        for t in self.tensors:
            self.accumulator_value.append(None if t is None else ndarray.array(
                np.full(t.shape, self.initial_accumulator_value), t.ctx))

    def update(self, grads, stream_handle=None):
        assert self.initiated is True
        params_size = len(self.params)
        assert params_size == len(grads)
        for i in range(params_size):
            if grads[i] == None:
                continue
            if self.params[i].on_gpu:
                assert isinstance(self.tensors[i], ndarray.NDArray)
                assert isinstance(
                    grads[i], (ndarray.NDArray, ndarray.IndexedSlices))
                if self.l2reg > 0:
                    gpu_op.add_l2_regularization(
                        self.tensors[i], grads[i], self.l2reg, stream_handle)
                gpu_op.adagrad_update(self.tensors[i], grads[i], self.accumulator_value[i], self.learning_rate, self.eps,
                                      stream_handle)
            else:
                if isinstance(grads[i], ndarray.IndexedSlices):
                    raise NotImplementedError
                else:
                    from ._base import DNNL_LIB
                    if DNNL_LIB['cpu_AdaGradOptimizerUpdate']:
                        from .cpu_links import adagrad_update as cpu_adagrad_update
                        if self.l2reg > 0:
                            from .cpu_links import add_l2_regularization as cpu_add_l2_regularization
                            cpu_add_l2_regularization(
                                self.tensors[i], grads[i], self.l2reg)
                        cpu_adagrad_update(
                            self.tensors[i], grads[i], self.accumulator_value[i], self.learning_rate, self.eps)
                    else:
                        prev_param = self.tensors[i].asnumpy()
                        grad = grads[i].asnumpy(
                        ) + self.l2reg * prev_param if self.l2reg > 0 else grads[i].asnumpy()
                        self.accumulator_value[i][:] = self.accumulator_value[i].asnumpy(
                        ) + np.power(grad, 2)
                        self.tensors[i][:] = \
                            prev_param - self.learning_rate * grad / \
                            (np.sqrt(
                                self.accumulator_value[i].asnumpy()) + self.eps)


class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7, l2reg=0):
        super(AdamOptimizer, self).__init__(learning_rate, l2reg)
        self.beta1 = beta1
        self.beta1_t = 1.0
        self.beta2 = beta2
        self.beta2_t = 1.0
        self.epsilon = epsilon
        self.name = "Adam"

    def get_config(self):
        return (ctypes.c_int(4), (ctypes.c_float * 4)(self.learning_rate, self.beta1, self.beta2, self.epsilon), ctypes.c_int(4))

    def initiate_states(self, config):
        super().initiate_states(config)
        self.m = []
        self.v = []
        for t in self.tensors:
            self.m.append(None if t is None else ndarray.array(
                np.zeros(t.shape), t.ctx))
            self.v.append(None if t is None else ndarray.array(
                np.zeros(t.shape), t.ctx))

    def update(self, grads, stream_handle=None):
        assert self.initiated is True
        params_size = len(self.tensors)
        assert params_size == len(grads)
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2
        for i in range(params_size):
            if grads[i] == None:
                continue
            if self.params[i].on_gpu:
                assert isinstance(self.tensors[i], ndarray.NDArray)
                assert isinstance(
                    grads[i], (ndarray.NDArray, ndarray.IndexedSlices))
                assert isinstance(self.m[i], ndarray.NDArray)
                assert isinstance(self.v[i], ndarray.NDArray)
                if self.l2reg > 0:
                    gpu_op.add_l2_regularization(
                        self.tensors[i], grads[i], self.l2reg, stream_handle)
                gpu_op.adam_update(self.tensors[i], grads[i], self.m[i], self.v[i], self.learning_rate, self.beta1,
                                   self.beta2, self.beta1_t, self.beta2_t, self.epsilon, stream_handle)
            else:
                if isinstance(grads[i], ndarray.IndexedSlices):
                    raise NotImplementedError
                else:
                    from ._base import DNNL_LIB
                    if DNNL_LIB['cpu_AdamOptimizerUpdate']:
                        from .cpu_links import adam_update as cpu_adam_update
                        if self.l2reg > 0:
                            from .cpu_links import add_l2_regularization as cpu_add_l2_regularization
                            cpu_add_l2_regularization(
                                self.tensors[i], grads[i], self.l2reg)
                        cpu_adam_update(self.tensors[i], grads[i], self.m[i], self.v[i], self.learning_rate, self.beta1,
                                        self.beta2, self.beta1_t, self.beta2_t, self.epsilon)
                    else:
                        prev_param = self.tensors[i].asnumpy()
                        grad = grads[i].asnumpy(
                        ) + self.l2reg * prev_param if self.l2reg > 0 else grads[i].asnumpy()
                        self.m[i][:] = self.beta1 * \
                            self.m[i].asnumpy() + (1 - self.beta1) * grad
                        self.v[i][:] = self.beta2 * self.v[i].asnumpy() + \
                            (1 - self.beta2) * grad * grad
                        mc = self.m[i].asnumpy() / (1 - self.beta1_t)
                        vc = self.v[i].asnumpy() / (1 - self.beta2_t)
                        self.tensors[i][:] = prev_param - \
                            self.learning_rate * mc / \
                            (np.sqrt(vc) + self.epsilon)


class AdamWOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7, weight_decay=0):
        super(AdamWOptimizer, self).__init__(learning_rate)
        self.beta1 = beta1
        self.beta1_t = 1.0
        self.beta2 = beta2
        self.beta2_t = 1.0
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.name = "AdamW"

    def get_config(self):
        return (ctypes.c_int(5), (ctypes.c_float * 5)(self.learning_rate, self.beta1, self.beta2, self.epsilon, self.weight_decay), ctypes.c_int(5))

    def initiate_states(self, config):
        super().initiate_states(config)
        self.m = []
        self.v = []
        for t in self.tensors:
            self.m.append(None if t is None else ndarray.array(
                np.zeros(t.shape), t.ctx))
            self.v.append(None if t is None else ndarray.array(
                np.zeros(t.shape), t.ctx))

    def update(self, grads, stream_handle=None):
        assert self.initiated is True
        params_size = len(self.tensors)
        assert params_size == len(grads)
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2
        for i in range(params_size):
            if grads[i] == None:
                continue
            if self.params[i].on_gpu:
                assert isinstance(self.tensors[i], ndarray.NDArray)
                assert isinstance(
                    grads[i], (ndarray.NDArray, ndarray.IndexedSlices))
                assert isinstance(self.m[i], ndarray.NDArray)
                assert isinstance(self.v[i], ndarray.NDArray)
                gpu_op.adamw_update(self.tensors[i], grads[i], self.m[i], self.v[i], self.learning_rate, self.beta1,
                                   self.beta2, self.beta1_t, self.beta2_t, self.epsilon, self.weight_decay, stream_handle)
            else:
                if isinstance(grads[i], ndarray.IndexedSlices):
                    raise NotImplementedError
                else:
                    prev_param = self.tensors[i].asnumpy()
                    grad = grads[i].asnumpy()
                    self.m[i][:] = self.beta1 * \
                        self.m[i].asnumpy() + (1 - self.beta1) * grad
                    self.v[i][:] = self.beta2 * self.v[i].asnumpy() + \
                        (1 - self.beta2) * grad * grad
                    mc = self.m[i].asnumpy() / (1 - self.beta1_t)
                    vc = self.v[i].asnumpy() / (1 - self.beta2_t)
                    update = mc / (np.sqrt(vc) + self.epsilon)
                    self.tensors[i][:] = prev_param - \
                        self.learning_rate * (update + self.weight_decay * self.tensors[i])

    def update_coefficients(self):
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2

    def update_per_grad(self, grad, i, stream_handle=None):
        assert self.initiated is True
        if grad == None:
            return
        if self.params[i].on_gpu:
            assert isinstance(self.tensors[i], ndarray.NDArray)
            assert isinstance(
                grad, (ndarray.NDArray, ndarray.IndexedSlices))
            assert isinstance(self.m[i], ndarray.NDArray)
            assert isinstance(self.v[i], ndarray.NDArray)
            gpu_op.adamw_update(self.tensors[i], grad, self.m[i], self.v[i], self.learning_rate, self.beta1,
                                self.beta2, self.beta1_t, self.beta2_t, self.epsilon, self.weight_decay, stream_handle)
        else:
            if isinstance(grad, ndarray.IndexedSlices):
                raise NotImplementedError
            else:
                prev_param = self.tensors[i].asnumpy()
                grad = grad.asnumpy()
                self.m[i][:] = self.beta1 * \
                    self.m[i].asnumpy() + (1 - self.beta1) * grad
                self.v[i][:] = self.beta2 * self.v[i].asnumpy() + \
                    (1 - self.beta2) * grad * grad
                mc = self.m[i].asnumpy() / (1 - self.beta1_t)
                vc = self.v[i].asnumpy() / (1 - self.beta2_t)
                update = mc / (np.sqrt(vc) + self.epsilon)
                self.tensors[i][:] = prev_param - \
                    self.learning_rate * (update + self.weight_decay * self.tensors[i])

class LambOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7, weight_decay=0):
        super(LambOptimizer, self).__init__(learning_rate)
        self.beta1 = beta1
        self.beta1_t = 1.0
        self.beta2 = beta2
        self.beta2_t = 1.0
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.name = "Lamb"

    def get_config(self):
        return (ctypes.c_int(5), (ctypes.c_float * 5)(self.learning_rate, self.beta1, self.beta2, self.epsilon, self.weight_decay), ctypes.c_int(5))

    def initiate_states(self, config):
        super().initiate_states(config)
        self.m = []
        self.v = []
        for t in self.tensors:
            self.m.append(None if t is None else ndarray.array(
                np.zeros(t.shape), t.ctx))
            self.v.append(None if t is None else ndarray.array(
                np.zeros(t.shape), t.ctx))

    def update(self, grads, stream_handle=None):
        assert self.initiated is True
        params_size = len(self.tensors)
        assert params_size == len(grads)
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2
        for i in range(params_size):
            if grads[i] == None:
                continue
            if self.params[i].on_gpu:
                assert isinstance(self.tensors[i], ndarray.NDArray)
                assert isinstance(
                    grads[i], (ndarray.NDArray, ndarray.IndexedSlices))
                assert isinstance(self.m[i], ndarray.NDArray)
                assert isinstance(self.v[i], ndarray.NDArray)
                gpu_op.lamb_update(self.tensors[i], grads[i], self.m[i], self.v[i], self.learning_rate, self.beta1,
                                   self.beta2, self.beta1_t, self.beta2_t, self.epsilon, self.weight_decay, stream_handle)
            else:
                if isinstance(grads[i], ndarray.IndexedSlices):
                    raise NotImplementedError
                else:
                    prev_param = self.tensors[i].asnumpy()
                    grad = grads[i].asnumpy()
                    self.m[i][:] = self.beta1 * \
                        self.m[i].asnumpy() + (1 - self.beta1) * grad
                    self.v[i][:] = self.beta2 * self.v[i].asnumpy() + \
                        (1 - self.beta2) * grad * grad
                    mc = self.m[i].asnumpy() / (1 - self.beta1_t)
                    vc = self.v[i].asnumpy() / (1 - self.beta2_t)
                    update = mc / (np.sqrt(vc) + self.epsilon)
                    norm2_param = np.sqrt(np.sum(np.power(self.tensors[i], 2)))
                    norm2_update = np.sqrt(np.sum(np.power(update, 2)))
                    self.tensors[i][:] = prev_param - \
                        self.learning_rate * norm2_param / norm2_update * (update + self.weight_decay * self.tensors[i])