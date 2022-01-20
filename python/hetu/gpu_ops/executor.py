""" library to take autodiff and execute a computation graph """
from __future__ import absolute_import
from .BatchNorm import Batch_NormalizationOp
import numpy as np
from scipy.sparse import spmatrix, coo_matrix
from .. import ndarray
from .._base import DNNL_LIB
from ..cpu_links import array_set as cpu_array_set
from .Variable import PlaceholderOp  # add for optimizer
from ..dataloader import DataloaderOp, GNNDataLoaderOp
from .AllReduceCommunicate import AllReduceCommunicateOp
from .ParameterServerCommunicate import ParameterServerCommunicateOp, ParameterServerSparsePullOp, parameterServerSparsePull_op
from .Sum import sum_op
from .DataTransfer import DataH2DOp, DataD2HOp, DataD2HSparseOp
from ..communicator.mpi_nccl_comm import ncclDataType_t, GroupStart, GroupEnd
from .EmbeddingLookUp import EmbeddingLookUp, EmbeddingLookUp_Gradient
from ..optimizer import OptimizerOp, Optimizer_per_grad_Op
from . import OnesLike
from ..stream import create_stream_handle, Event
from ..context import get_current_context, get_launch_config_by_traverse_nodes, assign_context_by_traverse_nodes, DeviceGroup
from .PipelineSend import PipelineSendOp
from .PipelineReceive import PipelineReceiveOp
from .Sum import SumOp
from .Split import SplitOp
from .Concatenate import ConcatenateOp
from .Dropout import DropoutOp
from .LayerNorm import Layer_NormalizationOp
from .OnesLike import OnesLikeOp
from operator import add
from functools import reduce
import ctypes
import os
from time import time
import pickle


def path_to_lib(name):
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_path = os.path.join(curr_path, '../../../build/lib/')
    return os.path.join(lib_path, name)


def wrapped_mpi_nccl_init(init_nccl=True, devices=None):
    from ..communicator.mpi_nccl_comm import mpi_communicator
    global mpi_comm
    global nccl_comm
    if 'mpi_comm' not in globals():
        mpi_comm = mpi_communicator(devices=devices)
        if 'nccl_comm' not in globals():
            nccl_comm = mpi_comm.ncclInit() if init_nccl else None
    return nccl_comm


def new_group_comm(devices_context=None):
    assert 'mpi_comm' in globals()
    global mpi_comm
    if devices_context is None:
        comm = mpi_comm.ncclInit()
    else:
        comm = mpi_comm.ncclGroupInit(devices_context)
    return comm


def get_mpi_communicate():
    global mpi_comm
    return mpi_comm


def get_nccl_communicate():
    global nccl_comm
    return nccl_comm


def get_worker_communicate():
    global ps_comm
    return ps_comm


def worker_init():
    global ps_comm
    ll = ctypes.cdll.LoadLibrary
    ps_comm = ll(path_to_lib("libps.so"))
    ps_comm.Init()


def worker_finish():
    ps_comm.Finalize()


def server_init():
    global ps_comm
    ll = ctypes.cdll.LoadLibrary
    ps_comm = ll(path_to_lib("libps.so"))
    ps_comm.Init()
    ps_comm.StartServer()


def server_finish():
    ps_comm.Finalize()


def scheduler_init():
    global ps_comm
    ll = ctypes.cdll.LoadLibrary
    ps_comm = ll(path_to_lib("libps.so"))
    ps_comm.Init()


def scheduler_finish():
    ps_comm.Finalize()


class HetuConfig(object):
    __slots__ = [
        'eval_node_list',
        'train_name',
        'val_name',
        'context',
        'seed',
        'np_rand',
        'comm_mode',
        'node_strategy',
        'context_launch',
        'ps_comm',
        'nccl_comm',
        'local_rank',
        'rank',
        'nrank',
        'p2p_stream',
        'comp_stream',
        'nccl_stream',
        'h2d_stream',
        'd2h_stream',
        'h2d_ops',
        'd2h_ops',
        'ps_map',
        'infer_ps_map',
        'dataloader_ops',
        'use_sparse_pull',
        'cstable_policy',
        'inference',
        'enable_lazy',
        'bsp',
        'prefetch',
        'cache_bound',
        'log_path',
        'my_eval_nodes',
        'param_allreduce_group',
        'placeholder_to_arr_map',
        'pipeline',
        'pipeline_rank',
        'pipeline_nrank',
        'pipeline_dp_rank',
        'use_preduce',
        'dynamic_memory',
        'layer_indices',
        'dist_strategy',
    ]

    def __init__(
        self,
        eval_node_list,
        train_name,
        val_name,
        ctx=None,
        seed=None,
        comm_mode=None,
        use_sparse_pull=True,
        cstable_policy=None,
        bsp=-1,
        prefetch=True,
        enable_lazy=True,
        cache_bound=100,
        log_path=None,
        pipeline=None,
        dynamic_memory=False,
        dist_strategy=None,
        use_preduce=False,
    ):
        '''
        context: default device context
        comm_mode: communication mode, should be one of the following
            None       -> Single GPU
            PS         -> Parameter Server
            AllRedeuce -> MPI AllReduce
            Hybrid     -> Parameter Server for Sparse Parameter and MPI AllReduce for Dense Parameter
        '''
        assert pipeline in (None, "gpipe", "pipedream", "hetpipe")
        self.pipeline = pipeline
        self.use_preduce = use_preduce

        self.eval_node_list = eval_node_list
        self.train_name = train_name
        self.val_name = val_name

        self.dynamic_memory = dynamic_memory

        # check context
        self.dist_strategy = dist_strategy
        node_cur_state_map, node_tar_state_map = None, None
        if self.dist_strategy is not None:
            if self.dist_strategy.use_dispatch:
                ctx = self.dist_strategy.set_raw_ctxs(eval_node_list)
            else:
                ctx, node_cur_state_map, node_tar_state_map = self.dist_strategy.set_raw_ctxs_n_states(
                    eval_node_list)
        elif ctx is None:
            ctx = get_current_context()
        assert ctx, 'Default context should be determined.'

        self.comm_mode = comm_mode
        self.node_strategy = {}
        local_gpu_devices = None
        context_launch = isinstance(ctx, DeviceGroup)
        self.context_launch = context_launch
        if context_launch:
            # with context usage
            launchMPI, launchPS, self.node_strategy, devices = get_launch_config_by_traverse_nodes(
                eval_node_list, ctx)
            local_gpu_devices = sorted(
                [dev.device_id for dev in devices if dev.local and ndarray.is_gpu_ctx(dev)])
            if not launchMPI and not launchPS:
                self.comm_mode = None
            elif launchMPI and (not launchPS or self.use_preduce):
                self.comm_mode = 'AllReduce'
            elif not launchMPI and launchPS:
                self.comm_mode = 'PS'
            else:
                self.comm_mode = 'Hybrid'
            # in pipeline or model parallel we have to initialize another p2p stream
            init_p2p_stream = len(devices) != len(ctx)

        # variables initialization
        self.seed = seed if seed is not None else np.int64(time())
        self.np_rand = np.random.RandomState(self.seed)

        # get attribute of communication mode
        self.ps_comm = None
        self.nccl_comm = None
        self.local_rank = None
        self.rank = None
        self.nrank = None
        ps_nrank = None
        if self.comm_mode == 'PS' or self.comm_mode == 'Hybrid':
            worker_init()
            self.ps_comm = get_worker_communicate()
            ps_rank = int(self.ps_comm.rank())
            ps_nrank = int(
                os.environ['DMLC_NUM_WORKER']) if 'DMLC_NUM_WORKER' in os.environ else 1
            topo_sort_register_ps(
                eval_node_list, self.ps_comm, self.comm_mode, self.seed, cstable_policy)
        if self.comm_mode == "Hybrid" or self.comm_mode == "AllReduce":
            self.nccl_comm = wrapped_mpi_nccl_init(devices=local_gpu_devices)
        elif context_launch:
            self.nccl_comm = wrapped_mpi_nccl_init(
                init_nccl=init_p2p_stream, devices=local_gpu_devices)
        if self.nccl_comm is not None:
            self.local_rank = self.nccl_comm.local_rank
            device_id = self.nccl_comm.dev_id
            self.rank = self.nccl_comm.rank
            self.nrank = self.nccl_comm.nrank
            if ps_nrank:
                assert ps_nrank == self.nrank
        elif self.comm_mode == 'PS':
            self.rank = ps_rank
            self.nrank = ps_nrank
            if context_launch:
                global mpi_comm
                self.local_rank = mpi_comm.local_rank
                device_id = mpi_comm.dev_id
        elif context_launch:
            assert len(devices) == 1
            device_id = devices.pop().device_id

        self.my_eval_nodes = eval_node_list
        self.p2p_stream = None
        self.param_allreduce_group = {}
        self.layer_indices = None
        if context_launch:
            # comm_mode is None <=> only 1 model parallel instance
            self.context = ndarray.gpu(device_id)
            if self.pipeline is not None:
                self.pipeline_rank, self.pipeline_nrank, self.pipeline_dp_rank = get_pipeline_stage_info(
                    eval_node_list, self.context)
            else:
                self.pipeline_rank, self.pipeline_nrank, self.pipeline_dp_rank = 0, 1, self.rank
            self.p2p_stream = create_stream_handle(
                self.context) if init_p2p_stream else None
            if node_cur_state_map is None:
                from ..context import parse_graph_with_dispatch
                node_cur_state_map, node_tar_state_map = parse_graph_with_dispatch(
                    eval_node_list)
            self.my_eval_nodes, self.param_allreduce_group, self.layer_indices = assign_context_by_traverse_nodes(
                eval_node_list, self.context, self.nccl_comm, self.p2p_stream, node_cur_state_map, node_tar_state_map)
            for param in self.param_allreduce_group:
                self.node_strategy[param] = 'AllReduce'
            if self.param_allreduce_group != {}:
                if self.comm_mode is None:
                    self.comm_mode = 'AllReduce'
                if self.comm_mode == 'PS':
                    self.comm_mode = 'Hybrid'
        else:
            self.pipeline_rank, self.pipeline_nrank, self.pipeline_dp_rank = 0, 1, self.rank
            self.context = ctx

        on_gpu = ndarray.is_gpu_ctx(self.context)

        self.nccl_stream = None
        if self.comm_mode == "Hybrid" or self.comm_mode == "AllReduce":
            if on_gpu:
                self.nccl_stream = create_stream_handle(self.context)
            self.nccl_comm = get_nccl_communicate()

        # define streams
        self.comp_stream = create_stream_handle(
            self.context) if on_gpu else None
        self.h2d_stream = create_stream_handle(
            self.context) if on_gpu else None
        self.d2h_stream = create_stream_handle(
            self.context) if on_gpu else None

        self.use_sparse_pull = use_sparse_pull if self.comm_mode == 'PS' or self.comm_mode == "Hybrid" else False
        self.cstable_policy = cstable_policy if self.comm_mode == 'PS' or self.comm_mode == "Hybrid" else None
        self.prefetch = prefetch if self.comm_mode == 'PS' or self.comm_mode == 'Hybrid' else False
        if self.cstable_policy is not None:
            self.cstable_policy = self.cstable_policy.upper()
            self.use_sparse_pull = False

        self.h2d_ops = {}
        self.d2h_ops = {}
        self.ps_map = {}
        self.infer_ps_map = {}
        self.enable_lazy = False and enable_lazy  # now we don't use lazy
        self.bsp = bsp
        self.cache_bound = int(cache_bound)
        if self.dynamic_memory:
            self.enable_lazy = True

        self.log_path = log_path
        if log_path is not None and (self.comm_mode == 'PS' or self.comm_mode == "Hybrid"):
            assert os.path.isdir(
                log_path), 'Need to specify a work directory to save logs.'
            self.ps_comm.startRecord(ctypes.c_char_p(bytes(log_path, 'utf-8')))

        self.placeholder_to_arr_map = dict()
        topo_sort_with_hook(self.my_eval_nodes, self)


class Executor(object):
    """Executor computes values for given set of nodes in computation graph."""

    def __init__(self, eval_node_dict, config=None, **kargs):
        """
        Parameters
        ----------
        eval_node_dict: dict of list of nodes whose values need to be computed.
        """
        if not isinstance(eval_node_dict, dict):
            eval_node_dict = {'default': eval_node_dict}
        train_name, val_name = None, None
        for k, v in eval_node_dict.items():
            if any([isinstance(node, OptimizerOp) for node in v]):
                # get the last subexecutor containing optimizer as train for ps op
                train_name = k
            else:
                # get the last subexecutor not containing optimizer as val for ps op
                val_name = k
        all_eval_nodes = []
        for n in reduce(add, eval_node_dict.values()):
            if n not in all_eval_nodes:
                all_eval_nodes.append(n)
        if config is None:
            config = HetuConfig(eval_node_list=all_eval_nodes,
                                train_name=train_name, val_name=val_name, **kargs)
        assert isinstance(
            config, HetuConfig), 'Config type %s invalid.' % str(type(config))

        self.eval_node_dict = eval_node_dict
        self.config = config

        def get_sub_executor(k):
            if config.pipeline == "gpipe" and k == "train":
                from .gpipe_subexecutor import SubExecutor4Gpipe
                return SubExecutor4Gpipe
            elif (config.pipeline == "pipedream" or config.pipeline == "hetpipe") and k == "train":
                from .pipedream_subexecutor import SubExecutor4Pipedream
                return SubExecutor4Pipedream
            return SubExecutor

        self.subexecutor = {k: get_sub_executor(k)(
            k, v, config) for k, v in eval_node_dict.items()}

        self.topo_order = find_topo_sort(config.my_eval_nodes)
        self.param_nodes = [node for node in self.topo_order if isinstance(
            node, PlaceholderOp) and node.trainable]
        self.comm_mode = self.config.comm_mode
        self.ps_comm = self.config.ps_comm
        self.local_rank = self.config.local_rank
        self.rank = self.config.rank

    def profile(self, feed_shapes, log_file, profiler='cpu', name='default'):
        self.subexecutor[name].profile(
            feed_shapes, log_file, profiler=profiler)

    def run(self, name='default', eval_node_list={}, feed_dict={}, convert_to_numpy_ret_vals=False, **kwargs):
        return self.subexecutor[name].run(eval_node_list, feed_dict, convert_to_numpy_ret_vals, **kwargs)

    @property
    def batch_num(self):
        assert len(
            self.subexecutor) == 1, 'Batch num should be used with only 1 subexecutor.'
        return list(self.subexecutor.values())[0].batch_num

    def get_batch_num(self, name='default'):
        return self.subexecutor[name].batch_num

    def save(self, file_path, file_name):
        assert os.path.isdir(
            file_path), 'Need to specify a work directory to save parameters.'
        state_dic={}
        if self.comm_mode in (None, 'AllReduce'):
            # when using allreduce, users need to specify the worker whose rank equals 0 to save
            for node in self.param_nodes:
                state_dic[node.name] = self.config.placeholder_to_arr_map[node].asnumpy()
        else:
            self.ps_comm.BarrierWorker()
            if self.config.rank == 0:
                for node in self.param_nodes:
                    if node.is_embed or self.comm_mode == 'PS':
                        node.event.sync()
                        nodeid = ctypes.c_int(node.id)
                        self.ps_comm.SaveParam(
                            nodeid, ctypes.c_char_p(bytes(file_path, 'utf-8')))
                        self.ps_comm.Wait(nodeid)
                    else:
                        state_dic[node.name] = self.config.placeholder_to_arr_map[node].asnumpy()
            self.ps_comm.BarrierWorker()
        
        with open(file_path+file_name, "wb") as writer:
            pickle.dump(state_dic, writer)

    def load(self, file_path, file_name):
        assert os.path.isdir(
            file_path), 'Need to specify a work directory to load parameters.'

        state_dic={}
        with open(file_path+file_name, "rb") as reader:
            state_dic = pickle.load(reader)

        if self.comm_mode in (None, 'AllReduce'):
            for node in self.param_nodes:
                if node.name in state_dic:
                    self.config.placeholder_to_arr_map[node][:]=state_dic[node.name]
        else:
            self.ps_comm.BarrierWorker()
            if self.config.rank == 0:
                for node in self.param_nodes:
                    if node.is_embed or self.comm_mode == 'PS':
                        node.event.sync()
                        nodeid = ctypes.c_int(node.id)
                        self.ps_comm.LoadParam(
                            nodeid, ctypes.c_char_p(bytes(file_path, 'utf-8')))
                        node.event.update()
            self.ps_comm.BarrierWorker()
            for node in self.topo_order:
                if isinstance(node, PlaceholderOp) and node.trainable and not node.is_embed:
                    if self.comm_mode == 'PS':
                        node.event.sync()
                        nodeid = ctypes.c_int(node.id)
                        self.ps_comm.Pull(
                            nodeid, self.config.ps_map[node].handle)
                        node.event.update()
                    else:
                        if node.name in state_dic:
                            self.config.placeholder_to_arr_map[node][:]=state_dic[node.name]
                elif isinstance(node, EmbeddingLookUp) and self.config.prefetch:
                    node.event.sync()
                    nodeid = ctypes.c_int(node.inputs[0].id)
                    self.ps_comm.SparsePull(nodeid, node.inputs[1].get_next_arr(
                        self.name).handle, self.config.ps_map[node.inputs[0]].handle)
                    node.event.update()
            self.ps_comm.BarrierWorker()

    def recordLoads(self):
        for node in self.config.ps_map:
            node.event.sync()
        self.ps_comm.getLoads()

    def reduceMean(self, arr, root=0):
        # only used for loss and accuracy, etc.
        # the communicator is formed by the context of loss
        if 'loss' in self.config.param_allreduce_group:
            comm = self.config.param_allreduce_group['loss']
            if comm.local_rank == -1:
                # in case local context is not in communicator
                return arr
            local_array = ndarray.array(arr, ndarray.cpu())
            comm.dlarrayNcclReduce(local_array, local_array, root)
            comm.stream.sync()
            return local_array.asnumpy() / comm.nrank
        else:
            return arr

    def __del__(self):
        if self.config.comp_stream is not None:
            self.config.comp_stream.sync()
        if self.config.h2d_stream is not None:
            self.config.h2d_stream.sync()
        if self.config.d2h_stream is not None:
            self.config.d2h_stream.sync()
        if self.config.nccl_stream is not None:
            self.config.nccl_stream.sync()
        for node in self.param_nodes:
            if node.event:
                node.event.sync()
        if self.comm_mode in ('PS', 'Hybrid'):
            worker_finish()


class SubExecutor(object):
    def __init__(self, name, eval_node_list, config):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        topo_order: list of nodes in topological order
        node_to_shape_map: dict from node to shape of the node
        node_to_arr_map: dict from node to ndarray.NDArray allocated for node
        feed_shapes: shapes of feed_dict from last run(...)
        """
        self.name = name
        self.eval_node_list = eval_node_list
        self.config = config
        inference = not any([isinstance(node, OptimizerOp) or isinstance(node, Optimizer_per_grad_Op)
                             for node in eval_node_list])
        self.inference = inference

        if config.pipeline:
            assert self.inference
            self.eval_node_list = []
            # Remove the last pipeline send on worker 1...n-1 ( not needed in inference stage) and the optimizer
            remove_send = 1 if config.pipeline_rank > 0 else 0
            for node in config.my_eval_nodes[::-1]:
                if remove_send and isinstance(node, PipelineSendOp):
                    remove_send = 0
                elif not isinstance(node, OptimizerOp):
                    self.eval_node_list.append(node)
            self.global_eval_nodes = eval_node_list
        elif config.p2p_stream:
            self.run_results_indices = [eval_node_list.index(
                node) if node in eval_node_list else -1 for node in config.my_eval_nodes]
            self.eval_node_list = config.my_eval_nodes
            self.global_eval_nodes = eval_node_list

        if inference == False:
            self.topo_order = find_topo_sort(self.eval_node_list)
        else:  # in inference phase
            if self.config.use_sparse_pull == True or self.config.cstable_policy is not None:
                # insert ps_sparse_pull_op
                self.topo_order = find_topo_sort_inference(self.eval_node_list)
                # fetch sparse parameter
                fetch_sparse_parameter_value(self.topo_order, self.config)
            else:
                self.topo_order = find_topo_sort(self.eval_node_list)

        self.topo_order = reorder_for_group(
            self.topo_order, self.config.layer_indices)

        # main structures, nodes' shapes and arrays
        self.node_to_shape_map = {}
        self.node_to_arr_map = {}

        # inherit from configurations
        self.comm_mode = self.config.comm_mode
        self.ps_comm = self.config.ps_comm
        self.nccl_comm = self.config.nccl_comm
        self.comp_stream = self.config.comp_stream
        self.h2d_stream = self.config.h2d_stream
        self.d2h_stream = self.config.d2h_stream
        self.nccl_stream = self.config.nccl_stream
        self.param_psval_map = self.config.infer_ps_map if self.inference else self.config.ps_map
        self.use_sparse_pull = self.config.use_sparse_pull
        self.cstable_policy = self.config.cstable_policy
        self.use_p2p = self.config.p2p_stream is not None
        self.dynamic_memory = self.config.dynamic_memory

        # assisting structures, improve performance
        self.need_feed_nodes = []
        self.param_nodes = []
        self.dataloader_nodes = []
        self.computing_nodes = []

        # structures related to memory pool, used when dynamic_memory == True
        self.node_outdeg_map = {}
        self.node_ref_cnt = {}
        self.memory_pool = {}
        for node in self.topo_order:
            self.node_outdeg_map[node] = 0
            self.node_ref_cnt[node] = None

        for node in self.topo_order:
            if isinstance(node, DataloaderOp) or isinstance(node, GNNDataLoaderOp):
                self.dataloader_nodes.append(node)
            elif isinstance(node, PlaceholderOp):
                if node.shape is None:
                    self.need_feed_nodes.append(node)
                elif node.trainable:
                    self.param_nodes.append(node)
            elif not ((self.use_sparse_pull or self.cstable_policy) and isinstance(node, EmbeddingLookUp) and self.config.prefetch):
                self.computing_nodes.append(node)
            for n in node.inputs:
                self.node_outdeg_map[n] += 1
        self.batch_num = set([node.get_batch_num(self.name)
                              for node in self.dataloader_nodes])
        assert len(self.batch_num) <= 1, 'Batch num not conform.'
        self.batch_num = None if len(
            self.batch_num) == 0 else self.batch_num.pop()
        self.init_need_allocation = (self.need_feed_nodes == []) and (
            self.dataloader_nodes == [])

        self.tensor_memory_allocated = 0
        self.change_topo = False

    def profile(self, feed_shapes, log_file, profiler='cpu'):
        # !!! we should profile before using distributed settings
        # !!! so here we don't consider multiple devices
        # !!! no self.use_p2p, no node reshape, no pipeline configuration
        # !!! not support sparse input now
        # !!! not support dynamic memory now
        assert profiler in ('cpu', 'gpu')
        assert len(feed_shapes) == len(self.need_feed_nodes) + \
            len(self.dataloader_nodes)

        need_reallocation = self.init_need_allocation

        for node, shape in feed_shapes.items():
            assert node in self.need_feed_nodes or node in self.dataloader_nodes
            local_realloc = shape != self.node_to_shape_map.get(node, None)
            need_reallocation = need_reallocation or local_realloc
            if local_realloc:
                self.node_to_arr_map[node] = ndarray.empty(
                    shape, ctx=node.ctx)

        if need_reallocation:
            self.init_need_allocation = False
            self.infer_shape(feed_shapes)
            self.memory_plan()

        from ..profiler import HetuProfiler
        if not hasattr(self, 'profiler'):
            self.profiler = HetuProfiler(
                self.computing_nodes, feed_shapes, self.node_to_arr_map, ctx=self.config.context)
        self.profiler.profile_n_log(log_file, profiler=profiler)

    def update_executor(self, eval_node_list):
        self.eval_node_list = eval_node_list
        inference = not any([isinstance(node, OptimizerOp)
                             for node in eval_node_list])
        self.inference = inference

        if self.config.p2p_stream and self.inference == True:
            raise NotImplementedError

        if inference == False:
            self.topo_order = find_topo_sort(self.eval_node_list)
        else:  # in inference phase
            if self.config.use_sparse_pull == True or self.config.cstable_policy is not None:
                # insert ps_sparse_pull_op
                self.topo_order = find_topo_sort_inference(self.eval_node_list)
                # fetch sparse parameter
                fetch_sparse_parameter_value(self.topo_order, self.config)
            else:
                self.topo_order = find_topo_sort(self.eval_node_list)

        self.topo_order = reorder_for_group(
            self.topo_order, self.config.layer_indices)

        # main structures, nodes' shapes and arrays
        self.node_to_shape_map = {}
        self.node_to_arr_map = {}

        # assisting structures, improve performance
        self.need_feed_nodes = []
        self.param_nodes = []
        self.dataloader_nodes = []
        self.computing_nodes = []

        # structures related to memory pool, used when dynamic_memory == True
        self.node_outdeg_map = {}
        self.node_ref_cnt = {}
        self.memory_pool = {}
        for node in self.topo_order:
            self.node_outdeg_map[node] = 0
            self.node_ref_cnt[node] = None

        for node in self.topo_order:
            if isinstance(node, DataloaderOp) or isinstance(node, GNNDataLoaderOp):
                self.dataloader_nodes.append(node)
            elif isinstance(node, PlaceholderOp):
                if node.shape is None:
                    self.need_feed_nodes.append(node)
                elif node.trainable:
                    self.param_nodes.append(node)
            elif not ((self.use_sparse_pull or self.cstable_policy) and isinstance(node, EmbeddingLookUp) and self.config.prefetch):
                self.computing_nodes.append(node)
            for n in node.inputs:
                self.node_outdeg_map[n] += 1
        self.batch_num = set([node.get_batch_num(self.name)
                              for node in self.dataloader_nodes])
        assert len(self.batch_num) <= 1, 'Batch num not conform.'
        self.batch_num = None if len(
            self.batch_num) == 0 else self.batch_num.pop()
        self.init_need_allocation = (self.need_feed_nodes == []) and (
            self.dataloader_nodes == [])

    def infer_shape(self, feed_shapes):
        """Given shapes of feed_dict nodes, infer shape for all nodes in graph.

        Implementation note:
        Iteratively calls node.infer_shape to infer shapes.
        Node shapes stored in self.node_to_shape_map.

        Parameters
        ----------
        feed_shapes: node->shapes mapping for feed_dict nodes.
        """

        def make_group():
            if len(grouping_nodes) > 1:
                GroupStart()
            temp_res = {node: ndarray.empty(
                (4,), ctx=node.ctx) for node in grouping_nodes}
            nccl_comm = self.config.nccl_comm
            p2p_stream = self.config.p2p_stream
            for node in grouping_nodes:
                if isinstance(node, PipelineSendOp):
                    shape = self.node_to_shape_map[node.inputs[0]]
                    if len(shape) < 4:
                        shape = [0] * (4 - len(shape)) + list(shape)
                    # construct and send
                    temp_res[node][:] = np.array(shape)
                    nccl_comm.dlarraySend(temp_res[node],
                                          ncclDataType_t.ncclFloat32,
                                          node.const_attr,
                                          p2p_stream)
                else:
                    nccl_comm.dlarrayRecv(temp_res[node],
                                          ncclDataType_t.ncclFloat32,
                                          node.const_attr,
                                          p2p_stream)
            if len(grouping_nodes) > 1:
                GroupEnd()
            p2p_stream.sync()
            for node in grouping_nodes:
                if isinstance(node, PipelineSendOp):
                    self.node_to_shape_map[node] = None
                else:
                    shape_arr = [int(x) for x in list(
                        temp_res[node].asnumpy()) if x != 0]
                    self.node_to_shape_map[node] = tuple(shape_arr)
            grouping_nodes.clear()

        self.node_to_shape_map = {}
        grouping_nodes = []
        cur_ind = -1
        for node in self.topo_order:
            if node in feed_shapes:
                self.node_to_shape_map[node] = tuple(feed_shapes[node])
            else:
                if isinstance(node, (PipelineSendOp, PipelineReceiveOp)):
                    if len(grouping_nodes) > 0 and self.config.layer_indices[node] != cur_ind:
                        make_group()
                    if len(grouping_nodes) == 0:
                        cur_ind = self.config.layer_indices[node]
                    grouping_nodes.append(node)
                    continue
                elif len(grouping_nodes) > 0:
                    make_group()
                input_shapes = [self.node_to_shape_map[n] for n in node.inputs]
                cur_shape = node.infer_shape(input_shapes)
                self.node_to_shape_map[node] = cur_shape if cur_shape is None else tuple(
                    cur_shape)
        if len(grouping_nodes) > 0:
            make_group()

    def from_memory_pool(self, key, node):
        return False
        if key in self.memory_pool:
            self.node_to_arr_map[node] = self.memory_pool[key].pop()
            if not len(self.memory_pool[key]):
                del self.memory_pool[key]
            return True
        else:
            return False

    def to_memory_pool(self, key, node):
        if isinstance(self.node_to_arr_map[node], ndarray.NDArray) and self.node_to_arr_map[node].no_free:
            self.node_to_arr_map[node] = None
            return
        if isinstance(self.node_to_arr_map[node], ndarray.NDArray):
            self.tensor_memory_allocated -= self.shape2size(key)
        del self.node_to_arr_map[node]
        return 
        if key not in self.memory_pool:
            self.memory_pool[key] = []
        self.memory_pool[key].append(self.node_to_arr_map[node])
        self.node_to_arr_map[node] = None

    def shape2size(self, shape):
        size = 4.0
        for s in shape:
            size *= s
        return size

    def memory_profile(self):
        size_MB = self.tensor_memory_allocated / 2**20
        print("Memory allocated: %.3fMB"%size_MB)
        return size_MB

    def node_memory_plan(self, node):
        """Allocates ndarray.NDArray for the specified node, used when dynamic_memory == True
        Parameters
        ----------
        """
        self.node_ref_cnt[node] = self.node_outdeg_map[node]
        shape = self.node_to_shape_map[node]
        if isinstance(node, PlaceholderOp):
            if self.config.placeholder_to_arr_map[node] is not None:
                self.node_to_arr_map[node] = self.config.placeholder_to_arr_map[node]
            elif node not in self.node_to_arr_map:
                self.node_to_arr_map[node] = None
        elif not isinstance(node, DataloaderOp) and not isinstance(node, GNNDataLoaderOp):
            # add for OptimizerOp and ParameterServerOp
            if shape is None:
                self.node_to_arr_map[node] = None
                return
            if isinstance(node, (EmbeddingLookUp_Gradient, DataD2HSparseOp)):
                if not self.from_memory_pool((shape, 'IndexedSlices'), node):
                    self.node_to_arr_map[node] = ndarray.IndexedSlices(
                        dense_shape=shape)
                return
            if isinstance(node, EmbeddingLookUp) and (self.use_sparse_pull or self.cstable_policy) and self.config.prefetch:
                self.node_to_arr_map[node] = self.param_psval_map[node.inputs[0]]
                return
            if isinstance(node, AllReduceCommunicateOp) and isinstance(node.inputs[0], EmbeddingLookUp_Gradient):
                self.node_to_arr_map[node] = ndarray.IndexedSlices(
                    dense_shape=shape)
                return
            if node.on_gpu:
                ln_bn_grad_nodes = ["Layer_Normalization_Gradient_of_DataOp", "Layer_Normalization_Gradient_of_ScaleOp",  
                                    "Layer_Normalization_Gradient_of_BiasOp", "Batch_Normalization_Gradient_of_DataOp",
                                    "Batch_Normalization_Gradient_of_ScaleOp", "Batch_Normalization_Gradient_of_BiasOp"]
                if node.inplace or node.op_type in ln_bn_grad_nodes:
                    self.node_to_arr_map[node] = ndarray.NDArray(None)
                elif self.inference and isinstance(node, DropoutOp):
                    self.node_to_arr_map[node] = self.node_to_arr_map[node.inputs[0]]
                else:
                    if not self.from_memory_pool(shape, node):
                        self.node_to_arr_map[node] = ndarray.empty(
                            shape, ctx=node.ctx)
                        self.tensor_memory_allocated += self.shape2size(shape)
            else:
                if not self.from_memory_pool(shape, node):
                    self.node_to_arr_map[node] = ndarray.empty(
                        shape, ctx=node.ctx)

    def memory_plan(self):
        """Allocates ndarray.NDArray for every node except feed_dict nodes.
        Parameters
        ----------
        """
        for node, shape in self.node_to_shape_map.items():
            if isinstance(node, PlaceholderOp):
                if self.config.placeholder_to_arr_map[node] is not None:
                    self.node_to_arr_map[node] = self.config.placeholder_to_arr_map[node]
                elif node not in self.node_to_arr_map:
                    self.node_to_arr_map[node] = None
            elif not isinstance(node, DataloaderOp) and not isinstance(node, GNNDataLoaderOp):
                # add for OptimizerOp and ParameterServerOp
                if shape is None:
                    self.node_to_arr_map[node] = None
                    continue
                if isinstance(node, (EmbeddingLookUp_Gradient, DataD2HSparseOp)):
                    self.node_to_arr_map[node] = ndarray.IndexedSlices(
                        dense_shape=shape)
                    continue
                if isinstance(node, EmbeddingLookUp) and (self.use_sparse_pull or self.cstable_policy) and self.config.prefetch:
                    self.node_to_arr_map[node] = self.param_psval_map[node.inputs[0]]
                    continue
                if isinstance(node, AllReduceCommunicateOp) and isinstance(node.inputs[0], EmbeddingLookUp_Gradient):
                    self.node_to_arr_map[node] = ndarray.IndexedSlices(
                        dense_shape=shape)
                    continue
                if node.on_gpu:
                    ln_bn_grad_nodes = ["Layer_Normalization_Gradient_of_DataOp", "Layer_Normalization_Gradient_of_ScaleOp",  
                                        "Layer_Normalization_Gradient_of_BiasOp", "Batch_Normalization_Gradient_of_DataOp",
                                        "Batch_Normalization_Gradient_of_ScaleOp", "Batch_Normalization_Gradient_of_BiasOp"]
                    if node.inplace or node.op_type in ln_bn_grad_nodes:
                        self.node_to_arr_map[node] = ndarray.NDArray(None)
                    elif self.inference and isinstance(node, DropoutOp):
                        self.node_to_arr_map[node] = self.node_to_arr_map[node.inputs[0]]
                    else:
                        self.node_to_arr_map[node] = ndarray.empty(
                            shape, ctx=node.ctx)
                else:
                    self.node_to_arr_map[node] = ndarray.empty(
                        shape, ctx=node.ctx)

    def run(self, eval_node_list={}, feed_dict={}, convert_to_numpy_ret_vals=False):
        """
        Parameters
        ----------
        feed_dict: a dictionary of node->np.ndarray supplied by user.
        convert_to_numpy_ret_vals: whether to convert ret vals to np.array

        Returns
        -------
        A list of values for nodes in eval_node_list. NDArray or np.ndarray.
        """
        assert len(feed_dict) == len(
            self.need_feed_nodes) or self.use_p2p, 'Feed dict invalid.'
        if eval_node_list != {} and eval_node_list != self.eval_node_list:
            self.update_executor(eval_node_list)

        feed_shapes = {}
        need_reallocation = self.init_need_allocation

        # get feed in values
        for node, value in feed_dict.items():
            if self.use_p2p and node not in self.need_feed_nodes:
                continue
            assert node in self.need_feed_nodes, 'Only allow feed in PlaceholderOp with no values, here got %s:%s.' % (
                str(type(node)), node.name)
            if node.reshaped:
                value = node.reshape_tensor(value)
            local_shape = tuple(value.shape)
            local_realloc = local_shape != self.node_to_shape_map.get(
                node, None)
            need_reallocation = need_reallocation or local_realloc
            if node.on_cpu:
                assert isinstance(value, (np.ndarray, spmatrix, ndarray.NDArray)), \
                    "feed_dict value type not supported"
                if isinstance(value, np.ndarray):
                    if local_realloc:
                        self.node_to_arr_map[node] = ndarray.empty(
                            local_shape, ctx=node.ctx)
                    self.node_to_arr_map[node][:] = value
                else:
                    self.node_to_arr_map[node] = value
            else:
                if isinstance(value, np.ndarray):
                    if local_realloc:
                        self.node_to_arr_map[node] = ndarray.array(
                            value, ctx=node.ctx)
                    else:
                        self.node_to_arr_map[node][:] = value
                elif isinstance(value, spmatrix):
                    value = coo_matrix(value)
                    value = ndarray.sparse_array(value.data,
                                                 (value.row, value.col), shape=local_shape, ctx=node.ctx)
                    self.node_to_arr_map[node] = value
                elif isinstance(value, ndarray.NDArray):
                    if value.ctx == node.ctx:
                        self.node_to_arr_map[node] = value
                    else:
                        if local_realloc:
                            self.node_to_arr_map[node] = ndarray.empty(
                                local_shape, ctx=node.ctx)
                        else:
                            self.node_to_arr_map[node][:] = value
                elif isinstance(value, ndarray.ND_Sparse_Array):
                    self.node_to_arr_map[node] = value
                else:
                    assert False, "feed_dict value type not supported"
            feed_shapes[node] = local_shape

        # get dataloader values
        for node in self.dataloader_nodes:
            local_shape = node.get_cur_shape(self.name)
            local_realloc = local_shape != self.node_to_shape_map.get(
                node, None)
            need_reallocation = need_reallocation or local_realloc
            self.node_to_arr_map[node] = node.get_arr(self.name)
            feed_shapes[node] = local_shape

        # in pipedream, we should retrieve the latest model parameter.
        if self.config.pipeline == "pipedream":
            self.node_to_arr_map.update(self.config.placeholder_to_arr_map)

        # reallocation, infer shapes and allocate memory
        if need_reallocation:
            self.init_need_allocation = False
            self.infer_shape(feed_shapes)
            if not self.dynamic_memory:
                self.memory_plan()

        # functions to free nodes when dynamic_memory == True
        def free_node(node):
            self.node_ref_cnt[node] = None
            key = self.node_to_shape_map[node]
            if key is not None:
                if isinstance(node, (EmbeddingLookUp_Gradient, DataD2HSparseOp)):
                    key = (key, 'IndexedSlices')
                self.to_memory_pool(key, node)
            else:
                del self.node_to_arr_map[node]

            if node.inplace: # if inplace node is freed, need to take care of its inputs node
                for n in node.inputs:
                    if n in self.computing_nodes and n not in self.eval_node_list \
                        and not (isinstance(n, AllReduceCommunicateOp) and isinstance(n.inputs[0], EmbeddingLookUp_Gradient)):
                        self.node_ref_cnt[n] -= 1
                        if self.node_ref_cnt[n] <= 0:
                            free_node(n)

        def end_node(node): # finish executing a node, should deal with its inputs
            if self.dynamic_memory and not node.inplace:
                for n in node.inputs:
                    if n in self.computing_nodes and n not in self.eval_node_list \
                        and not (isinstance(n, AllReduceCommunicateOp) and isinstance(n.inputs[0], EmbeddingLookUp_Gradient)):
                        self.node_ref_cnt[n] -= 1
                        if self.node_ref_cnt[n] <= 0:
                            free_node(n)

        # computing
        grouping_nodes = []
        cur_ind = -1

        def make_group():
            if len(grouping_nodes) > 1:
                GroupStart()
            nccl_comm = self.config.nccl_comm
            p2p_stream = self.config.p2p_stream
            for node in grouping_nodes:
                if isinstance(node, PipelineSendOp):
                    nccl_comm.dlarraySend(self.node_to_arr_map[node.inputs[0]],
                                          ncclDataType_t.ncclFloat32,
                                          node.const_attr,
                                          p2p_stream)
                else:
                    nccl_comm.dlarrayRecv(self.node_to_arr_map[node],
                                          ncclDataType_t.ncclFloat32,
                                          node.const_attr,
                                          p2p_stream)
            if len(grouping_nodes) > 1:
                GroupEnd()
            for node in grouping_nodes:
                node.event.record(p2p_stream)
            # Free nodes when dynamic_memory == True
            for n in grouping_nodes:
                end_node(n)
            grouping_nodes.clear()

        if not self.change_topo:
            self.change_topo = True
            index = -1
            out_idx = 999999
            for idx, node in enumerate(self.computing_nodes):
                shape = self.node_to_shape_map[node]
                if shape is not None:
                    size = self.shape2size(shape)/2**20
                    # print(node.name, size)
                    if size>200:
                        if len(node.inputs)>0 and len(node.inputs[0].inputs)>0 and "SoftmaxCrossEntropySparseGradientOp" in node.inputs[0].inputs[0].name:
                            index = idx
                            for i, n in enumerate(self.computing_nodes):
                                if node in n.inputs:
                                    out_idx=min(out_idx, i)
                                    #print(n.name, n.inputs, node)
                            #print(node.name, idx, out_idx)
            if index>0:
                n = self.computing_nodes.pop(index)
                self.computing_nodes.insert(out_idx-1, n)


        
        for node in self.computing_nodes:
            # print(node.name, [n.name for n in node.inputs], [self.node_ref_cnt[n] for n in node.inputs], end=" ")
            
            if self.dynamic_memory:
                # allocate memory for the node when dynamic_memory == True
                if self.node_ref_cnt[node] is None or need_reallocation:
                    self.node_memory_plan(node)
                for n in node.inputs:
                    if n not in self.node_to_arr_map:
                        self.node_memory_plan(n)

            # self.memory_profile()

            if node.on_cpu and isinstance(self.node_to_arr_map[node], ndarray.NDArray):
                if DNNL_LIB['cpu_ArraySet'] and not isinstance(node, DataD2HOp):
                    cpu_array_set(self.node_to_arr_map[node], 0.0)
                else:
                    # here we suppose not using DNNL_LIB
                    # self.node_to_arr_map[node][:] = np.zeros(self.node_to_shape_map[node]).astype(np.float32)
                    pass
            if isinstance(node, (PipelineSendOp, PipelineReceiveOp)):
                for n in node.inputs:
                    if n.event:
                        n.event.sync()
                if len(grouping_nodes) > 0 and self.config.layer_indices[node] != cur_ind:
                    make_group()
                if len(grouping_nodes) == 0:
                    cur_ind = self.config.layer_indices[node]
                grouping_nodes.append(node)
                continue
            else:
                if len(grouping_nodes) > 0:
                    make_group()

                for n in node.inputs:
                    if n.event:
                        n.event.sync()
                input_vals = [self.node_to_arr_map[n] for n in node.inputs]
                node_val = self.node_to_arr_map[node]

                if isinstance(node, (ParameterServerCommunicateOp, ParameterServerSparsePullOp)):
                    # Here we use d2h stream in ps op, since the stream is used for d2h data transfer.
                    # Please take care at this part.
                    node.compute(input_vals, node_val, self.d2h_stream)

                elif isinstance(node, AllReduceCommunicateOp):
                    node.compute(input_vals, node_val, self.nccl_stream)

                elif isinstance(node, DataH2DOp):
                    node.compute(input_vals, node_val, self.h2d_stream)

                elif isinstance(node, (DataD2HOp, DataD2HSparseOp)):
                    node.compute(input_vals, node_val, self.d2h_stream)

                elif isinstance(node, (DropoutOp, Batch_NormalizationOp)):
                    node.compute(input_vals, node_val,
                                 self.comp_stream, inference=self.inference)
                    if isinstance(node.event, Event):
                        # for d2h op / eval nodes / nodes before [allreduce or ps nodes or pipelinesend nodes]
                        node.event.record(self.comp_stream)

                else:
                    node.compute(input_vals, node_val, self.comp_stream)
                    if isinstance(node.event, Event):
                        # for d2h op / eval nodes / nodes before [allreduce or ps nodes or pipelinesend nodes]
                        node.event.record(self.comp_stream)

            # Free nodes when dynamic_memory == True
            end_node(node)

        if len(grouping_nodes) > 0:
            make_group()

        for n in self.eval_node_list:
            # every node in eval_node_list should have an event (except dataloader/optimizer...)
            if n.event:
                n.event.sync()

        # get results
        results = [self.node_to_arr_map[n] for n in self.eval_node_list]
        if convert_to_numpy_ret_vals:
            for i in range(len(results)):
                if results[i] is not None:
                    results[i] = results[i].asnumpy()

        # remap to original order in model parallel
        if self.config.pipeline:
            results = filter(lambda x: x[0] in self.global_eval_nodes,
                             zip(self.eval_node_list, results))
            results = [x[1] for x in results]
        elif self.use_p2p:
            new_results = [None for _ in self.global_eval_nodes]
            for i, j in enumerate(self.run_results_indices):
                new_results[j] = results[i]
            results = new_results

        return results


def gradients(output_node, node_list, insert_grad=None, return_all=False):
    """Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.
    insert_grad: used to assign gradient to output_node in model parallel.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.

    """
    from .ReduceMean import ReduceMeanOp
    from .BatchNorm import Batch_NormalizationOp
    from .LayerNorm import Layer_NormalizationOp
    from .AddElewise import AddOp
    from .AddConst import AddByConstOp
    from .Sum import SumOp
    # TODO: add support for Csrmm, Division, MatrixDot, Sigmoid, Sqrt, Tanh, Where
    backward2forward = {}  # key: backward node; value: tuple of forward nodes
    forward2backward = {}  # key: forward node; value: list of generated backward nodes
    if not isinstance(output_node, list):
        output_node = [output_node]
    if insert_grad is None:
        insert_grad = [OnesLike.oneslike_op(
            outnode) for outnode in output_node]
    elif not isinstance(insert_grad, list):
        insert_grad = [insert_grad]
    forward2backward[None] = insert_grad
    node_to_output_grads_list = {node: [grad]
                                 for node, grad in zip(output_node, insert_grad)}
    node_to_output_grad = {}
    # Traverse forward graph in reverse topological order
    reverse_topo_order = reversed(find_topo_sort(output_node))
    for node in reverse_topo_order:
        # here the ctx for embedding lookup is a workaround
        # TODO: when implement PS strategy for context semantics, modify here
        if isinstance(node, EmbeddingLookUp):
            output_grad = sum_node_list(
                node_to_output_grads_list[node], node_to_output_grads_list[node][0].raw_ctx)
        else:
            output_grad = sum_node_list(
                node_to_output_grads_list[node], node.raw_ctx)
        if output_grad is None:
            for n in node.inputs:
                if n not in node_to_output_grads_list:
                    node_to_output_grads_list[n] = []
            continue
        node_to_output_grad[node] = output_grad
        input_grads_list = node.gradient(output_grad)
        if input_grads_list is not None:
            # TODO: not consider following nodes in forward2backward, can be improved
            # DistGCN_15d, Division, MatrixDot, Sigmoid, Sqrt, Tanh, Where
            if isinstance(node, (AddOp, AddByConstOp, SumOp)):
                # these nodes don't generate new nodes
                forward2backward[node] = []
            else:
                forward2backward[node] = [
                    n for n in input_grads_list if n is not None]
            if isinstance(node, (ReduceMeanOp, Batch_NormalizationOp, Layer_NormalizationOp)):
                forward2backward[node].append(input_grads_list[0].inputs[0])
            if len(node_to_output_grads_list[node]) > 1:
                # sum op
                forward2backward[node].append(output_grad)
        for i in range(len(node.inputs)):
            if node.inputs[i] not in node_to_output_grads_list:
                node_to_output_grads_list[node.inputs[i]] = []
            # Calculate partial adjoint for input nodes.
            node_to_output_grads_list[node.inputs[i]].append(
                input_grads_list[i])
            need_target = True
            cur_key = input_grads_list[i]
            if cur_key not in backward2forward:
                need_target = False
                backward2forward[cur_key] = (node, [])
            if not isinstance(node.inputs[i], (AddOp, AddByConstOp, SumOp)):
                backward2forward[cur_key][1].append(
                    (node.inputs[i], need_target))

    grad_node_list = [node_to_output_grad[node] for node in node_list]
    if return_all:
        return grad_node_list, backward2forward, forward2backward
    else:
        return grad_node_list

##################
# Helper Methods #
##################


# this function synchronously initialize meta information and do the initialization on ps,
# Will modify PS worker state, PS server parmeter initialization
# Won't modify config, computation graph and executor state
def topo_sort_register_ps(node_list, ps_comm, comm_mode, seed, cstable_policy):
    visited = set()
    for node in node_list:
        if isinstance(node, OptimizerOp):
            opt = node.optimizer.get_config()

    def _topo_sort_register_ps(node):
        if node in visited:
            return
        visited.add(node)
        if isinstance(node, PlaceholderOp) and node.trainable and (comm_mode == "PS" or node.is_embed):
            node_type = int(node.is_embed)
            if node_type and cstable_policy is not None:
                node_type = 2
            node.initializer.init_on_ps(
                ps_comm, node.id, node_type, seed=seed + node.id, opt=opt)
        for n in node.inputs:
            _topo_sort_register_ps(n)

    for node in node_list:
        _topo_sort_register_ps(node)


def get_pipeline_stage_info(node_list, ctx):
    stage_index = {}

    def _is_pipeline_articulation(n0, n1):
        w0 = n0.raw_ctx._workers
        w1 = n1.raw_ctx._workers
        return len(w0) and len(w1) and w0 != w1

    def _traverse(node):
        if node not in stage_index:
            stage_index[node] = 0
        for n in node.inputs:
            _traverse(n)
            if _is_pipeline_articulation(node, n):
                stage_index[node] = max(stage_index[n] + 1, stage_index[node])
            else:
                stage_index[node] = max(stage_index[n], stage_index[node])

    for node in node_list:
        _traverse(node)

    # for a n stage pipeline, we have 2 * ( n - 1) pipeline articulation and 1 optimizer,
    # thus max(stage_index.values()) = 2n -1
    total_stage = (max(stage_index.values()) + 1) // 2
    total_stage = max(1, total_stage)  # handle corner case
    # find out my stage index, which is the biggest stage number in the forward pass
    my_stage = set()
    # dp rank (data parallel rank) is used to let dataloader know which part of data it should load
    dp_rank = set()
    for node, stage in stage_index.items():
        if ctx in node.raw_ctx._workers and stage < total_stage:
            my_stage.add(stage)
            dp_rank.add(node.raw_ctx._workers.index(ctx))
    my_stage = max(my_stage)
    assert my_stage >= 0
    assert len(dp_rank) == 1
    return my_stage, total_stage, dp_rank.pop()


def topo_sort_with_hook(node_list, config):
    visited = set()
    for node in node_list:
        topo_sort_dfs_with_hook(node, visited, config)


def topo_sort_dfs_with_hook(node, visited, config):
    if node in visited:
        return
    visited.add(node)
    node.backward_hook(config)
    # move param from node to config
    if isinstance(node, PlaceholderOp):
        config.placeholder_to_arr_map[node] = node.tensor_value
        node.tensor_value = None
    for n in node.inputs:
        topo_sort_dfs_with_hook(n, visited, config)
    node.forward_hook(config)


def find_topo_sort(node_list):
    """Given a list of nodes, return a topo ordering of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a
    topological sort.

    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)


def find_topo_sort_inference(node_list):
    topo_order = find_topo_sort(node_list)
    embedding_list = list()
    embedding_outputs = dict()
    embedding_cnt = dict()
    for node in topo_order:
        if isinstance(node, EmbeddingLookUp):
            embedding_outputs[node] = list()
            embedding_cnt[node] = 0
            embedding_list.append(node)
        else:
            for input_node in node.inputs:
                if isinstance(input_node, EmbeddingLookUp):
                    embedding_outputs[input_node].append(node)
                    embedding_cnt[input_node] += 1
    topo_order_inference = list()
    for node in topo_order:
        topo_order_inference.append(node)
        for embedding in embedding_list:
            if node in embedding_outputs[embedding]:
                embedding_cnt[embedding] -= 1
            if embedding_cnt[embedding] == 0:
                topo_order_inference.append(parameterServerSparsePull_op(
                    embedding, embedding_outputs[embedding]))
                embedding_list.remove(embedding)

    return topo_order_inference


def fetch_sparse_parameter_value(node_list, config):
    for node in node_list:
        if isinstance(node, ParameterServerSparsePullOp):
            node.forward_hook(config)


def fetch_dense_parameter_value(node_list, config):
    assert config.comm_mode in ('PS', 'Hybrid')
    topo_order = find_topo_sort(node_list)
    val_list = []
    # get var list
    for node in topo_order:
        if isinstance(node, PlaceholderOp) and node.trainable:
            val_list.append(node)
    for node in val_list:
        if config.use_sparse_pull and node.is_embed:
            continue
        else:
            pull_val = ndarray.empty(node.shape, ctx=ndarray.cpu(0))
            config.ps_comm.Pull(node.id, pull_val.handle)
            config.infer_ps_map[node] = pull_val
            config.placeholder_to_arr_map[node] = pull_val
        node.event.update()


def sum_node_list(node_list, ctx):
    """Custom sum func to avoid creating redundant nodes in Python sum func."""
    node_list = [n for n in node_list if n is not None]
    if node_list == []:
        return None
    elif len(node_list) == 1:
        return node_list[0]
    else:
        return sum_op(node_list, ctx=ctx)


def reorder_for_group(topo_order, layer_indices):
    if layer_indices is None:
        return topo_order
    # here we reorder for 2 reasons:
    # 1. group consecutive pipeline send/recv ops
    # 2. reorder pipeline send/recv ops according to grouping indices
    has_pipeline_ops = set([layer_indices[x] for x in layer_indices if isinstance(
        x, (PipelineSendOp, PipelineReceiveOp))])
    labels = {}
    for node in topo_order:
        if isinstance(node, (DataH2DOp, DataD2HOp, DataD2HSparseOp)):
            layer_indices[node] = layer_indices[node.inputs[0]] + 0.5
        cur_with_pipeline = layer_indices[node] in has_pipeline_ops
        if cur_with_pipeline and isinstance(node, SplitOp):
            labels[node] = 1
        elif isinstance(node, (PipelineSendOp, PipelineReceiveOp)):
            labels[node] = 2
        elif cur_with_pipeline and isinstance(node, (SumOp, ConcatenateOp)):
            labels[node] = 3
        else:
            labels[node] = 0

    topo_order = sorted(topo_order, key=lambda x: 10 *
                        layer_indices[x] + labels[x])
    return topo_order
