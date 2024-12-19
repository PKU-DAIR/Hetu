import signal
import math
import time
import os
import ast
import numpy as np
import hetu as ht
from queue import Queue
from typing import List, Dict, Any
from .utils import *
from .wrapper import *
from .straggler import *
from .strategy import StrategyModel
from .parallel_config import parse_multi_ds_parallel_config, config2ds, get_dg_from_union
from .data_utils.gpt import get_mask_and_position_ids, build_pretraining_data_loader
from .data_utils.llama import build_dist_data_loader, build_normal_data_loader
from .data_utils.bucket import sort_and_pack_for_global_batch

SLIDING_WINDOW = 2
PRE_PROFILING_ROUND = 5
WORKLOAD_BIAS_THRESHOLD = 0.1
LOG_FILE_PATH = "./elastic_log/straggler"
start_time = 0
end_time = 0
        
class TrainerCrucialOps(Args):
    def __init__(self, **kwargs):
        self.input_ids: ht.Tensor = kwargs["input_ids"]
        self.position_ids: ht.Tensor = kwargs["position_ids"]
        self.token_type_ids: ht.Tensor = kwargs["token_type_ids"]
        self.masked_lm_labels: ht.Tensor = kwargs["masked_lm_labels"]
        self.loss_op: ht.Tensor = kwargs["loss_op"]
        self.train_op: ht.Tensor = kwargs["train_op"]

class Trainer:
    def __init__(
        self, 
        model_wrapper: ModelWrapper, 
        optimizer_wrapper: OptimizerWrapper, 
        dataset_wrapper: DatasetWrapper
    ):
        self.model_wrapper = model_wrapper
        self.optimizer_wrapper = optimizer_wrapper
        self.dataset_wrapper = dataset_wrapper

        self.dataset = None
        self.data_iter = None
        self.is_built = False

    def build_data_iter(self, args):
        normal_dataloader = build_normal_data_loader(self.dataset, 0, args.global_batch_size)
        self.data_iter = iter(normal_dataloader)

    def create_dataset(self, args):
        self.dataset = self.dataset_wrapper.create_dataset(
            json_file=args.json_file,
            key=args.json_key,
            max_seq_len=args.seq_len,
            vocab_file=args.vocab_file,
            merge_file=args.merge_file)

    def build(self, args, ds_parallel_configs):
        assert self.is_built == False, "should only build once"
        # 1. Build Dataset & Data Iterator
        self.create_dataset(args)
        self.build_data_iter(args)
        # 2. Build Computation Graph
        with ht.graph("define_and_run", num_strategy=len(ds_parallel_configs)):
            if args.bf16:
                precision = "ht.bfloat16"
            else:
                precision = "ht.float32"
            with ht.autocast(eval(precision)):
                # print(f'{local_device}: use precision {precision}')
                self.create_define_graph(args, ds_parallel_configs)
        self.is_built = True

    def create_define_graph(self, args, ds_parallel_configs):
        pass

    def train(self, args):
        pass

    def run_plan(self, args):
        pass

class HotSPaTrainer(Trainer):
    def __init__(
        self, 
        model_wrapper: ModelWrapper, 
        optimizer_wrapper: OptimizerWrapper, 
        dataset_wrapper: DatasetWrapper
    ):
        super().__init__(model_wrapper, optimizer_wrapper, dataset_wrapper)

    def create_define_graph(self, args, ds_parallel_configs):
        # 1. Config Information
        input_ds_hierarchy, input_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'input')
        label_ds_hierarchy, label_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'label')
        dp_size = input_ds_hierarchy[0].get(0).get_dim(0)
        dummy_size = dp_size * args.seq_len # just a dummy shape, will change depend on real data shape

        # 2. Build Placeholders
        input_ids = ht.parallel_placeholder(ht.int64, global_shape=[dummy_size], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='input_ids')
        masked_lm_labels = ht.parallel_placeholder(ht.int64, global_shape=[dummy_size], ds_hierarchy=label_ds_hierarchy, device_group_hierarchy=label_dg_hierarchy, name='masked_lm_labels')
        position_ids = ht.parallel_placeholder(ht.int64, global_shape=[dummy_size], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='position_ids')
        token_type_ids = ht.parallel_placeholder(ht.int64, global_shape=[dummy_size], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='token_type_ids')
        
        # 3. Build Model Weight
        model = self.model_wrapper.create_model(ds_parallel_configs)

        # 4. Build Symbolic Shape
        config = self.model_wrapper.model_config
        config.cu_seqlens_list = []
        for block_id, block in enumerate(model.transformer.h):
            config.cu_seqlens_list.append(
                ht.parallel_placeholder(
                    ht.int32, 
                    global_shape=[dummy_size], 
                    ds_hierarchy=block.attn.qkv_dense.ds_union_map['split0_dup'], 
                    device_group_hierarchy=block.attn.qkv_dense.device_group_unions,
                    name=f'cu_seqlens_{block_id}'
                )
            )

        # just symbolic value, will change depend on real data
        config.multi_seq_lens_symbol = []
        config.multi_cp_group_symbol = []
        for i in range(len(input_ds_hierarchy)):
            cur_dp = input_ds_hierarchy[i].get(0).get_dim(0) # dp_i for strategy_i
            config.multi_seq_lens_symbol.append([input_ids.symbolic_shape[0] for _ in range(cur_dp)])
            config.multi_cp_group_symbol.append([ht.IntSymbol(i) for i in range(cur_dp)])
        config.max_seqlen_symbol = ht.IntSymbol(1)

        # 5. Build Forward Graph
        loss_op = model(input_ids=input_ids,
                        labels=masked_lm_labels)

        # 6. Build Backward Graph
        opt = self.optimizer_wrapper.create_optimizer(lr=args.lr)
        train_op = opt.minimize(loss_op)

        # 7. Record Ops
        if self.is_built == False:
            self.build_ops = TrainerCrucialOps(
                input_ids=input_ids,
                masked_lm_labels=masked_lm_labels,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                loss_op=loss_op,
                train_op=train_op
            )

    def train(
        self, 
        args, 
        ds_parallel_configs,
        local_device: ht.device,
        all_devices: ht.DeviceGroup,
        ):
        assert self.is_built == True and self.dataset != None and self.data_iter != None, \
            "must build graph and dataset before HotSPa training!"
        input_ds_hierarchy, input_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'input')
        label_ds_hierarchy, label_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'label')
        comm_args = TrainerCommAllArgs(
            input_ds_hierarchy=input_ds_hierarchy,
            input_device_group_hierarchy=input_dg_hierarchy,
            label_ds_hierarchy=label_ds_hierarchy,
            label_device_group_hierarchy=label_dg_hierarchy,
            local_device=local_device,
            all_devices=all_devices
        )
        if args.bf16:
            precision = "ht.bfloat16"
        else:
            precision = "ht.float32"

        self.feed_dicts = Queue()
        def cache_feed_dict(feed_dict):
            if self.feed_dicts.qsize() == 50:
                self.feed_dicts.get()
            self.feed_dicts.put(feed_dict)
        self.cache_feed_dict = cache_feed_dict

        print(f'{local_device}: the first step may take a few minutes to [generate topo & execution plan], please be patient...')
        for epoch in range(args.epochs):
            consumed_samples = 0 # should be reset when run next epoch
            for step in range(args.steps):
                # group and packing
                global_batch = next(self.data_iter).numpy()
                buckets = sort_and_pack_for_global_batch(global_batch, self.dataset.pad_id(), args.bucket_sizes, False)
                # enable parallelism hot switching
                for strategy_id in range(args.num_strategy + 1):
                    alloc_run_level = ht.run_level("alloc")
                    grad_run_level = ht.run_level("grad")
                    update_run_level = ht.run_level("update")
                    if strategy_id == 0:
                        cur_run_level = alloc_run_level
                        bucket = buckets[0]
                    elif strategy_id == len(ds_parallel_configs): # init graph
                        strategy_id = 0
                        cur_run_level = update_run_level
                        bucket = buckets[0]
                    else:
                        cur_run_level = grad_run_level
                        bucket = buckets[args.num_strategy - strategy_id]
                    # run sperate strategy for bucket
                    with ht.autocast(eval(precision)):
                        consumed_samples = self.run_plan(
                            args,
                            ds_parallel_configs,
                            comm_args = comm_args,
                            epoch = epoch,
                            step = step,
                            consumed_samples = consumed_samples, 
                            bucket = bucket, 
                            strategy_id = strategy_id, 
                            run_level = cur_run_level)

    def run_plan(
        self, 
        args,
        ds_parallel_configs,
        comm_args = None,
        epoch = 0,
        step = 0,
        consumed_samples = 0,
        bucket = None,
        strategy_id = 0, 
        run_level = 0):

        global start_time, end_time
        input_ds_hierarchy = comm_args.input_ds_hierarchy
        input_dg_hierarchy = comm_args.input_device_group_hierarchy
        label_ds_hierarchy = comm_args.label_ds_hierarchy
        label_dg_hierarchy = comm_args.label_device_group_hierarchy
        local_device = comm_args.local_device
        all_devices = comm_args.all_devices

        input_ds_union = input_ds_hierarchy[strategy_id]
        input_device_group_union = input_dg_hierarchy[strategy_id]
        label_ds_union = label_ds_hierarchy[strategy_id]
        label_device_group_union = label_dg_hierarchy[strategy_id]
        assert input_ds_union.hetero_dim == -3, "input hetero dim unsupported"
        assert label_ds_union.hetero_dim == -3, "label hetero dim unsupported"

        # device in same dp_group will read the same batch data, idx=-1 means no need to read data
        dup_group_idx, dup_group_num = -1, -1
        input_union_idx, input_device_group = get_dg_from_union(local_device, input_device_group_union)
        label_union_idx, label_device_group = get_dg_from_union(local_device, label_device_group_union)
        if input_device_group != None:
            local_device_idx = input_device_group.get_index(local_device)
            dup_group_idx = input_union_idx
            dup_group_num = len(input_device_group_union)
            if input_ds_union.hetero_dim == -3:
                dup_group_idx = input_ds_union.get(0).get_dup_group_index(local_device_idx)
                dup_group_num = input_ds_union.get(0).get_dim(0)
        elif label_device_group != None:
            local_device_idx = label_device_group.get_index(local_device)
            dup_group_idx = label_union_idx
            dup_group_num = len(label_device_group_union)
            if label_ds_union.hetero_dim == -3:
                dup_group_idx = label_ds_union.get(0).get_dup_group_index(local_device_idx)
                dup_group_num = label_ds_union.get(0).get_dim(0)
        else:
            dup_group_num = len(input_device_group_union)
            if input_ds_union.hetero_dim == -3:
                dup_group_num = input_ds_union.get(0).get_dim(0)
            # raise RuntimeError(f"device {local_device} not in input_device_group or label_device_group!")

        dp_rank = dup_group_idx
        dp_size = dup_group_num
        micro_batch_size = 1 # already packed
        input_bucket, label_bucket = bucket
        global_batch_size = input_bucket.packed_batch_size() // dp_size * dp_size
        if global_batch_size == 0: # empty input_bucket, just run a dummy batch with full padded
            input_bucket.make_dummy_batch()
            label_bucket.make_dummy_batch()
            global_batch_size = dp_size
        seq_len = input_bucket.max_seq_len()
        gbs_per_dp = global_batch_size // dp_size
        mbs_times_dp = micro_batch_size * dp_size
        assert global_batch_size % mbs_times_dp == 0, \
            f'gbs {global_batch_size} must could be divided by mbs {micro_batch_size} * dp {dp_size}'
        num_micro_batches = global_batch_size // mbs_times_dp

        # load data for each dp
        if dp_rank != -1:
            input_batch, label_batch = input_bucket.packed_batch(), label_bucket.packed_batch()
            cu_seqlens_list_batch = input_bucket.packed_cu_seqlens_list()
            input_list = []
            label_list = []
            cu_seqlens_list = []
            for begin_idx in range(dp_rank * micro_batch_size, global_batch_size, mbs_times_dp):
                assert begin_idx >= 0 and begin_idx+micro_batch_size <= len(input_batch) and begin_idx+micro_batch_size <= len(label_batch) and begin_idx+micro_batch_size <= len(cu_seqlens_list_batch), 'read micro batch error!!!'
                input_list += input_batch[begin_idx: begin_idx+micro_batch_size] # # batch_size * [seq_len], seq_len may different!
                label_list += label_batch[begin_idx: begin_idx+micro_batch_size]
                cu_seqlens_list += cu_seqlens_list_batch[begin_idx: begin_idx+micro_batch_size]
            input_list = [micro_batch.astype(np.int64) for micro_batch in input_list]
            label_list = [micro_batch.astype(np.int64) for micro_batch in label_list]
            num_micro_batches = len(input_list)
            max_seqlen_val = input_bucket.max_seq_len()
            config = self.model_wrapper.model_config
            config.max_seqlen_symbol.set_data(max_seqlen_val)
            feed_dict = {
                self.build_ops.input_ids: input_list,
                self.build_ops.masked_lm_labels: label_list,
            }
            for i in range(config.n_layer):
                feed_dict[config.cu_seqlens_list[i]] = [x.astype(np.int32) for x in cu_seqlens_list]
            self.cache_feed_dict(feed_dict)
        else: # fake data; feed_dict={} will cause segment fault?
            print("Need feed dict data!")
            os.killpg(0, signal.SIGTERM)
        if run_level == ht.run_level("alloc") or (len(ds_parallel_configs) == 1 and run_level == ht.run_level("update")):
            start_time = time.time()
        if args.run_memory_experiment and step == 0:
            os.environ['HETU_MEMORY_LOG_FILE'] = args.memory_file
        try:
            results = self.build_ops.train_op.graph.run(
                self.build_ops.loss_op, 
                [self.build_ops.loss_op, self.build_ops.train_op], 
                feed_dict = feed_dict, 
                num_micro_batches = num_micro_batches, 
                cur_strategy_id = strategy_id,
                run_level = run_level,
                grad_scale = 1.0)
        except RuntimeError as e:
            print(e)
            with open("./logs/exception.txt", 'w') as file:
                print(f"{local_device}:", file=file)
                print(e, file=file)
            os.killpg(0, signal.SIGTERM)
        if args.run_memory_experiment and step == 0:
            if 'HETU_MEMORY_LOG_FILE' in os.environ:
                del os.environ['HETU_MEMORY_LOG_FILE'] 
            return consumed_samples
        
        if run_level != ht.run_level("alloc"):
            consumed_samples += input_bucket.original_batch_size()

        if run_level == ht.run_level("update"):
            # NOTE: update new step end time
            end_time = time.time()
            if label_device_group != None:
                loss_out = results[0].numpy(force=True).mean()
                print(f"{local_device}: [Epoch {epoch}] (step {step}, consumed_samples = {consumed_samples}): loss = {loss_out:.3f}, time = {end_time - start_time:.4f}")

        return consumed_samples

class MalleusTrainer(Trainer):
    def __init__(
        self, 
        model_wrapper: ModelWrapper, 
        optimizer_wrapper: OptimizerWrapper, 
        dataset_wrapper: DatasetWrapper
    ):
        super().__init__(model_wrapper, optimizer_wrapper, dataset_wrapper)
        
        self.dataset = None
        self.is_built = False
        self.build_ctxs = None
        self.build_ops = None
        
        self.strategy_model = None
        self.strategy_models_pool = []
        self.ds_parallel_configs_pool = []
        
        self.local_straggler = None
        self.device_to_straggler_mapping = {}
    
    def train_data_iterator(self, dataset, consumed_samples, mbs, dp_rank, dp_size):
        train_dataloader = build_pretraining_data_loader(dataset, consumed_samples, mbs, dp_rank, dp_size)
        train_data_iterator = iter(train_dataloader)
        return train_data_iterator
    
    def create_dataset(self, args):
        '''Build training dataset.'''
        self.dataset = self.dataset_wrapper.create_dataset(
            json_file=args.json_file,
            key=args.json_key,
            max_seq_len=args.seq_len,
            vocab_file=args.vocab_file,
            merge_file=args.merge_file)
        return self.dataset    
    
    def build(self, args, ctxs, ds_parallel_configs):
        '''
        args:
            micro_batch_size
            seq_len
            lr
            json_file
            json_key
            vocab_file
            merge_file
        ctxs:
            bf16
            ...
        '''
        assert self.is_built == False, "should only build once"
        # Build Dataset
        self.create_dataset(args)
        # Build Computation Graph
        with ht.graph("define_and_run", num_strategy=len(ds_parallel_configs)):
            if ctxs.bf16:
                precision = "ht.bfloat16"
            else:
                precision = "ht.float32"
            with ht.autocast(eval(precision)):            
                self.create_define_graph(args, ds_parallel_configs)
        self.is_built = True
        self.build_ctxs: TrainerCtxs = ctxs
        # Cache initial strategies
        self.ds_parallel_configs_pool.extend(ds_parallel_configs)
        
    def add_strategy(self, args, ds_parallel_configs):
        '''
        args:
            micro_batch_size
            seq_len
            lr
        '''
        assert self.is_built == True, "must build graph before add strategy"
        print("Adding new strategies...")
        with ht.merge_strategy(target_graph="default", num_strategy=len(ds_parallel_configs)):
            if self.build_ctxs.bf16:
                precision = "ht.bfloat16"
            else:
                precision = "ht.float32"
            with ht.autocast(eval(precision)):            
                self.create_define_graph(args, ds_parallel_configs)
        
    def create_define_graph(self, args, ds_parallel_configs):  
        '''
        args:
            micro_batch_size
            seq_len
            lr
        '''
        # Config Information
        input_ds_hierarchy, input_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'input')
        label_ds_hierarchy, label_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'label')
        default_dp_size = input_ds_hierarchy[0].get(0).get_dim(0)
        default_mbs_times_dp = args.micro_batch_size * default_dp_size 
        default_seq_len = args.seq_len 
        
        # Build Placeholders 
        input_ids = ht.parallel_placeholder(ht.int64, global_shape=[default_mbs_times_dp, default_seq_len], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='input_ids')
        position_ids = ht.parallel_placeholder(ht.int64, global_shape=[default_mbs_times_dp, default_seq_len], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='position_ids')
        token_type_ids = ht.parallel_placeholder(ht.int64, global_shape=[default_mbs_times_dp, default_seq_len], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='token_type_ids')
        masked_lm_labels = ht.parallel_placeholder(ht.int64, global_shape=[default_mbs_times_dp, default_seq_len], ds_hierarchy=label_ds_hierarchy, device_group_hierarchy=label_dg_hierarchy, name='masked_lm_labels')
        
        # Build Symbolic Shape
        self.model_wrapper.model_config.mbs_times_dp_symbol.set_data(default_mbs_times_dp)
        self.model_wrapper.model_config.seq_len_symbol.set_data(default_seq_len)
        
        # Build Model Weight
        model = self.model_wrapper.create_model(ds_parallel_configs) 
        
        # Build Forward Graph
        loss_op = model(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            labels=masked_lm_labels
        )
        
        # Build Backward Graph
        opt = self.optimizer_wrapper.create_optimizer(lr=args.lr) 
        train_op = opt.minimize(loss_op)
        
        # Record Ops
        if self.is_built == False:
            self.build_ops = TrainerCrucialOps(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                masked_lm_labels=masked_lm_labels,
                loss_op=loss_op,
                train_op=train_op
            )
            
    def setup_stragglers(
        self, 
        args,
        local_device: ht.device, 
        all_devices: ht.DeviceGroup
    ):
        '''
        args:
            micro_batch_size
            seq_len
            hidden_size
        '''
        self.log_file_path = LOG_FILE_PATH
        suffix = "_" + str(all_devices.get_index(local_device)) + ".txt"
        self.log_file = self.log_file_path + suffix
        workload_info = WorkloadInfo(
            args.micro_batch_size,
            args.seq_len,
            args.hidden_size // args.tp
        )
        for device_idx in range(all_devices.num_devices):
            self.device_to_straggler_mapping[device_idx] = Straggler(
                all_devices.get(device_idx),
                self.log_file_path + "_" + str(device_idx) + ".txt",
                workload_info
            )
        self.local_straggler = self.device_to_straggler_mapping[all_devices.get_index(local_device)]
        self.local_straggler.begin_profile()
        print(f"{local_device}: pre-profiling straggler workload...")
        for _ in range(PRE_PROFILING_ROUND):
            self.local_straggler.run_profile()
        self.local_straggler.end_profile()
        ht.global_comm_barrier_rpc()
        all_workload_baseline_time = []
        for device_idx in range(all_devices.num_devices):
            straggler_info = Straggler.read_profile(self.log_file_path + "_" + str(device_idx) + ".txt", PRE_PROFILING_ROUND - 1)
            self.device_to_straggler_mapping[device_idx].workload_baseline_time = np.array(straggler_info).mean()
            all_workload_baseline_time.append(self.device_to_straggler_mapping[device_idx].workload_baseline_time)
        all_workload_baseline_time_mean = np.array(all_workload_baseline_time).mean()
        if abs(self.local_straggler.workload_baseline_time - all_workload_baseline_time_mean) / all_workload_baseline_time_mean >= WORKLOAD_BIAS_THRESHOLD:
            print(f"{local_device}: is heterogenous at the beginning")
        for device_idx, straggler in self.device_to_straggler_mapping.items():
            print(f"Device {device_idx} initial workload consuming time = {straggler.workload_baseline_time}")

    def detect_straggler_and_plan(
        self, 
        strategy_args: TrainerStrategyArgs, 
        comm_args: TrainerCommArgs
    ):
        used_devices_sr = {}
        suspended_devices_sr = {}
        unused_devices = []
        sliding_window = SLIDING_WINDOW
        
        ht.global_comm_barrier_rpc()
        for rank_idx in range(comm_args.all_devices.num_devices):
            device_idx = strategy_args.rank_to_device_mapping[rank_idx]
            curr_log_file = self.log_file_path + "_" + str(device_idx) + ".txt"
            if rank_idx in strategy_args.unused_rank_list:
                # 完全无法使用
                # TODO
                unused_devices.append(device_idx)
            else:
                straggler_info: List[float] = Straggler.read_profile(curr_log_file, sliding_window)
                straggler_compute_time = np.array(straggler_info).mean()
                if rank_idx in strategy_args.suspended_rank_list:
                    # 已经是straggler了
                    # 通过workload上的profile信息进行分析
                    suspended_devices_sr[device_idx] = (straggler_compute_time 
                        / self.device_to_straggler_mapping[device_idx].workload_baseline_time)
                else:
                    # 还不是straggler
                    # 通过compute stream上的profile信息进行分析
                    straggler_pipeline = 0
                    straggler_stage = 0
                    accumulate_ranks = 0
                    for i, stage_num in enumerate(strategy_args.hetero_stages):
                        if accumulate_ranks + stage_num * strategy_args.tp > rank_idx:
                            straggler_pipeline = i
                            straggler_stage = (rank_idx - accumulate_ranks) // strategy_args.tp
                            break
                        accumulate_ranks += stage_num * strategy_args.tp
                    # straggler_pipeline = rank_idx % (strategy_args.dp * strategy_args.tp) // strategy_args.tp
                    # straggler_stage = rank_idx // (strategy_args.dp * strategy_args.tp)
                    straggler_layers = strategy_args.hetero_layers[straggler_pipeline][straggler_stage]
                    straggler_mbn = strategy_args.hetero_micro_batch_num_list[straggler_pipeline]
                    curr_tp = 0
                    for curr_tp_rank_idx in range(rank_idx - rank_idx % strategy_args.tp, rank_idx - rank_idx % strategy_args.tp + strategy_args.tp):
                        if curr_tp_rank_idx not in strategy_args.suspended_rank_list:
                            curr_tp += 1
                    assert strategy_args.tp % curr_tp == 0, "hetero tp only support 1, 2, 4, 8"
                    alpha = self.build_ctxs.hetero_tp_alpha[int(math.log2(strategy_args.tp // curr_tp))]
                    straggler_ratio = ((straggler_compute_time / straggler_layers / straggler_mbn / alpha) 
                        / (self.build_ctxs.normal_compute_time / self.build_ctxs.normal_layers / self.build_ctxs.normal_mbn))
                    used_devices_sr[device_idx] = straggler_ratio          
        ht.global_comm_barrier_rpc()
        
        new_strategy_model = StrategyModel(
            self.build_ctxs,
            strategy_args, 
            used_devices_sr, 
            suspended_devices_sr, 
            unused_devices
        )
        
        need_switch = False
        is_cached = False
        if new_strategy_model != self.strategy_model:
            need_switch = True
        for strategy_model in reversed(self.strategy_models_pool):
            if strategy_model == new_strategy_model:
                self.strategy_model = strategy_model
                is_cached = True
                break
        if not is_cached:
            self.strategy_model = new_strategy_model
            self.strategy_models_pool.append(new_strategy_model)
        return need_switch
            
    def generate_new_strategies(self, args):
        strategies, ds_parallel_configs = self.strategy_model.make_plans()
        strategies_id = []
        new_ds_parallel_configs = []
        for i in range(len(strategies)):
            if ds_parallel_configs[i] in self.ds_parallel_configs_pool:
                strategies_id.append(self.ds_parallel_configs_pool.index(ds_parallel_configs[i]))
            else:
                strategies_id.append(len(self.ds_parallel_configs_pool))
                self.ds_parallel_configs_pool.append(ds_parallel_configs[i])
                new_ds_parallel_configs.append(ds_parallel_configs[i])
        if len(new_ds_parallel_configs) > 0:
            self.add_strategy(args, new_ds_parallel_configs)
        return strategies, strategies_id 
            
    def train(
            self, 
            args, 
            ds_parallel_configs, 
            local_device: ht.device,
            all_devices: ht.DeviceGroup,
            strategy_id: int = 0,
            elastic: bool = True
        ):
        assert self.is_built == True and self.dataset != None, \
            "must build graph and dataset before elastic train"
        initial_args = args
        initial_dataset_args = TrainerDatasetArgs(
            dataset=self.dataset,
            consumed_samples=0,
            steps=args.steps,
            epochs=args.epochs,
            step=0,
            epoch=0
        )
        initial_strategy_args = TrainerStrategyArgs(
            dp=args.dp,
            tp=args.tp,
            pp=args.pp,
            zero=args.zero,
            rank_to_device_mapping=args.rank_to_device_mapping,
            suspended_rank_list=args.suspended_rank_list,
            unused_rank_list=args.unused_rank_list,
            hetero_data=args.hetero_data,
            hetero_layers=args.hetero_layers,
            hetero_stages=args.hetero_stages,
            hetero_micro_batch_num_list=args.hetero_micro_batch_num_list
        )
        input_ds_hierarchy, input_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'input')
        label_ds_hierarchy, label_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'label')
        initial_comm_args = TrainerCommArgs(
            input_ds_union=input_ds_hierarchy[strategy_id],
            input_device_group_union=input_dg_hierarchy[strategy_id],
            label_ds_union=label_ds_hierarchy[strategy_id],
            label_device_group_union=label_dg_hierarchy[strategy_id],
            local_device=local_device,
            all_devices=all_devices
        )
        initial_envs = TrainerEnvs(
            run_straggler_experiment=args.run_straggler_experiment,
            run_memory_experiment=args.run_memory_experiment,
            straggler_file=args.straggler_file,
            memory_file=args.memory_file,
            elastic=elastic
        )
        if self.build_ctxs.bf16:
            precision = "ht.bfloat16"
        else:
            precision = "ht.float32"
        if elastic:
            self.setup_stragglers(args, local_device, all_devices)
        with ht.autocast(eval(precision)):
            is_complete, rest_of_dataset = self.run_plan(
                initial_args, 
                initial_dataset_args, 
                initial_strategy_args, 
                initial_comm_args, 
                initial_envs, 
                strategy_id=strategy_id, 
            )
        if not elastic:
            assert is_complete == True, "inelastic training wouldn't switch strategy"
            # TODO: save checkpoint
            return
        
        while not is_complete:
            strategies, strategies_id = self.generate_new_strategies(args)
            # TODO: find the best among all
            # Now directly use the best one of the ideal strategy model
            for i in range(1):
                strategy = strategies[i]
                strategy_id = strategies_id[i]
                input_ds_hierarchy, input_dg_hierarchy = parse_multi_ds_parallel_config([self.ds_parallel_configs_pool[strategy_id]], 'input')
                label_ds_hierarchy, label_dg_hierarchy = parse_multi_ds_parallel_config([self.ds_parallel_configs_pool[strategy_id]], 'label')
                comm_args = TrainerCommArgs(
                    input_ds_union=input_ds_hierarchy[0],
                    input_device_group_union=input_dg_hierarchy[0],
                    label_ds_union=label_ds_hierarchy[0],
                    label_device_group_union=label_dg_hierarchy[0],
                    local_device=local_device,
                    all_devices=all_devices
                )
                if self.build_ctxs.bf16:
                    precision = "ht.bfloat16"
                else:
                    precision = "ht.float32"
                with ht.autocast(eval(precision)):
                    is_complete, rest_of_dataset = self.run_plan(
                        initial_args, 
                        rest_of_dataset, 
                        strategy, 
                        comm_args, 
                        initial_envs, 
                        strategy_id=strategy_id, 
                    )
        
    def run_plan(
            self, 
            args, 
            dataset_args: TrainerDatasetArgs, 
            strategy_args: TrainerStrategyArgs, 
            comm_args: TrainerCommArgs,  
            envs: TrainerEnvs,
            strategy_id: int = 0, 
            run_level: int = ht.run_level("update")
        ):
        '''
        args:
            global_batch_size
            micro_batch_size
            seq_len
        dataset_args:
            dataset
            consumed_samples
            steps
            epochs
            step
            epoch
        strategy_args:
            dp
            tp
            pp
            rank_to_device_mapping 
            suspended_rank_list
            hetero_data
            hetero_micro_batch_num_list
        comm_args:
            input_ds_union
            input_device_group_union
            label_ds_union
            label_device_group_union
            local_device
            all_devices
        envs:
            run_straggler_experiment
            run_memory_experiment
            straggler_file
            memory_file
            elastic
        '''
        # Assertions
        assert self.is_built == True, "must build graph before run"
        assert comm_args.input_ds_union.hetero_dim == 0 or comm_args.input_ds_union.hetero_dim == -3, "input hetero dim unsupported"
        assert comm_args.label_ds_union.hetero_dim == 0 or comm_args.label_ds_union.hetero_dim == -3, "label hetero dim unsupported"
        if args.global_batch_size != self.model_wrapper.model_config.global_batch_size or args.seq_len != self.model_wrapper.model_config.seq_len:
            assert config.use_flash_attn == True, "symbolic shape can only used when flash attn is on for now"
            
        print(f"{comm_args.local_device}: start running strategy {strategy_id}")
        
        def get_dg_from_union(device, dg_union):
            for i, dg in enumerate(dg_union):
                if dg.contains(device):
                    return i, dg
            return None, None
        
        # Device in same dp_group will read the same batch data, idx = -1 means no need to read data.
        dup_group_idx, dup_group_num = -1, -1
        input_union_idx, input_device_group = get_dg_from_union(comm_args.local_device, comm_args.input_device_group_union)
        label_union_idx, label_device_group = get_dg_from_union(comm_args.local_device, comm_args.label_device_group_union)
        if input_device_group != None:
            local_device_idx = input_device_group.get_index(comm_args.local_device)
            dup_group_idx = input_union_idx
            dup_group_num = len(comm_args.input_device_group_union)
            if comm_args.input_ds_union.hetero_dim == -3:
                dup_group_idx = comm_args.input_ds_union.get(0).get_dup_group_index(local_device_idx)
                dup_group_num = comm_args.input_ds_union.get(0).get_dim(0)
        elif label_device_group != None:
            local_device_idx = label_device_group.get_index(comm_args.local_device)
            dup_group_idx = label_union_idx
            dup_group_num = len(comm_args.label_device_group_union)
            if comm_args.label_ds_union.hetero_dim == -3:
                dup_group_idx = comm_args.label_ds_union.get(0).get_dup_group_index(local_device_idx)
                dup_group_num = comm_args.label_ds_union.get(0).get_dim(0)
        else:
            # 其余local device都在中间stage上
            dup_group_num = len(comm_args.input_device_group_union)
            if comm_args.input_ds_union.hetero_dim == -3:
                dup_group_num = comm_args.input_ds_union.get(0).get_dim(0)
                
        # Homo
        dp_rank = dup_group_idx
        dp_size = dup_group_num
        assert dp_size == strategy_args.dp, f"dp conflict"
        seq_len = args.seq_len
        gbs_per_dp = args.global_batch_size // dp_size
        mbs_times_dp = args.micro_batch_size * dp_size
        assert args.global_batch_size % mbs_times_dp == 0, \
            f"gbs {args.global_batch_size} must be divided by mbs {args.micro_batch_size} * dp {dp_size}"
        num_micro_batches = args.global_batch_size // mbs_times_dp
        
        # Hetero TP
        is_suspended = False
        is_unused = False
        for suspended_rank in strategy_args.suspended_rank_list:
            if strategy_args.rank_to_device_mapping[suspended_rank] == comm_args.all_devices.get_index(comm_args.local_device):
                assert is_suspended == False, "multiple local device mapping"
                is_suspended = True
        for unused_rank in strategy_args.unused_rank_list:
            if strategy_args.rank_to_device_mapping[unused_rank] == comm_args.all_devices.get_index(comm_args.local_device):
                assert is_suspended == False, "can't be both suspended and unused"
                assert is_unused == False, "multiple local device mapping"
                is_unused = True      
        
        # Hetero Data
        if strategy_args.hetero_data:
            curr_rank_id = -1
            for rank_id, device_id in strategy_args.rank_to_device_mapping.items():
                if device_id == comm_args.all_devices.get_index(comm_args.local_device):
                    if curr_rank_id != -1:
                        assert False, "rank_to_device_mapping has duplicate keys"
                    curr_rank_id = rank_id
            assert curr_rank_id != -1, f"can't find device {comm_args.all_devices.get_index(comm_args.local_device)} in rank_to_device_mapping"
            # hetero_pipeline_num = curr_rank_id % (dp_size * strategy_args.tp) // strategy_args.tp
            # 找到所属的pipeline num
            accumulate_ranks = 0
            hetero_pipeline_num = -1
            for i, stage_num in enumerate(strategy_args.hetero_stages):
                accumulate_ranks += stage_num * strategy_args.tp
                if accumulate_ranks > curr_rank_id:
                    hetero_pipeline_num = i
                    break
            if hetero_pipeline_num == -1:
                # 说明是没有被用到的靠后的rank
                # 随便给一个pipeline编号即可
                assert is_unused or is_suspended, "can't figure out pipeline num"
                hetero_pipeline_num = 0
            num_micro_batches = strategy_args.hetero_micro_batch_num_list[hetero_pipeline_num]
            # re-assign
            gbs_per_dp = args.micro_batch_size * num_micro_batches
            mbs_times_dp = args.micro_batch_size * dp_size
                
        self.model_wrapper.model_config.mbs_times_dp_symbol.set_data(mbs_times_dp)
        self.model_wrapper.model_config.seq_len_symbol.set_data(seq_len)
        print(f'{comm_args.local_device}: dp_rank={dp_rank}, dp_size={dp_size}, gbs={args.global_batch_size}, mbs={args.micro_batch_size}, num_micro_batches={num_micro_batches}')
        
        # If dp_size * mbs changes, then should use the new dataloader.
        if dp_rank != -1:
            train_iter = self.train_data_iterator(
                dataset_args.dataset, 
                dataset_args.consumed_samples, 
                args.micro_batch_size, 
                dp_rank, 
                dp_size
            ) # need cache?
        else:
            train_iter = None
            
        def get_position_ids(gbs_per_dp, seq_len): 
            position_ids = np.arange(0, seq_len, dtype=np.int64) # [1, seq_len]
            position_ids = np.tile(position_ids, [gbs_per_dp, 1]) # [dp_size, seq_len]
            return position_ids
           
        # Start Training 
        if envs.elastic:
            self.local_straggler.begin_profile()
        curr_consumed_samples = dataset_args.consumed_samples
        detect_straggler_and_plan_cnt = 0
        for epoch in range(dataset_args.epoch, dataset_args.epochs):
            for step in range(dataset_args.step, dataset_args.steps):
                if envs.elastic: 
                    detect_straggler_and_plan_cnt += 1
                # load data for each dp
                if train_iter:
                    micro_batches = []
                    for _ in range(num_micro_batches):
                        micro_batch = next(train_iter)
                        micro_batches.append(micro_batch)
                    micro_batches = np.concatenate(micro_batches, axis=0) # [num_micro_batches, micro_batch_size, max_seq_len + 1]
                    # padding sequence
                    micro_batches = micro_batches.reshape(gbs_per_dp, -1) # [gbs_per_dp, seq_len + 1]
                    labels = micro_batches[:, 1:] # [gbs_per_dp, seq_len]
                    tokens = micro_batches[:, :-1] # [gbs_per_dp, seq_len]
                    _attention_mask, _position_ids = get_mask_and_position_ids(tokens, dataset_args.dataset.encoder.pad_id())
                    _token_type_ids = np.zeros([gbs_per_dp, seq_len])
                    feed_dict = {
                        self.build_ops.input_ids: tokens.astype(np.int64),
                        self.build_ops.position_ids: _position_ids.astype(np.int64), 
                        self.build_ops.token_type_ids: _token_type_ids.astype(np.int64),
                        self.build_ops.masked_lm_labels: labels.astype(np.int64),
                    }
                else: 
                    # fake data; feed_dict is empty will cause segment fault?
                    feed_dict = {
                        self.build_ops.input_ids: np.zeros([gbs_per_dp, seq_len]).astype(np.int64),
                        self.build_ops.position_ids: get_position_ids(gbs_per_dp, seq_len).astype(np.int64), 
                        self.build_ops.token_type_ids: np.zeros([gbs_per_dp, seq_len]).astype(np.int64),
                        self.build_ops.masked_lm_labels: np.zeros([gbs_per_dp, seq_len]).astype(np.int64),
                    }
                start_time = time.time()
                if envs.run_straggler_experiment and step == 5:
                    os.environ["HETU_STRAGGLER_LOG_FILE"] = envs.straggler_file
                if envs.run_memory_experiment and step == 0:
                    os.environ["HETU_MEMORY_LOG_FILE"] = envs.memory_file
                try:
                    # 由于目前热切换写在了run里
                    # 因此所有情况都需要跑一下run
                    # 但对于suspended和unused的rank这里实际上是空壳
                    # Hetu C++ Backend
                    results = self.build_ops.train_op.graph.run(
                        self.build_ops.loss_op, 
                        [self.build_ops.loss_op, self.build_ops.train_op], 
                        feed_dict = feed_dict, 
                        num_micro_batches = num_micro_batches, 
                        cur_strategy_id = strategy_id,
                        run_level = run_level,
                        grad_scale = 1.0
                    )
                    # NOTE: 实际上应该扫描一次alloc到update之间的所有数据
                    # grad_scale = 当前run的数据的batch_size除以总的这之间run加起来的batch_size
                    if is_suspended:
                        self.local_straggler.run_profile()
                except RuntimeError as e:
                    print(e)
                    os.killpg(0, signal.SIGTERM)
                if (envs.run_straggler_experiment and step == 5) or (args.run_memory_experiment and step == 0):
                    if "HETU_MEMORY_LOG_FILE" in os.environ:
                        del os.environ["HETU_MEMORY_LOG_FILE"] 
                    if "HETU_STRAGGLER_LOG_FILE" in os.environ:
                        del os.environ["HETU_STRAGGLER_LOG_FILE"] 
                    # TODO: 目前跑实验会直接强行终止
                    os.killpg(0, signal.SIGTERM)
                ht.global_comm_barrier_rpc()
                end_time = time.time()
                curr_consumed_samples += args.global_batch_size
                if run_level == ht.run_level("update"):
                    if label_device_group != None:
                        loss_out = results[0].numpy(force=True).mean()
                        print(f"{comm_args.local_device}: [Epoch {dataset_args.epoch}] (step {step}, consumed_samples = {curr_consumed_samples}): loss = {loss_out:.3f}, time = {end_time - start_time:.4f}")
                # detect stragglers and judge if the system need to switch strategy
                if envs.elastic and detect_straggler_and_plan_cnt == SLIDING_WINDOW + 1:
                    self.local_straggler.end_profile()
                    need_switch = self.detect_straggler_and_plan(strategy_args, comm_args)
                    if need_switch:
                        return False, TrainerDatasetArgs(
                            dataset=dataset_args.dataset,
                            consumed_samples=curr_consumed_samples,
                            steps=dataset_args.steps,
                            epochs=dataset_args.epochs,
                            step=step,
                            epoch=epoch
                        )
                    # 不需要切换而可以继续训练
                    # 重新开始一个SLIDING_WINDOW的profiling
                    self.local_straggler.begin_profile()
                    detect_straggler_and_plan_cnt = 0 
                # 总共训练的iter
            # epoch结束后要清空curr_consumed_samples
            curr_consumed_samples = 0
        return True, TrainerDatasetArgs(
            dataset=dataset_args.dataset,
            consumed_samples=curr_consumed_samples,
            steps=dataset_args.steps,
            epochs=dataset_args.epochs,
            step=step,
            epoch=epoch
        )