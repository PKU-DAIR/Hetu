import os
import time
import signal
import hetu as ht
import numpy as np
from types import SimpleNamespace
from .utils import DatasetWrapper, ModelWrapper, OptimizerWrapper, TrainerConfig
from .batch_scheduler import global_batch_scheduler, make_micro_batch
from utils import parse_multi_ds_parallel_config, write_to_csv, read_from_csv
from data_utils import build_bucket_global_data_loader, Encoder
from .planner import BalanceDynamicDispatcher, GroupDynamicDispatcher
from hetu.peft.lora import MultiLoraModel

class DatasetContext:
    def __init__(
        self,
        dataset,
        steps,
        epochs
    ):
        self.dataset = dataset
        self.consumed_samples = 0
        self.steps = steps
        self.epochs = epochs
        self.step = 0
        self.epoch = 0

class Trainer:
    def __init__(
        self,
        args,
        dataset_wrapper: DatasetWrapper,
        pretrained_model_wrapper: ModelWrapper,
        optimizer_wrapper: OptimizerWrapper,
        trainer_config: TrainerConfig,
        cost_model,
        strategy_configs,
        max_tokens_list=None,
        dataset_ctxs=None,
        use_packing=False,
    ):
        self.dataset_wrapper = dataset_wrapper
        self.pretrained_model_wrapper = pretrained_model_wrapper
        self.config = pretrained_model_wrapper.model_config
        self.optimizer_wrapper = optimizer_wrapper
        self.strategy_configs = strategy_configs
        self.cost_model = cost_model
        self.trainer_config = trainer_config
        self.train_task_num = trainer_config.train_task_num
        
        self.dataset_ctxs = dataset_ctxs
        self.train_dataset_pool = {}
        self.build_ops = None
        self.pad_id = None
        self.precision = None
        self.max_epochs = 0
        self.ngpus = 0
        self.max_steps = 0
        self.use_packing = use_packing

        if max_tokens_list is not None:
            self.max_tokens_list = max_tokens_list
        else:
            self.max_tokens_list = list(map(int, args.max_tokens.split(','))) \
                                    if isinstance(args.max_tokens, str) \
                                    else args.max_tokens
        
        # logging
        self.total_tokens = 0
        self.valid_tokens = 0
        self.total_run_times = []
        self.schedule_times = []

    def create_dataset(self, args):
        """Build train dataset."""
        if self.dataset_ctxs is not None:
            return

        self.dataset_ctxs = []
        encoder_args = {
            'key': 'text',
            'rank': 0,
            'make_vocab_size_divisible_by': 128,
            'tensor_model_parallel_size': 1,
            'vocab_extra_ids': 0,
            'tokenizer_type': 'GPT2BPETokenizer',
            'vocab_file': args.vocab_file,
            'merge_file': args.merge_file,
        }
        encoder_args = SimpleNamespace(**encoder_args)
        encoder = Encoder(encoder_args)

        for i in range(self.train_task_num):
            task_config = self.trainer_config.task_configs[i]
            if task_config.dataset_name == "" or os.environ.get('EXPR_CUSTOM_DISTRIBUTION') == 'TRUE' and i > 0:
                train_dataset = None
            elif self.train_dataset_pool.get((task_config.dataset_name, task_config.context_length), None) is not None:
                train_dataset = self.train_dataset_pool[(task_config.dataset_name, task_config.context_length)]
            else:
                train_dataset = self.dataset_wrapper.create_dataset(
                    dataset_name=task_config.dataset_name,
                    key=task_config.json_key,
                    max_seq_len=task_config.context_length,
                    vocab_file=args.vocab_file,
                    merge_file=args.merge_file,
                    encoder=encoder
                )
                self.train_dataset_pool[(task_config.dataset_name, task_config.context_length)] = train_dataset
            dataset_ctx = DatasetContext(
                dataset=train_dataset,
                steps=task_config.steps,
                epochs=task_config.epochs
            )
            self.max_epochs = max(self.max_epochs, task_config.epochs)
            self.max_steps = max(self.max_steps, task_config.steps)
            self.dataset_ctxs.append(dataset_ctx)
        self.pad_id = self.dataset_ctxs[0].dataset.encoder.pad_id()

    def train_data_iterator(
        self,
        dataset,
        consumed_samples,
        gbs,
        min_seq_length=256,
        max_seq_length=16384
    ):
        train_dataloader = build_bucket_global_data_loader(dataset, consumed_samples, gbs, min_seq_length, max_seq_length)
        train_data_iterator = iter(train_dataloader)
        return train_data_iterator

    def get_custom_global_batch(self, seq_distribution, max_seq_len, pad_id):
        global_batch = []
        for seq_len, num in seq_distribution.items():
            for _ in range(num):
                global_batch.append([1] * (seq_len) + [pad_id] * (max_seq_len + 1 - seq_len))
        return global_batch

    def build_model(self, args, ds_parallel_configs):
        # Build dataset
        self.create_dataset(args)
        print("create dataset done")
        # Build model
        # '''
        with ht.graph("define_and_run", num_strategy=1):
            if args.bf16:
                precision = "ht.bfloat16"
            else:
                precision = "ht.float32"
            self.precision = eval(precision)
            with ht.autocast(eval(precision)):
                self.create_define_and_run_graph(args, ds_parallel_configs)
        # '''
    
    def create_define_and_run_graph(self, args, ds_parallel_configs):
        input_ds_hierarchy, input_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'input')
        label_ds_hierarchy, label_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'label')
        task_ds_hierarchy, task_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'task_batch_idxs')
        
        # 获取默认的seq_len, dp, mbs_times_dp, batch_offset, batch_size
        default_seq_len = args.max_seq_length
        default_dp = input_ds_hierarchy[0].get(0).get_dim(0)
        default_mbs_times_dp = default_dp
        dummy_size = default_mbs_times_dp * default_seq_len
        default_batch_offset = 0
        default_batch_size = 1
        
        # 构建placeholder
        if self.use_packing:
            input_ids = ht.parallel_placeholder(ht.int64, global_shape=[dummy_size], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='input_ids')
            masked_lm_labels = ht.parallel_placeholder(ht.int64, global_shape=[dummy_size], ds_hierarchy=label_ds_hierarchy, device_group_hierarchy=label_dg_hierarchy, name='masked_lm_labels')
        else:
            input_ids = ht.parallel_placeholder(ht.int64, global_shape=[default_mbs_times_dp, default_seq_len], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='input_ids')
            masked_lm_labels = ht.parallel_placeholder(ht.int64, global_shape=[default_mbs_times_dp, default_seq_len], ds_hierarchy=label_ds_hierarchy, device_group_hierarchy=label_dg_hierarchy, name='masked_lm_labels')
        self.config.task_batch_idxs = []
        for i in range(self.train_task_num):
            self.config.task_batch_idxs.append(
                ht.parallel_placeholder(
                    ht.int64,
                    global_shape=[default_batch_offset, default_batch_size, default_batch_size],
                    ds_hierarchy=task_ds_hierarchy,
                    device_group_hierarchy=task_dg_hierarchy,
                    name='task_batch_idxs_task{}'.format(i), is_cpu=True
                )
            )

        # 设置symbolic shape
        self.config.dp_symbol.set_data(default_dp)
        self.config.train_task_num = self.train_task_num
        # 创建预训练模型
        pretrained_model = self.pretrained_model_wrapper.create_model(ds_parallel_configs=ds_parallel_configs)
        # 创建微调模型
        peft_configs = [task_config.lora_config for task_config in self.trainer_config.task_configs]
        model = MultiLoraModel(
            model=pretrained_model,
            peft_configs=peft_configs,
            config=self.pretrained_model_wrapper.model_config
        )
        self.config.cu_seqlens_list = []
        if self.use_packing:
            for block_id, block in enumerate(pretrained_model.transformer.h):
                self.config.cu_seqlens_list.append(
                    ht.parallel_placeholder(
                        ht.int32,
                        global_shape=[dummy_size], 
                        ds_hierarchy=block.attn.q_proj.base_layer.ds_union_map['split0_dup'], 
                        device_group_hierarchy=block.attn.q_proj.base_layer.device_group_unions,
                        name=f'cu_seqlens_{block_id}'
                    )
                )
            self.config.max_seqlen_symbol = ht.IntSymbol(1)
        print(f"start to build model...")
        # 构建静态图
        if self.use_packing:
            loss = model(
                input_ids=input_ids,
                labels=masked_lm_labels,
                # cu_seqlens_list=cu_seqlens_list
            )
        else:
            loss = model(
                input_ids=input_ids,
                labels=masked_lm_labels,
            )
        print(f"build model end...")
        # build optimizer
        opt = self.optimizer_wrapper.create_optimizer(lr=args.lr)
        train_op = opt.minimize(loss)
        # build ops
        self.build_ops = {
            'input_ids': input_ids,
            'masked_lm_labels': masked_lm_labels,
            # 'task_batch_idxs': task_batch_idxs,
            # 'cu_seqlens_list': cu_seqlens_list,
            'loss': loss,
            'train_op': train_op
        }
    
    def run(self, args):
        assert len(self.strategy_configs) == 1, \
            'currently we only support single strategy'
        strategy_id = 0
        strategy_config = self.strategy_configs[strategy_id]
        self.ngpus = strategy_config.get_num_gpus()
        local_device = ht.local_device()
        all_devices = ht.global_device_group()
        local_host_name = os.environ['HETU_LOCAL_HOSTNAME']

        gpu_id = all_devices.get_index(local_device)
        scheme_id = strategy_config.get_scheme_id(gpu_id)
        dup_group_idx = strategy_config.get_local_dp_id(gpu_id)
        stage_id = strategy_config.get_stage_id(gpu_id)
        print(f"scheme_id = {scheme_id}")
        self.config.dp_symbol.set_data(strategy_config.num_pipeline)
        
        # train_iter
        train_iter_list = []
        train_task_num = self.trainer_config.train_task_num
        for i in range(train_task_num):
            task_ctx = self.dataset_ctxs[i]
            if task_ctx.dataset is None:
                train_iter_list.append(None)
            else:
                train_iter_list.append(self.train_data_iterator(task_ctx.dataset, task_ctx.consumed_samples,
                                                                self.trainer_config.task_configs[i].global_batch_size,
                                                                args.min_seq_length, args.max_seq_length))
        if os.environ.get('EXPR_CUSTOM_DISTRIBUTION') == 'ON':
            task_seq_len_distribution = {0 : {8192: 14, 16384: 2}}
            global_batch_size_list = [sum(task_seq_len_distribution[task_id].values()) for task_id in sorted(task_seq_len_distribution.keys())]
        elif os.environ.get('EXPR_CUSTOM_DISTRIBUTION') == 'OFF':
            task_seq_len_distribution = None
            global_batch_size_list = [task_config.global_batch_size for task_config in self.trainer_config.task_configs]
        else:
            raise ValueError(f"EXPR_CUSTOM_DISTRIBUTION should be ON or OFF, but got {os.environ.get('EXPR_CUSTOM_DISTRIBUTION')}")
        
        # Dynamic Batch Dispatcher
        data_dispatch_pattern = os.environ.get('EXPR_DATA_DISPATCH')
        print(f"create dynamic {data_dispatch_pattern} batch planner...")
        if data_dispatch_pattern == 'GROUP':
            dynamic_planner = GroupDynamicDispatcher(
                self.cost_model,
                args.num_layers,
                train_task_num,
                global_batch_size_list,
                strategy_config,
                self.max_tokens_list,
                scheme_id,
                local_device_idx=local_device.index,
            )
        elif data_dispatch_pattern == 'BALANCE':
            dynamic_planner = BalanceDynamicDispatcher(
                self.cost_model,
                args.num_layers,
                train_task_num,
                global_batch_size_list,
                strategy_config,
                self.max_tokens_list,
                scheme_id,
                local_device_idx=local_device.index,
            )
        else:
            raise ValueError(f"EXPR_DATA_DISPATCH should be GROUP or BALANCE, but got {data_dispatch_pattern}")

        # warm up
        if os.environ.get('DP_BUCKET') == 'ON' and \
           os.environ.get('EXPR_SIMULATE') != 'ON':
            print(f"{local_device}: warmup begin...")
            warmup_step = 5
            for _ in range(warmup_step):
                num_micro_batches = strategy_config.get_pp_degree(scheme_id)
                warmup_micro_batch = make_micro_batch(1, self.max_tokens_list[scheme_id], train_task_num=train_task_num)
                cur_batch_size = warmup_micro_batch.batch_size
                cur_seq_len = warmup_micro_batch.seq_length
                cur_batch_offset_list = warmup_micro_batch.batch_offset_list
                cur_batch_size_list = warmup_micro_batch.batch_size_list
                task_batch_idxs_list = [[] for _ in range(train_task_num)]

                for task_id in range(train_task_num):
                    task_batch_idxs = np.zeros([cur_batch_offset_list[task_id], cur_batch_size_list[task_id], cur_batch_size], dtype=np.int64)
                    task_batch_idxs_list[task_id] = [task_batch_idxs for _ in range(num_micro_batches)]
                input_ids_list = [np.zeros([cur_batch_size, cur_seq_len]).astype(np.int64) for _ in range(num_micro_batches)]
                masked_lm_labels_list = [np.zeros([cur_batch_size, cur_seq_len]).astype(np.int64) for _ in range(num_micro_batches)]
                feed_dict = {
                    self.build_ops['input_ids']: input_ids_list,
                    self.build_ops['masked_lm_labels']: masked_lm_labels_list,
                }
                for i in range(train_task_num):
                    feed_dict[self.config.task_batch_idxs[i]] = task_batch_idxs_list[i]
                with ht.autocast(self.precision):
                    _ = self.build_ops['train_op'].graph.run(
                        self.build_ops['loss'],
                        [self.build_ops['loss'], self.build_ops['train_op']],
                        feed_dict=feed_dict,
                        num_micro_batches = num_micro_batches,
                        cur_strategy_id = strategy_id,
                        run_level = ht.run_level("update"),
                        grad_scale = 1.0)
            print(f"{local_device}: warmup end...")
        
        # train
        for epoch in range(self.max_epochs):
            for step in range(self.max_steps):
                multi_task_global_batch_map = {}
                for task_id in range(train_task_num):
                    if self.dataset_ctxs[task_id].step >= self.dataset_ctxs[task_id].steps or \
                        self.dataset_ctxs[task_id].epoch >= self.dataset_ctxs[task_id].epochs:
                        continue
                    if train_iter_list[task_id] is None:
                        continue

                    if os.environ.get('EXPR_CUSTOM_DISTRIBUTION') == 'ON':
                        seq_len_distribution = task_seq_len_distribution[task_id]
                        global_batch = self.get_custom_global_batch(seq_len_distribution, args.max_seq_length, self.pad_id)
                    elif os.environ.get('EXPR_CUSTOM_DISTRIBUTION') == 'OFF':
                        try:
                            global_batch = next(train_iter_list[task_id])
                        except StopIteration:
                            train_iter_list[task_id] = self.train_data_iterator(self.dataset_ctxs[task_id].dataset, 0,
                                                                                self.trainer_config.task_configs[task_id].global_batch_size,
                                                                                args.min_seq_length, args.max_seq_length)
                            global_batch = next(train_iter_list[task_id])
                    else:
                        raise ValueError(f"EXPR_CUSTOM_DISTRIBUTION should be ON or OFF, but got {os.environ.get('EXPR_CUSTOM_DISTRIBUTION')}")
                    multi_task_global_batch_map[task_id] = global_batch
                    self.dataset_ctxs[task_id].consumed_samples += len(global_batch)
                micro_batches_list, schedule_time = global_batch_scheduler(args, multi_task_global_batch_map, train_task_num,
                                                                           self.pad_id, strategy_config, dup_group_idx,
                                                                           self.max_tokens_list[scheme_id], strategy_config.get_num_scheme(),
                                                                           scheme_id, dynamic_planner, step=step)
                # special case for group dispatch
                if not micro_batches_list:
                    step -= 1
                    continue
                self.schedule_times.append(schedule_time)
                print(f"step = {step}")

                # 统计tokens
                if step > 0 or epoch > 0:
                    for micro_batch in micro_batches_list:
                        batch_data = np.array(micro_batch.batch_data)
                        self.total_tokens += micro_batch.batch_size * micro_batch.seq_length
                        self.valid_tokens += np.sum(batch_data != self.pad_id)

                if os.environ.get('EXPR_SIMULATE') == 'TRUE':
                    self.total_run_times.append(0)
                    continue
                
                # prepare feed_dict
                input_ids_list = []
                masked_lm_labels_list = []
                task_batch_idxs_list = [[] for _ in range(train_task_num)]
                for micro_batch in micro_batches_list:
                    cur_batch_size = micro_batch.batch_size
                    cur_batch_data = np.array(micro_batch.batch_data).reshape(cur_batch_size, -1)
                    cur_seq_len = micro_batch.seq_length
                    cur_batch_offset_list = micro_batch.batch_offset_list
                    cur_batch_size_list = micro_batch.batch_size_list
                    for task_id in range(train_task_num):
                        task_batch_idxs = np.zeros([cur_batch_offset_list[task_id], cur_batch_size_list[task_id], cur_batch_size], dtype=np.int64)
                        task_batch_idxs_list[task_id].append(task_batch_idxs)
                    labels = cur_batch_data[:, 1:]
                    tokens = cur_batch_data[:, :-1]
                    if dup_group_idx != -1:
                        input_ids_list.append(tokens.astype(np.int64))
                        masked_lm_labels_list.append(labels.astype(np.int64))
                    else:
                        input_ids_list.append(np.zeros([cur_batch_size, cur_seq_len]).astype(np.int64))
                        masked_lm_labels_list.append(np.zeros([cur_batch_size, cur_seq_len]).astype(np.int64))
                feed_dict = {
                    self.build_ops['input_ids']: input_ids_list,
                    self.build_ops['masked_lm_labels']: masked_lm_labels_list,
                }
                for i in range(train_task_num):
                    feed_dict[self.config.task_batch_idxs[i]] = task_batch_idxs_list[i]

                iter_time = 0
                run_level = ht.run_level("update")
                ht.global_comm_barrier_rpc()
                start_time = time.time()
                try:
                    with ht.autocast(self.precision):
                        results = self.build_ops['train_op'].graph.run(
                            self.build_ops['loss'],
                            [self.build_ops['loss'], self.build_ops['train_op']],
                            feed_dict=feed_dict,
                            num_micro_batches = len(micro_batches_list),
                            cur_strategy_id = strategy_id,
                            run_level = run_level,
                            grad_scale = 1.0
                        )
                except RuntimeError as e:
                    print(e)
                    os.killpg(0, signal.SIGTERM)
                end_time = time.time()
                iter_time += end_time - start_time
                if (step > 0 or epoch > 0) and (run_level == ht.run_level("compute_only") or run_level == ht.run_level("update")):
                    print(f"iter {step}: {iter_time:.3f}s")
                    self.total_run_times.append(iter_time)
                    if run_level == ht.run_level("compute_only"):
                        if os.environ.get("EXPR_EFFECTIVENESS") == "ON":
                            if not os.path.exists("effective_logs"):
                                os.makedirs("effective_logs")
                            with open(f"effective_logs/{local_host_name}-{local_device.index}.txt", 'a') as f:
                                f.write(f"{iter_time}\n")
                        
                        if os.environ.get('EXPR_CASE_STUDY') == 'ON':
                            with open(f"case_study/{data_dispatch_pattern}-{os.environ.get('DP_BUCKET')}/{local_host_name}-{local_device.index}.txt", 'a') as f:
                                f.write(f"{iter_time}\n")
                                f.write("\n")
                        ht.global_comm_barrier_rpc()
                # TODO: consumed samples of each task
                if stage_id == strategy_config.get_pp_degree(scheme_id) and run_level == ht.run_level("update"):
                    loss_out = results[0].numpy(force=True).mean()
                    consumed_samples = np.sum(self.dataset_ctxs[i].consumed_samples for i in range(train_task_num))
                    print(f"{local_device}: [Epoch {epoch}] (step {step}, consumed_samples = {consumed_samples}): loss = {loss_out:.3f}, time = {iter_time:.4f}")
        print(f"token_num = {dynamic_planner.token_num}, valid_num = {dynamic_planner.valid_token_num}, padding_ratio = {1 - dynamic_planner.valid_token_num / dynamic_planner.token_num}, max schedule time = {max(self.schedule_times)}, min schedule time = {min(self.schedule_times)}, avg schedule time = {np.mean(self.schedule_times)}")
        
        if gpu_id == 0:
            if not os.path.exists("temp"):
                os.makedirs("temp")
        
        if stage_id == strategy_config.get_pp_degree(scheme_id):
            # total_run_time = np.mean(self.total_run_times)
            total_run_time = self.total_run_times
            schedule_time = np.mean(self.schedule_times)
            run_time_entry = {
                'dp': strategy_config.get_dp_degree(scheme_id),
                'tp': strategy_config.get_tp_degree(scheme_id),
                'pp': strategy_config.get_pp_degree(scheme_id),
                'total_run_time': total_run_time,
                'schedule_time': schedule_time
            }
            print(f"total run time num = {len(total_run_time)}")
            write_to_csv(run_time_entry, f"temp/run_time_{strategy_config.get_num_scheme()}_{local_host_name}_{local_device.index}")
        ht.global_comm_barrier_rpc()
        if gpu_id == 0:
            print(f"handler: {local_host_name}")
            total_cnt = dynamic_planner.token_num 
            valid_cnt = dynamic_planner.valid_token_num
            run_times = []
            schedule_times = []
            run_time_file_names = [f for f in os.listdir('temp') if f.startswith(f"run_time_{strategy_config.get_num_scheme()}")]
            for run_time_file_name in run_time_file_names:
                rows = read_from_csv(f"temp/{run_time_file_name}")
                if len(rows) == 0:
                    continue
                run_times.append(rows[0]['total_run_time'])
                schedule_times.append(rows[0]['schedule_time'])
                os.remove(f"temp/{run_time_file_name}")
            log_entry = {
                'dp': strategy_config.dps,
                'tp': strategy_config.tps,
                'pp': strategy_config.pps,
                'max_tokens': self.max_tokens_list,
                'valid_tokens': valid_cnt,
                'run_time': np.min(run_times),
                'gpu_seconds': np.min(run_times) * self.ngpus,
                # 'schedule_time': np.min(schedule_times)
            }
            if os.environ.get('EXPR_SENSITIVITY') == 'ON':
                with open('sensitivity_result.txt', 'a') as f:
                    f.write(f"{args.bucket_num}\n")
                    f.write(f"{(dynamic_planner.token_num - dynamic_planner.valid_token_num) / dynamic_planner.token_num}\n")
                    f.write(f"{np.min(run_times, axis=0).tolist()}\n")
                    f.write("\n")
            write_to_csv(log_entry, f"exp_result/e2e/result.csv")
            print(f"total_cnt = {total_cnt}, valid_cnt = {valid_cnt}, mean_run_time = {np.min(run_times)}, mean_schedule_time = {np.min(schedule_times)}")
