import os
import signal
import hetu as ht
import numpy as np
from trainer.utils import ModelWrapper, OptimizerWrapper, TrainerConfig
from utils import parse_multi_ds_parallel_config
from trainer.batch_scheduler import make_micro_batch
from hetu.peft.lora import MultiLoraModel

class Profiler:
    def __init__(
        self,
        args,
        pretrained_model_wrapper: ModelWrapper,
        optimizer_wrapper: OptimizerWrapper,
        trainer_config: TrainerConfig,
        ds_parallel_configs,
    ):
        self.pretrained_model_wrapper = pretrained_model_wrapper
        self.config = self.pretrained_model_wrapper.model_config
        self.optimizer_wrapper = optimizer_wrapper
        self.trainer_config = trainer_config
        self.ds_parallel_configs = ds_parallel_configs
        
        self.build_ops = None
        self.profile_steps = args.profile_steps
        self.warmup_steps = args.warmup_steps
        self.train_task_num = self.trainer_config.train_task_num
        self.precision = None
        self.sequence_parallel = args.sequence_parallel
        self.dp = 1
        self.pp = 1
        self.tp = args.tp
        self.ngpus = self.dp * self.pp * self.tp
        
        # logging
        self.block_stream_time = []
        self.total_stream_time = []

    def build_model(self, args, ds_parallel_configs):
        # Build model
        with ht.graph("define_and_run", num_strategy=1):
            if args.bf16:
                precision = "ht.bfloat16"
            else:
                precision = "ht.float32"
            self.precision = eval(precision)
            with ht.autocast(eval(precision)):
                self.create_define_and_run_graph(args, ds_parallel_configs)
    
    def create_define_and_run_graph(self, args, ds_parallel_configs):
        input_ds_hierarchy, input_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'input')
        label_ds_hierarchy, label_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'label')
        task_ds_hierarchy, task_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'task_batch_idxs')
        
        # 获取默认的seq_len, dp, mbs_times_dp, batch_offset, batch_size
        default_seq_len = args.default_seq_len
        default_dp = self.dp
        default_mbs_times_dp = args.default_mbs * default_dp
        default_batch_offset = 0
        default_batch_size = args.default_mbs
        
        # 构建placeholder
        input_ids = ht.parallel_placeholder(ht.int64, global_shape=[default_mbs_times_dp, default_seq_len], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='input_ids')
        masked_lm_labels = ht.parallel_placeholder(ht.int64, global_shape=[default_mbs_times_dp, default_seq_len], ds_hierarchy=label_ds_hierarchy, device_group_hierarchy=label_dg_hierarchy, name='masked_lm_labels')
        self.config.task_batch_idxs = []
        for i in range(self.trainer_config.train_task_num):
            self.config.task_batch_idxs.append(
                ht.parallel_placeholder(
                    ht.int64,
                    global_shape=[default_batch_offset, default_batch_size, default_batch_size],
                    ds_hierarchy=task_ds_hierarchy,
                    device_group_hierarchy=task_dg_hierarchy,
                    name='task_batch_idxs_task{}'.format(i),
                    is_cpu=True
                )
            )
        # 设置symbolic shape
        self.config.dp_symbol = ht.IntSymbol(default_dp)
        self.config.train_task_num = self.trainer_config.train_task_num
        # 创建预训练模型
        pretrained_model = self.pretrained_model_wrapper.create_model(ds_parallel_configs=ds_parallel_configs)
        # 创建微调模型
        peft_configs = [task_config.lora_config for task_config in self.trainer_config.task_configs]
        model = MultiLoraModel(
            model=pretrained_model,
            peft_configs=peft_configs,
            config=self.config
        )
        # 构建静态图
        loss = model(
            input_ids=input_ids,
            labels=masked_lm_labels
        )
        # build optimizer
        opt = self.optimizer_wrapper.create_optimizer(lr=args.lr)
        train_op = opt.minimize(loss)
        # build ops
        self.build_ops = {
            'input_ids': input_ids,
            'masked_lm_labels': masked_lm_labels,
            'loss': loss,
            'train_op': train_op
        }
    
    def profile(self, mbs, seq_len):
        self.block_time = []
        self.total_stream_time = []
        for step in range(self.profile_steps + self.warmup_steps):
            print(f"profiler: step = {step}")
            profile_micro_batch = make_micro_batch(mbs, seq_len, train_task_num=self.train_task_num)
            batch_data = np.array(profile_micro_batch.batch_data).reshape(mbs, -1)
            labels = batch_data[:, 1:]
            tokens = batch_data[:, :-1]
            task_batch_idxs_list = []
            for train_task_idx in range(self.train_task_num):
                task_batch_idxs_i = np.zeros([profile_micro_batch.batch_offset_list[train_task_idx], profile_micro_batch.batch_size_list[train_task_idx], profile_micro_batch.batch_size], dtype=np.int64)
                task_batch_idxs_list.append(task_batch_idxs_i)
            feed_dict = {
                self.build_ops['input_ids']: tokens.astype(np.int64),
                self.build_ops['masked_lm_labels']: labels.astype(np.int64),
            }
            for i in range(self.train_task_num):
                feed_dict[self.config.task_batch_idxs][i] = task_batch_idxs_list[i]
            run_level = ht.run_level("update")
            try:
                with ht.autocast(self.precision):
                    with ht.profiler(enabled=True, record_shapes=False) as profiler:
                        _ = self.build_ops['train_op'].graph.run(
                            self.build_ops['loss'],
                            [self.build_ops['loss'], self.build_ops['train_op']],
                            feed_dict=feed_dict,
                            num_micro_batches=1,
                            cur_strategy_id=0,
                            run_level = run_level,
                            grad_scale=1.0)
                if step >= self.warmup_steps:
                    self.total_stream_time.append(float(profiler.summary()['graph_view'][11][1])) # total-time-stream
                    self.block_time.append(float(profiler.summary()['graph_view'][12][1])) # block time
            except RuntimeError as e:
                print(e)
                os.killpg(0, signal.SIGTERM)