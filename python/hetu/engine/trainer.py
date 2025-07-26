import os
import time
import signal
import logging
import hetu
import numpy as np
from omegaconf import OmegaConf, DictConfig
from typing import Union, Optional, Callable, List, Any, Dict, Tuple
from hetu.engine.trainer_config import TrainingConfig
from hetu.models.utils.model_utils import PreTrainedModel
from hetu.data.tokenizers.utils import BaseTokenizer
from hetu.data.tokenizers.tokenizer import build_tokenizer
from hetu.data.data_collator import DataCollatorForLanguageModel
from hetu.data.dataloader import build_data_loader
from hetu.data.dataset import JsonDataset
from hetu.data.bucket import get_sorted_batch_and_len, get_input_and_label_buckets
from hetu.data.utils import convert_parquet_to_json
from hetu.data import IGNORE_INDEX
from hetu.utils.parallel import read_ds_parallel_config, parse_multi_ds_parallel_config, get_dg_from_union
from hetu.engine.utils import TrainerCommAllArgs
from .wrapper import ModelWrapper, OptimizerWrapper, ModelWrapperFromConfig
from torch.profiler import profile, ProfilerActivity
from torch.utils.data import DataLoader, Dataset

MAX_TP_DEGREE = 16

class TrainerStates:
    """
    Class to hold the states during training process.
    
    Args:
        loss_op: Loss operation tensor.
        train_op: Training operation tensor.
        input_ids: Input token IDs tensor.
        lm_labels: Language model labels tensor.
        loss_mask: Optional mask for loss calculation.
        position_ids: Optional position IDs tensor.
        token_type_ids: Optional token type IDs tensor.
        num_tokens: Number of tokens processed.
        loss_sum: Accumulated loss sum.
        **kwargs: Additional state variables.
    """
    def __init__(
        self,
        loss_op: hetu.Tensor,
        train_op: hetu.Tensor,
        input_ids: hetu.Tensor,
        lm_labels: hetu.Tensor,
        loss_mask: hetu.Tensor = None,
        position_ids: hetu.Tensor = None,
        token_type_ids: hetu.Tensor = None,
        num_tokens: int = 0,
        loss_sum: float = 0.0,
        **kwargs,
    ):
        self.loss_op = loss_op
        self.train_op = train_op
        self.input_ids = input_ids
        self.lm_labels = lm_labels
        self.loss_mask = loss_mask
        self.position_ids = position_ids
        self.token_type_ids = token_type_ids
        self.num_tokens = num_tokens
        self.loss_sum = loss_sum

class Trainer:
    """
    Trainer class for training language models with distributed parallelism.
    
    Args:
        pretrain_config: Configuration for training.
        model: Model to train.
        tokenizer: Tokenizer for processing text.
        optimizer: Optimizer for training.
        train_dataset: Optional dataset for training.
        data_collator: Optional custom data collator.
        **kwargs: Additional arguments.
    """
    def __init__(
        self,
        pretrain_config: TrainingConfig,
        model: Union[PreTrainedModel, ModelWrapper, ModelWrapperFromConfig],
        tokenizer: Union[BaseTokenizer, DictConfig],
        optimizer: Union[OptimizerWrapper, DictConfig],
        train_dataset: Optional[Dataset] = None,
        data_collator: Optional[Callable] = None,
        **kwargs,
    ):
        ds_parallel_configs = kwargs.get("ds_parallel_configs", None)
        
        self.pretrain_config = pretrain_config
        self.ds_config = pretrain_config.ds_parallel
        if self.pretrain_config.packing and self.pretrain_config.micro_batch_size:
            raise ValueError("micro_batch_size is only valid when packing is off")

        self.model = model
        self.model_config = None
        if isinstance(optimizer, DictConfig):
            optimizer = OmegaConf.to_container(optimizer, resolve=True)
            self.optimizer = OptimizerWrapper(optimizer)
        else:
            self.optimizer = optimizer
        self.train_dataset = train_dataset
        
        if self.pretrain_config.bf16:
            self.precision = "hetu.bfloat16"
        else:
            self.precision = "hetu.float32"
        
        # build tokenizer from tokenizer type
        if isinstance(tokenizer, DictConfig):
            tokenizer_type = tokenizer.type
            tokenizer_kwargs = OmegaConf.to_container(tokenizer, resolve=True)
            tokenizer = build_tokenizer(tokenizer_type, **tokenizer_kwargs)
        self.tokenizer = tokenizer
        
        self.data_collator = (
            data_collator
            if data_collator is not None
            else DataCollatorForLanguageModel(tokenizer)
        )
        
        self.epoch = 0
        self.consumed_samples = 0
        self.data_iter = None
        self.trainer_states = None
        self.is_model_built = False
        
        if ds_parallel_configs is None:
            ds_parallel_config_path = self.ds_config.ds_parallel_config_path
            ds_parallel_config_name = self.ds_config.ds_parallel_config_name
            ds_parallel_config = os.path.join(ds_parallel_config_path, ds_parallel_config_name)
            if ds_parallel_config is None:
                raise ValueError("ds_parallel_config is required.")
            ds_parallel_configs = read_ds_parallel_config(ds_parallel_config)
        self.ds_parallel_configs = ds_parallel_configs
        self.num_strategy = len(ds_parallel_configs)
        
        local_device = hetu.local_device()
        all_devices = hetu.global_device_group()
        
        input_ds_hierarchy, input_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'input')
        label_ds_hierarchy, label_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'label')
        self.comm_args = TrainerCommAllArgs(
            input_ds_hierarchy=input_ds_hierarchy,
            input_dg_hierarchy=input_dg_hierarchy,
            label_ds_hierarchy=label_ds_hierarchy,
            label_dg_hierarchy=label_dg_hierarchy,
            local_device=local_device,
            all_devices=all_devices,
        )
        
        # Initialize loss tracking for plotting
        if self.pretrain_config.plot_loss:
            self.loss_history = []
            self.step_history = []
            self.fig = None
            self.ax = None

    def get_train_data_loader(self) -> DataLoader:
        """
        Get a data loader for training.
        
        Returns:
            DataLoader for training data.
            
        Raises:
            ValueError: If required dataset paths are not provided.
        """
        if self.train_dataset is None:
            dataset_path = self.pretrain_config.train_dataset_path
            dataset_text_field = self.pretrain_config.dataset_text_field
            if dataset_path is None or dataset_text_field is None:
                raise ValueError("train_dataset_path and dataset_text_field are required.")
            if dataset_path.endswith(".parquet"):
                dataset_path = convert_parquet_to_json(dataset_path)
            self.train_dataset = JsonDataset(dataset_path, dataset_text_field, self.tokenizer, self.pretrain_config.max_seq_length)
            self.data_collator = self.data_collator
        return build_data_loader(
            self.train_dataset,
            self.consumed_samples,
            self.pretrain_config.global_load_size,
            self.pretrain_config.data_load_level,
            self.data_collator
        )

    def build(self):
        """
        Build the trainer model.
        
        Raises:
            ValueError: If trainer has already been built.
        """
        if self.is_model_built:
            raise ValueError("Trainer has already been built.")
        with hetu.graph("define_and_run", num_strategy=self.num_strategy):
            with hetu.autocast(eval(self.precision)):
                self.trainer_states = self.create_define_graph()
        self.is_model_built = True
    
    def create_define_graph(self) -> TrainerStates:
        """
        Create and define the computational graph for training.
        
        Returns:
            TrainerStates: States for the training process.
        """
        dcp_size = self.comm_args.input_ds_hierarchy[0].get(0).get_dim(0)
        input_ids = hetu.parallel_placeholder(
            hetu.int64,
            global_shape=[dcp_size * MAX_TP_DEGREE],
            ds_hierarchy=self.comm_args.input_ds_hierarchy,
            device_group_hierarchy=self.comm_args.input_dg_hierarchy,
            name='input_ids'
        )
        lm_labels = hetu.parallel_placeholder(
            hetu.int64,
            global_shape=[dcp_size * MAX_TP_DEGREE],
            ds_hierarchy=self.comm_args.label_ds_hierarchy,
            device_group_hierarchy=self.comm_args.label_dg_hierarchy,
            name='lm_labels'
        )
        
        if isinstance(self.model, (ModelWrapper, ModelWrapperFromConfig)):
            self.model = self.model.create_model(self.ds_parallel_configs)

        self._init_config_symbol()
        self.model_config = self.model.config
        
        loss = self.model(
            input_ids=input_ids,
            labels=lm_labels,
        )
        
        self.optimizer = self.optimizer.create_optimizer()
        
        train_op = self.optimizer.minimize(loss)
        
        return TrainerStates(
            loss_op=loss,
            train_op=train_op,
            input_ids=input_ids,
            lm_labels=lm_labels,
        )
    
    def _init_config_symbol(self):
        """
        Initialize configuration symbols for the model.
        Sets up sequence length symbols and communication patterns for distributed training.
        """
        self.model.config.multi_seq_lens_symbol = []
        self.model.config.multi_cp_group_symbol = []
        for i in range(len(self.comm_args.input_ds_hierarchy)):
            dcp_size = self.comm_args.input_ds_hierarchy[i].get(0).get_dim(0)
            # 例如[32, 32, 32, 48, 48, 32, 32, 32]
            # 表示各个dcp分到的seq len
            self.model.config.multi_seq_lens_symbol.append([hetu.IntSymbol(1) for _ in range(dcp_size)])
            # 例如[0, 0, 0, 1, 1, 2, 2, 2] 
            # 相同编号的在一个cp group中
            self.model.config.multi_cp_group_symbol.append([hetu.IntSymbol(1) for _ in range(dcp_size)])
        
        self.model.config.cu_seqlens_list = []
        for block_id, block in enumerate(self.model.model.h):
            self.model.config.cu_seqlens_list.append(
                hetu.parallel_placeholder(
                    hetu.int32, 
                    global_shape=[dcp_size * MAX_TP_DEGREE], 
                    ds_hierarchy=block.attn.qkv_dense.ds_union_map['split0_dup'], 
                    device_group_hierarchy=block.attn.qkv_dense.device_group_unions,
                    name=f'cu_seqlens_{block_id}'
                ) if self.pretrain_config.packing else None
            )
        self.model.config.max_seqlen_symbol = (
            hetu.IntSymbol(1)
            if self.pretrain_config.packing else None
        )
        self.model.config.packing = self.pretrain_config.packing
    
    def _train(
        self,
        feed_dict: Dict[hetu.Tensor, np.ndarray],
        int_symbol_dict: Dict[hetu.IntSymbol, int],
        num_micro_batches: int,
        strategy_id: int,
        num_tokens: int,
        run_level: int = hetu.run_level("update"),
    ) -> Tuple[List, int]:
        """
        Execute a training step.
        
        Args:
            feed_dict: Dictionary mapping tensors to their input values.
            int_symbol_dict: Dictionary mapping integer symbols to their values.
            num_micro_batches: Number of micro-batches.
            strategy_id: ID of the parallel strategy.
            num_tokens: Number of tokens in the batch.
            run_level: Run level for the execution.
            
        Returns:
            Tuple containing results and accumulated token count.
            
        Raises:
            RuntimeError: If execution fails.
        """
        try:
            with hetu.autocast(eval(self.precision)):
                results, accumulate_token_num = self.trainer_states.train_op.graph.run(
                    self.trainer_states.loss_op, 
                    [self.trainer_states.loss_op, self.trainer_states.train_op],
                    feed_dict = feed_dict, 
                    int_symbol_dict = int_symbol_dict, 
                    num_micro_batches = num_micro_batches, 
                    cur_strategy_id = strategy_id,
                    run_level = run_level,
                    num_tokens = num_tokens,
                )
        except RuntimeError as e:
            logging.error(e)
            with open("./logs/exception.txt", 'w') as file:
                print("device rank:", self.comm_args.all_devices.get_index(self.comm_args.local_device), file=file)
                print(e, file=file)
            os.killpg(0, signal.SIGTERM)
        return results, accumulate_token_num
    
    def extract_strategy(
        self,
        cp_list: List[int],
        strategy_id: int,
    ) -> Tuple[int, int, int]:
        """
        Extract the communication and parallelism strategy for the current device.
        
        Args:
            cp_list: List of communication parallelism degrees.
            strategy_id: ID of the parallel strategy.
            
        Returns:
            Tuple containing:
            - cp_id: Communication parallelism ID.
            - dp_group_id: Data parallelism group ID.
            - pipeline_id: Pipeline parallelism ID.
        """
        input_ds_union = self.comm_args.input_ds_hierarchy[strategy_id]
        input_dg_union = self.comm_args.input_dg_hierarchy[strategy_id]
        label_ds_union = self.comm_args.label_ds_hierarchy[strategy_id]
        label_dg_union = self.comm_args.label_dg_hierarchy[strategy_id]
        if not self.pretrain_config.packing:
            assert input_ds_union.hetero_dim == 0 or input_ds_union.hetero_dim == -3, "input hetero dim unsupported"
            assert label_ds_union.hetero_dim == 0 or label_ds_union.hetero_dim == -3, "label hetero dim unsupported"
        
        local_device = self.comm_args.local_device
        all_devices = self.comm_args.all_devices
        
        dp_size = len(cp_list)
        dcp_size = sum(cp_list)
        
        data_union_idx, data_dcp_rank, data_dp_rank = -1, -1, -1
        input_union_idx, input_device_group = get_dg_from_union(local_device, input_dg_union)
        label_union_idx, label_device_group = get_dg_from_union(local_device, label_dg_union)
        if input_device_group is not None:
            data_union_idx = input_union_idx
        elif label_device_group is not None:
            data_union_idx = label_union_idx
        
        if data_union_idx != -1:
            data_dcp_rank = data_union_idx
            accumulate_cp = 0
            for i, cp in enumerate(cp_list):
                accumulate_cp += cp
                if accumulate_cp > data_dcp_rank:
                    data_dp_rank = i
                    break

        pipeline_id, dp_group_id = None, None
        # 兼容传统的同构策略
        if not self.ds_config.hetero:
            for cp in (cp_list):
                assert cp == cp_list[0], "homo setting should have the same cp degree"
            pipeline_id = all_devices.get_index(local_device) % (dcp_size * self.ds_config.gpus_per_stage) // self.ds_config.gpus_per_stage
            dp_group_id = pipeline_id // cp_list[0]
        # 异构策略
        else:
            # ---- hetero tp ----
            rank_to_device_mapping = {}
            if self.ds_config.rank_to_device_mapping is None:
                # 默认identity映射
                for idx in range(all_devices.num_devices):
                    rank_to_device_mapping[idx] = idx
            else:   
                rank_to_device_mapping = self.ds_config.rank_to_device_mapping
            unused_rank_list = self.ds_config.unused_rank
            if unused_rank_list is not None:
                for unused_rank in unused_rank_list:
                    if rank_to_device_mapping[unused_rank] == all_devices.get_index(local_device):
                        # 直接在此处返回不去run
                        return -1, -1, -1
            cur_rank_id = -1
            for rank_id, device_id in rank_to_device_mapping.items():
                if device_id == all_devices.get_index(local_device):
                    if cur_rank_id != -1:
                        assert False, "rank_to_device_mapping has duplicate keys"
                    cur_rank_id = rank_id
            assert cur_rank_id != -1, f"can't find device {all_devices.get_index(local_device)} in rank_to_device_mapping"
            # ---- hetero pipeline ----
            if self.ds_config.hetero_layers is None:
                # 默认均分stage
                pp = all_devices.num_devices // dcp_size // self.ds_config.gpus_per_stage
                hetero_stages = [pp for _ in range(dcp_size)]
            else:
                hetero_stages = [len(pipeline) for pipeline in self.ds_config.hetero_layers]
                assert len(hetero_stages) == dcp_size, f"len of hetero_stages should be equal to dcp={dcp_size}"
            accumulate_ranks = 0
            for i, stage_num in enumerate(hetero_stages):
                accumulate_ranks += stage_num * self.ds_config.gpus_per_stage
                if accumulate_ranks > cur_rank_id:
                    pipeline_id = i
                    break
            assert pipeline_id != None, "can't figure out pipeline num"
            accumulate_cp = 0
            for i, cp in enumerate(cp_list):
                accumulate_cp += cp
                if accumulate_cp > pipeline_id:
                    dp_group_id = i
                    break
            if not self.pretrain_config.packing:
                # 检测含有data的device的属性是否一致
                # dp_group_id和pipeline_id是每个rank都有的
                # data_dp_rank和data_dcp_rank是只有含data的rank有的
                if data_dp_rank != -1:
                    assert data_dp_rank == dp_group_id, f"data_dp_rank={data_dp_rank} should be equal to dp_group_id={dp_group_id}"
                if data_dcp_rank != -1:
                    assert data_dcp_rank == pipeline_id, f"data_dcp_rank={data_dcp_rank} should be equal to pipeline_id={pipeline_id}"
        
        # dp group内部的cp idx
        cp_id = pipeline_id - sum(cp_list[:dp_group_id])
        # seqlen_store = kv_store_client.register_dict('seqlen_store')
        
        accumulate_cp = cp_list[0]
        cp_cnt = 0
        for i, symbol in enumerate(self.model.config.multi_cp_group_symbol[strategy_id]):
            if i < accumulate_cp:
                symbol.set_data(cp_cnt)
            else:
                cp_cnt += 1
                accumulate_cp += cp_list[cp_cnt]
        
        if self.pretrain_config.packing:
            logging.info(
                f"{local_device}: " + \
                f"rank={all_devices.get_index(local_device)}, dp_size={dp_size}, dcp_size={dcp_size}, " + \
                f"dp_group_id={dp_group_id}, pipeline_id={pipeline_id}, cp_id={cp_id}"
            )
        else:
            logging.info(
                f"{local_device}: " + \
                f"rank={all_devices.get_index(local_device)}, dp_size={dp_size}, dcp_size={dcp_size}, " + \
                f"data_dp_rank={data_dp_rank}, data_dcp_rank={data_dcp_rank}, " + \
                f"dp_group_id={dp_group_id}, pipeline_id={pipeline_id}, cp_id={cp_id}"
            )

        logging.info(f"runtime cp list is {cp_list}")

        return cp_id, dp_group_id, pipeline_id

    def prepare_feed_dict(
        self,
        global_batch: Dict[str, np.ndarray],
        global_batch_size: int,
        cp_list: List[int],
        cp_id: int,
        dp_group_id: int,
        pipeline_id: int,
        strategy_id: int,
    ) -> Tuple[Dict[hetu.Tensor, np.ndarray], Dict[hetu.IntSymbol, int], int]:
        """
        Prepare feed dict for training.
        
        Args:
            global_batch: Batch of data to process.
            global_batch_size: Size of the global batch.
            cp_list: List of communication parallelism degrees.
            cp_id: Communication parallelism ID.
            dp_group_id: Data parallelism group ID.
            pipeline_id: Pipeline parallelism ID.
            strategy_id: ID of the parallel strategy.
            
        Returns:
            Tuple containing:
            - feed_dict: Dictionary mapping tensors to their input values.
            - int_symbol_dict: Dictionary mapping integer symbols to their values.
            - num_micro_batches: Number of micro-batches.
        """
        dp_size = len(cp_list)
        dcp_size = sum(cp_list)

        gbs_per_dp, num_micro_batches = None, None
        seq_len, seq_lens = None, None
        if not self.ds_config.hetero:
            gbs_per_dp = global_batch_size // dp_size
            if not self.pretrain_config.packing:
                assert gbs_per_dp % self.pretrain_config.micro_batch_size == 0, \
                    f'gbs_per_dp={gbs_per_dp} must be divided by mbs={self.pretrain_config.micro_batch_size}'
                assert self.pretrain_config.max_seq_length % cp_list[0] == 0, \
                    f'gsl={self.pretrain_config.max_seq_length} must be divided by cp={cp_list[0]}'
                
                num_micro_batches = gbs_per_dp // self.pretrain_config.micro_batch_size
                self.accumulate_micro_batch_num = [(i * gbs_per_dp // self.pretrain_config.micro_batch_size) for i in range(0, dp_size + 1)]
                seq_len = self.pretrain_config.max_seq_length // cp_list[0]
                seq_lens = [seq_len] * dcp_size
                inner_group_seq_lens = [seq_len] * cp_list[0]
            else:
                accumulate_seq_id = [(i * gbs_per_dp) for i in range(0, dp_size + 1)]
        elif not self.pretrain_config.packing:
            accumulate_micro_batch_num = [0,]
            if self.ds_config.micro_batch_num_list is None:
                # 默认均分micro batch
                num_micro_batches = global_batch_size // self.pretrain_config.micro_batch_size // dp_size
                for i in range(dp_size):
                    accumulate_micro_batch_num.append(accumulate_micro_batch_num[-1] + num_micro_batches)
            else:
                micro_batch_num_list = self.ds_config.micro_batch_num_list
                assert len(micro_batch_num_list) == dp_size, f"len of micro_batch_num_list should be equal to dp={dp_size}"
                num_micro_batches = micro_batch_num_list[dp_group_id]
                for i in range(dp_size):
                    accumulate_micro_batch_num.append(accumulate_micro_batch_num[-1] + micro_batch_num_list[i])
            self.accumulate_micro_batch_num = accumulate_micro_batch_num
            gbs_per_dp = self.pretrain_config.micro_batch_size * num_micro_batches
                
            if self.ds_config.seq_len_list is None:
                # 默认均分seq len
                seq_lens = []
                for cp in cp_list:
                    assert self.pretrain_config.max_seq_length % cp == 0, \
                        f'gsl={self.pretrain_config.max_seq_length} must be divided by cp={cp}'
                    seq_lens.extend([self.pretrain_config.max_seq_length // cp] * cp)
                inner_group_seq_lens = [seq_len] * cp_list[dp_group_id]
            else:
                seq_lens = self.ds_config.seq_len_list
                assert len(seq_lens) == dcp_size, f"len of seq_len_list should be equal to dcp={dcp_size}"
                inner_group_seq_lens = seq_lens[sum(cp_list[:dp_group_id]): sum(cp_list[:dp_group_id + 1])]
            seq_len = seq_lens[pipeline_id]
        else:
            # TODO: 需要给出具体切分global batch的策略
            # 目前均匀分配
            gbs_per_dp = global_batch_size // dp_size
            accumulate_seq_id = [(i * gbs_per_dp) for i in range(0, dp_size + 1)]
        
        logging.info(
            f"gbs = {global_batch_size}, mbs = {self.pretrain_config.micro_batch_size}, " + \
            f"num_micro_batches = {num_micro_batches}, seq_len = {seq_len}"
        )
        logging.info(f"runtime seq_lens is {seq_lens}")

        int_symbol_dict = {}
        if not self.pretrain_config.packing:
            inner_group_accumulate_seq_lens = []
            inner_group_accumulate_length = 0
            for length in inner_group_seq_lens:
                inner_group_accumulate_seq_lens.append(inner_group_accumulate_length) 
                inner_group_accumulate_length += length
            begin_seq_len = inner_group_accumulate_seq_lens[cp_id]
            begin_seq_id = self.accumulate_micro_batch_num[dp_group_id] * self.pretrain_config.micro_batch_size
            end_seq_id = self.accumulate_micro_batch_num[dp_group_id + 1] * self.pretrain_config.micro_batch_size
            assert end_seq_id - begin_seq_id == gbs_per_dp, "batch size mismatches"
            
            for i, symbol in enumerate(self.model.config.multi_seq_lens_symbol[strategy_id]):
                symbol.set_data(seq_lens[i])

            tokens = global_batch["input_ids"][begin_seq_id: end_seq_id, begin_seq_len: begin_seq_len + seq_len].reshape(-1) # [gbs_per_dp, seq_len]
            labels = global_batch["labels"][begin_seq_id: end_seq_id, begin_seq_len: begin_seq_len + seq_len].reshape(-1) # [gbs_per_dp, seq_len]
            feed_dict = {
                self.trainer_states.input_ids: tokens.astype(np.int64),
                self.trainer_states.lm_labels: labels.astype(np.int64),
            }
        else:
            # workaround: 目前只是为了测试CP + packing是否能正常运行
            # 直接按sorted_batch里的顺序分
            sorted_batch, sorted_len = get_sorted_batch_and_len(global_batch, self.tokenizer.pad_id)
            logging.debug(f"{self.comm_args.local_device}: {len(sorted_batch)} seqs sorted lens is {sorted_len}")
                
            # 为该dp group构建对应的bucket
            begin_seq_id = accumulate_seq_id[dp_group_id]
            end_seq_id = accumulate_seq_id[dp_group_id + 1]
            batch_indices = list(range(begin_seq_id, end_seq_id))
            self.model.config.max_seqlen_symbol.set_data(sorted_len[batch_indices[-1]] - 1) 
            # 获取input_bucket和label_bucket
            bucket = get_input_and_label_buckets(
                sorted_batch, 
                self.tokenizer.pad_id, 
                batch_indices, 
                self.pretrain_config.max_seq_length, 
                alignment=1, # packing后seqlen是1的整数倍 
                valid_alignment=cp_list[dp_group_id]*self.ds_config.gpus_per_stage # packing的每个original seqlen都是cp size * tp的整数倍
            )
            # 贪心packing
            bucket.pack_data()
            # packing后SYM方式切割得到各cp rank上的data
            bucket.generate_cp_pack_data(cp_list[dp_group_id])
                
            input_batch, label_batch = bucket.cp_packed_batch(cp_id)["input_ids"], bucket.cp_packed_batch(cp_id)["labels"]
            cu_seqlens_list = bucket.cp_packed_cu_seqlens_list()
            logging.info(f"{self.comm_args.local_device}: input shape list is {[input.shape for input in input_batch]}, cu_seqlens_list = {cu_seqlens_list}") # cu_seqlens_list w/o CP = {input_bucket.packed_cu_seqlens_list()}")
            seqlen_list = bucket.cp_packed_seqlen_list()
                
            # 设置symbol
            for enum_cp_id, enum_pipeline_id in enumerate(range(sum(cp_list[:dp_group_id]), sum(cp_list[:dp_group_id+1]))):
                enum_symbol = self.model.config.multi_seq_lens_symbol[strategy_id][enum_pipeline_id]
                int_symbol_dict[enum_symbol] = seqlen_list[enum_cp_id]
            '''
            # 每个cp idx为0的负责把该cp的seqlen_list放到kv store中
            if cp_id == 0:
                seqlen_store.put(f"dp_group_{dp_group_id}", seqlen_list)
            ht.global_comm_barrier_rpc()
            # 设置symbol
            int_symbol_dict = {}
            for enum_dp_group_id in range(dp_size):
                for enum_cp_id in range(cp_list[enum_dp_group_id]):
                    enum_pipeline_id = sum(cp_list[:enum_dp_group_id]) + enum_cp_id
                    enum_symbol = config.multi_seq_lens_symbol[strategy_id][enum_pipeline_id]
                    enum_seqlen_list = seqlen_store.get(f"dp_group_{enum_dp_group_id}")
                    assert enum_cp_id in enum_seqlen_list, f"cannot find {enum_cp_id} in {enum_seqlen_list}"
                    int_symbol_dict[enum_symbol] = enum_seqlen_list[enum_cp_id]
                    print(f"pipeline {enum_pipeline_id} (in dp group {enum_dp_group_id}) micro batches corresponded seqlen list is {enum_seqlen_list[enum_cp_id]}")
            '''
                
            # 构造feed dict
            # 形状为num_micro_batches * [seq_len]格式的
            num_micro_batches = len(input_batch)
            input_list = [micro_batch.astype(np.int64) for micro_batch in input_batch] 
            label_list = [micro_batch.astype(np.int64) for micro_batch in label_batch] 
            feed_dict = {
                self.trainer_states.input_ids: input_list,
                self.trainer_states.lm_labels: label_list,
            }
            for i in range(self.model.config.num_hidden_layers):
                feed_dict[self.model.config.cu_seqlens_list[i]] = [x.astype(np.int32) for x in cu_seqlens_list]
        
        return feed_dict, int_symbol_dict, num_micro_batches

    def train_data_iterator(self):
        """
        Get an iterator for the training data.
        
        This method creates and returns a data iterator from the training data loader.
        If an iterator already exists, it returns the existing one.
        
        Returns:
            Iterator over the training dataset.
        """
        dataloader = self.get_train_data_loader()
        train_data_iter = iter(dataloader)
        self.data_iter = train_data_iter
        return train_data_iter

    def train(
        self,
        cp_list: Optional[List[int]] = None,
        strategy_id: Optional[int] = 0,
        run_level: int = hetu.run_level("update"),
    ):
        """
        Execute the training process.
        
        This method handles the main training loop, processing batches, computing loss,
        and updating model parameters according to the specified parallel strategy.
        
        Args:
            cp_list: Optional list of communication parallelism degrees.
                   If None, uses the default from ds_config.
            strategy_id: ID of the parallel strategy to use.
            run_level: The execution level (update, grad, or forward).
        
        Raises:
            AssertionError: If flash attention is not enabled when using symbolic shapes.
        """
        assert self.model.config.use_flash_attn == True, "symbolic shape can only used when flash attn is on for now"
        
        label_device_group_union = self.comm_args.label_dg_hierarchy[strategy_id]
        _, label_device_group = get_dg_from_union(self.comm_args.local_device, label_device_group_union)
        
        if cp_list is None:
            cp_list = self.ds_config.cp_list
        
        cp_id, dp_group_id, pipeline_id = self.extract_strategy(
            cp_list,
            strategy_id,
        )
        
        if cp_id == -1 and dp_group_id == -1 and pipeline_id == -1:
            logging.info(f"{self.comm_args.local_device}: Device is marked as unused, skipping training")
            return

        train_iter = self.train_data_iterator() if self.data_iter is None else self.data_iter
        
        for step in range(self.pretrain_config.steps):
            # load data for each dp
            global_batch = next(train_iter)
            global_batch_size = len(global_batch["input_ids"])
            global_labels = global_batch["labels"]
            num_tokens = np.sum(global_labels != IGNORE_INDEX)
            
            feed_dict, int_symbol_dict, num_micro_batches = self.prepare_feed_dict(
                global_batch,
                global_batch_size,
                cp_list,
                cp_id,
                dp_group_id,
                pipeline_id,
                strategy_id,
            )

            start_time = time.time()
            if self.pretrain_config.torch_profile and (
                step >= self.pretrain_config.start_profile_step and
                step < self.pretrain_config.end_profile_step
            ):
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    results, self.trainer_states.accumulate_token_num = self._train(
                        feed_dict,
                        int_symbol_dict,
                        num_micro_batches,
                        strategy_id,
                        num_tokens,
                        run_level,
                    )
                    if step == self.pretrain_config.end_profile_step - 1:
                        os.makedirs(self.pretrain_config.profile_save_path, exist_ok=True)
                        prof.export_chrome_trace(f"{self.pretrain_config.profile_save_path}/trace_{self.comm_args.local_device}.json")
            else:
                results, self.trainer_states.accumulate_token_num = self._train(
                    feed_dict,
                    int_symbol_dict,
                    num_micro_batches,
                    strategy_id,
                    num_tokens,
                    run_level,
                )
            end_time = time.time()
            
            self.consumed_samples += global_batch_size
            if run_level in (hetu.run_level("update"), hetu.run_level("grad")):
                if label_device_group != None:
                    self.trainer_states.loss_sum += results[0].numpy(force=True).sum()
            
            if run_level == hetu.run_level("update"):
                if label_device_group != None:
                    logging.debug(f"step {step}: loss sum is {self.trainer_states.loss_sum}, accumulate token num is {self.trainer_states.accumulate_token_num}")
                    loss_out = self.trainer_states.loss_sum / self.trainer_states.accumulate_token_num
                    
                    # Store loss for plotting if enabled
                    if self.pretrain_config.plot_loss:
                        self.loss_history.append(loss_out)
                        self.step_history.append(step)
                        self.plot_training_loss()

                    self.trainer_states.loss_sum = 0.0
                    logging.info(f"{self.comm_args.local_device}: [Epoch {self.epoch}] (step {step}, consumed_samples = {self.consumed_samples}): loss = {loss_out:.3f}, time = {end_time - start_time:.4f}")

    def save_model(
        self,
        output_dir: Optional[str] = None,
        save_dtype: hetu.dtype = hetu.float32,
    ):
        """
        Save the trained model to disk.
        
        Args:
            output_dir: Directory to save the model to. If None, uses the path from 
                       pretrain_config.output_dir.
            save_dtype: Data type to use when saving model weights.
        """
        if output_dir is None:
            output_dir = self.pretrain_config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir, dtype=save_dtype)
        conf = OmegaConf.structured(self.pretrain_config)
        OmegaConf.save(conf, f"{output_dir}/training_config.yaml")

    def plot_training_loss(self):
        """
        Plot and save the training loss history.
        
        This method creates a plot of the training loss over time and saves it to disk.
        The plot is only generated on the first device that has labels to avoid duplicate plots.
        Requires matplotlib to be installed.
        """
        if not self.pretrain_config.plot_loss:
            return
            
        label_device_group_union = self.comm_args.label_dg_hierarchy[0]
        _, label_device_group = get_dg_from_union(self.comm_args.local_device, label_device_group_union)
        
        # Only plot on the first device that has labels
        if label_device_group is None or label_device_group.get_index(self.comm_args.local_device) != 0:
            return

        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
            
        if self.fig is None:
            self.fig, self.ax = plt.figure(figsize=(10, 6)), None
            
        plt.clf()
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(self.step_history, self.loss_history, 'b-', label='Training Loss')
        self.ax.set_xlabel('Training Steps')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Training Loss over Time')
        self.ax.legend(loc='upper right')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Force x-axis to use integer ticks for steps
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        
        # Create the directory if it doesn't exist
        output_dir = self.pretrain_config.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # Save the plot
        plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=300)
        
        # Update plot every 10 steps or as configured
        update_freq = getattr(self.pretrain_config, 'plot_update_freq', 10)
        if len(self.step_history) % update_freq == 0:
            plt.pause(0.1)
