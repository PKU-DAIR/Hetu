import os
import signal
import math
import hetu as ht
from hetu_llama import LLaMALMHeadModel
from hetu.utils.parallel import config2ds
from gpt_config import GPTConfig
from data_utils import GPTJsonDataset, get_mask_and_position_ids, build_pretraining_data_loader
import numpy as np
import time
import argparse
import json
import socket
import pynvml
import ast
from queue import Queue
import ptvsd

local_device = None
all_devices = None

def distributed_init(args):
    global local_device, all_devices
    hostname = socket.gethostname()
    os.environ['HETU_LOCAL_HOSTNAME'] = hostname
    ht.init_comm_group(args.ngpus, server_address=args.server_addr+":"+args.server_port)
    local_device = ht.local_device()
    all_devices = ht.global_device_group()
    if local_device.index == 0:
        print(f'local_device: {local_device}, all_devices: {all_devices}')
    # used for debug
    # ptvsd.enable_attach(address =('127.0.0.1', 4000 + all_devices.get_index(local_device)))
    # ptvsd.wait_for_attach()

def read_ds_parallel_config(args):
    # read ds_parallel_config from json file
    print(f'{local_device}: load ds_parallel_config from: {args.ds_parallel_config}')
    config_paths = args.ds_parallel_config.split(',')
    assert len(config_paths) == args.num_strategy, \
      f'ds_parallel_config num should equal to num_strategy {args.num_strategy}'
    ds_parallel_configs = []
    for config_path in config_paths:
        ds_parallel_config = json.load(open(config_path, 'r'))
        zero = ds_parallel_config['zero']
        # assign zero to all variables
        config_queue = Queue()
        for value in ds_parallel_config.values():
            config_queue.put(value)
        while (not config_queue.empty()):
            config = config_queue.get()
            if type(config) == dict:
                if 'type' in config:
                    if config['type'] == 'variable' and 'zero' not in config:
                        config['zero'] = zero
                else:
                    for value in config.values():
                        config_queue.put(value)
        ds_parallel_configs.append(ds_parallel_config)
    return ds_parallel_configs

def get_multi_ds_parallel_config(ds_parallel_configs, module_name, _range=-1):
    multi_ds_parallel_config = []
    for ds_parallel_config in ds_parallel_configs:
        config_queue = Queue()
        config_queue.put(ds_parallel_config)
        while (not config_queue.empty()):
            config = config_queue.get()
            if module_name in config:
                multi_ds_parallel_config.append(config[module_name])
                break
            else:
                for value in config.values():
                    if type(value) == dict:
                        if "range" in value and (_range < value["range"][0] or _range > value["range"][-1]):
                            continue
                        config_queue.put(value)
    assert len(multi_ds_parallel_config) == len(ds_parallel_configs), 'ds_parallel_configs parse error!'
    return multi_ds_parallel_config

def parse_multi_ds_parallel_config(ds_parallel_configs, module_name, _range=-1):
    ds_hierarchy = []
    dg_hierarchy = []
    multi_ds_parallel_config = get_multi_ds_parallel_config(ds_parallel_configs, module_name, _range)
    for ds_parallel_config in multi_ds_parallel_config:
        ds_union, dg_union = config2ds(ds_parallel_config)
        ds_hierarchy.append(ds_union)
        dg_hierarchy.append(dg_union)
    return ds_hierarchy, dg_hierarchy

def train_dataset_provider(args):
    """Build train dataset."""
    train_dataset = GPTJsonDataset(
        json_file=args.json_file,
        key=args.json_key,
        max_seq_len=args.seq_length,
        vocab_file=args.vocab_file,
        merge_file=args.merge_file)
    return train_dataset

def get_position_ids(gbs_per_dp, seq_len): 
    position_ids = np.arange(0, seq_len, dtype=np.int64) # [1, seq_len]
    position_ids = np.tile(position_ids, [gbs_per_dp, 1]) # [dp_size, seq_len]
    return position_ids

def train_data_iterator(dataset, consumed_samples, mbs, dp_rank, dp_size):
    # print(f'new dataloader: consumed_samples = {consumed_samples}')
    train_dataloader = build_pretraining_data_loader(dataset, consumed_samples, mbs, dp_rank, dp_size)
    train_data_iterator = iter(train_dataloader)
    return train_data_iterator

def profile_memory(device_index = 0):
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    # 查询设备名称
    device_name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
    print("Device", device_index, ":", device_name)
    # 查询显存信息
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_memory = memory_info.total / 1024 / 1024  # 总显存大小（MB）
    used_memory = memory_info.used / 1024 / 1024   # 已使用的显存大小（MB）
    free_memory = memory_info.free / 1024 / 1024   # 剩余的显存大小（MB）
    print("Total Memory:", total_memory, "MiB")
    print("Used Memory:", used_memory, "MiB")
    print("Free Memory:", free_memory, "MiB")
    
def get_dg_from_union(device, dg_union):
    for i, dg in enumerate(dg_union):
        if dg.contains(device):
            return i, dg
    return None, None

def pretrain(args):
    ds_parallel_configs = read_ds_parallel_config(args)

    config = GPTConfig(vocab_size=args.vocab_size, 
                       n_positions=args.seq_length,
                       n_ctx=args.seq_length,
                       n_embd=args.hidden_size,
                       ffn_hidden_size=args.ffn_hidden_size,
                       n_layer=args.num_hidden_layers, 
                       n_head=args.num_attention_heads, 
                       seq_len=args.seq_length,
                       resid_pdrop=args.dropout_prob,
                       embd_pdrop=args.dropout_prob,
                       attn_pdrop=args.dropout_prob,
                       activation_function=args.hidden_act,
                       global_batch_size=args.global_batch_size,
                       use_flash_attn=args.use_flash_attn,
                       )
    
    # simple check for gpt blocks range
    ranges = []
    for _, block_config in ds_parallel_configs[0]['gpt']['blocks'].items():
        ranges.append(block_config['range'])
    assert ranges[0][0] == 0 and ranges[-1][-1] == config.num_hidden_layers-1, \
        f"gpt blocks range: {ranges} is conflict with num_hidden_layers: {config.num_hidden_layers}!"

    # Hetu model definition
    model = LLaMALMHeadModel(config=config, ds_parallel_configs=ds_parallel_configs)
    
    input_ds_hierarchy, input_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'input')
    label_ds_hierarchy, label_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'label')
    # print(f'input_ds: {input_ds}, label_ds: {label_ds}')
    
    # mbs_times_dp = args.global_batch_size // config.num_micro_batches
    dp_size = input_ds_hierarchy[0].get(0).get_dim(0)
    mbs_times_dp = args.micro_batch_size * dp_size
        
    # todo: assign multi dg_hierarchy, and instantiate only one placement_group
    input_ids = ht.parallel_placeholder(ht.int64, global_shape=[mbs_times_dp, config.seq_len], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='input_ids')
    # position_ids = ht.parallel_placeholder(ht.int64, global_shape=[mbs_times_dp, config.seq_len], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='position_ids')
    # token_type_ids = ht.parallel_placeholder(ht.int64, global_shape=[mbs_times_dp, config.seq_len], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='token_type_ids')
    # attention_mask = ht.parallel_placeholder(ht.float32, global_shape=[mbs_times_dp, config.seq_len], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='attention_mask')
    masked_lm_labels = ht.parallel_placeholder(ht.int64, global_shape=[mbs_times_dp, config.seq_len], ds_hierarchy=label_ds_hierarchy, device_group_hierarchy=label_dg_hierarchy, name='masked_lm_labels')

    config.mbs_times_dp_symbol = ht.IntSymbol(mbs_times_dp)
    config.seq_len_symbol = input_ids.symbolic_shape[1]

    # print(f'{local_device}: build model begin...')
    loss = model(input_ids=input_ids,
                 # position_ids=position_ids,
                 # attention_mask=attention_mask,
                 # token_type_ids=token_type_ids,
                 labels=masked_lm_labels)
    # print(f'{local_device}: build model end...')

    loss_mean = loss

    print(f'{local_device}: optimizer minimize begin...')
    # opt = ht.SGDOptimizer(lr=args.lr, momentum = 0.0)
    opt = ht.AdamOptimizer(lr=args.lr)
    train_op = opt.minimize(loss_mean)
    print(f'{local_device}: optimizer minimize end...')
    
    print(f'{local_device}: build dataset begin...')
    train_dataset = train_dataset_provider(args)
    print(f'{local_device}: build dataset end...')
    # return

    def run_plan(epoch = 0,
                 steps = args.steps,
                 consumed_samples = 0,
                 global_batch_size = args.global_batch_size,
                 micro_batch_size = args.micro_batch_size,
                 seq_len = args.seq_length,
                 strategy_id = 0, 
                 run_level = 0):       
        if global_batch_size != args.global_batch_size or seq_len != args.seq_length:
            assert config.use_flash_attn == True, "symbolic shape can only used when flash attn is on for now"
        
        input_ds_union = input_ds_hierarchy[strategy_id]
        input_device_group_union = input_dg_hierarchy[strategy_id]
        label_ds_union = label_ds_hierarchy[strategy_id]
        label_device_group_union = label_dg_hierarchy[strategy_id]
        assert input_ds_union.hetero_dim == 0 or input_ds_union.hetero_dim == -3, "input hetero dim unsupported"
        assert label_ds_union.hetero_dim == 0 or label_ds_union.hetero_dim == -3, "label hetero dim unsupported"

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
            # 其余local device都在中间stage上
            dup_group_num = len(input_device_group_union)
            if input_ds_union.hetero_dim == -3:
                dup_group_num = input_ds_union.get(0).get_dim(0)
            # raise RuntimeError(f"device {local_device} not in input_device_group or label_device_group!")

        dp_rank = dup_group_idx
        dp_size = dup_group_num
        gbs_per_dp = global_batch_size // dp_size
        mbs_times_dp = micro_batch_size * dp_size
        assert global_batch_size % mbs_times_dp == 0, \
            f'gbs {global_batch_size} must could be divided by mbs {micro_batch_size} * dp {dp_size}'
        num_micro_batches = global_batch_size // mbs_times_dp
        
        # heterogenous tp
        rank_to_device_mapping = ast.literal_eval(args.rank_to_device_mapping)
        unused_rank_list = ast.literal_eval(args.unused_rank)
        for unused_rank in unused_rank_list:
            if rank_to_device_mapping[unused_rank] == all_devices.get_index(local_device):
                pass
                # TODO: 不运行
                # 目前是进入到exec graph阶段发现不在pipeline中才不运行的
        
        # heterogenous pipeline 
        if args.hetero_data:
            curr_rank_id = -1
            for rank_id, device_id in rank_to_device_mapping.items():
                if device_id == all_devices.get_index(local_device):
                    if curr_rank_id != -1:
                        assert False, "rank_to_device_mapping has duplicate keys"
                    curr_rank_id = rank_id
            assert curr_rank_id != -1, f"can't find device {all_devices.get_index(local_device)} in rank_to_device_mapping"
            if args.hetero_stages == "[]":
                pp = all_devices.num_devices // dp_size // args.hetero_stage_gpus
                hetero_stages = [pp for _ in range(dp_size)]
            else:
                hetero_stages = ast.literal_eval(args.hetero_stages)
            accumulate_ranks = 0
            hetero_pipeline_num = -1
            for i, stage_num in enumerate(hetero_stages):
                accumulate_ranks += stage_num * args.hetero_stage_gpus
                if accumulate_ranks > curr_rank_id:
                    hetero_pipeline_num = i
                    break
            # assert hetero_pipeline_num != -1, "can't figure out pipeline num"
            # -1说明是没有被用到的靠后的rank
            # hetero_pipeline_num = curr_rank_id % (dp_size * args.hetero_stage_gpus) // args.hetero_stage_gpus
            # adjust micro_batch_size
            '''
            batch_ratio = 8
            if hetero_pipeline_num == 0:
                micro_batch_size = int(global_batch_size // num_micro_batches / batch_ratio)
            else:
                micro_batch_size = global_batch_size // num_micro_batches - int(global_batch_size // num_micro_batches / batch_ratio)
            '''
            # adjust num_micro_batches
            '''
            normal_micro_batches = args.normal_micro_batches
            if hetero_pipeline_num == 0:
                num_micro_batches = global_batch_size // micro_batch_size - normal_micro_batches * (dp_size - 1)
                assert num_micro_batches > 0, f"straggler num_micro_batches should > 0, but find it is {num_micro_batches}"
            else:
                num_micro_batches = normal_micro_batches
            '''
            # hetero_pipeline_num是-1的话data随便取都无关紧要
            num_micro_batches = ast.literal_eval(args.micro_batch_num_list)[hetero_pipeline_num]
            # re-assign
            gbs_per_dp = micro_batch_size * num_micro_batches
            mbs_times_dp = micro_batch_size * dp_size
                
        config.mbs_times_dp_symbol.set_data(mbs_times_dp)
        config.seq_len_symbol.set_data(seq_len)
        print(f'{local_device}: dp_rank={dp_rank}, dp_size={dp_size}, gbs={global_batch_size}, mbs={micro_batch_size}, num_micro_batches={num_micro_batches}')

        # if dp_size * mbs changes, then should use the new dataloader
        # start_time = time.time()
        if dp_rank != -1 and hetero_pipeline_num != -1:
            train_iter = train_data_iterator(train_dataset, consumed_samples, micro_batch_size, dp_rank, dp_size) # need cache?
        else:
            train_iter = None
        # end_time = time.time()
        # print(f'{local_device}: create dataloader cost {end_time - start_time} s')

        # profile_memory()

        for step in range(steps):
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
                _attention_mask, _position_ids = get_mask_and_position_ids(tokens, train_dataset.encoder.pad_id())
                _token_type_ids = np.zeros([gbs_per_dp, seq_len])

                feed_dict = {
                    input_ids: tokens.astype(np.int64),
                    # position_ids: _position_ids.astype(np.int64), 
                    # token_type_ids: _token_type_ids.astype(np.int64),
                    # attention_mask: _attention_mask.astype(np.int64),
                    masked_lm_labels: labels.astype(np.int64),
                }
            else: # fake data; feed_dict={} will cause segment fault?
                feed_dict = {
                    input_ids: np.zeros([gbs_per_dp, seq_len]).astype(np.int64),
                    # position_ids: get_position_ids(gbs_per_dp, seq_len).astype(np.int64), 
                    # token_type_ids: np.zeros([gbs_per_dp, seq_len]).astype(np.int64),
                    # attention_mask: np.zeros([gbs_per_dp, seq_len]).astype(np.float32),
                    masked_lm_labels: np.zeros([gbs_per_dp, seq_len]).astype(np.int64),
                }
            # print(f"{local_device}: strategy_id = {strategy_id}, gbs = {global_batch_size}, mbs = {micro_batch_size}, seq_len = {seq_len} run begin")
            start_time = time.time()
            try:
                results = train_op.graph.run(loss_mean, 
                                                [loss_mean, train_op], 
                                                feed_dict = feed_dict, 
                                                num_micro_batches = num_micro_batches, 
                                                cur_strategy_id = strategy_id,
                                                run_level = run_level,
                                                grad_scale = 1.0)
            except RuntimeError as e:
                print(e)
                with open("./logs/exception.txt", 'w') as file:
                    print("device index:", local_device.index, file=file)
                    print(e, file=file)
                os.killpg(0, signal.SIGTERM)
            end_time = time.time()
            consumed_samples += global_batch_size
            # print(f"{local_device}: strategy_id = {strategy_id}, gbs = {global_batch_size}, mbs = {micro_batch_size}, seq_len = {seq_len} run end, consumed_samples = {consumed_samples}")
            # NOTE: 实际上应该扫描一次alloc到update之间的所有数据
            # grad_scale = 当前run的数据的batch_size除以总的这之间run加起来的batch_size
            if run_level == ht.run_level("update"):
                if label_device_group != None:
                    loss_out = results[0].numpy(force=True).mean()
                    print(f"{local_device}: [Epoch {epoch}] (step {step}, consumed_samples = {consumed_samples}): loss = {loss_out:.3f}, time = {end_time - start_time:.4f}")
                # switch只跑一个iter
                return consumed_samples
        return consumed_samples
            
    # 热切换
    def test_switch(): 
        consumed_samples = 0 # should be reset when run next epoch
        args.hetero_data = True
        args.hetero_stage_gpus = args.before_hetero_stage_gpus
        args.hetero_stages = args.before_hetero_stages
        args.micro_batch_num_list = args.before_micro_batch_num_list
        args.rank_to_device_mapping = args.before_rank_to_device_mapping
        args.unused_rank = args.before_unused_rank
        consumed_samples = run_plan(epoch = 0,
                                    steps = args.steps,
                                    consumed_samples = consumed_samples, 
                                    global_batch_size = args.global_batch_size, 
                                    micro_batch_size = args.micro_batch_size, 
                                    seq_len = args.seq_length, 
                                    strategy_id = 0, 
                                    run_level = ht.run_level("update"))
        args.hetero_stage_gpus = args.after_hetero_stage_gpus
        args.hetero_stages = args.after_hetero_stages
        args.micro_batch_num_list = args.after_micro_batch_num_list
        args.rank_to_device_mapping = args.after_rank_to_device_mapping
        args.unused_rank = args.after_unused_rank
        consumed_samples = run_plan(epoch = 0,
                                    steps = args.steps,
                                    consumed_samples = consumed_samples, 
                                    global_batch_size = args.global_batch_size, 
                                    micro_batch_size = args.micro_batch_size, 
                                    seq_len = args.seq_length, 
                                    strategy_id = 1, 
                                    run_level = ht.run_level("update"))
    
    test_switch()

if __name__ == '__main__':
    print("Run hetu single switch training")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hetero_pipeline", action="store_true", help="use heterogenous pipeline."
    )
    parser.add_argument(
        "--hetero_data", action="store_true", help="use heterogenous data for each heterogenous pipeline."
    )
    parser.add_argument(
        '--before_hetero_stage_gpus', type=int, default=2, help='num of gpus of a single stage'
    )
    parser.add_argument(
        '--after_hetero_stage_gpus', type=int, default=2, help='num of gpus of a single stage'
    )
    parser.add_argument(
        '--before_hetero_stages', type=str, default="[]", help='hetero stages.'
    )
    parser.add_argument(
        '--after_hetero_stages', type=str, default="[]", help='hetero stages.'
    )
    parser.add_argument(
        "--before_micro_batch_num_list", type=str, help='micro batch num list.'
    )
    parser.add_argument(
        "--after_micro_batch_num_list", type=str, help='micro batch num list.'
    )
    parser.add_argument(
        "--before_rank_to_device_mapping", type=str, help='rank to device mapping.'
    )
    parser.add_argument(
        "--after_rank_to_device_mapping", type=str, help='rank to device mapping.'
    )
    parser.add_argument(
        "--before_unused_rank", type=str, help='unused rank.'
    )
    parser.add_argument(
        "--after_unused_rank", type=str, help='unused rank.'
    )
    parser.add_argument(
        '--gpu_id', type=int, default=0, help='Id of GPU to run.'
    )
    parser.add_argument(
        "--ds_parallel_config", default="ds_parallel_config/dp2_tp2_pp2.json", type=str, help="ds parallel config json file"
    )
    parser.add_argument(
        "--num_strategy", type=int, default=1, help="multi ds num"
    )
    parser.add_argument(
        "--global_batch_size", type=int, default=64, help="Training batch size global"
    )
    parser.add_argument(
        "--micro_batch_size", type=int, default=2, help="Training batch size each micro batch"
    )
    parser.add_argument(
        "--dataset", type=str, default='wikicorpus_en', help="Dataset used to train."
    )
    parser.add_argument(
        "--json_file", type=str, help='data json format file path'
    )
    parser.add_argument(
        "--json_key", type=str, help='json key for tokens'
    )
    parser.add_argument(
        "--vocab_file", type=str, help='gpt vocab file path'
    )
    parser.add_argument(
        "--merge_file", type=str, help='gpt merge file path'
    )
    parser.add_argument(
        "--vocab_size", type=int, default=30522, help="Total number of vocab"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
    )
    parser.add_argument(
        "--ffn_hidden_size", type=int, default=-1, help="FFN hidden size of transformer model",
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=12, help="Number of layers"
    )
    parser.add_argument(
        "-a",
        "--num_attention_heads",
        type=int,
        default=12,
        help="Number of attention heads",
    )
    parser.add_argument(
        "-s", "--seq_length", type=int, default=128, help="Maximum sequence len"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=4, help="Number of epochs"
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="Number of steps for each epoch",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate of adam"
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )
    parser.add_argument(
        "--hidden_act", type=str, default='gelu', help="Hidden activation to use."
    )
    parser.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout rate."
    )
    parser.add_argument(
        "--use_flash_attn", action="store_true", help="Use Flash Attention."
    )    
    parser.add_argument(
        "--bf16", action="store_true", help="Use bfloat16."
    )
    parser.add_argument(
        "--server_addr", type=str, default='127.0.0.1', help="server's address"
    )
    parser.add_argument(
        "--server_port", type=str, default='23457', help="server's port"
    ) 
    parser.add_argument(
        "--ngpus", type=int, default=8, help="num of gpus"
    ) 
    args = parser.parse_args()
    pynvml.nvmlInit()
    print("Hetu distributed init")
    distributed_init(args)
    with ht.graph("define_and_run", num_strategy=args.num_strategy):
        if args.bf16:
            precision = "ht.bfloat16"
        else:
            precision = "ht.float32"
        print(f'{local_device}: use precision {precision}')
        with ht.autocast(eval(precision)):            
            pretrain(args)
            print(f'{local_device}: train hetu ds parallel end...')