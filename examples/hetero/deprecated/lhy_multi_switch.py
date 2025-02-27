import os
import hetu as ht
from hetu_gpt_multi_ds_parallel_symbolic import GPTLMHeadModel
from hetu.utils.parallel import config2ds, get_device_index
from gpt_config import GPTConfig
from data_utils import GPTJsonDataset, get_mask_and_position_ids, build_pretraining_data_loader
import numpy as np
import time
import argparse
import json
import socket
import pynvml
from queue import Queue

local_device = None
all_devices = None

def distributed_init(use_two_node: bool = False):
    if use_two_node:
        hostname = socket.gethostname()
        if hostname == 'job-4e4cb411-1139-4f15-b221-5a30f1760a2b-master-0':
            os.environ['HETU_LOCAL_HOSTNAME'] = 'a100-0'
        elif hostname == 'job-4e4cb411-1139-4f15-b221-5a30f1760a2b-worker-0':
            os.environ['HETU_LOCAL_HOSTNAME'] = 'a100-1'
        else:
            raise ValueError(f"Unknown hostname: {hostname}")

    global local_device, all_devices
    ht.init_comm_group(8)
    local_device = ht.local_device()
    all_devices = ht.global_device_group()
    if local_device.index == 0:
        print(f'local_device: {local_device}, all_devices: {all_devices}')

def read_ds_parallel_config(args):
    # read ds_parallel_config from json file
    print(f'{local_device}: load ds_parallel_config from: {args.ds_parallel_config}')
    config_paths = args.ds_parallel_config.split(',')
    assert len(config_paths) == args.num_strategy, \
      f'ds_parallel_config num should equal to num_strategy {args.num_strategy}'
    ds_parallel_configs = []
    for config_path in config_paths:
        ds_parallel_config = json.load(open(config_path, 'r'))
        # ds_parallel_config = json.load(open('./ds_parallel_config/dp2_tp2_pp2.json', 'r'))
        # ds_parallel_config = json.load(open('./ds_parallel_config/dp2_tp4.json', 'r'))
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
    multi_ds = []
    device_groups = []
    multi_ds_parallel_config = get_multi_ds_parallel_config(ds_parallel_configs, module_name, _range)
    for ds_parallel_config in multi_ds_parallel_config:
        ds, device_group = config2ds(ds_parallel_config)
        multi_ds.append(ds)
        device_groups.append(device_group)
    return multi_ds, device_groups

def train_dataset_provider(args):
    """Build train dataset."""
    train_dataset = GPTJsonDataset(
        json_file=args.json_file,
        key=args.json_key,
        max_seq_len=args.seq_length,
        vocab_file=args.vocab_file,
        merge_file=args.merge_file)
    return train_dataset

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

def pretrain(args):
    ds_parallel_configs = read_ds_parallel_config(args)
    num_epochs = args.epochs
    lr = args.lr

    config = GPTConfig(vocab_size=args.vocab_size, 
                       n_positions=args.seq_length,
                       n_ctx=args.seq_length,
                       n_embd=args.hidden_size,
                       n_layer=args.num_hidden_layers, 
                       n_head=args.num_attention_heads, 
                       seq_len=args.seq_length,
                       resid_pdrop=args.dropout_prob,
                       embd_pdrop=args.dropout_prob,
                       attn_pdrop=args.dropout_prob,
                       activation_function=args.hidden_act,
                       global_batch_size=args.global_batch_size,
                       num_micro_batches=args.num_micro_batches,
                       use_flash_attn=args.use_flash_attn,
                       )
    
    # simple check for gpt blocks range
    ranges = []
    for _, block_config in ds_parallel_configs[0]['gpt']['blocks'].items():
        ranges.append(block_config['range'])
    assert ranges[0][0] == 0 and ranges[-1][-1] == config.num_hidden_layers-1, \
        f"gpt blocks range: {ranges} is conflict with num_hidden_layers: {config.num_hidden_layers}!"

    # Hetu model definition
    model = GPTLMHeadModel(config=config, ds_parallel_configs=ds_parallel_configs)
    
    input_multi_ds, input_device_groups = parse_multi_ds_parallel_config(ds_parallel_configs, 'input')
    label_multi_ds, label_device_groups = parse_multi_ds_parallel_config(ds_parallel_configs, 'label')
    # print(f'input_ds: {input_ds}, label_ds: {label_ds}')
    
    mbs_times_dp = config.global_batch_size // config.num_micro_batches
        
    # todo: assign multi device_groups, and instantiate only one placement_group
    input_ids = ht.parallel_placeholder(ht.int64, global_shape=[mbs_times_dp, config.seq_len], multi_ds=input_multi_ds, device_groups=input_device_groups, name='input_ids')
    position_ids = ht.parallel_placeholder(ht.int64, global_shape=[mbs_times_dp, config.seq_len], multi_ds=input_multi_ds, device_groups=input_device_groups, name='position_ids')
    token_type_ids = ht.parallel_placeholder(ht.int64, global_shape=[mbs_times_dp, config.seq_len], multi_ds=input_multi_ds, device_groups=input_device_groups, name='token_type_ids')
    attention_mask = ht.parallel_placeholder(ht.float32, global_shape=[mbs_times_dp, config.seq_len], multi_ds=input_multi_ds, device_groups=input_device_groups, name='attention_mask')
    masked_lm_labels = ht.parallel_placeholder(ht.int64, global_shape=[mbs_times_dp, config.seq_len], multi_ds=label_multi_ds, device_groups=label_device_groups, name='masked_lm_labels')

    config.mbs_times_dp_symbol = ht.IntSymbol(mbs_times_dp)
    config.seq_len_symbol = input_ids.symbolic_shape[1]

    # print(f'{local_device}: build model begin...')
    loss = model(input_ids=input_ids,
                 position_ids=position_ids,
                 attention_mask=attention_mask,
                 token_type_ids=token_type_ids,
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

    def run_plan(consumed_samples = 0,
                 global_batch_size = config.global_batch_size,
                 micro_batch_size = 2,
                 seq_len = config.seq_len,
                 strategy_id = 0, 
                 run_level = 0):       
        if global_batch_size != config.global_batch_size or seq_len != config.seq_len:
            assert config.use_flash_attn == True, "symbolic shape can only used when flash attn is on for now"
        
        input_ds = input_multi_ds[strategy_id]
        input_device_group = input_device_groups[strategy_id]
        label_ds = label_multi_ds[strategy_id]
        label_device_group = label_device_groups[strategy_id]

        # device in same dp_group will read the same batch data
        dup_group_idx, dup_group_num = -1, -1
        if input_device_group.contains(local_device):
            local_device_idx = input_device_group.get_index(local_device)
            dup_group_idx = input_ds.get_dup_group_index(local_device_idx)
            dup_group_num = input_ds.get_dim(0)
        elif label_device_group.contains(local_device):
            local_device_idx = label_device_group.get_index(local_device)
            dup_group_idx = label_ds.get_dup_group_index(local_device_idx)
            dup_group_num = label_ds.get_dim(0)
        else:
            raise RuntimeError(f"device {local_device} not in input_device_group or label_device_group!")
        # print(f'local deivce: {local_device}, local_device_idx: {local_device_idx}, dup_group_idx: {dup_group_idx}, dup_group_num: {dup_group_num}')

        dp_rank = dup_group_idx
        dp_size = dup_group_num
        gbs_per_dp = global_batch_size // dp_size
        mbs_times_dp = micro_batch_size * dp_size
        assert global_batch_size % mbs_times_dp == 0, \
            f'gbs {global_batch_size} must could be divided by mbs {micro_batch_size} * dp {dp_size}'
        num_micro_batches = global_batch_size // mbs_times_dp
        config.mbs_times_dp_symbol.set_data(mbs_times_dp)
        config.seq_len_symbol.set_data(seq_len)

        # if dp_size * mbs changes, then should use the new dataloader
        # start_time = time.time()
        train_iter = train_data_iterator(train_dataset, consumed_samples, micro_batch_size, dp_rank, dp_size) # need cache?
        # end_time = time.time()
        # print(f'{local_device}: create dataloader cost {end_time - start_time} s')

        # profile_memory()

        # load data for each dp
        micro_batches = []
        for _ in range(num_micro_batches):
            micro_batch = next(train_iter)
            micro_batches.append(micro_batch)
        micro_batches = np.concatenate(micro_batches, axis=0) # [num_micro_batches, micro_batch_size, max_seq_len + 1]
        # todo: use the real seq_len
        micro_batches = micro_batches.reshape(gbs_per_dp, -1)[:, :seq_len+1] # [gbs_per_dp, seq_len + 1]
        labels = micro_batches[:, 1:] # [gbs_per_dp, seq_len]
        tokens = micro_batches[:, :-1] # [gbs_per_dp, seq_len]
        _attention_mask, _position_ids = get_mask_and_position_ids(tokens, train_dataset.encoder.pad_id())
        _token_type_ids = np.zeros([gbs_per_dp, seq_len])

        start_time = time.time()
        feed_dict = {
            input_ids: tokens.astype(np.int64),
            position_ids: _position_ids.astype(np.int64), 
            token_type_ids: _token_type_ids.astype(np.int64),
            attention_mask: _attention_mask.astype(np.int64),
            masked_lm_labels: labels.astype(np.int64),
        }
        print(f"{local_device}: strategy_id = {strategy_id}, gbs = {global_batch_size}, mbs = {micro_batch_size}, seq_len = {seq_len} run begin")
        results = train_op.graph.run(loss_mean, 
                                        [loss_mean, train_op], 
                                        feed_dict = feed_dict, 
                                        num_micro_batches = num_micro_batches, 
                                        cur_strategy_id = strategy_id,
                                        run_level = run_level,
                                        grad_scale = 1.0) 
        consumed_samples += global_batch_size
        print(f"{local_device}: strategy_id = {strategy_id}, gbs = {global_batch_size}, mbs = {micro_batch_size}, seq_len = {seq_len} run end, consumed_samples = {consumed_samples}")
        # NOTE: 实际上应该扫描一次alloc到update之间的所有数据
        # grad_scale = 当前run的数据的batch_size除以总的这之间run加起来的batch_size
        end_time = time.time()
        if run_level == ht.run_level("update"):
            if label_device_group.contains(local_device):
                loss_out = results[0].numpy(force=True).mean()
                print(f"{local_device}: loss = {loss_out} and time = {end_time - start_time}")
        return consumed_samples
    
    '''
    run_plan(global_batch_size = 2, seq_len = 32, strategy_id = 4, run_level = ht.run_level("update"))
    run_plan(global_batch_size = 4, seq_len = 16, strategy_id = 0, run_level = ht.run_level("update"))
    run_plan(global_batch_size = 2, seq_len = 32, strategy_id = 4, run_level = ht.run_level("update"))
    '''
    
    '''
    run_plan(global_batch_size = 2, seq_len = 32, strategy_id = 4, run_level = ht.run_level("topo"))
    # run_plan(global_batch_size = 8, seq_len = 128, strategy_id = 0, run_level = ht.run_level("topo"))
    run_plan(global_batch_size = 16, seq_len = 8, strategy_id = 1, run_level = ht.run_level("topo"))
    run_plan(global_batch_size = 8, seq_len = 64, strategy_id = 2, run_level = ht.run_level("topo"))
    run_plan(global_batch_size = 4, seq_len = 16, strategy_id = 3, run_level = ht.run_level("topo"))
    run_plan(global_batch_size = 2, seq_len = 32, strategy_id = 4, run_level = ht.run_level("topo"))     
    print("--------- run topo end ---------")
    ''' 
    
    # 单次切换
    def test_single_switch():
        consumed_samples = 0
        consumed_samples = run_plan(consumed_samples, global_batch_size = 16, micro_batch_size = 2, seq_len = 32, strategy_id = 0, run_level = ht.run_level("update"))
        consumed_samples = run_plan(consumed_samples, global_batch_size = 16, micro_batch_size = 2, seq_len = 32, strategy_id = 0, run_level = ht.run_level("update"))
        consumed_samples = run_plan(consumed_samples, global_batch_size = 16, micro_batch_size = 2, seq_len = 32, strategy_id = 0, run_level = ht.run_level("update"))
    
    # 单轮样例 
    def test_single_round(): 
        consumed_samples = 0 # should be reset when run next epoch
        consumed_samples = run_plan(consumed_samples, global_batch_size = 2, micro_batch_size = 2, seq_len = 32, strategy_id = 4, run_level = ht.run_level("alloc"))
        # consumed_samples = run_plan(consumed_samples, global_batch_size = 8, micro_batch_size = 2, seq_len = 128, strategy_id = 0, run_level = ht.run_level("grad"))
        consumed_samples = run_plan(consumed_samples, global_batch_size = 16, micro_batch_size = 2,  seq_len = 8, strategy_id = 1, run_level = ht.run_level("grad"))
        consumed_samples = run_plan(consumed_samples, global_batch_size = 8, micro_batch_size = 2, seq_len = 64, strategy_id = 2, run_level = ht.run_level("grad"))
        consumed_samples = run_plan(consumed_samples, global_batch_size = 4, micro_batch_size = 2, seq_len = 16, strategy_id = 3, run_level = ht.run_level("grad"))
        consumed_samples = run_plan(consumed_samples, global_batch_size = 2, micro_batch_size = 2, seq_len = 32, strategy_id = 4, run_level = ht.run_level("update"))
    
    # 多轮样例
    def test_multi_round():
        # 在tp8上分配fp32参数和bf16参数
        # bf16参数切换到dp8并用累计梯度的方式训练、切换到tp8然后更新
        # 第一个循环会很慢是因为要算topo & 算切换方案 & 进行很多cudaMalloc
        consumed_samples = 0 # should be reset when run next epoch
        for round in range(10):
            consumed_samples = run_plan(consumed_samples, global_batch_size = 8, micro_batch_size = 4, seq_len = 256, strategy_id = 4, run_level = ht.run_level("alloc"))
            # consumed_samples = run_plan(consumed_samples, global_batch_size = 8, micro_batch_size = 2, seq_len = 1024, strategy_id = 0, run_level = ht.run_level("grad"))
            consumed_samples = run_plan(consumed_samples, global_batch_size = 64, micro_batch_size = 4, seq_len = 512, strategy_id = 2, run_level = ht.run_level("grad"))
            consumed_samples = run_plan(consumed_samples, global_batch_size = 4, micro_batch_size = 2, seq_len = 128, strategy_id = 3, run_level = ht.run_level("grad"))
            consumed_samples = run_plan(consumed_samples, global_batch_size = 16, micro_batch_size = 2, seq_len = 256, strategy_id = 2, run_level = ht.run_level("grad"))
            consumed_samples = run_plan(consumed_samples, global_batch_size = 8, micro_batch_size = 4, seq_len = 32, strategy_id = 4, run_level = ht.run_level("update"))
            print(f"round {round} finished, consumed_samples = {consumed_samples}")
    
    # memory debug
    def test_memory():
        consumed_samples = 0 # should be reset when run next epoch
        for round in range(10):
            consumed_samples = run_plan(consumed_samples, global_batch_size = 8, micro_batch_size = 4, seq_len = 256, strategy_id = 4, run_level = ht.run_level("alloc"))
            consumed_samples = run_plan(consumed_samples, global_batch_size = 64, micro_batch_size = 4, seq_len = 512, strategy_id = 2, run_level = ht.run_level("alloc"))
            consumed_samples = run_plan(consumed_samples, global_batch_size = 4, micro_batch_size = 2, seq_len = 128, strategy_id = 3, run_level = ht.run_level("alloc"))
            consumed_samples = run_plan(consumed_samples, global_batch_size = 16, micro_batch_size = 2, seq_len = 256, strategy_id = 2, run_level = ht.run_level("alloc"))
            consumed_samples = run_plan(consumed_samples, global_batch_size = 8, micro_batch_size = 4, seq_len = 32, strategy_id = 4, run_level = ht.run_level("alloc"))
            print(f"round {round} finished, consumed_samples = {consumed_samples}")
    
    # test_single_switch()
    # test_single_round()
    test_multi_round()
    # test_memory()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_two_node", action="store_true", help="use 2x8 gpus to run script."
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
        "--num_micro_batches", type=int, default=1, help="Training micro batches num for pipeline parallel"
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
    parser.add_argument("-e", "--epochs", type=int,
                        default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate of adam")
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
    args = parser.parse_args()
    pynvml.nvmlInit()
    distributed_init(args.use_two_node)
    with ht.graph("define_and_run", num_strategy=args.num_strategy):
        if args.bf16:
            precision = "ht.bfloat16"
        else:
            precision = "ht.float32"
        print(f'{local_device}: use precision {precision}')
        with ht.autocast(eval(precision)):            
            pretrain(args)
            print(f'{local_device}: train hetu ds parallel end...')