import os
import signal
import hetu as ht
from hetu_gpt_multi_ds_parallel_symbolic import GPTLMHeadModel
from hetu.nn.modules.parallel_multi_ds import config2ds, get_device_index
from gpt_config import GPTConfig
from load_data import DataLoaderForGPT
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
        if hostname == 'job-b9f7b317-d1ec-4c6a-b4e4-a03c9e3b1d19-master-0':
            os.environ['HETU_LOCAL_HOSTNAME'] = 'A100-1'
        elif hostname == 'job-b9f7b317-d1ec-4c6a-b4e4-a03c9e3b1d19-worker-0':
            os.environ['HETU_LOCAL_HOSTNAME'] = 'A100-2'
        else:
            raise ValueError(f"Unknown hostname: {hostname}")

    global local_device, all_devices
    ht.init_comm_group(8)
    local_device = ht.local_device()
    all_devices = ht.global_device_group()
    if local_device.index == 0:
        pass
        # print(f'local_device: {local_device}, all_devices: {all_devices}')

def read_ds_parallel_config(args):
    # read ds_parallel_config from json file
    # print(f'{local_device}: load ds_parallel_config from: {args.ds_parallel_config}')
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
                        if "range" in value and (_range < value["range"][0] or _range > value["range"][1]):
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

def get_position_ids(dp_size, seq_len): 
    position_ids = np.arange(0, seq_len, dtype=np.int64) # [1, seq_len]
    position_ids = np.tile(position_ids, [dp_size, 1]) # [dp_size, seq_len]
    return position_ids

def get_causal_mask(seq_len, global_batch_size, num_heads, distributed_states, device_group):
    raise NotImplementedError

def get_mask(seq_len, global_batch_size, num_heads, distributed_states, device_group):
    raise NotImplementedError

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
    assert ranges[0][0] == 0 and ranges[-1][1] == config.num_hidden_layers-1, \
        f"gpt blocks range: {ranges} is conflict with num_hidden_layers: {config.num_hidden_layers}!"

    # Hetu model definition
    model = GPTLMHeadModel(config=config, ds_parallel_configs=ds_parallel_configs)
    
    input_multi_ds, input_device_groups = parse_multi_ds_parallel_config(ds_parallel_configs, 'input')
    label_multi_ds, label_device_groups = parse_multi_ds_parallel_config(ds_parallel_configs, 'label')
    # print(f'input_ds: {input_ds}, label_ds: {label_ds}')
    
    micro_batch_size = config.global_batch_size // config.num_micro_batches
        
    # todo: assign multi device_groups, and instantiate only one placement_group
    input_ids = ht.parallel_placeholder(ht.int64, global_shape=[micro_batch_size, config.seq_len], multi_ds=input_multi_ds, device_groups=input_device_groups, name='input_ids')
    position_ids = ht.parallel_placeholder(ht.int64, global_shape=[micro_batch_size, config.seq_len], multi_ds=input_multi_ds, device_groups=input_device_groups, name='position_ids')
    token_type_ids = ht.parallel_placeholder(ht.int64, global_shape=[micro_batch_size, config.seq_len], multi_ds=input_multi_ds, device_groups=input_device_groups, name='token_type_ids')
    attention_mask = ht.parallel_placeholder(ht.float32, global_shape=[micro_batch_size, config.seq_len], multi_ds=input_multi_ds, device_groups=input_device_groups, name='attention_mask')
    masked_lm_labels = ht.parallel_placeholder(ht.int64, global_shape=[micro_batch_size, config.seq_len], multi_ds=label_multi_ds, device_groups=label_device_groups, name='masked_lm_labels')

    config.mbs_times_dp_symbol = ht.IntSymbol(micro_batch_size)
    config.seq_len_symbol = input_ids.symbolic_shape[1]

    # print(f'{local_device}: build model begin...')
    loss, lm_logits = model(input_ids=input_ids,
                            position_ids=position_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=masked_lm_labels)
    # print(f'{local_device}: build model end...')

    loss_mean = loss

    # print(f'{local_device}: optimizer minimize begin...')
    # opt = ht.SGDOptimizer(lr=args.lr, momentum = 0.0)
    opt = ht.AdamOptimizer(lr=args.lr)
    train_op = opt.minimize(loss_mean)
    # print(f'{local_device}: optimizer minimize end...')
    
    # return

    def run_plan(global_batch_size = config.global_batch_size,
                 seq_len = config.seq_len,
                 strategy_id = 0, 
                 run_level = 0):       
        if global_batch_size != config.global_batch_size or seq_len != config.seq_len:
            assert config.use_flash_attn == True, "symbolic shape can only used when flash attn is on for now"
        # todo: may also have multiple config.num_micro_batches
        config.mbs_times_dp_symbol.set_data(global_batch_size // config.num_micro_batches)
        config.seq_len_symbol.set_data(seq_len)
        
        dp_size = global_batch_size // input_multi_ds[strategy_id].get_dim(0)
        input_ds = input_multi_ds[strategy_id]
        input_device_group = input_device_groups[strategy_id]
        label_ds = label_multi_ds[strategy_id]
        label_device_group = label_device_groups[strategy_id]
        # device in same dp_group will read the same batch data
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

        # profile_memory()

        for i in range(1000):
            if i % dup_group_num != dup_group_idx:
                continue
            start_time = time.time()
            feed_dict = {
                input_ids: np.zeros([dp_size, seq_len]).astype(np.int64),
                position_ids: get_position_ids(dp_size, seq_len).astype(np.int64), 
                token_type_ids: np.zeros([dp_size, seq_len]).astype(np.int64),
                attention_mask: np.zeros([dp_size, seq_len]).astype(np.float32),
                masked_lm_labels: np.zeros([dp_size, seq_len]).astype(np.int64),
            }
            # print(f"{local_device}: strategy_id = {strategy_id}, dp_size = {dp_size}, seq_len = {seq_len} run begin")
            results = train_op.graph.run(loss_mean, 
                                         [loss_mean, lm_logits, train_op], 
                                         feed_dict = feed_dict, 
                                         num_micro_batches = config.num_micro_batches, 
                                         cur_strategy_id = strategy_id,
                                         run_level = run_level,
                                         grad_scale = 1.0) 
            # print(f"{local_device}: strategy_id = {strategy_id}, dp_size = {dp_size}, seq_len = {seq_len} run end")
            # NOTE: 实际上应该扫描一次alloc到update之间的所有数据
            # grad_scale = 当前run的数据的batch_size除以总的这之间run加起来的batch_size
            end_time = time.time()
            if run_level == ht.run_level("update"):
                if label_device_group.contains(local_device):
                    loss_out = results[0].numpy(force=True).mean()
                    print(f"{local_device}: loss = {loss_out} and time = {end_time - start_time}")
            else:
                print(f"{local_device}: time = {end_time - start_time}")
            return
    
    # 单次切换实验
    def run_experiment():
        try:
            if args.bf16:
                # 保险起见先在tp8/tp16上分配float32的参数以防止爆显存
                run_plan(global_batch_size = 64, seq_len = 32, strategy_id = args.num_strategy - 1, run_level = ht.run_level("alloc"))
                run_plan(global_batch_size = 64, seq_len = 32, strategy_id = args.from_strategy, run_level = ht.run_level("alloc"))
                os.environ['HETU_SWITCH_LOG_FILE'] = args.switch_file
                run_plan(global_batch_size = 64, seq_len = 32, strategy_id = args.to_strategy, run_level = ht.run_level("alloc"))
                if 'HETU_SWITCH_LOG_FILE' in os.environ:
                    del os.environ['HETU_SWITCH_LOG_FILE']
            else:
                run_plan(global_batch_size = 64, seq_len = 32, strategy_id = args.from_strategy, run_level = ht.run_level("alloc"))
                os.environ['HETU_SWITCH_LOG_FILE'] = args.switch_file
                run_plan(global_batch_size = 64, seq_len = 32, strategy_id = args.to_strategy, run_level = ht.run_level("alloc"))
                if 'HETU_SWITCH_LOG_FILE' in os.environ:
                    del os.environ['HETU_SWITCH_LOG_FILE']
        except RuntimeError as e:
            print("OOM occurs, we just skipped")
            # with open(args.switch_file, "a") as file:
            #    file.write(" OOM") # may occur multiple OOM because of async write (the file format will be ugly)
            os.killpg(0, signal.SIGTERM)
                
    def test_single_switch():
        for _ in range(10):
            run_plan(global_batch_size = 64, seq_len = 32, strategy_id = 0, run_level = ht.run_level("grad"))
            run_plan(global_batch_size = 64, seq_len = 32, strategy_id = 5, run_level = ht.run_level("grad"))
    
    # 单轮样例 
    def test_single_round(): 
        run_plan(global_batch_size = 2, seq_len = 32, strategy_id = 4, run_level = ht.run_level("alloc"))
        # run_plan(global_batch_size = 8, seq_len = 128, strategy_id = 0, run_level = ht.run_level("grad"))
        run_plan(global_batch_size = 16, seq_len = 8, strategy_id = 1, run_level = ht.run_level("grad"))
        run_plan(global_batch_size = 8, seq_len = 64, strategy_id = 2, run_level = ht.run_level("grad"))
        run_plan(global_batch_size = 4, seq_len = 16, strategy_id = 3, run_level = ht.run_level("grad"))
        run_plan(global_batch_size = 2, seq_len = 32, strategy_id = 4, run_level = ht.run_level("update"))
    
    # 多轮样例
    def test_multi_round():
        # 在tp8上分配fp32参数和bf16参数
        # bf16参数切换到dp8并用累计梯度的方式训练、切换到tp8然后更新
        # 第一个循环会很慢是因为要算topo & 算切换方案 & 进行很多cudaMalloc
        for round in range(10):
            run_plan(global_batch_size = 32, seq_len = 32, strategy_id = 4, run_level = ht.run_level("alloc"))
            # run_plan(global_batch_size = 8, seq_len = 128, strategy_id = 0, run_level = ht.run_level("grad"))
            run_plan(global_batch_size = 128, seq_len = 8, strategy_id = 1, run_level = ht.run_level("grad"))
            run_plan(global_batch_size = 32, seq_len = 64, strategy_id = 2, run_level = ht.run_level("grad"))
            run_plan(global_batch_size = 64, seq_len = 16, strategy_id = 3, run_level = ht.run_level("grad"))
            run_plan(global_batch_size = 32, seq_len = 32, strategy_id = 4, run_level = ht.run_level("update"))
            print(f"round {round} finished")
    
    run_experiment()
    # test_single_switch()
    # test_multi_round()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from_strategy", type=int, default=0
    )
    parser.add_argument(
        "--to_strategy", type=int, default=0
    )
    parser.add_argument(
        "--switch_file", type=str, default="experiments/result.txt"
    )
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
        # print(f'{local_device}: use precision {precision}')
        with ht.autocast(eval(precision)):            
            pretrain(args)
            # print(f'{local_device}: train hetu ds parallel end...')