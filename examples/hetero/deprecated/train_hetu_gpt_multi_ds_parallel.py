import os
import hetu as ht
from hetu_gpt_multi_ds_parallel import GPTLMHeadModel
from hetu.utils.parallel import config2ds
from gpt_config import GPTConfig
from load_data import DataLoaderForGPT
import numpy as np
import time
import argparse
import json
import socket
from queue import Queue

local_device = None
all_devices = None

def distributed_init(use_two_node: bool = False):
    if use_two_node:
        hostname = socket.gethostname()
        if hostname == 'job-4e4cb411-1139-4f15-b221-5a30f1760a2b-master-0':
            os.environ['HETU_LOCAL_HOSTNAME'] = 'worker-0'
        elif hostname == 'job-4e4cb411-1139-4f15-b221-5a30f1760a2b-worker-0':
            os.environ['HETU_LOCAL_HOSTNAME'] = 'worker-1'
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
                       # n_inner=4*args.hidden_size,
                       resid_pdrop=args.dropout_prob,
                       embd_pdrop=args.dropout_prob,
                       attn_pdrop=args.dropout_prob,
                       activation_function=args.hidden_act,
                       global_batch_size=args.global_batch_size,
                       num_micro_batches=args.num_micro_batches,
                       use_flash_attn=args.use_flash_attn,
                       )
    # Input data file names definition
    # dict_seqlen2predlen = {128:20, 512:80}
    # assert config.seq_len == 128 or config.seq_len == 512, "seq_len must be 128 or 512, got " + config.seq_len
    # pred_len = dict_seqlen2predlen[config.max_position_embeddings]
    pred_len = config.seq_len
    dataset = args.dataset
    if dataset not in ['wikicorpus_en', 'wiki_books']:
        raise(NotImplementedError)
    # file_dir = '../bert/data/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/%s/'%dataset
    file_dir = './data/'    
    file_name_format = dataset + '_training_%d.hdf5'
    train_file_num = 1
    train_files = [file_dir + file_name_format%file_id for file_id in range(train_file_num)]

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
    
    micro_batch_size = config.global_batch_size // config.num_micro_batches
    dp_size = config.global_batch_size // input_multi_ds[0].get_dim(0)
    # print(f'''{local_device}: 3d parallel config: 
    #       global_batch_size={config.global_batch_size}, num_micro_batches={config.num_micro_batches}, micro_batch_size={micro_batch_size}, dp_size={dp_size},
    #       num_layers={config.num_hidden_layers}, hidden_size={config.hidden_size}, num_heads={config.num_attention_heads}, seq_length={config.n_positions}''')
        
    # todo: assign multi device_groups, and instantiate only one placement_group
    input_ids = ht.parallel_placeholder(ht.int64, global_shape=[micro_batch_size, config.seq_len], multi_ds=input_multi_ds, device_groups=input_device_groups, name='input_ids')
    token_type_ids = ht.parallel_placeholder(ht.int64, global_shape=[micro_batch_size, config.seq_len], multi_ds=input_multi_ds, device_groups=input_device_groups, name='token_type_ids')
    attention_mask = ht.parallel_placeholder(ht.float32, global_shape=[micro_batch_size, config.seq_len], multi_ds=input_multi_ds, device_groups=input_device_groups, name='attention_mask')
    masked_lm_labels = ht.parallel_placeholder(ht.int64, global_shape=[micro_batch_size, config.seq_len], multi_ds=label_multi_ds, device_groups=label_device_groups, name='masked_lm_labels')

    # print(f'{local_device}: build model begin...')
    loss, lm_logits = model(input_ids=input_ids,
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
    
    # return

    # use the first ds, just adapt for run, will changed for dynamic shape & distributed states in future...
    input_ds = input_multi_ds[0]
    input_device_group = input_device_groups[0]
    label_ds = label_multi_ds[0]
    label_device_group = label_device_groups[0]
    # return
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

    global_step_num = 0
    for ep in range(num_epochs):
        step_num = 0
        for train_file in train_files:
            # walkaround: dataloader读取来的seq_len必定是128的, 这里暂时手动处理下
            # dataloader = DataLoaderForGPT(train_file, dp_size, pred_len)
            required_batch_size = dp_size * config.seq_len // 128
            dataloader = DataLoaderForGPT(train_file, required_batch_size, pred_len)
            # todo: 保证dataloader.batch_num是dp的倍数
            for i in range(dataloader.batch_num):
                if i % dup_group_num != dup_group_idx:
                    continue
                start_time = time.time()
                batch_data = dataloader.get_batch(i)
                feed_dict = {
                    input_ids: batch_data['input_ids'].astype(np.int64).reshape([dp_size, config.seq_len]),
                    token_type_ids: batch_data['token_type_ids'].astype(np.int64).reshape([dp_size, config.seq_len]),
                    attention_mask: batch_data['attention_mask'].astype(np.float32).reshape([dp_size, config.seq_len]),
                    masked_lm_labels: batch_data['masked_lm_labels'].astype(np.int64).reshape([dp_size, config.seq_len]),
                    # loss_position_sum: np.array([np.where(batch_data['masked_lm_labels'].reshape(-1, 1)!=-1)[0].shape[0]]).astype(np.float32), # shape=[1,]
                }
                results = train_op.graph.run(loss_mean, [loss_mean, lm_logits, train_op], feed_dict = feed_dict, num_micro_batches = config.num_micro_batches)
                end_time = time.time()
                if label_device_group.contains(local_device):
                    loss_out = results[0].numpy(force=True).mean()
                    print('%s: [Epoch %d] (Iteration %d): Loss = %.3f, Time = %.4f'%(local_device, ep, step_num, loss_out, end_time-start_time))
                step_num += 1
                global_step_num += 1
                # return
                # if global_step_num == 20:
                #     return

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