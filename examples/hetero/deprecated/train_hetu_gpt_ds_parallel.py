import os
import hetu as ht
from hetu_gpt_ds_parallel_sp import GPTLMHeadModel
# from hetu_gpt_ds_parallel import GPTLMHeadModel
from hetu.nn.modules.parallel_ds import config2ds
from hetu.utils.checkpoint import load_file, save_file, temp_load, temp_save,\
                                  temp_load_split, temp_save_split,load_model, save_model, save_checkpoint
from gpt_config import GPTConfig
from data_utils import GPTJsonDataset, get_mask_and_position_ids, build_pretraining_data_loader
import numpy as np
import time
import argparse
import json
import socket
from queue import Queue

local_device = None
all_devices = None

def distributed_init(args, use_multi_node: bool = False):
    if use_multi_node:
        hostname = socket.gethostname()
        # zhiyuan A100
        # note tencent A800 should use LOCAL_HOSTNAME instead
        os.environ['HETU_LOCAL_HOSTNAME'] = os.environ['HOSTNAME']
        '''
        if hostname == 'job-4e4cb411-1139-4f15-b221-5a30f1760a2b-master-0':
           os.environ['HETU_LOCAL_HOSTNAME'] = 'worker-0'
        elif hostname == 'job-4e4cb411-1139-4f15-b221-5a30f1760a2b-worker-0':
           os.environ['HETU_LOCAL_HOSTNAME'] = 'worker-1'
        else:
           raise ValueError(f"Unknown hostname: {hostname}")
        '''

    global local_device, all_devices
    ht.init_comm_group(args.ngpus, server_address=args.server_addr+":"+args.server_port)
    local_device = ht.local_device()
    all_devices = ht.global_device_group()
    # print(local_device, os.environ['HETU_LOCAL_HOSTNAME'])
    if local_device.index == 0:
        print(f'local_device: {local_device}, all_devices: {all_devices}')

def read_ds_parallel_config(args):
    # read ds_parallel_config from json file
    ds_parallel_config = json.load(open(args.ds_parallel_config, 'r'))
    # ds_parallel_config = json.load(open('./ds_parallel_config/dp2_tp2_pp2.json', 'r'))
    # ds_parallel_config = json.load(open('./ds_parallel_config/dp2_tp4.json', 'r'))
    print(f'{local_device}: load ds_parallel_config from: {args.ds_parallel_config}')
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
    # print(f'{local_device}: ds_parallel_config: {ds_parallel_config}')
    return ds_parallel_config

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

def pretrain(args):
    ds_parallel_config = read_ds_parallel_config(args)

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
                       use_flash_attn=args.use_flash_attn,
                       )

    # simple check for gpt blocks range
    ranges = []
    for _, block_config in ds_parallel_config['gpt']['blocks'].items():
        ranges.append(block_config['range'])
    assert ranges[0][0] == 0 and ranges[-1][-1] == config.num_hidden_layers-1, \
        f"gpt blocks range: {ranges} is conflict with num_hidden_layers: {config.num_hidden_layers}!"

    # Hetu model definition
    model = GPTLMHeadModel(config=config, ds_parallel_config=ds_parallel_config)

    input_ds, input_device_group = config2ds(ds_parallel_config['input'])
    label_ds, label_device_group = config2ds(ds_parallel_config['label'])
    # print(f'input_ds: {input_ds}, label_ds: {label_ds}')
    
    global_batch_size = args.global_batch_size
    micro_batch_size = args.micro_batch_size
    seq_len = args.seq_length

    dp_size = input_ds.get_dim(0)
    mbs_times_dp = micro_batch_size * dp_size    

    input_ids = ht.parallel_placeholder(ht.int64, global_shape=[mbs_times_dp, config.seq_len], ds=input_ds, device_group=input_device_group, name='input_ids')
    position_ids = ht.parallel_placeholder(ht.int64, global_shape=[mbs_times_dp, config.seq_len], ds=input_ds, device_group=input_device_group, name='position_ids')
    # token_type_ids = ht.parallel_placeholder(ht.int64, global_shape=[mbs_times_dp, config.seq_len], ds=input_ds, device_group=input_device_group, name='token_type_ids')
    attention_mask = ht.parallel_placeholder(ht.float32, global_shape=[mbs_times_dp, config.seq_len], ds=input_ds, device_group=input_device_group, name='attention_mask')
    masked_lm_labels = ht.parallel_placeholder(ht.int64, global_shape=[mbs_times_dp, config.seq_len], ds=label_ds, device_group=label_device_group, name='masked_lm_labels')

    print(f'{local_device}: build model begin...')
    loss = model(input_ids=input_ids,
                 position_ids=position_ids,
                 attention_mask=attention_mask,
                 # token_type_ids=token_type_ids,
                 labels=masked_lm_labels)
    print(f'{local_device}: build model end...')

    loss_mean = loss

    print(f'{local_device}: optimizer minimize begin...')
    # opt = ht.SGDOptimizer(lr=args.lr, momentum = 0.0)
    opt = ht.AdamOptimizer(lr=args.lr)
    train_op = opt.minimize(loss_mean)
    print(f'{local_device}: optimizer minimize end...')

    # return

    print(f'{local_device}: build dataset begin...')
    train_dataset = train_dataset_provider(args)
    print(f'{local_device}: build dataset end...')

    load_st = time.time()
    # temp_load(model, opt, "./checkpoint/temp3", config=config, local_device=local_device)
    # load_model(model, "./checkpoint/temp2", config=config, local_device=local_device)
    # temp_load_split(model, opt, "./checkpoint/temp3", config=config, local_device=local_device)
    load_ed = time.time()
    print("Checkpoint Loaded.")
    print('%s: Load_Time = %.4f'%(local_device, load_ed - load_st))

    # return
    # device in same dp_group will read the same batch data, idx=-1 means no need to read data
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
        dup_group_num = input_ds.get_dim(0)

    dp_rank = dup_group_idx
    dp_size = dup_group_num
    gbs_per_dp = global_batch_size // dp_size
    mbs_times_dp = micro_batch_size * dp_size
    assert global_batch_size % mbs_times_dp == 0, \
        f'gbs {global_batch_size} must could be divided by mbs {micro_batch_size} * dp {dp_size}'
    num_micro_batches = global_batch_size // mbs_times_dp
    print(f'{local_device}: dp_rank={dp_rank}, dp_size={dp_size}, gbs={global_batch_size}, mbs={micro_batch_size}, num_micro_batches={num_micro_batches}')

    consumed_samples = 0
    if dp_rank != -1:
        train_iter = train_data_iterator(train_dataset, consumed_samples, micro_batch_size, dp_rank, dp_size) # need cache?
    else:
        train_iter = None

    for ep in range(args.epochs):
        for step in range(args.steps):
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
                # _token_type_ids = np.zeros([gbs_per_dp, seq_len])

                feed_dict = {
                    input_ids: tokens.astype(np.int64),
                    position_ids: _position_ids.astype(np.int64),
                    # token_type_ids: _token_type_ids.astype(np.int64),
                    attention_mask: _attention_mask.astype(np.int64),
                    masked_lm_labels: labels.astype(np.int64),
                }
            else: # fake data; feed_dict={} will cause segment fault?
                feed_dict = {
                    input_ids: np.zeros([gbs_per_dp, seq_len]).astype(np.int64),
                    position_ids: get_position_ids(gbs_per_dp, seq_len).astype(np.int64),
                    # token_type_ids: np.zeros([gbs_per_dp, seq_len]).astype(np.int64),
                    attention_mask: np.zeros([gbs_per_dp, seq_len]).astype(np.float32),
                    masked_lm_labels: np.zeros([gbs_per_dp, seq_len]).astype(np.int64),
                }
            # run exec graph
            start_time = time.time()
            results = train_op.graph.run(loss_mean, 
                                        [loss_mean, train_op], 
                                        feed_dict = feed_dict, 
                                        num_micro_batches = num_micro_batches)
            end_time = time.time()
            consumed_samples += global_batch_size
            if label_device_group.contains(local_device):
                loss_out = results[0].numpy(force=True).mean()
                print('%s: [Epoch %d] (Iteration %d, consumed_samples = %d): Loss = %.3f, Time = %.4f'%(local_device, ep, step, consumed_samples, loss_out, end_time-start_time))

                if step == 10 and dp_rank == 0:
                    os.system('nvidia-smi')
    
    save_st = time.time()
    # save_model(model, "./checkpoint/temp", config=config, local_device=local_device, save_dtype=ht.float32)
    # temp_save(model, opt, "./checkpoint/temp3", config=config, local_device=local_device, save_dtype=ht.float32)
    save_ed = time.time()
    print("Checkpoint Saved.")
    print('%s: Save_Time = %.4f'%(local_device, save_ed - save_st))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_multi_node", action="store_true", help="use multi node (like 2x8 gpus) to run script."
    )
    parser.add_argument(
        '--gpu_id', type=int, default=0, help='Id of GPU to run.'
    )
    parser.add_argument(
        "--ds_parallel_config", default="ds_parallel_config/dp2_tp2_pp2.json", type=str, help="ds parallel config json file"
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
    parser.add_argument(
        "--rank", type=int, default=0, help="the rank of process"
    ) 
    args = parser.parse_args()
    distributed_init(args, args.use_multi_node)
    with ht.graph("define_and_run"):
        if args.bf16:
            precision = "ht.bfloat16"
        else:
            precision = "ht.float32"
        print(f'{local_device}: use precision {precision}')
        with ht.autocast(eval(precision)):            
            pretrain(args)
            print(f'{local_device}: train hetu ds parallel end...')