import os
import hetu as ht
from hetu_gpt_ds_parallel_lora import GPTLMHeadModel
from hetu.nn.modules.parallel_ds import config2ds
from hetu.utils.checkpoint import load_file, save_file, load_model, save_model, save_checkpoint
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
        if hostname == 'job-e44df83d-4af0-4fbf-b066-b4650867451d-master-0':
            os.environ['HETU_LOCAL_HOSTNAME'] = 'a100-0'
        elif hostname == 'job-e44df83d-4af0-4fbf-b066-b4650867451d-worker-0':
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

def pretrain(args):
    ds_parallel_config = read_ds_parallel_config(args)
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
                       lora_dtype=args.lora_dtype,
                       dqtype=args.dqtype
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
    for _, block_config in ds_parallel_config['gpt']['blocks'].items():
        ranges.append(block_config['range'])
    assert ranges[0][0] == 0 and ranges[-1][1] == config.num_hidden_layers-1, \
        f"gpt blocks range: {ranges} is conflict with num_hidden_layers: {config.num_hidden_layers}!"

    # Hetu model definition
    
    model = GPTLMHeadModel(config=config, ds_parallel_config=ds_parallel_config)
    model = ht.nn.LoRAModel(model, 8, qtype=config.dqtype, lora_alpha = 1)

    print("begin load.")
    # load_model(model, "./checkpoint/temp3", config=config, local_device=local_device)
    print("Load end.")

    input_ds, input_device_group = config2ds(ds_parallel_config['input'])
    label_ds, label_device_group = config2ds(ds_parallel_config['label'])
    # print(f'input_ds: {input_ds}, label_ds: {label_ds}')
    
    micro_batch_size = config.global_batch_size // config.num_micro_batches
    dp_size = config.global_batch_size // input_ds.get_dim(0)
    # print(f'''{local_device}: 3d parallel config: 
    #       global_batch_size={config.global_batch_size}, num_micro_batches={config.num_micro_batches}, micro_batch_size={micro_batch_size}, dp_size={dp_size},
    #       num_layers={config.num_hidden_layers}, hidden_size={config.hidden_size}, num_heads={config.num_attention_heads}, seq_length={config.n_positions}''')
        
    input_ids = ht.parallel_placeholder(ht.int64, global_shape=[micro_batch_size, config.seq_len], ds=input_ds, device_group=input_device_group, name='input_ids')
    token_type_ids = ht.parallel_placeholder(ht.int64, global_shape=[micro_batch_size, config.seq_len], ds=input_ds, device_group=input_device_group, name='token_type_ids')
    attention_mask = ht.parallel_placeholder(config.dqtype, global_shape=[micro_batch_size, config.seq_len], ds=input_ds, device_group=input_device_group, name='attention_mask')
    masked_lm_labels = ht.parallel_placeholder(ht.int64, global_shape=[micro_batch_size, config.seq_len], ds=label_ds, device_group=label_device_group, name='masked_lm_labels')

    print(f'{local_device}: build model begin...')
    loss = model(input_ids=input_ids,
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
    num_epochs = 1
    for ep in range(num_epochs):
        step_num = 0
        for train_file in train_files:
            # walkaround: dataloader读取来的seq_len必定是128的, 这里暂时手动处理下
            # dataloader = DataLoaderForGPT(train_file, dp_size, pred_len)
            required_batch_size = dp_size * config.seq_len // 128
            dataloader = DataLoaderForGPT(train_file, required_batch_size, pred_len)
            # todo: 保证dataloader.batch_num是dp的倍数
            for i in range(20):
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
                results = train_op.graph.run(loss_mean, [loss_mean, train_op], feed_dict = feed_dict, num_micro_batches = config.num_micro_batches)
                end_time = time.time()
                if label_device_group.contains(local_device):
                    loss_out = results[0].numpy(force=True).mean()
                    print('%s: [Epoch %d] (Iteration %d): Loss = %.3f, Time = %.4f'%(local_device, ep, step_num, loss_out, end_time-start_time))
                step_num += 1
                global_step_num += 1
                # return
    print("Finish Finetune")
    # save_checkpoint(model, "./checkpoint/temp2", config=config, local_device=local_device)
    # save_model(model, "./checkpoint/temp4", config=config, local_device=local_device, save_dtype=ht.nfloat4, only_lora=True)
    # save_model(model, "./checkpoint/temp3", config=config, local_device=local_device, only_lora=True)

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
    parser.add_argument(
        "--lora_dtype", type=str, default='fp32', help="Base Model dtype."
    )
    parser.add_argument(
        "--dqtype", type=str, default='fp32', help="Dequantization type."
    )
    args = parser.parse_args()
    distributed_init(args.use_two_node)
    with ht.graph("define_and_run"):
        if args.bf16:
            precision = "ht.bfloat16"
        else:
            precision = "ht.float32"
        print(f'{local_device}: use precision {precision}')
        with ht.autocast(eval(precision)):            
            pretrain(args)
            print(f'{local_device}: train hetu ds parallel end...')