from tqdm import tqdm
import os
import math
import logging
import hetu as ht
from hetu_gpt_3d_parallel import GPTLMHeadModel
from gpt_config import GPTConfig
from load_data import DataLoaderForGPT
import numpy as np
import time
import argparse

ds_dup = ht.DistributedStates(4, {-1: 4}, [-1])
ds_split0 = ht.DistributedStates(4, {0: 4}, [0])
ds_split0_dup = ht.DistributedStates(4, {-1: 2, 0: 2}, [0, -1])
ds_dup_split1 = ht.DistributedStates(4, {-1: 2, 1: 2}, [-1, 1])
ds_split01 = ht.DistributedStates(4, {0: 2, 1: 2}, [0, 1])

ht.init_comm_group()
local_device = ht.local_device()
all_devices = ht.global_device_group()
device_group0 = ht.DeviceGroup([all_devices.get(0), all_devices.get(1), 
                                all_devices.get(2), all_devices.get(3)]) # pp stage0 (dp=2, tp=2)
device_group1 = ht.DeviceGroup([all_devices.get(4), all_devices.get(5), 
                                all_devices.get(6), all_devices.get(7)]) # pp stage0 (dp=2, tp=2)
device_groups = [device_group0, device_group1]
for device_group in device_groups:
    if device_group.contains(local_device):
        local_device_index = device_group.get_index(local_device)
devices_num = device_groups[0].num_devices

def pretrain(args):
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
                       dp=args.dp,
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

    # Hetu model definition
    model = GPTLMHeadModel(config=config, device_groups=device_groups)

    micro_batch_size = config.global_batch_size // config.num_micro_batches
    dp_size = config.global_batch_size // config.dp
    print(f'''{local_device}: 3d parallel config: 
          global_batch_size={config.global_batch_size}, num_micro_batches={config.num_micro_batches}, micro_batch_size={micro_batch_size}, 
          dp={config.dp}, num_layers={config.num_hidden_layers}, hidden_size={config.hidden_size}, num_heads={config.num_attention_heads}, seq_length={config.n_positions}''')
    # return
    input_ids = ht.parallel_placeholder(ht.int64, global_shape=[micro_batch_size, config.seq_len], ds=ds_split0_dup, device_group=device_groups[0], name='input_ids')
    token_type_ids = ht.parallel_placeholder(ht.int64, global_shape=[micro_batch_size, config.seq_len], ds=ds_split0_dup, device_group=device_groups[0], name='token_type_ids')
    attention_mask = ht.parallel_placeholder(ht.float32, global_shape=[micro_batch_size, config.seq_len], ds=ds_split0_dup, device_group=device_groups[0], name='attention_mask')
    masked_lm_labels = ht.parallel_placeholder(ht.int64, global_shape=[micro_batch_size, config.seq_len], ds=ds_split0_dup, device_group=device_groups[1], name='masked_lm_labels')

    print(f'{local_device}: build model begin...')
    loss, lm_logits = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=masked_lm_labels)
    print(f'{local_device}: build model end...')

    loss_mean = loss

    print(f'{local_device}: optimizer minimize begin...')
    # opt = ht.SGDOptimizer(lr=args.lr, momentum = 0.0)
    opt = ht.AdamOptimizer(lr=args.lr)

    train_op = opt.minimize(loss_mean)
    print(f'{local_device}: optimizer minimize end...')

    # return
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
                # device 0, 1 读取第偶数个batch; device 2, 3 读取第奇数个batch
                if local_device_index < devices_num / 2 and i % 2 != 0:
                    continue
                if local_device_index >= devices_num / 2 and i % 2 != 1:
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
                # 统计masked_lm_labels中为-1的个数及占比
                # print(f'{local_device}: masked_lm_labels=-1 num={np.where(batch_data["masked_lm_labels"].reshape(-1, 1)==-1)[0].shape[0]}, 
                #       ratio={np.where(batch_data["masked_lm_labels"].reshape(-1, 1)==-1)[0].shape[0]/(batch_data["masked_lm_labels"].reshape(-1, 1).shape[0])}')
                results = train_op.graph.run(loss_mean, [loss_mean, lm_logits, train_op], feed_dict = feed_dict, num_micro_batches = config.num_micro_batches)
                end_time = time.time()
                if device_groups[1].contains(local_device):
                    loss_out = results[0].numpy(force=True).mean()
                    print('%s: [Epoch %d] (Iteration %d): Loss = %.3f, Time = %.4f'%(local_device, ep, step_num, loss_out, end_time-start_time))
                step_num += 1
                global_step_num += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu_id', type=int, default=0, help='Id of GPU to run.'
    )
    parser.add_argument(
        "--global_batch_size", type=int, default=64, help="Training batch size global"
    )
    parser.add_argument(
        "--num_micro_batches", type=int, default=1, help="Training micro batches num for pipeline parallel"
    )
    parser.add_argument(
        "--dp", type=int, default=1, help="data parallel degrees"
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
        "--use_flash_attn", type=bool, default=False, help="Use Flash Attention."
    )
    parser.add_argument(
        "--bf16", action="store_true", help="Use bfloat16."
    )    
    args = parser.parse_args()
    with ht.graph("define_and_run"):
        if args.bf16:
            precision = "ht.bfloat16"
        else:
            precision = "ht.float32"
        print(f'{local_device}: use precision {precision}')
        with ht.autocast(eval(precision)):            
            pretrain(args)
            print(f'{local_device}: train hetu 3d parallel end...')
