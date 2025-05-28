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
from hetu.nn.modules.parallel import parallel_data_provider
from hetu.utils.checkpoint import load_checkpoint, save_checkpoint
from transformers import GPT2Tokenizer
import ptvsd # used for debug

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
local_group_index = 0
for i, device_group in enumerate(device_groups):
    if device_group.contains(local_device):
        local_group_index = i
        local_device_index = device_group.get_index(local_device)
devices_num = device_groups[0].num_devices

# used for debug
# ptvsd.enable_attach(address =('127.0.0.1', 4000 + local_group_index * devices_num + local_device_index))
# ptvsd.wait_for_attach()

def pretrain(args):
    
    with ht.graph("define_and_run"):
        
        num_epochs = args.epochs
        lr = args.lr
        
        tp = 2
        vocab_size = args.vocab_size
        while vocab_size % tp != 0:
            vocab_size += 1

        config = GPTConfig(vocab_size=vocab_size, 
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
                        dp=args.dp
                        )
        
        max_len = 5
        
        # ---- Hetu model definition ----
        model = GPTLMHeadModel(config=config, device_groups=device_groups)
        
        # ---- Load tokenizer ----
        tokenizer = GPT2Tokenizer.from_pretrained('./checkpoint/HuggingFace')
        
        # ---- Load checkpoint ----
        load_checkpoint(model, "./checkpoint/HuggingFace", config=config, local_device=local_device)
        # print("Load the model successfully, the components are:", model.state_dict().keys())
        # You could also see the values by model.state_dict().values()

        micro_batch_size = config.global_batch_size // config.num_micro_batches
        dp_size = config.global_batch_size // config.dp
        print(f'''{local_device}: 3d parallel config: 
            global_batch_size={config.global_batch_size}, num_micro_batches={config.num_micro_batches}, micro_batch_size={micro_batch_size}, 
            dp={config.dp}, num_layers={config.num_hidden_layers}, hidden_size={config.hidden_size}, num_heads={config.num_attention_heads}, seq_length={config.n_positions}''')
        
        # return
        input_ids = ht.parallel_placeholder(ht.int64, global_shape=[micro_batch_size, max_len], ds=ds_split0_dup, device_group=device_groups[0], name='input_ids')
        # token_type_ids = ht.parallel_placeholder(ht.int64, global_shape=[micro_batch_size, config.seq_len], ds=ds_split0_dup, device_group=device_groups[0], name='token_type_ids')
        token_type_ids = None
        attention_mask = ht.parallel_placeholder(ht.float32, global_shape=[micro_batch_size, max_len], ds=ds_split0_dup, device_group=device_groups[0], name='attention_mask')
        masked_lm_labels = ht.parallel_placeholder(ht.int64, global_shape=[micro_batch_size, max_len], ds=ds_split0_dup, device_group=device_groups[1], name='masked_lm_labels')

        print(f'{local_device}: build model begin...')
        loss, lm_logits = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                labels=masked_lm_labels)
        print(f'{local_device}: build model end...')

        loss_mean = loss

        print(f'{local_device}: optimizer minimize begin...')
        opt = ht.SGDOptimizer(lr=args.lr, momentum = 0.0)
        train_op = opt.minimize(loss_mean)
        print(f'{local_device}: optimizer minimize end...')
        
        input = ['Hello, I am a',
                "Good morning! Today is",
                "There is a question about",
                "Where can I find the",
                'Hello, I am a',
                "Good morning! Today is",
                "There is a question about",
                "Where can I find the"]
        encoded_input = tokenizer(input, return_tensors='np')

        for _ in range(10):
            for i in range(len(input) // dp_size):
                # device 0, 1 读取第偶数个batch; device 2, 3 读取第奇数个batch
                if local_device_index < devices_num / 2 and i % 2 != 0:
                    continue
                if local_device_index >= devices_num / 2 and i % 2 != 1:
                    continue
                # start_time = time.time()
                start = dp_size * i
                end = dp_size * (i + 1)
                feed_dict = {
                    input_ids: encoded_input['input_ids'][start:end,:].astype(np.int64),
                    attention_mask: np.ones((dp_size, max_len)).astype(np.float32),
                    masked_lm_labels: encoded_input['input_ids'][start:end,:].astype(np.int64),
                }                                                                                                            
                results = train_op.graph.run(loss_mean, [loss_mean, lm_logits, train_op], feed_dict = feed_dict, num_micro_batches = config.num_micro_batches)
                # end_time = time.time()
                if device_groups[1].contains(local_device):
                    loss_out = results[0].numpy(force=True).mean()
                    # print(f'device = {local_device}, lm_logits = {results[1].numpy(force=True)[:, -1, :]}, loss = {loss_out}')
                    print(f'device = {local_device}, loss = {loss_out}')
            
    save_checkpoint(model, "./checkpoint/temp", config=config, local_device=local_device)
    # print(f"device = {local_device}, test weight = {model.state_dict()['transformer.h.5.mlp.parallel_mlp.dense_4h_to_h.weight']}")
    print(f'device = {local_device}, save model sucessfully!')
    

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
    args = parser.parse_args()
    pretrain(args)