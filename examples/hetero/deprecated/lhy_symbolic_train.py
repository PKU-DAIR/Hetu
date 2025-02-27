from tqdm import tqdm
import os
import math
import logging
import hetu as ht
from hetu_gpt_3d_parallel_symbolic import GPTLMHeadModel
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


def get_position_ids(begin, end, global_batch_size, ds):
    
    # position_ids: [1, seq_len]
    position_ids = np.arange(begin, end, dtype=np.int64) # pos: [idx, ]
    position_ids = np.tile(position_ids, [global_batch_size, 1]) # shape: [b, seq_len]
    return parallel_data_provider(position_ids, ds, local_device_index)

def get_causal_mask(begin, end, max_len, global_batch_size, num_heads, ds):
    
    bias = np.tril(np.ones((max_len, max_len), dtype=np.int64).reshape(
                    1, 1, max_len, max_len))
    bias = np.tile(bias[:, :, begin:end, :], (global_batch_size, num_heads, 1, 1))
    return parallel_data_provider(bias, ds, local_device_index)

def get_mask(seq_len, global_batch_size, num_heads, ds):
    
    masked_value = -1e4
    mask = np.full((global_batch_size, num_heads, seq_len, seq_len), masked_value, dtype=np.float32)
    return parallel_data_provider(mask, ds, local_device_index)

def pretrain(args):
    
    with ht.graph("define_and_run"):
        
        num_epochs = args.epochs
        lr = args.lr
        
        max_tp = 2
        vocab_size = args.vocab_size
        while vocab_size % max_tp != 0:
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
                        num_micro_batches=args.num_micro_batches,
                        dp=args.dp
                        )

        print(f'''{local_device}: 3d parallel config: 
            num_micro_batches={config.num_micro_batches}, 
            dp={config.dp}, num_layers={config.num_hidden_layers}, hidden_size={config.hidden_size}, num_heads={config.num_attention_heads}, seq_length={config.n_positions}''')
        
        # 一个暂时的设置，之后会根据数据自动更改
        max_len = 5
        global_batch_size = 4
        micro_batch_size = global_batch_size // config.num_micro_batches
        dp_size = global_batch_size // config.dp
        
        # ---- Placeholder ----
        input_ids = ht.parallel_placeholder(ht.int64, global_shape=[micro_batch_size, max_len], ds=ds_split0_dup, device_group=device_groups[0], name='input_ids')
        position_ids = ht.parallel_placeholder(ht.int64, global_shape=[micro_batch_size, max_len], ds=ds_split0_dup, device_group=device_groups[0], name='position_ids')
        causal_mask = (ht.parallel_placeholder(ht.int64, global_shape=[micro_batch_size, config.n_head, max_len, max_len], ds=ds_split01, device_group=device_groups[0], name='causal_mask0'),
                        ht.parallel_placeholder(ht.int64, global_shape=[micro_batch_size, config.n_head, max_len, max_len], ds=ds_split01, device_group=device_groups[1], name='causal_mask1'))    
        mask = (ht.parallel_placeholder(ht.float32, global_shape=[micro_batch_size, config.n_head, max_len, max_len], ds=ds_split01, device_group=device_groups[0], name='mask0'),
                ht.parallel_placeholder(ht.float32, global_shape=[micro_batch_size, config.n_head, max_len, max_len], ds=ds_split01, device_group=device_groups[1], name='mask1'))    
        # token_type_ids = ht.parallel_placeholder(ht.int64, global_shape=[micro_batch_size, config.seq_len], ds=ds_split0_dup, device_group=device_groups[0], name='token_type_ids')
        # attention_mask = ht.parallel_placeholder(ht.float32, global_shape=[micro_batch_size, max_len], ds=ds_split0_dup, device_group=device_groups[0], name='attention_mask')
        token_type_ids = None
        attention_mask = None
        masked_lm_labels = ht.parallel_placeholder(ht.int64, global_shape=[micro_batch_size, max_len], ds=ds_split0_dup, device_group=device_groups[1], name='masked_lm_labels')
        vocab_start_index = ht.parallel_placeholder(ht.int64, global_shape=[micro_batch_size, 1], ds=ds_split0_dup, device_group=device_groups[0], name='vocab_start_index')
        
        # ---- Bind symbolic shape ----
        # config.micro_batch_size_symbol = input_ids.symbolic_shape[0] * config.dp
        config.micro_batch_size_symbol = ht.IntSymbol(2)
        config.seq_len_symbol = input_ids.symbolic_shape[1]
        
        # ---- Hetu model definition ----
        model = GPTLMHeadModel(config=config, device_groups=device_groups)
        
        # ---- Load tokenizer ----
        tokenizer = GPT2Tokenizer.from_pretrained('./checkpoint/HuggingFace')
        
        # ---- Load checkpoint ----
        load_checkpoint(model, "./checkpoint/HuggingFace", config=config, local_device=local_device)
        # print("Load the model successfully, the components are:", model.state_dict().keys())
        # You could also see the values by model.state_dict().values()

        print(f'{local_device}: build model begin...')
        loss, lm_logits = model(input_ids=input_ids,
                                position_ids=position_ids,
                                causal_mask=causal_mask,
                                mask=mask,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                labels=masked_lm_labels,
                                vocab_start_index=vocab_start_index)
        print(f'{local_device}: build model end...')

        loss_mean = loss

        print(f'{local_device}: optimizer minimize begin...')
        opt = ht.SGDOptimizer(lr=args.lr, momentum = 0.0)
        train_op = opt.minimize(loss_mean)
        print(f'{local_device}: optimizer minimize end...')
        
        total_devices_num = 8
        encoded_inputs = []
        seq_lens = []
        parallel_plan = []
        input = ['Hello, I am',
                "Good morning! Today",
                "There is a question",
                "Where can I find",
                'Hello, I am',
                "Good morning! Today",
                "There is a question",
                "Where can I find"]
        encoded_inputs.append(tokenizer(input, return_tensors='np'))
        seq_lens.append(4)
        parallel_plan.append({'pp': 2, 'dp': 2, 'tp': 2})
        input = ['Hello, I am a',
                "Good morning! Today is",
                "There is a question about",
                "Where can I find the",
                'Cool, you are so',
                "Well done! The thing",
                "I have so many things",
                "Why the child is going",
                'I like playing tennis and',
                "Do you think this apple",
                "This sounds good for me",
                "Have you ever been to",
                'It is my pleasure to',
                "I think people are all",
                "Is there any chance that",
                "We are good friends and"]
        encoded_inputs.append(tokenizer(input, return_tensors='np'))
        seq_lens.append(5)
        parallel_plan.append({'pp': 2, 'dp': 4, 'tp': 1})

        for round in range(len(seq_lens)):
            encoded_input = encoded_inputs[round]
            global_batch_size = len(encoded_input['input_ids'])
            seq_len = seq_lens[round]
            pp = parallel_plan[round]['pp']
            dp = parallel_plan[round]['dp']
            tp = parallel_plan[round]['tp']
            # 当前local device取第几个dp的小batch
            dp_num = local_device_index % (total_devices_num // pp)
            dp_num = dp_num // (total_devices_num // pp // dp)
            dp_size = global_batch_size // dp
            start = dp_size * dp_num
            end = dp_size * (dp_num + 1)
            config.micro_batch_size_symbol.set_data(global_batch_size // pp)
            # feed_dict
            feed_dict = {
                input_ids: encoded_input['input_ids'][start:end,:].astype(np.int64),
                position_ids: get_position_ids(0, seq_len, global_batch_size, ht.DistributedStates(total_devices_num // pp, {-1: tp, 0: dp}, [0, -1])), # [batch_size, seq_len]
                # attention_mask: np.ones((dp_size, seq_len)).astype(np.float32),
                masked_lm_labels: encoded_input['input_ids'][start:end,:].astype(np.int64),
                vocab_start_index: np.ones((dp_size, 1)).astype(np.int64) * (vocab_size // tp * (local_device_index % tp))
            }    
            # for now, we only consider pp = 2
            for device_group_num in range(2):
                feed_dict[causal_mask[device_group_num]] = get_causal_mask(0, seq_len, seq_len, global_batch_size, config.n_head, ht.DistributedStates(4, {0: dp, 1: tp}, [0, 1])) # [batch_size, num_heads, seq_len, seq_len]    
                feed_dict[mask[device_group_num]] = get_mask(seq_len, global_batch_size, config.n_head, ht.DistributedStates(4, {0: dp, 1: tp}, [0, 1])) # [batch_size, num_heads, seq_len, seq_len]                                                                                                    
            # start_time = time.time()
            results = train_op.graph.run(loss_mean, [loss_mean, lm_logits, train_op], feed_dict = feed_dict, num_micro_batches = config.num_micro_batches)
            # end_time = time.time()
            # for now, we only consider pp = 2
            if device_groups[1].contains(local_device):
                loss_out = results[0].numpy(force=True).mean()
                # print(f'device = {local_device}, lm_logits = {results[1].numpy(force=True)[:, -1, :]}, loss = {loss_out}')
                print(f'device = {local_device}, round = {round}, loss = {loss_out}')
        
    save_checkpoint(model, "./checkpoint/temp", config=config, local_device=local_device)
    # print(f"device = {local_device}, test weight = {model.state_dict()['transformer.h.5.mlp.parallel_mlp.dense_4h_to_h.weight']}")
    print(f'device = {local_device}, save model sucessfully!')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu_id', type=int, default=0, help='Id of GPU to run.'
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