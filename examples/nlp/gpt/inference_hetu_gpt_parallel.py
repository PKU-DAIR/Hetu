from tqdm import tqdm
import os
import math
import logging
import hetu as ht
# from hetu_gpt_parallel import GPTLMHeadModel
from hetu_gpt_parallel_inference import GPTLMHeadModel
from gpt_config import GPTConfig
from load_data import DataLoaderForGPT
import numpy as np
import time
import argparse
from hetu.nn.modules.parallel import parallel_data_provider
from hetu.utils.checkpoint import load_checkpoint
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
all_device_group = ht.DeviceGroup([all_devices.get(0), all_devices.get(1), all_devices.get(2), all_devices.get(3)])
local_device_index = all_device_group.get_index(local_device)
devices_num = all_device_group.num_devices

# used for debug
# ptvsd.enable_attach(address =('127.0.0.1', 4000 + local_device_index))
# ptvsd.wait_for_attach()


def dynamic_fill_input(x, y):
    x[:, :y.shape[1]] = y
    return x


def get_position_ids(begin, end, global_batch_size, ds):
    
    # position_ids: [1, seq_len]
    position_ids = np.arange(begin, end, dtype=np.int64) # pos: [idx, ]
    position_ids = np.tile(position_ids, [global_batch_size, 1]) # shape: [b, 1]
    return parallel_data_provider(position_ids, ds, local_device_index)


def get_causal_mask(begin, end, max_len, global_batch_size, num_heads, ds):
    
    bias = np.tril(np.ones((max_len, max_len), dtype=np.int64).reshape(
                    1, 1, max_len, max_len))
    bias = np.tile(bias[:, :, begin:end, :], (global_batch_size, num_heads, 1, 1))
    return parallel_data_provider(bias, ds, local_device_index)


# todo: 目前只支持贪心sample
def sample(lm_logits, cur_len):
    
    lm_logits_data = lm_logits.numpy(force=True) # [batch_size, seq_len, vocab_size]
    # print(f'device = {local_device}, lm_logits:', lm_logits_data[:, cur_len - 1, :])
    token_ids_data = np.argmax(lm_logits_data[:, cur_len - 1, :], axis=-1).reshape(-1, 1) # [batch_size, 1]
    return token_ids_data


# 第一次运行，从prefix得到kv cache
# 从第二次运行开始，每次得到一个新的token
def run_graph(feed_dict, kv_cache, lm_logits, cur_len, debug_info=[]):
    
    kv_cache_len = len(kv_cache)
    fetches_key_t = [key_t for (key_t, _) in kv_cache]
    fetches_value = [value for (_, value) in kv_cache]
    fetches = [lm_logits] + fetches_key_t + fetches_value
    fetches += debug_info
    
    # print("FeedDict:", feed_dict)
    results = lm_logits.graph.run(lm_logits, fetches, feed_dict = feed_dict)
    
    new_token_ids_numpy = sample(results[0], cur_len) 
    kv_cache_tuple = ()
    for i in range(kv_cache_len):
        kv_cache_tuple += ((results[1 + i].numpy(force=True), results[1 + kv_cache_len + i].numpy(force=True)),) 
        # print("key_t shape:", kv_cache_tuple[i][0].shape, "value shape:", kv_cache_tuple[i][1].shape)
        
    for i in range(len(debug_info)):
        print(f'device = {local_device}, debug_info[{i}]:', results[i - len(debug_info)].numpy(force=True))
        
    return  kv_cache_tuple, new_token_ids_numpy
    
    
def inference(args):

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
                       # n_inner=4*args.hidden_size,
                       resid_pdrop=args.dropout_prob,
                       embd_pdrop=args.dropout_prob,
                       attn_pdrop=args.dropout_prob,
                       activation_function=args.hidden_act,
                       global_batch_size=args.global_batch_size,
                       num_micro_batches=args.num_micro_batches,
                       dp=args.dp
                       )

    # ---- Hetu model definition ----
    model = GPTLMHeadModel(config=config, device_group=all_device_group)
    
    # ---- Load tokenizer ----
    tokenizer = GPT2Tokenizer.from_pretrained('./checkpoint/HuggingFace')
    
    # ---- Load checkpoint ----
    load_checkpoint(model, "./checkpoint/HuggingFace", config=config, device_index=local_device_index)
    print("Load the model successfully, the components are:", model.state_dict().keys())
    # You could also see the values by model.state_dict().values()

    assert config.num_micro_batches == 1, f'2d parallel must set num_micro_batches = 1, got {config.num_micro_batches}'
    batch_size = config.global_batch_size // config.dp
    # there will be some random bugs if set max_len larger (still working on it)
    max_len = 20 # dynamic seq_len will pad to max_len
    print(f'{local_device}: 2d parallel config: global_batch_size={config.global_batch_size}, dp={config.dp}, batch_size={batch_size}')

    # ---- Build graph ----
    # Placeholders for kv_cache
    prefix_ids = ht.parallel_placeholder(ht.int64, global_shape=[config.global_batch_size, max_len], ds=ds_split0_dup, device_group=all_device_group)
    # prefix_type_ids = ht.parallel_placeholder(ht.int64, global_shape=[config.global_batch_size, max_len], ds=ds_split0_dup, device_group=all_device_group)
    prefix_type_ids = None
    # attention_mask其实也可以不用
    prefix_attention_mask = ht.parallel_placeholder(ht.float32, global_shape=[config.global_batch_size, max_len], ds=ds_split0_dup, device_group=all_device_group)
    prefix_position_ids = ht.parallel_placeholder(ht.int64, global_shape=[config.global_batch_size, max_len], ds=ds_split0_dup, device_group=all_device_group)
    prefix_causal_mask = ht.parallel_placeholder(ht.int64, global_shape=[config.global_batch_size, config.n_head, max_len, max_len], ds=ds_split01, device_group=all_device_group)   
    # Placeholders for generation
    token_ids = ht.parallel_placeholder(ht.int64, global_shape=[config.global_batch_size, 1], ds=ds_split0_dup, device_group=all_device_group)
    # token_type_ids = ht.parallel_placeholder(ht.int64, global_shape=[config.global_batch_size, 1], ds=ds_split0_dup, device_group=all_device_group)
    token_type_ids = None
    # attention_mask其实也可以不用
    token_attention_mask = ht.parallel_placeholder(ht.float32, global_shape=[config.global_batch_size, max_len], ds=ds_split0_dup, device_group=all_device_group)
    token_position_ids = ht.parallel_placeholder(ht.int64, global_shape=[config.global_batch_size, 1], ds=ds_split0_dup, device_group=all_device_group)
    token_causal_mask = ht.parallel_placeholder(ht.int64, global_shape=[config.global_batch_size, config.n_head, 1, max_len], ds=ds_split01, device_group=all_device_group)   
    past_key_values = ()
    head_dim = config.hidden_size // config.num_attention_heads
    for i in range(config.n_layer):
        # append a tuple (key_t placeholder, value placeholder)
        past_key_values += ((ht.parallel_placeholder(ht.float32, global_shape=[config.global_batch_size, config.n_head, head_dim, max_len], ds=ds_split01, device_group=all_device_group),
                            ht.parallel_placeholder(ht.float32, global_shape=[config.global_batch_size, config.n_head, max_len, head_dim], ds=ds_split01, device_group=all_device_group)),)
    
    print(f'{local_device}: build kv_cache model begin...')
    kv_cache, lm_logits = model(input_ids=prefix_ids, # dynamic runtime shape [batch_size, seq_len], padding to [batch_size, max_len]
                            position_ids=prefix_position_ids, # dynamic runtime shape [batch_size, seq_len], padding to [batch_size, max_len]
                            attention_mask=prefix_attention_mask, # dynamic runtime shape [batch_size, seq_len], padding to [batch_size, max_len]
                            causal_mask=prefix_causal_mask, # dynamic runtime shape [batch_size, num_heads, seq_len, seq_len], padding to [batch_size, num_heads, max_len, max_len]
                            token_type_ids=prefix_type_ids, # dynamic runtime shape [batch_size, seq_len], padding to [batch_size, max_len]
                            use_cache=True)
    print(f'{local_device}: build kv_cache model end...')
    
    # kv_cache: dynamic runtime shape ((key_t, value), (key_t, value), ...)
    # ([batch_size, num_heads, head_dim, seq_len], [batch_size, num_heads, seq_len, head_dim]) * layers_num
    # padding seq_len to max_len
    
    # lm_logits: dynamic runtime shape [batch_size, seq_len, vocab_size]
    # padding seq_len to max_len
    
    print(f'{local_device}: build generation model begin...')
    new_key_values, new_lm_logits = model(input_ids=token_ids, # [batch_size, 1]
                            position_ids=token_position_ids, # [batch_size, 1]
                            attention_mask=token_attention_mask, # dynamic runtime shape [batch_size, seq_len], padding to [batch_size, max_len]
                            causal_mask=token_causal_mask, # dynamic runtime shape [batch_size, num_heads, 1, seq_len], padding to [batch_size, num_heads, 1, max_len]
                            token_type_ids=token_type_ids, # [batch_size, 1]
                            past_key_values=past_key_values, # dynamic runtime shape
                            use_cache=True)
    print(f'{local_device}: build generation model end...')
    
    # new_key_values: dynamic runtime shape ((key_t, value), (key_t, value), ...)
    # ([batch_size, num_heads, head_dim, seq_len + 1], [batch_size, num_heads, seq_len + 1, head_dim]) * layers_num
    # padding seq_len + 1 to max_len
    
    # new_lm_logits: [batch_size, 1, vocab_size]

    # ---- Run dataset and generate ----
    # todo: 目前只支持batch内等长的inference，不支持dynamic batching
    input = ['Hello, I am a',
             "Good morning! Today is",
             "There is a question about",
             "Where can I find the"]
    history = [_ for _ in input]
    encoded_input = tokenizer(input, return_tensors='np')
    
    # todo: 保证dataloader.batch_num是dp(2)的倍数
    for i in range(len(input) // batch_size):
        
        # device 0, 1 读取第偶数个batch; device 2, 3 读取第奇数个batch
        if local_device_index < devices_num / 2 and i % 2 != 0:
            continue
        if local_device_index >= devices_num / 2 and i % 2 != 1:
            continue
        start = batch_size * i
        end = batch_size * (i + 1)
        
        # -- Run kv_cache --
        # 样例: token初始长度为5
        basic_len = encoded_input['input_ids'].shape[-1]
        cur_len = basic_len
        # assume initial dynamic shape is [batch_size, 5]
        dynamic_shape = [batch_size, basic_len]
        kv_cache_feed_dict = {
            prefix_ids: dynamic_fill_input(np.zeros((batch_size, max_len)).astype(np.int64), encoded_input['input_ids'][start:end,:].astype(np.int64)),
            # prefix_type_ids: np.zeros((batch_size, max_len)).astype(np.int64),
            prefix_attention_mask: np.ones((batch_size, max_len)).astype(np.int64),
            prefix_position_ids: get_position_ids(0, max_len, config.global_batch_size, ds_split0_dup), # [batch_size, max_len]   
            prefix_causal_mask: get_causal_mask(0, max_len, max_len, config.global_batch_size, config.n_head, ds_split01) # [batch_size, num_heads, max_len, max_len]               
        }
        # numpy -> ht.NDArray with dynamic shape
        for key in kv_cache_feed_dict:
            value = kv_cache_feed_dict[key]
            if key == prefix_causal_mask:
                kv_cache_feed_dict[key] = ht.numpy_to_NDArray(value, dynamic_shape=[value.shape[0], value.shape[1], basic_len, basic_len])
            else:
                kv_cache_feed_dict[key] = ht.numpy_to_NDArray(value, dynamic_shape=dynamic_shape)
        # run graph
        kv_cache_tuple, new_token_ids_numpy = run_graph(kv_cache_feed_dict, kv_cache, lm_logits, cur_len)
        new_tokens = tokenizer.batch_decode(new_token_ids_numpy)
        print(f"device = {local_device}, kv_cache_len = {cur_len}, new generated batched tokens = id: {new_token_ids_numpy}, old tokens: {history[start:end]}, new tokens: {new_tokens}")
        cur_len += 1
        for piece in range(start, end):
            history[piece] += new_tokens[piece - start]
        
        # -- Run generation --
        # todo: 如果batch中某一条碰到eos应该停下
        while cur_len < max_len:
            generation_feed_dict = {
                token_ids: new_token_ids_numpy.astype(np.int64), # [batch_size, 1]
                # token_type_ids: np.zeros((batch_size, 1)).astype(np.int64), # [batch_size, 1], 这里type都一样，随便
                token_attention_mask: np.ones((batch_size, max_len)).astype(np.int64), # [batch_size, max_len] 
                token_position_ids: get_position_ids(cur_len - 1, cur_len, config.global_batch_size, ds_split0_dup), # [batch_size, 1]   
                token_causal_mask: get_causal_mask(cur_len - 1, cur_len, max_len, config.global_batch_size, config.n_head, ds_split01) # [batch_size, num_heads, 1, max_len]               
            }
            # numpy -> ht.NDArray with dynamic shape
            for key in generation_feed_dict:
                value = generation_feed_dict[key]
                if key == token_attention_mask:
                    generation_feed_dict[key] = ht.numpy_to_NDArray(value, dynamic_shape=[batch_size, cur_len])
                if key == token_causal_mask:
                    generation_feed_dict[key] = ht.numpy_to_NDArray(value, dynamic_shape=[value.shape[0], value.shape[1], value.shape[2], cur_len])
            # add kv_cache to feed_dict
            key_t_shape = kv_cache_tuple[0][0].shape
            value_shape = kv_cache_tuple[0][1].shape
            key_t_dynamic_shape = [key_t_shape[0], key_t_shape[1], key_t_shape[2], cur_len - 1]
            value_dynamic_shape = [value_shape[0], value_shape[1], cur_len - 1, value_shape[3]]
            for j in range(config.n_layer):
                generation_feed_dict[past_key_values[j][0]] = ht.numpy_to_NDArray(kv_cache_tuple[j][0], dynamic_shape=key_t_dynamic_shape) # key_t
                generation_feed_dict[past_key_values[j][1]] = ht.numpy_to_NDArray(kv_cache_tuple[j][1], dynamic_shape=value_dynamic_shape) # value        
            # run graph
            kv_cache_tuple, new_token_ids_numpy = run_graph(generation_feed_dict, new_key_values, new_lm_logits, 1)
            new_tokens = tokenizer.batch_decode(new_token_ids_numpy)
            print(f"device = {local_device}, batch = {i}, past_len = {cur_len}, new generated batched tokens = id: {new_token_ids_numpy}, old tokens: {history[start:end]}, new tokens: {new_tokens}")
            cur_len += 1
            for piece in range(start, end):
                history[piece] += new_tokens[piece - start]
            


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
    with ht.graph("define_and_run"):
        inference(args)

     
    # eager graph generation   
    '''
    new_past_key_values, lm_logits = model(input_ids=feed_dict['input_ids'][:, :basic_len],
                            attention_mask=feed_dict['attention_mask'][:, :basic_len],
                            token_type_ids=feed_dict['token_type_ids'][:, :basic_len],
                            use_cache=True)
    # todo: 如果batch中某一条碰到eos应该停下
    while cur_len < max_len:
        cur_len += 1
        lm_logits_result = lm_logits.graph.run([lm_logits])[0]
        new_token_ids_data = sample(lm_logits_result)
        outputs.append(new_token_ids_data[:, -1])
        new_token_ids = ht.from_numpy_parallel(new_token_ids_data, lm_logits.distributed_states, device_group=all_device_group)
        new_past_key_values, lm_logits = model(input_ids=new_token_ids,
                                attention_mask=feed_dict['attention_mask'][:, :cur_len],
                                token_type_ids=feed_dict['token_type_ids'][:, :cur_len],
                                past_key_values=new_past_key_values,
                                use_cache=True)
    return outputs
    '''