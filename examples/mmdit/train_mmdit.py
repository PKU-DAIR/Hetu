import os
import signal
import time
import argparse
import socket
import pynvml
import ast
import json
import numpy as np
import hetu as ht

from arugments import add_all_args
from mmdit_config import MMDiTConfig
from model.hetu_mmdit import MMDitMHeadModel
from data_utils import LLaMAJsonDataset, build_data_loader, get_sorted_batch_and_len, build_fake_batch_and_len, get_input_and_label_buckets, LLaMaDatasetConfig, build_tokenizer, BlendedHetuDatasetBuilder
from parallel_utils import read_ds_parallel_config, parse_multi_ds_parallel_config, convert_strategy, generate_mmdit_model_ds_parallel_config
from typing import List
from diffusion import create_diffusion

local_device = None
all_devices = None
tokenizer = None
ds_parallel_config_path = "./ds_parallel_config/"
alignment = 128

def distributed_init(args):
    global local_device, all_devices
    hostname = socket.gethostname()
    os.environ['HETU_LOCAL_HOSTNAME'] = hostname
    ht.init_comm_group(args.ngpus, server_address = args.server_addr + ":" + args.server_port)
    local_device = ht.local_device()
    all_devices = ht.global_device_group()
    if local_device.index == 0:
        print(f'local_device: {local_device}, all_devices: {all_devices}')


def train_dataset_provider(args):
    global tokenizer
    args.make_vocab_size_divisible_by = 128
    tokenizer = build_tokenizer(args)
    config = LLaMaDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.max_seq_len,
        blend=args.data_path,
        blend_per_split=[None, None, None],
        split=args.split,
        path_to_cache=args.data_cache_path,
        tokenizer=tokenizer,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        vocab_size=args.vocab_size,
    )
    train_val_test_num_samples = [args.epochs * args.steps * args.global_batch_size, 0, 0]
    train_ds, valid_ds, test_ds = BlendedHetuDatasetBuilder(
        LLaMAJsonDataset,
        train_val_test_num_samples,
        config
    ).build()

    return train_ds

def train_dataloader_provider(train_ds, consumed_samples, global_batch_size):
    data_loader = build_data_loader(train_ds, consumed_samples, global_batch_size)

    return iter(data_loader)
  
def get_dg_from_union(device, dg_union):
    for i, dg in enumerate(dg_union):
        if dg.contains(device):
            return i, dg
    return None, None


def pretrain(args):
    
    multi_tp_pp_list = args.multi_tp_pp_list

    llm_num_strategy = len(multi_tp_pp_list)
    assert len(multi_tp_pp_list) == 1, "currently only support one strategy for each model"

    multi_dp_size = [len(tp_pp_list) for tp_pp_list in multi_tp_pp_list]

    multi_gpu_pos = []
    multi_config_file_path = []
    for strategy_id in range(llm_num_strategy):
        # 获取GPU的位置
        # 原则是不让tp跨机并尽可能贪心地让pp跨机
        layers_tp_groups, gpu_pos = convert_strategy(multi_tp_pp_list[strategy_id], args.ngpus, args.num_hidden_layers)
        config_file_path = ds_parallel_config_path + f"strategy_{strategy_id}.txt"
        # print("layers_tp_groups", layers_tp_groups)
        generate_mmdit_model_ds_parallel_config(args.ngpus, args.num_hidden_layers, layers_tp_groups, config_file_path)
        print(f"Strategy {strategy_id}, gpu positions are: {gpu_pos}")
        multi_gpu_pos.append(gpu_pos)
        multi_config_file_path.append(config_file_path)
    ds_parallel_configs = read_ds_parallel_config(",".join(multi_config_file_path), llm_num_strategy)

    print(ds_parallel_config_path)

    # 1. Config Information
    text_input_ds_hierarchy, text_input_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'text_input')
    image_input_ds_hierarchy, image_input_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'image_input')
    label_ds_hierarchy, label_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'label')

    # 2. Build Placeholders
    dp_size = text_input_ds_hierarchy[0].get(0).get_dim(0)
    dummy_size = dp_size * args.max_seq_len
    # embed_dim = args.patch_size * args.patch_size * 3
    C, D, H, W = 3, 128, 128, 128
    # mbs_times_dp = dp_size * args.micro_batch_size
    image_inputs = ht.parallel_placeholder(ht.float32, global_shape=[dp_size, C, D, H, W], ds_hierarchy=image_input_ds_hierarchy, device_group_hierarchy=image_input_dg_hierarchy, name='input_ids')
    text_ids = ht.parallel_placeholder(ht.int64, global_shape=[dp_size, args.max_seq_len], ds_hierarchy=text_input_ds_hierarchy, device_group_hierarchy=text_input_dg_hierarchy, name='input_ids')
    time_steps = ht.parallel_placeholder(ht.float32, global_shape=[dp_size, args.frequency_embedding_size], ds_hierarchy=text_input_ds_hierarchy, device_group_hierarchy=text_input_dg_hierarchy, name='input_ids')
    sqrt_alphas_cumprod = ht.parallel_placeholder(ht.float32, global_shape=[dp_size,1,1,1,1], ds_hierarchy=text_input_ds_hierarchy, device_group_hierarchy=text_input_dg_hierarchy, name='input_ids')
    sqrt_one_minus_alphas_cumprod = ht.parallel_placeholder(ht.float32, global_shape=[dp_size,1,1,1,1], ds_hierarchy=text_input_ds_hierarchy, device_group_hierarchy=text_input_dg_hierarchy, name='input_ids')
    # position_ids = ht.parallel_placeholder(ht.int64, global_shape=[2 * dummy_size, args.hidden_size], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='position_ids')
    # token_type_ids = ht.parallel_placeholder(ht.int64, global_shape=[dummy_size], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='token_type_ids')
    # attention_mask = ht.parallel_placeholder(ht.float32, global_shape=[dummy_size], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='attention_mask')
    # loss_mask =  ht.parallel_placeholder(ht.float32, global_shape=[dummy_size], ds_hierarchy=label_ds_hierarchy, device_group_hierarchy=label_dg_hierarchy, name='loss_mask')
    masked_lm_labels = ht.parallel_placeholder(ht.float32, global_shape=[dp_size, args.block_out_channels[-1], 15, 15, 15], ds_hierarchy=label_ds_hierarchy, device_group_hierarchy=label_dg_hierarchy, name='masked_lm_labels')

    # 3. Build Model Weight

    
    print("hidden_size", args.hidden_size)
    mmdit_config = MMDiTConfig(
        vocab_size=args.vocab_size, 
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        n_layer=args.num_hidden_layers, 
        n_head=args.num_attention_heads, 
        resid_pdrop=args.dropout_prob,
        embd_pdrop=args.dropout_prob,
        attn_pdrop=args.dropout_prob,
        activation_function=args.hidden_act,
        use_flash_attn=args.use_flash_attn,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        frequency_embedding_size = args.frequency_embedding_size,
        block_out_channels = args.block_out_channels,
        layers_per_block = args.layers_per_block,
        dqtype = "bf16" if args.bf16 else "fp32"
    )

    mmdit_config.mbs_times_dp_symbol = ht.IntSymbol(0)
    mmdit_config.text_seq_len_symbol = ht.IntSymbol(0)
    mmdit_config.vision_seq_len_symbol = ht.IntSymbol(0)

    mmdit_config.mbs_times_dp_symbol.set_data(1)
    # print("dp size", dp_size)
    mmdit_config.text_seq_len_symbol.set_data(args.max_seq_len)
    mmdit_config.vision_seq_len_symbol.set_data(343)

    print(f'{local_device}: init model begin...')
    mmdit_model = MMDitMHeadModel(config = mmdit_config, ds_parallel_configs = ds_parallel_configs)
    print(f'{local_device}: init model end...')

    # 4. Build Symbolic Shape

    mmdit_config.cu_seqlens_list = []
    for block_id, block in enumerate(mmdit_model.transformer.h):
        mmdit_config.cu_seqlens_list.append(
            ht.parallel_placeholder(
                ht.int32, 
                global_shape=[dummy_size], 
                ds_hierarchy=block.attn.qkv_dense.ds_union_map['split0_dup'], 
                device_group_hierarchy=block.attn.qkv_dense.device_group_unions,
                name=f'cu_seqlens_{block_id}'
            )
        )

    # just symbolic value, will change depend on real data
    mmdit_config.multi_seq_lens_symbol = []
    mmdit_config.multi_cp_group_symbol = []
    for i in range(len(label_ds_hierarchy)):
        cur_dp = label_ds_hierarchy[i].get(0).get_dim(0)
        mmdit_config.multi_seq_lens_symbol.append([text_ids.symbolic_shape[0] for _ in range(cur_dp)])
        mmdit_config.multi_cp_group_symbol.append([ht.IntSymbol(i) for i in range(cur_dp)])
    mmdit_config.max_seqlen_symbol = ht.IntSymbol(1)


    # 5. Build Forward Graph
    print(f'{local_device}: build model begin...')
    loss = mmdit_model(
        image_inputs = image_inputs,
        text_ids = text_ids,
        time_steps = time_steps,
        sqrt_alphas_cumprod = sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod,
        # position_ids=position_ids,
        # image_mask=image_mask,
        # video_mask=video_mask,
        # attention_mask=attention_mask,
        # loss_mask=loss_mask,
        # token_type_ids=token_type_ids,
        labels=masked_lm_labels
    )
    print(f'{local_device}: build model end...')

    # 6. Build Backward Graph
    print(f'{local_device}: optimizer minimize begin...')
    # opt = ht.SGDOptimizer(lr=args.lr, momentum = 0.0)
    opt = ht.AdamOptimizer(init_lr=args.lr, max_lr=args.lr, min_lr=args.lr, lr_warmup_steps=0, lr_decay_steps=1, lr_decay_style="constant")
    # opt = ht.AdamOptimizer(lr=args.lr)
    train_op = opt.minimize(loss)
    print(f'{local_device}: optimizer minimize end...')

    print(f'{local_device}: build dataset begin...')
    train_dataset = train_dataset_provider(args)
    print(f'{local_device}: build dataset end...')     

    print(f"build diffusion begin...")
    diffusion = create_diffusion()
    print(f"build diffusion end...")   

    def start():
        consumed_samples = 0
        for epoch in range(args.epochs):
            strategy_id = 0
            consumed_samples = run_plan("train", epoch, consumed_samples, strategy_id, args.max_seq_len)
            return consumed_samples

    def run_plan(
        self,
        epoch = 0,
        consumed_samples = 0,
        strategy_id = 0,
        max_padded_seqlen = None,
    ):     
        dp_size = multi_dp_size[strategy_id]
        # tp_pp_list = multi_tp_pp_list[strategy_id]
        gpu_pos = multi_gpu_pos[strategy_id]
        gpu_id = all_devices.get_index(local_device)


        dp_id, stage_id, is_vision = None, None, False
        print("gpu", gpu_id, gpu_pos, (gpu_id in gpu_pos) )
        if gpu_id in gpu_pos:
            dp_id, stage_id, is_vision = gpu_pos[gpu_id].dp_id, gpu_pos[gpu_id].stage_id, gpu_pos[gpu_id].is_vision
            assert dp_id < dp_size, "dp size mismatches"
        

        print(f"{local_device}: gpu_id = {gpu_id}, dp_id = {dp_id}, stage_id = {stage_id}, is_vision = {is_vision}")

        if dp_id != None:
            train_iter = train_dataloader_provider(train_dataset, consumed_samples, args.global_batch_size)


        # load data for each dp
        for step in range(args.steps):
            input_batch, label_batch, cu_seqlens_list = None, None, None

            print("dp_id", dp_id)
            if dp_id != None:
                global_batch = next(train_iter).numpy()
                sorted_batch, sorted_len = get_sorted_batch_and_len(global_batch, tokenizer.pad)
                # padding
                assert args.global_batch_size % dp_size == 0, "global_batch_size should be divided by dp_size when padding"
                batch_size_per_dp = args.global_batch_size // dp_size
                batch_indices = list(range(batch_size_per_dp * dp_id, batch_size_per_dp * (dp_id + 1)))
                assert max_padded_seqlen, "padding should provide the max seqlen after padding"
                mmdit_config.max_seqlen_symbol.set_data(max_padded_seqlen - 1) 
                input_bucket, label_bucket = get_input_and_label_buckets(sorted_batch, tokenizer.pad, batch_indices, max_padded_seqlen, alignment)
                input_bucket.pad_data()
                label_bucket.pad_data()
                input_batch, label_batch = input_bucket.padded_batch(), label_bucket.padded_batch()
                cu_seqlens_list = input_bucket.padded_cu_seqlens_list()
            if input_batch == None or len(input_batch) < 1: 
                raise NotImplementedError("currently not support GPUs with no data")
            else:

                def generate_random_images(num_images=5):
                    images = []
                    for _ in range(num_images):
                        # height = np.random.randint(50, 500)  # 随机高度
                        # width = np.random.randint(50, 500)   # 随机宽度
                        
                        depth = D
                        height = H
                        width = W
                        image = np.random.randint(0, 256, (depth, height, width, 3), dtype=np.uint8)  # RGB 图片
                        # images.append((image, (height, width)))
                        images.append(image)
                    return images

                dp_size = 1
                images = generate_random_images(dp_size)
                image_list = [np.array(images).astype(np.float32).reshape(-1, 3, D, H, W)]
                labels_list = [np.random.randint(0, 1, (dp_size, args.block_out_channels[-1], 15, 15, 15)).astype(np.float32)]
                text_list = [np.random.randint(0, 100, (dp_size, args.max_seq_len,)).astype(np.int64)]
                time_steps_list = [np.random.randint(0, 100, (dp_size,))]
                sqrt_alphas_cumprod_list, sqrt_one_minus_alphas_cumprod_list = diffusion.q_sample(t=time_steps_list[0])
                sqrt_alphas_cumprod_list = [sqrt_alphas_cumprod_list.astype(np.float32)]
                sqrt_one_minus_alphas_cumprod_list = [sqrt_one_minus_alphas_cumprod_list.astype(np.float32)]
                time_steps_list[0] = time_steps_list[0].repeat(args.frequency_embedding_size).reshape(-1, args.frequency_embedding_size).astype(np.float32)
                print("time_steps_list", time_steps_list[0].shape)
                print("labels_list", labels_list[0].shape)
                
                feed_dict = {
                    image_inputs: image_list,
                    text_ids: text_list,
                    time_steps: time_steps_list,
                    sqrt_alphas_cumprod: sqrt_alphas_cumprod_list,
                    sqrt_one_minus_alphas_cumprod: sqrt_one_minus_alphas_cumprod_list,
                    masked_lm_labels: labels_list
                }


            start_time = time.time()
            try:
                results = train_op.graph.run(
                    loss, 
                    [loss, train_op], 
                    feed_dict=feed_dict, 
                    num_micro_batches=1, 
                    cur_strategy_id=strategy_id,
                    run_level = ht.run_level("update")
                )    
            except RuntimeError as e:
                print(e)
                with open("./logs/exception.txt", 'w') as file:
                    print(f"{local_device}:", file=file)
                    print(e, file=file)
                os.killpg(0, signal.SIGTERM)


        end_time = time.time()
        consumed_samples += args.global_batch_size
        # 如果在pipeline的最后一个stage上那么就打印loss
        if stage_id == multi_tp_pp_list[0][dp_id][1] - 1 and len(results) > 0:
            loss_out = results[0].numpy(force=True).mean()
            print(f"{local_device}: [Epoch {epoch}] (step {step}, consumed_samples = {consumed_samples}): loss = {loss_out:.3f}, time = {end_time - start_time:.4f}")
        
        return consumed_samples

    start()
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Training Configuration")

    # 添加各个模块的参数
    parser = add_all_args(parser=parser)
    # 解析参数
    args = parser.parse_args()
    distributed_init(args)

    print("Local device world rank is", all_devices.get_index(local_device))
    args.rank = all_devices.get_index(local_device)


    args.multi_tp_pp_list = ast.literal_eval(args.multi_tp_pp_list)
    assert len(args.multi_tp_pp_list) >= 1, "there should be at least one strategy"

    with ht.graph("define_and_run", num_strategy=len(args.multi_tp_pp_list)):
        if args.bf16:
            precision = "ht.bfloat16"
        else:
            precision = "ht.float32"
        print(f'{local_device}: use precision {precision}')
        with ht.autocast(eval(precision)):            
            pretrain(args)
            print(f'{local_device}: train hetu ds parallel end...')




