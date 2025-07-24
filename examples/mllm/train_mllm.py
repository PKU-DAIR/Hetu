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

from torch.profiler import profile, ProfilerActivity
import torch
from arugments import add_all_args
from model.hetu_mllm import MLLMModel
from mllm_config import MLLMConfig, VisionConfig, LLaMAConfig
from data_utils import LLaMAJsonDataset, build_data_loader, build_tokenizer, HetuMLLMProcessor, HetuImageProcessor
from data_utils import get_sorted_by_text_and_video, get_sorted_batch_and_len, get_input_and_label_buckets, get_bucket
from data_utils.multimodal_dataset import NExTQADataset
from parallel_utils import read_ds_parallel_config, parse_multi_ds_parallel_config, convert_strategy, generate_mllm_model_ds_parallel_config
from typing import List
from analyze_FLOPs_memory import analyze_FLOPs_memory


local_device = None
all_devices = None
tokenizer = None
ds_parallel_config_path = "./ds_parallel_config/"
alignment = 128
IMAGE_TOKEN = -200
profiler = None

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
    # config = LLaMaDatasetConfig(
    #     random_seed=args.seed,
    #     sequence_length=args.max_seq_len,
    #     blend=args.data_path,
    #     blend_per_split=[None, None, None],
    #     split=args.split,
    #     path_to_cache=args.data_cache_path,
    #     tokenizer=tokenizer,
    #     reset_position_ids=False,
    #     reset_attention_mask=False,
    #     eod_mask_loss=False,
    #     vocab_size=args.vocab_size,
    # )
    # train_val_test_num_samples = [args.epochs * args.steps * args.global_batch_size, 0, 0]
    # train_ds, valid_ds, test_ds = BlendedHetuDatasetBuilder(
    #     LLaMAJsonDataset,
    #     train_val_test_num_samples,
    #     config
    # ).build()
    train_ds = NExTQADataset(tokenizer = tokenizer, args = args)
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
    global profiler
    
    vision_multi_tp_pp_list = args.vision_multi_tp_pp_list
    llm_multi_tp_pp_list = args.llm_multi_tp_pp_list

    vision_num_strategy = len(vision_multi_tp_pp_list)
    llm_num_strategy = len(llm_multi_tp_pp_list)
    assert vision_num_strategy == 1 and llm_num_strategy == 1, "currently only support one strategy for each model"

    vision_multi_dp_size = [len(tp_pp_list) for tp_pp_list in vision_multi_tp_pp_list]
    llm_multi_dp_size = [len(tp_pp_list) for tp_pp_list in llm_multi_tp_pp_list]

    # assert vision_multi_dp_size == llm_multi_dp_size, "vision and llm should have the same number of data parallelism"

    multi_gpu_pos = []
    multi_config_file_path = []
    for strategy_id in range(llm_num_strategy):
        # 获取GPU的位置
        # 原则是不让tp跨机并尽可能贪心地让pp跨机
        vision_layers_tp_groups, llm_layers_tp_groups, gpu_pos = convert_strategy(vision_multi_tp_pp_list[strategy_id], llm_multi_tp_pp_list[strategy_id], args.ngpus, args.num_hidden_layers)
        config_file_path = ds_parallel_config_path + f"strategy_{strategy_id}.txt"
        # print("layers_tp_groups", layers_tp_groups)
        generate_mllm_model_ds_parallel_config(args.ngpus, args.vision_num_layers, args.num_hidden_layers, vision_layers_tp_groups, llm_layers_tp_groups,config_file_path)
        print(f"Strategy {strategy_id}, gpu positions are: {gpu_pos}")
        multi_gpu_pos.append(gpu_pos)
        multi_config_file_path.append(config_file_path)
    ds_parallel_configs = read_ds_parallel_config(",".join(multi_config_file_path), llm_num_strategy)

    # 1. Config Information
    text_input_ds_hierarchy, text_input_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'text_input')
    image_input_ds_hierarchy, image_input_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'image_input')
    label_ds_hierarchy, label_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'label')

    # 2. Build Placeholders
    dp_size = text_input_ds_hierarchy[0].get(0).get_dim(0)
    dummy_size = dp_size * args.max_seq_len
    embed_dim = args.patch_size * args.patch_size * args.temporal_patch_size * args.in_channels
    # mbs_times_dp = dp_size * args.micro_batch_size
    print("dp_size", dp_size)
    print("args.max_seq_len", args.max_seq_len)
    print("dummy_size", dummy_size)
    print("embed_dim", embed_dim)
    image_inputs = ht.parallel_placeholder(ht.float32, global_shape=[2 * dummy_size, embed_dim], ds_hierarchy=image_input_ds_hierarchy, device_group_hierarchy=image_input_dg_hierarchy, name='input_ids')
    text_ids = ht.parallel_placeholder(ht.int64, global_shape=[2 * dummy_size], ds_hierarchy=text_input_ds_hierarchy, device_group_hierarchy=text_input_dg_hierarchy, name='input_ids')
    # position_ids = ht.parallel_placeholder(ht.int64, global_shape=[2 * dummy_size, args.hidden_size], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='position_ids')
    image_mask = ht.parallel_placeholder(ht.int64, global_shape=[2 * dummy_size, args.hidden_size], ds_hierarchy=text_input_ds_hierarchy, device_group_hierarchy=text_input_dg_hierarchy, name='image_mask')
    video_mask = ht.parallel_placeholder(ht.int64, global_shape=[2 * dummy_size, args.hidden_size], ds_hierarchy=text_input_ds_hierarchy, device_group_hierarchy=text_input_dg_hierarchy, name='video_mask')
    # token_type_ids = ht.parallel_placeholder(ht.int64, global_shape=[dummy_size], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='token_type_ids')
    # attention_mask = ht.parallel_placeholder(ht.float32, global_shape=[dummy_size], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='attention_mask')
    loss_mask =  ht.parallel_placeholder(ht.float32, global_shape=[2 * dummy_size], ds_hierarchy=label_ds_hierarchy, device_group_hierarchy=label_dg_hierarchy, name='loss_mask')
    masked_lm_labels = ht.parallel_placeholder(ht.int64, global_shape=[dummy_size + dummy_size], ds_hierarchy=label_ds_hierarchy, device_group_hierarchy=label_dg_hierarchy, name='masked_lm_labels')

    

    # 3. Build Model Weight
    vision_config = VisionConfig(
        in_channels = 3,
        patch_size = args.patch_size,
        temporal_patch_size=args.temporal_patch_size,
        num_hidden_layers = args.num_hidden_layers,
        num_attention_heads = args.num_attention_heads,
        embed_dim = args.vision_embed_dim,
        hidden_size = args.hidden_size,  # LLM对应的embed_dim, 需要在Vision最后将embed_dim转化为hidden size
        use_flash_attn=args.use_flash_attn,
        hidden_dropout = args.vision_dropout,
        attention_dropout = args.vision_dropout,
        mlp_dim=args.vision_mlp_dim,
        acitvation_func = args.hidden_act,
        dqtype = "bf16" if args.bf16 else "fp32"
    )

    llm_config = LLaMAConfig(
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
        dqtype = "bf16" if args.bf16 else "fp32"
    )

    print(f'{local_device}: init model begin...')
    mllm_model = MLLMModel(vision_config = vision_config, llm_config = llm_config, ds_parallel_configs = ds_parallel_configs)
    print(f'{local_device}: init model end...')

    # 4. Build Symbolic Shape
    vision_config.cu_seqlens_list = []
    llm_config.cu_seqlens_list = []
    for block_id, block in enumerate(mllm_model.vision.h):
        vision_config.cu_seqlens_list.append(
            ht.parallel_placeholder(
                ht.int32, 
                global_shape=[dummy_size], 
                ds_hierarchy=block.attn.qkv_dense.ds_union_map['split0_dup'], 
                device_group_hierarchy=block.attn.qkv_dense.device_group_unions,
                name=f'cu_seqlens_{block_id}'
            )
        )

    for block_id, block in enumerate(mllm_model.llm.h):
        llm_config.cu_seqlens_list.append(
            ht.parallel_placeholder(
                ht.int32, 
                global_shape=[dummy_size], 
                ds_hierarchy=block.attn.qkv_dense.ds_union_map['split0_dup'], 
                device_group_hierarchy=block.attn.qkv_dense.device_group_unions,
                name=f'cu_seqlens_{block_id}'
            )
        )

    # just symbolic value, will change depend on real data
    vision_config.multi_seq_lens_symbol = []
    vision_config.multi_cp_group_symbol = []
    for i in range(len(image_input_ds_hierarchy)):
        cur_dp = image_input_ds_hierarchy[i].get(0).get_dim(0) # dp_i for strategy_i
        vision_config.multi_seq_lens_symbol.append([image_inputs.symbolic_shape[0] for _ in range(cur_dp)])
        vision_config.multi_cp_group_symbol.append([ht.IntSymbol(i) for i in range(cur_dp)])
    vision_config.max_seqlen_symbol = ht.IntSymbol(1)


    llm_config.multi_seq_lens_symbol = []
    llm_config.multi_cp_group_symbol = []
    for i in range(len(label_ds_hierarchy)):
        cur_dp = label_ds_hierarchy[i].get(0).get_dim(0)
        llm_config.multi_seq_lens_symbol.append([text_ids.symbolic_shape[0] for _ in range(cur_dp)])
        llm_config.multi_cp_group_symbol.append([ht.IntSymbol(i) for i in range(cur_dp)])
    llm_config.max_seqlen_symbol = ht.IntSymbol(1)

    llm_config.packing_slice_list = []
    assert len(image_input_ds_hierarchy) == len(text_input_ds_hierarchy), "image_input_ds_hierarchy and text_input_ds_hierarchy should have the same length"
    for i in range(len(image_input_ds_hierarchy)):
        vision_dp = image_input_ds_hierarchy[i].get(0).get_dim(0)
        llm_config.packing_slice_list.append([])
        for j in range(vision_dp):
            llm_dp = text_input_ds_hierarchy[i].get(j).get_dim(0)
            llm_config.packing_slice_list[i].append([ht.IntSymbol(k * k) for k in range(llm_dp + 1)])


    # 5. Build Forward Graph
    print(f'{local_device}: build model begin...')
    loss = mllm_model(
        image_inputs = image_inputs,
        text_ids = text_ids,
        # position_ids=position_ids,
        image_mask=image_mask,
        video_mask=video_mask,
        # attention_mask=attention_mask,
        loss_mask=loss_mask,
        # token_type_ids=token_type_ids,
        labels=masked_lm_labels
    )
    print(f'{local_device}: build model end...')

    # 6. Build Backward Graph
    print(f'{local_device}: optimizer minimize begin...')
    # opt = ht.SGDOptimizer(lr=args.lr, momentum = 0.0)
    opt = ht.AdamOptimizer(init_lr=args.lr, max_lr=args.lr, min_lr=args.lr, lr_warmup_steps=0, lr_decay_steps=1, lr_decay_style="constant")
    train_op = opt.minimize(loss)
    print(f'{local_device}: optimizer minimize end...')

    print(f'{local_device}: build dataset begin...')
    train_dataset = train_dataset_provider(args)
    print(f'{local_device}: build dataset end...')        


    def start():
        global profiler
        consumed_samples = 0
        
        print("args.torch_profile", args.torch_profile)
        if args.torch_profile:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CUDA
                ],
            )
            print("profiler", profiler)
        for epoch in range(args.epochs):
            strategy_id = 0

            # if profiler is not None:
                # profiler.start()

            consumed_samples = run_plan("train", epoch, consumed_samples, strategy_id, args.max_seq_len)
            # if profiler is not None:
                # profiler.stop()
                # profiler.export_chrome_trace(f"/home/pkuhetu/njw1123/hetu_mm/examples/mllm/tensorboard_log/trace_{local_device}.json")
            return consumed_samples

    def run_plan(
        self,
        epoch = 0,
        consumed_samples = 0,
        strategy_id = 0,
        max_padded_seqlen = None,
    ):     
        global profiler
        vision_dp_size = vision_multi_dp_size[strategy_id]
        llm_dp_size = llm_multi_dp_size[strategy_id]
        print("vision_dp_size", vision_dp_size)
        print("llm_dp_size", llm_dp_size)
        # tp_pp_list = multi_tp_pp_list[strategy_id]
        gpu_pos = multi_gpu_pos[strategy_id]
        print
        gpu_id = all_devices.get_index(local_device)


        vision_dp_id, llm_dp_id, stage_id, is_vision = None, None, None, False
        print("gpu", gpu_id, gpu_pos, (gpu_id in gpu_pos) )
        if gpu_id in gpu_pos:
            dp_id, stage_id, is_vision = gpu_pos[gpu_id].dp_id, gpu_pos[gpu_id].stage_id, gpu_pos[gpu_id].is_vision
            print("dp_id", dp_id, stage_id, is_vision)
            if is_vision:
                vision_dp_id = dp_id
                assert vision_dp_id < vision_dp_size, "dp size mismatches"
                if vision_dp_id < llm_dp_size:
                    llm_dp_id = vision_dp_id
                else:
                    llm_dp_id = 0
            else:
                llm_dp_id = dp_id
                assert llm_dp_id < llm_dp_size, "dp size mismatches"
                if llm_dp_id < vision_dp_size:
                    vision_dp_id = llm_dp_id
                else:
                    vision_dp_id = 0
        
        print("vision_dp_id", vision_dp_id)
        print("llm_dp_id", llm_dp_id)

        print(f"{local_device}: gpu_id = {gpu_id}, dp_id = {dp_id}, stage_id = {stage_id}, is_vision = {is_vision}")

        if dp_id != None:
            train_iter = train_dataloader_provider(train_dataset, consumed_samples, args.global_batch_size)


        # load data for each dp
        for step in range(args.steps):
            try:
                global_batch = next(train_iter)
            except StopIteration:
                print("训练数据迭代完成")
                break
            except Exception as e:
                print(f"获取训练数据时发生错误: {e}")
                raise
            vision_config.max_seqlen_symbol.set_data(args.vision_max_seqlen) 
            llm_config.max_seqlen_symbol.set_data(args.text_max_seqlen)   
            print('batching method is ', args.batching_method)
            if args.batching_method == 0: # padding
                raise NotImplementedError("padding is not supported")
            elif args.batching_method == 1: # greedy packing
                sorted_batch = get_sorted_by_text_and_video(global_batch, tokenizer.pad)
                bucket = get_bucket(sorted_batch, tokenizer.pad, args.vision_max_seqlen, args.text_max_seqlen, IMAGE_TOKEN, alignment)
                bucket.pack_data(args.batching_method, vision_dp_size, llm_dp_size)
                vision_input_batch = bucket.get_packed_vision_batch()
                llm_input_batch = bucket.get_packed_llm_batch()
                llm_label_batch = bucket.get_packed_llm_label_batch()
                vision_cu_seqlens_list = bucket.get_packed_vision_cu_seqlens_list()
                llm_cu_seqlens_list = bucket.get_packed_llm_cu_seqlens_list()
                llm_input_mask = bucket.get_llm_input_mask()
                assert vision_dp_size == llm_dp_size, "vision_dp_size should be equal to the length of llm_dp_size"
                llm_micro_batch_per_dp = len(llm_input_batch) // llm_dp_size
                print("llm_micro_batch_per_dp", llm_micro_batch_per_dp)
                llm_start_idx = llm_dp_id * llm_micro_batch_per_dp
                llm_end_idx = (llm_dp_id + 1) * llm_micro_batch_per_dp      
                # 批量处理数据
                vision_list = [np.array(vision_input_batch[i], dtype=np.float32) for i in range(llm_start_idx, llm_end_idx)]
                text_list = [np.array(llm_input_batch[i], dtype=np.int64) for i in range(llm_start_idx, llm_end_idx)]
                text_mask_list = [np.array(llm_input_mask[i], dtype=np.int64) for i in range(llm_start_idx, llm_end_idx)]
                label_list = [np.array(llm_label_batch[i], dtype=np.int64) for i in range(llm_start_idx, llm_end_idx)]
                vision_cu_seqlens_list = [np.array(vision_cu_seqlens_list[i], dtype=np.int32) for i in range(llm_start_idx, llm_end_idx)]
                llm_cu_seqlens_list = [np.array(llm_cu_seqlens_list[i], dtype=np.int32) for i in range(llm_start_idx, llm_end_idx)]
                # 计算vision_dp_size和llm_dp_size的最小公倍数
            elif args.batching_method == 2 or args.batching_method == 3: # 混合packing
                sorted_batch = get_sorted_by_text_and_video(global_batch, tokenizer.pad)
                bucket = get_bucket(sorted_batch, tokenizer.pad, args.vision_max_seqlen, args.text_max_seqlen, IMAGE_TOKEN, alignment)
                bucket.pack_data(args.batching_method, vision_dp_size, llm_dp_size)
                vision_input_batch = bucket.get_packed_vision_batch()
                llm_input_batch = bucket.get_packed_llm_batch()
                llm_label_batch = bucket.get_packed_llm_label_batch()
                vision_cu_seqlens_list = bucket.get_packed_vision_cu_seqlens_list()
                llm_cu_seqlens_list = bucket.get_packed_llm_cu_seqlens_list()
                llm_input_mask = bucket.get_llm_input_mask()
                packed_llm_packing_slice_list = bucket.get_packed_llm_packing_slice_list()
                packed_vision_packing_slice_list = bucket.get_packed_vision_packing_slice_list()
                print("vision dp size", len(vision_input_batch))
                print("llm dp size", len(llm_input_batch))
                vision_seq_len_list = [vision_input_batch[i].shape[0] for i in range(len(vision_input_batch))]
                llm_seq_len_list = [llm_input_batch[i].shape[0] for i in range(len(llm_input_batch))]
                print("vision_seq_len_list", vision_seq_len_list)
                print("llm_seq_len_list", llm_seq_len_list)
                vision_micro_batch_per_dp = len(vision_input_batch) // vision_dp_size
                llm_micro_batch_per_dp = len(llm_input_batch) // llm_dp_size
                print("vision_micro_batch_per_dp", vision_micro_batch_per_dp)
                print("llm_micro_batch_per_dp", llm_micro_batch_per_dp)
                assert vision_micro_batch_per_dp == llm_micro_batch_per_dp, "vision_micro_batch_per_dp should be equal to llm_micro_batch_per_dp"
                print("all vision_cu_seqlens_list", vision_cu_seqlens_list)
                print("all llm_cu_seqlens_list", llm_cu_seqlens_list)
                vision_indices = [i for i in range(len(vision_input_batch)) if i % vision_dp_size == vision_dp_id]
                llm_indices = [i for i in range(len(llm_input_batch)) if i % llm_dp_size == llm_dp_id]
                vision_list = [np.array(vision_input_batch[i], dtype=np.float32) for i in vision_indices[:vision_micro_batch_per_dp]]
                text_list = [np.array(llm_input_batch[i], dtype=np.int64) for i in llm_indices[:llm_micro_batch_per_dp]]
                text_mask_list = [np.array(llm_input_mask[i], dtype=np.int64) for i in llm_indices[:llm_micro_batch_per_dp]]
                loss_mask_list = [np.ones_like(mask).astype(np.float32) - np.array(mask, copy=True).astype(np.float32) for mask in text_mask_list]
                # loss_mask_list = [np.array(mask, copy=True).astype(np.float32) for mask in text_mask_list]
                label_list = [np.array(llm_label_batch[i], dtype=np.int64) for i in llm_indices[:llm_micro_batch_per_dp]]
                vision_cu_seqlens_list = [np.array(vision_cu_seqlens_list[i], dtype=np.int32) for i in vision_indices[:vision_micro_batch_per_dp]]
                llm_cu_seqlens_list = [np.array(llm_cu_seqlens_list[i], dtype=np.int32) for i in llm_indices[:llm_micro_batch_per_dp]]
                # packed_llm_packing_slice_list = [np.array(packed_llm_packing_slice_list[i], dtype=np.int32) for i in llm_indices[:llm_micro_batch_per_dp]]
                packed_vision_packing_slice_list_preprocessed = []
                for i in range(vision_dp_size):
                    packed_vision_packing_slice_list_preprocessed.append([])
                    for j in range(llm_dp_size + 1):
                        packed_vision_packing_slice_list_preprocessed[i].append([])

                for idx, packing_slice in enumerate(packed_vision_packing_slice_list):
                    vision_dp_id = idx % vision_dp_size
                    for j in range(llm_dp_size + 1):
                        packed_vision_packing_slice_list_preprocessed[vision_dp_id][j].append(packing_slice[j])
                print("packed_vision_packing_slice_list_preprocessed", packed_vision_packing_slice_list_preprocessed)

            else:
                raise ValueError("Invalid batching method")

            analyze_FLOPs_memory(vision_seq_len_list, llm_seq_len_list, vision_config, llm_config)

            for vision in vision_list:
                print(f"vision shape is {vision.shape}")
            for text in text_list:
                print(f"text shape is {text.shape}")
            for i, text_mask in enumerate(text_mask_list):
                # Create contiguous array by using np.zeros and filling it
                text_mask_expanded = np.zeros((text_mask.shape[0], args.hidden_size), dtype=text_mask.dtype)
                text_mask_expanded[:,:] = text_mask[:,np.newaxis]
                text_mask_list[i] = text_mask_expanded
                print(f"text_mask shape is {text_mask_expanded.shape}")
            for i, loss_mask_ in enumerate(loss_mask_list):
                print(f"loss_mask shape is {loss_mask_.shape}")
            for label in label_list:
                print(f"label shape is {label.shape}")
            for i, vision_cu_seqlens in enumerate(vision_cu_seqlens_list):
                print(f"vision_cu_seqlens is {vision_cu_seqlens}")
                assert(vision_cu_seqlens[-1] == vision_list[i].shape[0]), "vision_cu_seqlens should be equal to the sum of the product of image_grid_thws"
            for i, llm_cu_seqlens in enumerate(llm_cu_seqlens_list):
                print(f"llm_cu_seqlens is {llm_cu_seqlens}")
                assert(llm_cu_seqlens[-1] == text_list[i].shape[0]), "llm_cu_seqlens should be equal to the sum of the product of image_grid_thws"
                assert(llm_cu_seqlens[-1] == label_list[i].shape[0]), "llm_cu_seqlens should be equal to the sum of the product of image_grid_thws"
            

            print("image_inputs", vision_list[0], np.mean(vision_list[0]))
            print("text_ids", text_list[0], np.mean(text_list[0]))
            print("image_mask", text_mask_list[0], np.mean(text_mask_list[0]))
            print("masked_lm_labels", label_list[0], np.mean(label_list[0]))
            print("loss_mask", loss_mask_list[0], np.sum(loss_mask_list[0]))
            feed_dict = {
                image_inputs: vision_list,
                text_ids: text_list,
                image_mask: text_mask_list,
                loss_mask: loss_mask_list,
                masked_lm_labels: label_list
            }

            for i in range(vision_config.num_hidden_layers):
                feed_dict[vision_config.cu_seqlens_list[i]] = vision_cu_seqlens_list
            for i in range(llm_config.num_hidden_layers):
                feed_dict[llm_config.cu_seqlens_list[i]] = llm_cu_seqlens_list

            # loss_mask_list = []
            # for idx, label in enumerate(label_list):
            #     micro_batch_loss_mask = np.zeros_like(label, dtype=np.float32)
            #     print("micro_batch_loss_mask", micro_batch_loss_mask.shape)
            #     print("text_mask_list", text_mask_list[idx].shape)
            #     # micro_batch_loss_mask[cu_seqlens_list[idx][0]:cu_seqlens_list[idx][1]] = 1
            #     loss_mask_list.append(micro_batch_loss_mask)
            # feed_dict[loss_mask] = loss_mask_list



            int_symbol_dict = {}
            # print("packed_vision_packing_slice_list", packed_vision_packing_slice_list)

            for i in range(vision_dp_size):
                for j in range(llm_dp_size + 1):
                    int_symbol_dict[llm_config.packing_slice_list[strategy_id][i][j]] = packed_vision_packing_slice_list_preprocessed[i][j]

            # print("int_symbol_dict", int_symbol_dict)
            # print("feed_dict", feed_dict)
            
            start_time = time.time()
            with ht.profiler(enabled = True, use_cuda = True, record_shapes = True, profile_memory = True) as profile:
                if profiler is not None:
                    profiler.start()    
                try:
                    results = train_op.graph.run(
                        loss, 
                        [loss, train_op], 
                        feed_dict = feed_dict, 
                        int_symbol_dict = int_symbol_dict, 
                        num_micro_batches = len(vision_list), 
                        cur_strategy_id = strategy_id,
                        run_level = ht.run_level("update"),
                    )
                except RuntimeError as e:
                    print(e)
                    with open("./logs/exception.txt", 'w') as file:
                        print(f"{local_device}:", file=file)
                        print(e, file=file)
                    os.killpg(0, signal.SIGTERM) 
                if profiler is not None:
                    profiler.stop()   


            end_time = time.time()
            consumed_samples += args.global_batch_size
            # 如果在pipeline的最后一个stage上那么就打印loss
            # print(tp_pp_list[dp_id])
            if stage_id == llm_multi_tp_pp_list[0][dp_id][1] - 1 and len(results) > 0 and results[0] is not None:
                loss_out = results[0].numpy(force=True).mean()
                print(f"{local_device}: [Epoch {epoch}] (step {step}, consumed_samples = {consumed_samples}): loss = {loss_out:.3f}, time = {end_time - start_time:.4f}")
        
        print("profiler is ", profiler)
        if profiler is not None:
            print("profiler is not None" )
            profiler.export_chrome_trace(f"/home/pkuhetu/njw1123/hetu_mm/examples/mllm/tensorboard_log/trace_{local_device}.json")
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

    args.llm_multi_tp_pp_list = ast.literal_eval(args.llm_multi_tp_pp_list)
    assert len(args.llm_multi_tp_pp_list) >= 1, "there should be at least one strategy"
    args.vision_multi_tp_pp_list = ast.literal_eval(args.vision_multi_tp_pp_list)
    assert len(args.vision_multi_tp_pp_list) >= 1, "there should be at least one strategy"
    assert len(args.llm_multi_tp_pp_list) == len(args.vision_multi_tp_pp_list), "llm and vision should have the same number of strategies"

    with ht.graph("define_and_run", num_strategy=len(args.llm_multi_tp_pp_list)):
        if args.bf16:
            precision = "ht.bfloat16"
        else:
            precision = "ht.float32"
        print(f'{local_device}: use precision {precision}')
        with ht.autocast(eval(precision)):            
            pretrain(args)
            print(f'{local_device}: train hetu ds parallel end...')




