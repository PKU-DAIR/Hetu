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
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from arugments import add_all_args
from model.hetu_vision import VITModel
from vit_config import VisionConfig
from data_utils import LLaMAJsonDataset, build_data_loader, get_sorted_batch_and_len, build_fake_batch_and_len, get_input_and_label_buckets, LLaMaDatasetConfig, build_tokenizer, BlendedHetuDatasetBuilder, HetuMLLMProcessor, HetuImageProcessor
from parallel_utils import  read_ds_parallel_config, parse_multi_ds_parallel_config, convert_strategy, generate_vit_ds_parallel_config
from typing import List

local_device = None
all_devices = None
tokenizer = None
ds_parallel_config_path = "./ds_parallel_config/"
alignment = 128
IMAGE_TOKEN = -200


def distributed_init(args):
    global local_device, all_devices
    hostname = socket.gethostname()
    os.environ['HETU_LOCAL_HOSTNAME'] = hostname
    print(f"server_addr: {args.server_addr}, server_port: {args.server_port}")
    ht.init_comm_group(args.ngpus, server_address = args.server_addr + ":" + args.server_port)
    local_device = ht.local_device()
    all_devices = ht.global_device_group()
    # if local_device.index == 0:
    print(f'local_device: {local_device}, all_devices: {all_devices}')


def patchify(images, patch_size):
    # images: (B, H, W, C)
    B, H, W, C = images.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by patch size"

    num_patches_h = H // patch_size
    num_patches_w = W // patch_size

    # Reshape and permute to (B, num_patches, patch_area * C)
    patches = images.reshape(B, num_patches_h, patch_size, num_patches_w, patch_size, C)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)  # (B, num_patches_h, num_patches_w, patch_size, patch_size, C)
    patches = patches.reshape(B, num_patches_h * num_patches_w, patch_size * patch_size * C)


    start = 0
    vision_cu_seqlens_list = [0]
    for i in range(patches.shape[0]):
        start += patches.shape[1]
        vision_cu_seqlens_list.append(start)
    patches = patches.reshape(-1, patches.shape[-1])

    return patches, vision_cu_seqlens_list


def train_dataset_provider(args):

    traindir = os.path.join(args.data_path, 'train')
    normalize = transforms.Normalize(0.5, 0.5)

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])) 

    return train_dataset

def train_dataloader_provider(train_ds, consumed_samples, global_batch_size):
    data_loader = build_data_loader(train_ds, consumed_samples, global_batch_size)

    return iter(data_loader)
  
def get_dg_from_union(device, dg_union):
    for i, dg in enumerate(dg_union):
        if dg.contains(device):
            return i, dg
    return None, None


def pretrain(args):
    
    vision_multi_tp_pp_list = args.vision_multi_tp_pp_list

    vision_num_strategy = len(vision_multi_tp_pp_list)
    assert vision_num_strategy == 1, "currently only support one strategy for each model"

    vision_multi_dp_size = [len(tp_pp_list) for tp_pp_list in vision_multi_tp_pp_list]

    multi_gpu_pos = []
    multi_config_file_path = []
    for strategy_id in range(vision_num_strategy):
        # 获取GPU的位置
        # 原则是不让tp跨机并尽可能贪心地让pp跨机
        vision_layers_tp_groups, gpu_pos = convert_strategy(vision_multi_tp_pp_list[strategy_id], args.ngpus, args.vision_num_layers)
        config_file_path = ds_parallel_config_path + f"strategy_{strategy_id}.json"
        generate_vit_ds_parallel_config(args.ngpus, args.vision_num_layers, vision_layers_tp_groups,config_file_path)
        print(f"Strategy {strategy_id}, gpu positions are: {gpu_pos}")
        multi_gpu_pos.append(gpu_pos)
        multi_config_file_path.append(config_file_path)
    ds_parallel_configs = read_ds_parallel_config(",".join(multi_config_file_path), vision_num_strategy)

    print(ds_parallel_config_path)

    # 1. Config Information
    image_input_ds_hierarchy, image_input_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'image_input')
    label_ds_hierarchy, label_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'label')

    # 2. Build Placeholders
    dp_size = image_input_ds_hierarchy[0].get(0).get_dim(0)
    dummy_size = args.max_seq_len
    embed_dim = args.patch_size * args.patch_size * 3
    # mbs_times_dp = dp_size * args.micro_batch_size
    print("image_input_ds_hierarchy", image_input_ds_hierarchy)
    print("image_input_dg_hierarchy", image_input_dg_hierarchy)
    image_inputs = ht.parallel_placeholder(ht.float32, global_shape=[args.micro_batch_size * dummy_size, embed_dim], ds_hierarchy=image_input_ds_hierarchy, device_group_hierarchy=image_input_dg_hierarchy, name='input_ids')
    #image_mask = ht.parallel_placeholder(ht.int64, global_shape=[2 * dummy_size, args.vision_embed_dim], ds_hierarchy=image_input_ds_hierarchy, device_group_hierarchy=image_input_dg_hierarchy, name='image_mask')
    #video_mask = ht.parallel_placeholder(ht.int64, global_shape=[2 * dummy_size, args.vision_embed_dim], ds_hierarchy=image_input_ds_hierarchy, device_group_hierarchy=image_input_dg_hierarchy, name='video_mask')
    # token_type_ids = ht.parallel_placeholder(ht.int64, global_shape=[dummy_size], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='token_type_ids')
    # attention_mask = ht.parallel_placeholder(ht.float32, global_shape=[dummy_size], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='attention_mask')
    # loss_mask =  ht.parallel_placeholder(ht.float32, global_shape=[dummy_size], ds_hierarchy=label_ds_hierarchy, device_group_hierarchy=label_dg_hierarchy, name='loss_mask')
    masked_lm_labels = ht.parallel_placeholder(ht.int64, global_shape=[args.micro_batch_size * dummy_size, args.vision_embed_dim], ds_hierarchy=label_ds_hierarchy, device_group_hierarchy=label_dg_hierarchy, name='masked_lm_labels')

    

    # 3. Build Model Weight
    vision_config = VisionConfig(
        in_channels = 3,
        patch_size = args.patch_size,
        num_hidden_layers = args.vision_num_layers,
        num_attention_heads = args.vision_num_heads,
        micro_batch_size = args.micro_batch_size * dp_size,
        embed_dim = args.vision_embed_dim,
        hidden_size = args.vision_embed_dim, # LLM对应的embed_dim, 需要在Vision最后将embed_dim转化为hidden size
        use_flash_attn=args.use_flash_attn,
        max_position_embeddings = args.max_position_embeddings,
        hidden_dropout = args.vision_dropout,
        attention_dropout = args.vision_dropout,
        mlp_dim=args.vision_mlp_dim,
        acitvation_func = args.hidden_act,
        dqtype = "bf16" if args.bf16 else "fp32"
    )


    print(f'{local_device}: init model begin...')
    vit_model = VITModel(vision_config = vision_config, ds_parallel_configs = ds_parallel_configs)
    print(f'{local_device}: init model end...')

    # 4. Build Symbolic Shape
    vision_config.cu_seqlens_list = []
    for block_id, block in enumerate(vit_model.vision.h):
        vision_config.cu_seqlens_list.append(
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
        print("vision cur dp", image_input_ds_hierarchy[i].get(0).get_dim(0))
        cur_dp = image_input_ds_hierarchy[i].get(0).get_dim(0) # dp_i for strategy_i
        vision_config.multi_seq_lens_symbol.append([image_inputs.symbolic_shape[0] for _ in range(cur_dp)])
        vision_config.multi_cp_group_symbol.append([ht.IntSymbol(i) for i in range(cur_dp)])
    vision_config.max_seqlen_symbol = ht.IntSymbol(1)



    # 5. Build Forward Graph
    print(f'{local_device}: build model begin...')
    loss = vit_model(
        image_inputs = image_inputs,
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
    train_op = opt.minimize(loss)
    print(f'{local_device}: optimizer minimize end...')

    print(f'{local_device}: build dataset begin...')
    train_dataset = train_dataset_provider(args)
    print(f'{local_device}: build dataset end...')        

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
        dp_size = vision_multi_dp_size[strategy_id]
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
                global_batch = next(train_iter)
                input_batch = global_batch[0].numpy().transpose(0, 2, 3, 1)
                label_batch = global_batch[1].numpy()
                assert args.global_batch_size % dp_size == 0, "global_batch_size should be divided by dp_size when padding"
                batch_size_per_dp = args.global_batch_size // dp_size
                batch_indices = list(range(batch_size_per_dp * dp_id, batch_size_per_dp * (dp_id + 1)))
                input_batch = input_batch[batch_indices]
                label_batch = label_batch[batch_indices]
                assert max_padded_seqlen, "padding should provide the max seqlen after padding"
                vision_config.max_seqlen_symbol.set_data(max_padded_seqlen - 1) 

            if input_batch is None or len(input_batch) < 1: 
                raise NotImplementedError("currently not support GPUs with no data")
            else:
            
                # processor = HetuMLLMProcessor(image_processor=HetuImageProcessor(), tokenizer=tokenizer, chat_template=None)
                # text_inputs_data, image_inputs_data, videos_inputs_data = processor(input_batch, ["hello world hello world hello world <|image_pad|>"], None)
                input_batch, vision_cu_seqlens_list = patchify(input_batch, args.patch_size)
                print("input_batch", input_batch.shape)
                print("label_batch", label_batch.shape)
                image_list = [np.array(input_batch).astype(np.float32)]
                label_list = [np.array(label_batch).astype(np.int64)]

                feed_dict = {
                    image_inputs: image_list,
                    # image_mask: image_mask_list,
                    masked_lm_labels: label_list,
                }

                print("image_list", image_list[0].shape)
                print("vision_cu_seqlens_list", vision_cu_seqlens_list)


                for i in range(vision_config.num_hidden_layers):
                    # feed_dict[vision_config.cu_seqlens_list[i]] = [np.array([0, max_padded_seqlen]).astype(np.int32) for x in cu_seqlens_list]
                    feed_dict[vision_config.cu_seqlens_list[i]] = [np.array(vision_cu_seqlens_list).astype(np.int32)]
                # loss_mask_list = []
                # for idx, label in enumerate(label_list):
                #     micro_batch_loss_mask = np.zeros_like(label, dtype=np.float32)
                #     micro_batch_loss_mask[cu_seqlens_list[idx][0]:cu_seqlens_list[idx][1]] = 1
                #     loss_mask_list.append(micro_batch_loss_mask)
                # feed_dict[loss_mask] = loss_mask_list

            start_time = time.time()
            print("args.torch_profile", args.torch_profile)
            print("step", step)
            if args.torch_profile != 0 and step == 0:
                # with profile(activities=[ProfilerActivity.CUDA]) as prof:
                # torch.cuda.cudart().cudaProfilerStart()
                try:
                    print("feed_dict", feed_dict)
                    print("start run", time.time())
                    print("train_op graph", train_op.graph)

                    results = train_op.graph.run(
                        loss, 
                        [loss, train_op], 
                        feed_dict=feed_dict, 
                        int_symbol_dict=None,
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
                # torch.cuda.cudart().cudaProfilerStop()
                # prof.export_chrome_trace(f"/home/gehao/njw1123/hetu_mm/examples/mllm/trace_{local_device}.json")
                print("train done!")
            else:
                try:
                    results = train_op.graph.run(
                        loss, 
                        [loss, train_op], 
                        feed_dict=feed_dict, 
                        int_symbol_dict=None,
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
            # print(tp_pp_list[dp_id])
            if stage_id == vision_multi_tp_pp_list[0][dp_id][1] - 1 and len(results) > 0 and results[0][0] is not None:
                loss_out = results[0][0].numpy(force=True).mean()
                # print("results", results)
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

    print("vision_multi_tp_pp_list", args.vision_multi_tp_pp_list)

    args.vision_multi_tp_pp_list = ast.literal_eval(args.vision_multi_tp_pp_list)
    assert len(args.vision_multi_tp_pp_list) >= 1, "there should be at least one strategy"

    with ht.graph("define_and_run", num_strategy=len(args.vision_multi_tp_pp_list)):
        if args.bf16:
            precision = "ht.bfloat16"
        else:
            precision = "ht.float32"
        print(f'{local_device}: use precision {precision}')
        with ht.autocast(eval(precision)):            
            pretrain(args)
            print(f'{local_device}: train hetu ds parallel end...')




