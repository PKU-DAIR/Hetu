import os
import signal
import math
import time
import argparse
import json
import socket
import pynvml
import ast
import ptvsd
import numpy as np
import hetu as ht
from queue import Queue
from torch.profiler import profile, ProfilerActivity
from hetu.models.llama import LlamaLMHeadModel, LlamaConfig
from hetu.utils.data import HetuJsonDataset, build_data_loader
from hetu.utils.parallel import distributed_init, read_ds_parallel_config, parse_multi_ds_parallel_config

local_device = None
all_devices = None

def train_dataset_provider(args):
    train_dataset = HetuJsonDataset(
        json_file=args.json_file,
        key=args.json_key,
        max_seq_len=args.global_seq_len,
        vocab_file=args.vocab_file,
        merge_file=args.merge_file)
    return train_dataset

def train_data_iterator(dataset, consumed_samples, global_batch_size=None, global_token_num=None):
    dataloader = build_data_loader(dataset, consumed_samples, global_batch_size=global_batch_size, global_token_num=global_token_num)
    train_data_iter = iter(dataloader)
    return train_data_iter
    
def get_dg_from_union(device, dg_union):
    for i, dg in enumerate(dg_union):
        if dg.contains(device):
            return i, dg
    return None, None

def pretrain(args):
    ds_parallel_configs = read_ds_parallel_config(args.ds_parallel_config, args.num_strategy)

    config = LlamaConfig(
        vocab_size=args.vocab_size, 
        n_embd=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        n_layer=args.num_hidden_layers, 
        n_head=args.num_attention_heads, 
        resid_pdrop=args.dropout_prob,
        embd_pdrop=args.dropout_prob,
        attn_pdrop=args.dropout_prob,
        activation_function=args.hidden_act,
        use_flash_attn=args.use_flash_attn
    )
    
    assert config.use_flash_attn == True, "symbolic shape can only used when flash attn is on for now"
    # Simple check for llama blocks range
    ranges = []
    for _, block_config in ds_parallel_configs[0]['llama']['blocks'].items():
        ranges.append(block_config['range'])
    assert ranges[0][0] == 0 and ranges[-1][-1] == config.num_hidden_layers - 1, \
        f'llama blocks range: {ranges} is conflict with num_hidden_layers: {config.num_hidden_layers}!'

    # Hetu model definition
    model = LlamaLMHeadModel(config=config, ds_parallel_configs=ds_parallel_configs)

    input_ds_hierarchy, input_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'input')
    label_ds_hierarchy, label_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'label')
    
    config.packing = False
    config.multi_seq_lens_symbol = []
    config.multi_cp_group_symbol = []
    for i in range(len(input_ds_hierarchy)):
        dcp_size = input_ds_hierarchy[i].get(0).get_dim(0)
        # 例如[32, 32, 32, 48, 48, 32, 32, 32]
        # 表示各个dcp分到的seq len
        config.multi_seq_lens_symbol.append([ht.IntSymbol(1) for _ in range(dcp_size)])
        # 例如[0, 0, 0, 1, 1, 2, 2, 2] 
        # 相同编号的在一个cp group中
        config.multi_cp_group_symbol.append([ht.IntSymbol(1) for _ in range(dcp_size)])
    # 非packing不需要用到
    config.cu_seqlens_list = []
    for block_id, block in enumerate(model.transformer.h):
        config.cu_seqlens_list.append(None)
    config.max_seqlen_symbol = None
        
    # todo: remove the global_shape
    # now just offer a shape that can be divided by dcp * tp_max size
    input_ids = ht.parallel_placeholder(ht.int64, global_shape=[dcp_size * 16], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='input_ids')
    loss_mask =  ht.parallel_placeholder(ht.float32, global_shape=[dcp_size * 16], ds_hierarchy=label_ds_hierarchy, device_group_hierarchy=label_dg_hierarchy, name='loss_mask')
    masked_lm_labels = ht.parallel_placeholder(ht.int64, global_shape=[dcp_size * 16], ds_hierarchy=label_ds_hierarchy, device_group_hierarchy=label_dg_hierarchy, name='masked_lm_labels')

    print(f'{local_device}: build model begin...')
    loss = model(
        input_ids=input_ids,
        loss_mask=loss_mask,
        labels=masked_lm_labels
    )
    print(f'{local_device}: build model end...')

    print(f'{local_device}: optimizer minimize begin...')
    opt = ht.AdamOptimizer(init_lr=args.lr, max_lr=args.lr, min_lr=args.lr, lr_warmup_steps=0, lr_decay_steps=1000, lr_decay_style="constant")
    train_op = opt.minimize(loss)
    print(f'{local_device}: optimizer minimize end...')
    
    print(f'{local_device}: build dataset begin...')
    train_dataset = train_dataset_provider(args)
    print(f'{local_device}: build dataset end...')
    
    def hetu_train(
        feed_dict,
        num_micro_batches,
        strategy_id,
        run_level=ht.run_level("update")
    ):    
        try:
            results = train_op.graph.run(
                loss, 
                [loss, train_op], 
                feed_dict = feed_dict, 
                num_micro_batches = num_micro_batches, 
                cur_strategy_id = strategy_id,
                run_level = run_level,
            )
        except RuntimeError as e:
            print(e)
            with open("./logs/exception.txt", 'w') as file:
                print("device rank:", all_devices.get_index(local_device), file=file)
                print(e, file=file)
            os.killpg(0, signal.SIGTERM)
        return results

    def run_plan(
        cp_list,
        epoch = 0,
        steps = args.steps,
        consumed_samples = 0,
        global_batch_size = args.global_batch_size,
        micro_batch_size = args.micro_batch_size,
        global_seq_len = args.global_seq_len,
        strategy_id = 0, 
        run_level = 0
    ):  
             
        assert config.use_flash_attn == True, "symbolic shape can only used when flash attn is on for now"
        input_ds_union = input_ds_hierarchy[strategy_id]
        input_device_group_union = input_dg_hierarchy[strategy_id]
        label_ds_union = label_ds_hierarchy[strategy_id]
        label_device_group_union = label_dg_hierarchy[strategy_id]
        assert input_ds_union.hetero_dim == 0 or input_ds_union.hetero_dim == -3, "input hetero dim unsupported"
        assert label_ds_union.hetero_dim == 0 or label_ds_union.hetero_dim == -3, "label hetero dim unsupported"

        dp_size = len(cp_list)
        dcp_size = sum(cp_list)
        data_union_idx, data_dcp_rank, data_dp_rank = -1, -1, -1
        input_union_idx, input_device_group = get_dg_from_union(local_device, input_device_group_union)
        label_union_idx, label_device_group = get_dg_from_union(local_device, label_device_group_union)
        if input_device_group != None:
            data_union_idx = input_union_idx
        elif label_device_group != None:
            data_union_idx = label_union_idx
        if data_union_idx != -1:
            data_dcp_rank = data_union_idx
            accumulate_cp = 0
            for i, cp in enumerate(cp_list):
                accumulate_cp += cp
                if accumulate_cp > data_dcp_rank:
                    data_dp_rank = i
                    break
        
        pipeline_id, dp_group_id = None, None
        gbs_per_dp, num_micro_batches = None, None
        seq_len, seq_lens, inner_group_seq_lens = None, None, None
        # 兼容传统的同构策略
        if not args.hetero:
            gbs_per_dp = global_batch_size // dp_size
            assert gbs_per_dp % micro_batch_size == 0, \
                f'gbs_per_dp={gbs_per_dp} must be divided by mbs={micro_batch_size}'
            num_micro_batches = gbs_per_dp // micro_batch_size
            for cp in (cp_list):
                assert cp == cp_list[0], "homo setting should have the same cp degree"
            assert global_seq_len % cp_list[0] == 0, \
                f'gsl={global_seq_len} must be divided by cp={cp_list[0]}'
            pipeline_id = all_devices.get_index(local_device) % (dcp_size * args.gpus_per_stage) // args.gpus_per_stage
            dp_group_id = pipeline_id // cp_list[0]
            accumulate_micro_batch_num = [(i * gbs_per_dp // micro_batch_size) for i in range(0, dp_size + 1)]
            seq_len = global_seq_len // cp_list[0]
            seq_lens = [seq_len] * dcp_size
            inner_group_seq_lens = [seq_len] * cp_list[0]
        # 异构策略
        else:
            # ---- hetero tp ----
            rank_to_device_mapping = {}
            if args.rank_to_device_mapping == "":
                # 默认identity映射
                for idx in range(all_devices.num_devices):
                    rank_to_device_mapping[idx] = idx
            else:   
                rank_to_device_mapping = ast.literal_eval(args.rank_to_device_mapping)
            unused_rank_list = ast.literal_eval(args.unused_rank)
            for unused_rank in unused_rank_list:
                if rank_to_device_mapping[unused_rank] == all_devices.get_index(local_device):
                    # 直接在此处返回不去run
                    return
            curr_rank_id = -1
            for rank_id, device_id in rank_to_device_mapping.items():
                if device_id == all_devices.get_index(local_device):
                    if curr_rank_id != -1:
                        assert False, "rank_to_device_mapping has duplicate keys"
                    curr_rank_id = rank_id
            assert curr_rank_id != -1, f"can't find device {all_devices.get_index(local_device)} in rank_to_device_mapping"
            # ---- hetero pipeline ----
            if args.hetero_layers == "":
                # 默认均分stage
                pp = all_devices.num_devices // dcp_size // args.gpus_per_stage
                hetero_stages = [pp for _ in range(dcp_size)]
            else:
                hetero_stages = [len(pipeline) for pipeline in ast.literal_eval(args.hetero_layers)]
                assert len(hetero_stages) == dcp_size, f"len of hetero_stages should be equal to dcp={dcp_size}"
            accumulate_ranks = 0
            for i, stage_num in enumerate(hetero_stages):
                accumulate_ranks += stage_num * args.gpus_per_stage
                if accumulate_ranks > curr_rank_id:
                    pipeline_id = i
                    break
            assert pipeline_id != None, "can't figure out pipeline num"
            accumulate_cp = 0
            for i, cp in enumerate(cp_list):
                accumulate_cp += cp
                if accumulate_cp > pipeline_id:
                    dp_group_id = i
                    break
            # ---- hetero batch ----
            accumulate_micro_batch_num = [0,]
            if args.micro_batch_num_list == "":
                # 默认均分micro batch
                num_micro_batches = global_batch_size // micro_batch_size // dp_size
                for i in range(dp_size):
                    accumulate_micro_batch_num.append(accumulate_micro_batch_num[-1] + num_micro_batches)
            else:
                micro_batch_num_list = ast.literal_eval(args.micro_batch_num_list)
                assert len(micro_batch_num_list) == dp_size, f"len of micro_batch_num_list should be equal to dp={dp_size}"
                num_micro_batches = micro_batch_num_list[dp_group_id]
                for i in range(dp_size):
                    accumulate_micro_batch_num.append(accumulate_micro_batch_num[-1] + micro_batch_num_list[i])
            gbs_per_dp = micro_batch_size * num_micro_batches
            # ---- hetero seqlen ----
            if args.seq_len_list == "":
                # 默认均分seq len
                seq_lens = []
                for cp in cp_list:
                    assert global_seq_len % cp == 0, \
                        f'gsl={global_seq_len} must be divided by cp={cp}'
                    seq_lens.extend([global_seq_len // cp] * cp)
                seq_len = seq_lens[pipeline_id]
                inner_group_seq_lens = [seq_len] * cp_list[dp_group_id]
            else:
                seq_lens = ast.literal_eval(args.seq_len_list)
                assert len(seq_lens) == dcp_size, f"len of seq_len_list should be equal to dcp={dcp_size}"
                seq_len = seq_lens[pipeline_id]
                inner_group_seq_lens = seq_lens[sum(cp_list[:dp_group_id]): sum(cp_list[:dp_group_id + 1])]
            # 检测含有data的device的属性是否一致
            # dp_group_id和pipeline_id是每个rank都有的
            # data_dp_rank和data_dcp_rank是只有含data的rank有的
            if data_dp_rank != -1:
                assert data_dp_rank == dp_group_id, f"data_dp_rank={data_dp_rank} should be equal to dp_group_id={dp_group_id}"
            if data_dcp_rank != -1:
                assert data_dcp_rank == pipeline_id, f"data_dcp_rank={data_dcp_rank} should be equal to pipeline_id={pipeline_id}"
        
        print(
            f"{local_device}: " + \
            f"rank={all_devices.get_index(local_device)}, dp_size={dp_size}, dcp_size={dcp_size}, " + \
            f"data_dp_rank={data_dp_rank}, data_dcp_rank={data_dcp_rank}, " + \
            f"dp_group_id={dp_group_id}, pipeline_id={pipeline_id}, " + \
            f"gbs={global_batch_size}, mbs={micro_batch_size}, gsl={global_seq_len}, " + \
            f"num_micro_batches={num_micro_batches}, seq_len={seq_len}" \
        )

        train_iter = train_data_iterator(train_dataset, consumed_samples, global_batch_size=args.global_batch_size)
        print("runtime seq_lens is", seq_lens, "and runtime cp list is", cp_list)    
        for i, symbol in enumerate(config.multi_seq_lens_symbol[strategy_id]):
            symbol.set_data(seq_lens[i])
        accumulate_cp = cp_list[0]
        cp_cnt = 0
        for i, symbol in enumerate(config.multi_cp_group_symbol[strategy_id]):
            if i < accumulate_cp:
                symbol.set_data(cp_cnt)
            else:
                cp_cnt += 1
                accumulate_cp += cp_list[cp_cnt]
                
        inner_group_accumulate_seq_lens = []
        inner_group_accumulate_length = 0
        for length in inner_group_seq_lens:
            inner_group_accumulate_seq_lens.append(inner_group_accumulate_length) 
            inner_group_accumulate_length += length

        for step in range(steps):
            # load data for each dp
            global_batch = np.array(next(train_iter)) # [gbs, gsl + 1]
            inner_group_seq_idx = pipeline_id - sum(cp_list[:dp_group_id])
            begin_seq_len = inner_group_accumulate_seq_lens[inner_group_seq_idx]
            begin_seq_id = accumulate_micro_batch_num[dp_group_id] * micro_batch_size
            end_seq_id = accumulate_micro_batch_num[dp_group_id + 1] * micro_batch_size
            assert end_seq_id - begin_seq_id == gbs_per_dp, "batch size mismatches"
            labels = global_batch[begin_seq_id: end_seq_id, begin_seq_len + 1: begin_seq_len + seq_len + 1].reshape(-1) # [gbs_per_dp, seq_len]
            tokens = global_batch[begin_seq_id: end_seq_id, begin_seq_len: begin_seq_len + seq_len].reshape(-1) # [gbs_per_dp, seq_len]
            feed_dict = {
                input_ids: tokens.astype(np.int64),
                loss_mask: np.ones_like(tokens, dtype=np.float32),
                masked_lm_labels: labels.astype(np.int64),
            }
            # print(f"{local_device}: strategy_id = {strategy_id}, gbs = {global_batch_size}, mbs = {micro_batch_size}, seq_len = {seq_len} run begin")
            start_time = time.time()
            if args.torch_profile != 0 and step == 1:
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    results = hetu_train(feed_dict, num_micro_batches, strategy_id)
                prof.export_chrome_trace(f"trace/trace_{local_device}.json")
            else:
                results = hetu_train(feed_dict, num_micro_batches, strategy_id)
            end_time = time.time()
            consumed_samples += global_batch_size
            # print(f"{local_device}: strategy_id = {strategy_id}, gbs = {global_batch_size}, mbs = {micro_batch_size}, seq_len = {seq_len} run end, consumed_samples = {consumed_samples}")
            # NOTE: 实际上应该扫描一次alloc到update之间的所有数据
            # grad_scale = 当前run的数据的batch_size除以总的这之间run加起来的batch_size
            if run_level == ht.run_level("update"):
                if label_device_group != None:
                    loss_out = results[0].numpy(force=True).mean()
                    print(f"{local_device}: [Epoch {epoch}] (step {step}, consumed_samples = {consumed_samples}): loss = {loss_out:.3f}, time = {end_time - start_time:.4f}")
        return consumed_samples
    
    # 单轮样例 
    def test(): 
        strategy_id = 0
        cp_list = ast.literal_eval(args.cp_list)
        assert len(cp_list) >= 1, "cp list shouldn't be empty"
        consumed_samples = 0 # should be reset when run next epoch
        consumed_samples = run_plan(
            cp_list,
            epoch = 0,
            steps = args.steps,
            consumed_samples = consumed_samples, 
            global_batch_size = args.global_batch_size, 
            micro_batch_size = args.micro_batch_size, 
            global_seq_len = args.global_seq_len,
            strategy_id = strategy_id, 
            run_level = ht.run_level("update")
        )
    
    test()

if __name__ == '__main__':
    print("Run hetu training")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--torch_profile", type=int, default=0, help="use pytorch profiler"
    )
    parser.add_argument(
        "--hetero", action="store_true", help="use hetero training strategy."
    )
    parser.add_argument(
        '--gpus_per_stage', type=int, default=2, help='num of gpus of a single stage (to degree)'
    )
    parser.add_argument(
        '--hetero_layers', type=str, default="", help='hetero layers.'
    )
    parser.add_argument(
        "--rank_to_device_mapping", type=str, default="", help='rank to device mapping.'
    )
    parser.add_argument(
        "--unused_rank", type=str, default="[]", help='unused rank.'
    )
    parser.add_argument(
        "--cp_list", type=str, default="[]", help='cp list.'
    )
    parser.add_argument(
        "--seq_len_list", type=str, default="", help='seq len list.'
    )
    parser.add_argument(
        "--micro_batch_num_list", type=str, default="", help='micro batch num list.'
    )
    parser.add_argument(
        "--ds_parallel_config", default="ds_parallel_config/dp2_tp2_pp2.json", type=str, help="ds parallel config json file"
    )
    parser.add_argument(
        "--num_strategy", type=int, default=1, help="multi ds num"
    )
    parser.add_argument(
        "-s", "--global_seq_len", type=int, default=128, help="Maximum sequence len"
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
        "--ffn_hidden_size", type=int, default=-1, help="FFN hidden size of transformer model",
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
    args = parser.parse_args()
    pynvml.nvmlInit()
    print("Hetu distributed init")
    local_device, all_devices = distributed_init(args.ngpus, args.server_addr, args.server_port)
    print("Local device world rank is", all_devices.get_index(local_device))
    with ht.graph("define_and_run", num_strategy=args.num_strategy):
        if args.bf16:
            precision = "ht.bfloat16"
        else:
            precision = "ht.float32"
        print(f'{local_device}: use precision {precision}')
        with ht.autocast(eval(precision)):            
            pretrain(args)
            print(f'{local_device}: train hetu ds parallel end...')
