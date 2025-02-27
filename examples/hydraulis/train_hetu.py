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
from hetu_llama import LLaMALMHeadModel
from llama_config import LLaMAConfig
from data_utils import LLaMAJsonDataset, build_data_loader, get_sorted_batch_and_len, build_fake_batch_and_len, get_input_and_label_buckets, LLaMaDatasetConfig, build_tokenizer, BlendedHetuDatasetBuilder
from parallel_utils import read_ds_parallel_config, parse_multi_ds_parallel_config, convert_strategy, generate_ds_parallel_config
from strategy import get_strategy_max_seqlen, find_optimal_strategy

local_device = None
all_devices = None
tokenizer = None
ds_parallel_config_path = "./ds_parallel_config/"
alignment = 128

def distributed_init(args):
    global local_device, all_devices
    if 'HETU_LOCAL_HOSTNAME' not in os.environ:
        # 通过socket获取主机名并设置环境变量
        hostname = socket.gethostname()
        os.environ['HETU_LOCAL_HOSTNAME'] = hostname
    else:
        print(f"Environment variable 'HETU_LOCAL_HOSTNAME' already set: {os.environ['HETU_LOCAL_HOSTNAME']}")
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
    samples_per_step = args.global_batch_size
    # workaround for fixed token num sampler
    if samples_per_step == -1:
        samples_per_step = args.global_token_num // args.max_seq_len
    train_val_test_num_samples = [args.epochs * args.steps * samples_per_step, 0, 0]
    train_ds, valid_ds, test_ds = BlendedHetuDatasetBuilder(
        LLaMAJsonDataset,
        train_val_test_num_samples,
        config
    ).build()
    return train_ds

def train_dataloader_provider(train_ds, tokenizer, consumed_samples, global_batch_size=None, global_token_num=None):
    data_loader = build_data_loader(train_ds, tokenizer, consumed_samples, global_batch_size=global_batch_size, global_token_num=global_token_num)
    return iter(data_loader)
  
def get_dg_from_union(device, dg_union):
    for i, dg in enumerate(dg_union):
        if dg.contains(device):
            return i, dg
    return None, None

def pretrain(args):
    # Generate & read configs
    with open(args.strategy_pool, 'r') as f:
        strategy_pool = json.load(f)
    multi_tp_pp_list = args.multi_tp_pp_list
    num_strategy = len(multi_tp_pp_list)
    multi_dp_size = [len(tp_pp_list) for tp_pp_list in multi_tp_pp_list]
    multi_gpu_pos = []
    multi_config_file_path = []
    multi_match_id_list = []
    multi_max_seqlen_list = []
    multi_dp_representive_gpu = []
    
    # 默认策略list中第一个放optimizer的同构的strategy
    os_tp, os_pp = multi_tp_pp_list[0][0]
    os_dp = args.ngpus // os_tp // os_pp
    for tp_pp in multi_tp_pp_list[0]:
        assert tp_pp[0] == os_tp and tp_pp[1] == os_pp, "must ensure the first strategy is a homo optimizer strategy"
    
    for strategy_id in range(num_strategy):
        # 获取当前异构dp策略下每个tp+pp子策略在pool中的id以及其支持的最大seq长度
        match_id_list = []
        max_seqlen_list = []
        dp_representive_gpu = {}
        for tp_pp in multi_tp_pp_list[strategy_id]:
            tp = tp_pp[0]
            pp = tp_pp[1]
            match_id = None
            for i, data in enumerate(strategy_pool['strategies']):
                if data['tp'] == tp and data['pp'] == pp:
                    match_id = i
                    break
            assert match_id != None, f"can't find tp{tp}pp{pp} in the strategy pool, please use the strategy within the pool"
            match_id_list.append(match_id)
            max_seqlen = get_strategy_max_seqlen(strategy_pool, match_id, os_dp_tp_pp=(os_dp, os_tp, os_pp))
            aligned_max_seqlen = max_seqlen // alignment * alignment
            max_seqlen_list.append(aligned_max_seqlen)
        multi_match_id_list.append(match_id_list)
        multi_max_seqlen_list.append(max_seqlen_list)
        print(f"Strategy {strategy_id}, match strategy id list: {match_id_list} and max seqlen list: {max_seqlen_list}")
        # 获取GPU的位置
        # 原则是不让tp跨机并尽可能贪心地让pp跨机
        layers_tp_groups, gpu_pos = convert_strategy(multi_tp_pp_list[strategy_id], args.ngpus, args.num_hidden_layers)
        config_file_path = ds_parallel_config_path + f"strategy_{strategy_id}.txt"
        generate_ds_parallel_config(args.ngpus, layers_tp_groups, config_file_path)
        print(f"Strategy {strategy_id}, gpu positions are: {gpu_pos}")
        multi_gpu_pos.append(gpu_pos)
        multi_config_file_path.append(config_file_path)
        # 找到每个dp中编号最小的gpu_id
        # 后面需要用这些gpu代表去跑决策算法
        for cur_gpu_id, cur_pos in gpu_pos.items():
            if cur_pos.dp_id not in dp_representive_gpu:
                dp_representive_gpu[cur_pos.dp_id] = cur_gpu_id
            else:
                dp_representive_gpu[cur_pos.dp_id] = min(dp_representive_gpu[cur_pos.dp_id], cur_gpu_id)
        print(f"Strategy {strategy_id}, DP representive gpu:", dp_representive_gpu)
        multi_dp_representive_gpu.append(dp_representive_gpu)
        
    ht.global_comm_barrier_rpc() 
    ds_parallel_configs = read_ds_parallel_config(",".join(multi_config_file_path), num_strategy)
    config = LLaMAConfig(
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
    model = LLaMALMHeadModel(config=config, ds_parallel_configs=ds_parallel_configs)

    input_ds_hierarchy, input_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'input')
    label_ds_hierarchy, label_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'label')
    # todo: remove the global_shape
    # now just offer a shape that can be divided by dp * tp_max size
    input_ids = ht.parallel_placeholder(ht.int64, global_shape=[multi_dp_size[0] * 16], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='input_ids')
    # position_ids = ht.parallel_placeholder(ht.int64, global_shape=[multi_dp_size[0] * 16], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='position_ids')
    # token_type_ids = ht.parallel_placeholder(ht.int64, global_shape=[multi_dp_size[0] * 16], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='token_type_ids')
    # attention_mask = ht.parallel_placeholder(ht.float32, global_shape=[multi_dp_size[0] * 16], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='attention_mask')
    loss_mask =  ht.parallel_placeholder(ht.float32, global_shape=[multi_dp_size[0] * 16], ds_hierarchy=label_ds_hierarchy, device_group_hierarchy=label_dg_hierarchy, name='loss_mask')
    masked_lm_labels = ht.parallel_placeholder(ht.int64, global_shape=[multi_dp_size[0] * 16], ds_hierarchy=label_ds_hierarchy, device_group_hierarchy=label_dg_hierarchy, name='masked_lm_labels')
    config.cu_seqlens_list = []
    for block_id, block in enumerate(model.transformer.h):
        config.cu_seqlens_list.append(
            ht.parallel_placeholder(
                ht.int32, 
                global_shape=[multi_dp_size[0] * 16], 
                ds_hierarchy=block.attn.qkv_dense.ds_union_map['split0_dup'], 
                device_group_hierarchy=block.attn.qkv_dense.device_group_unions,
                name=f'cu_seqlens_{block_id}'
            )
        )
    
    # 设置symbol
    # cp恒等于1
    config.multi_seq_lens_symbol = []
    config.multi_cp_group_symbol = []
    for i in range(len(input_ds_hierarchy)):
        assert multi_dp_size[i] == input_ds_hierarchy[i].get(0).get_dim(0), "dp size mismatches"
        # 例如[32, 32, 32, 48, 48, 32, 32, 32]
        # 表示各个dp分到的seq len
        # Hydraulis中mbs恒等于1
        # 其即是input_ids的shape 0
        config.multi_seq_lens_symbol.append([input_ids.symbolic_shape[0] for _ in range(multi_dp_size[i])])
        # 例如[0, 0, 0, 1, 1, 2, 2, 2] 
        # 相同编号的在一个cp group中
        # Hydraulis中我们不使用cp
        config.multi_cp_group_symbol.append([ht.IntSymbol(i) for i in range(multi_dp_size[i])])
    # run plan时再根据当前GPU所在的tp pp组合来确定
    config.max_seqlen_symbol = ht.IntSymbol(1)

    print(f'{local_device}: build model begin...')
    loss = model(
        input_ids=input_ids,
        # position_ids=position_ids,
        # attention_mask=attention_mask,
        loss_mask=loss_mask,
        # token_type_ids=token_type_ids,
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
    
    def get_strategy_info(
        strategy_id
    ):
        dp_size = multi_dp_size[strategy_id]
        tp_pp_list = multi_tp_pp_list[strategy_id]
        max_seqlen_list = multi_max_seqlen_list[strategy_id]
        match_id_list = multi_match_id_list[strategy_id]
        gpu_pos = multi_gpu_pos[strategy_id]
        dp_representive_gpu = multi_dp_representive_gpu[strategy_id]
        gpu_id = all_devices.get_index(local_device)
        
        assert gpu_id in gpu_pos, f"gpu {gpu_id} is not included in this training"
        dp_id, stage_id = gpu_pos[gpu_id].dp_id, gpu_pos[gpu_id].stage_id
        assert dp_id < dp_size, "dp size mismatches"
        
        return dp_size, tp_pp_list, max_seqlen_list, dp_id, stage_id
    
    def hetu_train(
        feed_dict,
        num_micro_batches,
        compute_strategy_id,
        optimize_strategy_id,
        run_level=ht.run_level("update")
    ):
        # return None
        try:
            results = train_op.graph.run(
                loss, 
                [loss, train_op], 
                feed_dict=feed_dict, 
                num_micro_batches=num_micro_batches, 
                compute_strategy_id=compute_strategy_id,
                optimize_strategy_id=optimize_strategy_id,
                run_level = run_level
            )
        except RuntimeError as e:
            print(e)
            with open("./logs/exception.txt", 'w') as file:
                print(f"{local_device}:", file=file)
                print(e, file=file)
            os.killpg(0, signal.SIGTERM)
        return results

    def run_plan(
        compute_only = 0,
        epoch = 0,
        consumed_samples = 0,
        compute_strategy_id_list = [0,],
        optimize_strategy_id = 0,
        warm_up = False,
        batching_method = 4, 
        max_padded_seqlen = None,
        fake_seqlens = []
    ):     
        # batching_method
        # 0 means padding
        # 1 means unblanced assigned packing (maybe not a proper baseline)
        # 2 means greedy packing with static shape
        # 3 means greedy packing with dynamic shape
        # 4 means hydraulis packing
        if batching_method <= 1:
            assert len(compute_strategy_id_list) == 1, "batching method <= 1 only support single strategy"
        assert max(compute_strategy_id_list) < num_strategy, "compute strategy out of range"
        if max_padded_seqlen:
            max_padded_seqlen % alignment == 0, "max_padded_seqlen should be aligned"
            
        if warm_up:
            for compute_strategy_id in compute_strategy_id_list:
                print(f"{local_device}: warm up for compute strategy {compute_strategy_id} begin...")
                dp_size, tp_pp_list, max_seqlen_list, dp_id, stage_id = get_strategy_info(compute_strategy_id)
                num_micro_batches = max([pp for (tp, pp) in tp_pp_list])
                # packing
                if batching_method == 4:
                    max_seqlen = max_seqlen_list[dp_id]
                    print(f"{local_device}: warm up with max_seqlen = {max_seqlen}")
                # padding (or original greedy packing with static or dynamic shape)
                else:
                    assert max_padded_seqlen, "you should provide the max seqlen when doing padding or static-shape packing"
                    max_seqlen = max_padded_seqlen
                assert max_seqlen % alignment == 0, "max seqlen should already be aligned"
                config.max_seqlen_symbol.set_data(max_seqlen)
                packed_cu_seqlens_list = [np.array([0, max_seqlen], dtype=np.int32)] * num_micro_batches
                input_list = [np.zeros((max_seqlen,), dtype=np.int64)] * num_micro_batches
                label_list = [np.zeros((max_seqlen,), dtype=np.int64)] * num_micro_batches
                loss_mask_list = [np.zeros((max_seqlen,), dtype=np.float32)] * num_micro_batches
                feed_dict = {
                    input_ids: input_list,
                    masked_lm_labels: label_list,
                    loss_mask: loss_mask_list
                }
                for i in range(config.n_layer):
                    feed_dict[config.cu_seqlens_list[i]] = packed_cu_seqlens_list
                hetu_train(feed_dict, num_micro_batches, compute_strategy_id, optimize_strategy_id, run_level=ht.run_level("compute_only") if compute_only else ht.run_level("update"))
                print(f"{local_device}: warm up end")
            return
        
        # build dataloader and get sequence parallel degree
        assert (args.global_batch_size == -1 and args.global_token_num != -1) \
            or (args.global_batch_size != -1 and args.global_token_num == -1), "should only use one of the args: global_batch_size & global_token_num"
        if args.global_batch_size != -1:
            train_iter = train_dataloader_provider(train_dataset, tokenizer, consumed_samples, global_batch_size=args.global_batch_size)
        if args.global_token_num != -1:
            train_iter = train_dataloader_provider(train_dataset, tokenizer, consumed_samples, global_token_num=args.global_token_num)
        
        # 默认用compute_strategy_id_list中的第一项
        optimal_compute_strategy_id = compute_strategy_id_list[0]
        dp_size, tp_pp_list, max_seqlen_list, dp_id, stage_id = get_strategy_info(compute_strategy_id_list[0])
            
        for _ in range(args.begin_step):
            next(train_iter)    
        for step in range(args.begin_step, args.steps):
            # load data for each dp
            input_batch, label_batch, cu_seqlens_list = None, None, None
            if len(fake_seqlens) > 0:
                sorted_batch, sorted_len = build_fake_batch_and_len(fake_seqlens, tokenizer.pad)
            else:
                try:
                    global_batch = np.array(next(train_iter))
                except StopIteration:
                    print(f"{local_device}: Running out of data, stop training")
                    return consumed_samples
                # print("global batch shape is", global_batch.shape)
                sorted_batch, sorted_len = get_sorted_batch_and_len(global_batch, tokenizer.pad)
            print(f"{local_device}: {len(sorted_batch)} seqs sorted lens is {sorted_len}")
            
            # packing
            if batching_method > 0:
                # unbalanced seqs assignment
                if batching_method == 1:
                    assert args.global_batch_size != -1 and args.global_batch_size % dp_size == 0, "global_batch_size should be divided by dp_size"
                    batch_size_per_dp = args.global_batch_size // dp_size
                    batch_indices = list(range(batch_size_per_dp * dp_id, batch_size_per_dp * (dp_id + 1)))
                    batching_option_matrix = None
                if batching_method >= 2:
                    optimal_compute_strategy_id, estimated_cost_1, batch_indices, estimated_cost_2, batching_option_matrix = find_optimal_strategy(
                        compute_strategy_id_list, multi_dp_size, multi_tp_pp_list, multi_max_seqlen_list, 
                        multi_match_id_list, multi_gpu_pos, multi_dp_representive_gpu, 
                        all_devices.get_index(local_device), batching_method, strategy_pool, sorted_len
                    )
                    dp_size, tp_pp_list, max_seqlen_list, dp_id, stage_id = get_strategy_info(optimal_compute_strategy_id)
                # Question: 每个micro batch的实际的max_seqlen都不一样
                # FlashAttn的这一属性的设置是否对性能有明显的影响有待探究
                # 目前暂时将其设置成当前轮次所处理的最大的seqlen
                config.max_seqlen_symbol.set_data(sorted_len[batch_indices[-1]] - 1) 
                # print(f"{local_device}: {optimal_compute_strategy_id}-th strategy {dp_id}-th dp local batch indices is {batch_indices}, estimated cost is {estimated_cost_1}")
                strategy_max_seqlen = max_seqlen_list[dp_id] 
                static_shape = False
                if batching_method == 2 or batching_method == 3:
                    assert max_padded_seqlen, "static-shape packing should provide the max seqlen after packing"
                    strategy_max_seqlen = max_padded_seqlen
                    if batching_method == 2:
                        static_shape = True
                input_bucket, label_bucket = get_input_and_label_buckets(sorted_batch, tokenizer.pad, batch_indices, strategy_max_seqlen, alignment)
                input_bucket.pack_data(batching_option_matrix, static_shape)
                label_bucket.pack_data(batching_option_matrix, static_shape)
                input_batch, label_batch = input_bucket.packed_batch(), label_bucket.packed_batch()
                cu_seqlens_list = input_bucket.packed_cu_seqlens_list()
                print(f"{local_device}: {optimal_compute_strategy_id}-th strategy {dp_id}-th dp seqlens after packed is {[len(seq) for seq in input_batch]}, estimated cost is {estimated_cost_2}")
            
            # padding
            if batching_method == 0:
                assert args.global_batch_size != -1 and args.global_batch_size % dp_size == 0, "global_batch_size should be divided by dp_size"
                batch_size_per_dp = args.global_batch_size // dp_size
                batch_indices = list(range(batch_size_per_dp * dp_id, batch_size_per_dp * (dp_id + 1)))
                assert max_padded_seqlen, "padding should provide the max seqlen after padding"
                config.max_seqlen_symbol.set_data(max_padded_seqlen) 
                input_bucket, label_bucket = get_input_and_label_buckets(sorted_batch, tokenizer.pad, batch_indices, max_padded_seqlen, alignment)
                input_bucket.pad_data()
                label_bucket.pad_data()
                input_batch, label_batch = input_bucket.padded_batch(), label_bucket.padded_batch()
                cu_seqlens_list = input_bucket.padded_cu_seqlens_list()
            
            # build feed_dict
            if input_batch == None or len(input_batch) < 1: 
                raise NotImplementedError("currently not support GPUs with no data")
            else:
                input_list = [micro_batch.astype(np.int64) for micro_batch in input_batch] # batch_size * [seq_len]
                label_list = [micro_batch.astype(np.int64) for micro_batch in label_batch] # batch_size * [seq_len]
                # key : value = tensor : NDArrayList
                feed_dict = {
                    input_ids: input_list,
                    masked_lm_labels: label_list
                }
                for i in range(config.n_layer):
                    feed_dict[config.cu_seqlens_list[i]] = [x.astype(np.int32) for x in cu_seqlens_list]
                    # feed_dict[config.cu_seqlens_list[i]] = [np.array([0, max_padded_seqlen]).astype(np.int32) for x in cu_seqlens_list]
                loss_mask_list = []
                for idx, label in enumerate(label_list):
                    micro_batch_loss_mask = np.zeros_like(label, dtype=np.float32)
                    micro_batch_loss_mask[cu_seqlens_list[idx][0]:cu_seqlens_list[idx][-1]] = 1
                    loss_mask_list.append(micro_batch_loss_mask)
                feed_dict[loss_mask] = loss_mask_list

            start_time = time.time()
            if args.torch_profile != 0 and step == 0:
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    results = hetu_train(feed_dict, len(input_batch), optimal_compute_strategy_id, optimize_strategy_id, run_level=ht.run_level("compute_only") if compute_only else ht.run_level("update"))
                prof.export_chrome_trace(f"trace/trace_{local_device}.json")
            else:
                results = hetu_train(feed_dict, len(input_batch), optimal_compute_strategy_id, optimize_strategy_id, run_level=ht.run_level("compute_only") if compute_only else ht.run_level("update"))
            end_time = time.time()
            
            consumed_samples += len(sorted_batch)
            # 如果在pipeline的最后一个stage上那么就打印loss
            if stage_id == tp_pp_list[dp_id][1] - 1 and len(results) > 0:
                loss_out = results[0].numpy(force=True).mean()
                print(f"{local_device}: [Epoch {epoch}] (step {step}, consumed_samples = {consumed_samples}): loss = {loss_out:.3f}, time = {end_time - start_time:.4f}")
        return consumed_samples
    
    # 运行
    def test(
        compute_strategy_id_list=[0,],
        optimize_strategy_id=0,
        warm_up=True
    ): 
        if warm_up:
            run_plan(
                compute_only=args.compute_only,
                warm_up=True, 
                batching_method=args.batching_method,
                max_padded_seqlen=args.max_seq_len,
                compute_strategy_id_list=compute_strategy_id_list,
                optimize_strategy_id=optimize_strategy_id
            )
        for epoch in range(args.epochs):
            consumed_samples = 0 # should be reset when run next epoch
            consumed_samples = run_plan(
                compute_only=args.compute_only,
                epoch=epoch, 
                consumed_samples=consumed_samples,
                compute_strategy_id_list=compute_strategy_id_list,
                optimize_strategy_id=optimize_strategy_id,
                batching_method=args.batching_method,
                max_padded_seqlen=args.max_seq_len,
                fake_seqlens=ast.literal_eval(args.fake_seqlens)
            )
    
    compute_strategy_id_list = list(range(len(args.multi_tp_pp_list)))
    optimize_strategy_id = 0
    test(compute_strategy_id_list=compute_strategy_id_list, optimize_strategy_id=optimize_strategy_id, warm_up=args.warm_up)

if __name__ == '__main__':
    print("Run hetu training")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fake_seqlens", type=str, default="[]", help="seqlen list of fake data"
    )
    parser.add_argument(
        "--torch_profile", type=int, default=0, help="use pytorch profiler"
    )
    parser.add_argument(
        "--compute_only", type=int, default=0, help="use compute only"
    )
    parser.add_argument(
        "--warm_up", type=int, default=0, help="use warm up"
    )
    parser.add_argument(
        "--batching_method", type=int, default=4, help="batching method"
    )
    parser.add_argument(
        "--strategy_pool", type=str, default="./strategy/strategy_pool.json", help="json path to the strategy pool"
    )
    parser.add_argument(
        "--multi_tp_pp_list", type=str, default="[]", help="multi hetero dp strategy list"
    )
    parser.add_argument(
        "--global_batch_size", type=int, default=-1, help="global training batch size"
    )
    parser.add_argument(
        "--global_token_num", type=int, default=-1, help="global training token num"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=4096, help="maximum sequence length in the whole dataset"
    )
    parser.add_argument(
        "--data_path", type=str, nargs='+', help='The blend string, consisting of either a single dataset or a flattened sequential sequence of weight-dataset pairs'
    )
    parser.add_argument(
        "--data_cache_path", type=str, help='Where all re-useable dataset indices are to be cached'
    )
    parser.add_argument(
        "--tokenizer_type", type=str, help='tokenizer type'
    )
    parser.add_argument(
        "--split", type=str, help='The split string, a comma separated weighting for the dataset splits when drawing samples from a single distribution'
    )
    parser.add_argument(
        "--vocab_file", type=str, help='llama vocab file path'
    )
    parser.add_argument(
        "--merge_file", type=str, help='llama merge file path'
    )
    parser.add_argument(
        "--vocab_size", type=int, default=30522, help="total number of vocab"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="hidden size of transformer model",
    )
    parser.add_argument(
        "--ffn_hidden_size", type=int, default=-1, help="ffn hidden size of transformer model",
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=12, help="number of layers"
    )
    parser.add_argument(
        "--num_attention_heads", type=int, default=32, help="number of attention heads",
    )
    parser.add_argument(
        "--epochs", type=int, default=4, help="number of epochs"
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="number of steps for each epoch",
    )
    parser.add_argument(
        "--begin_step", type=int, default=0, help="number of step to begin",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="learning rate of adam"
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam"
    )
    parser.add_argument(
        "--hidden_act", type=str, default='gelu', help="hidden activation to use."
    )
    parser.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout rate."
    )
    parser.add_argument(
        "--use_flash_attn", action="store_true", help="use Flash Attention."
    )    
    parser.add_argument(
        "--bf16", action="store_true", help="use bfloat16."
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
        "--seed", type=int, default=12345, help="random seed"
    ) 
    args = parser.parse_args()
    print("Hetu distributed init")
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

