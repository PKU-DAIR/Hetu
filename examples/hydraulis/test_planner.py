import os
import signal
import time
import argparse
import socket
import pynvml
import ast
import json
import numpy as np
import multiprocessing
from tqdm import tqdm
from multiprocessing import Pool
from data_utils import LLaMAJsonDataset, build_data_loader, get_sorted_batch_and_len, build_fake_batch_and_len, get_input_and_label_buckets
from strategy import get_strategy_max_seqlen, dynamic_strategy, batching_strategy

# Global variables to store large data
global_sorted_lens = []
# Assuming these variables are large and constant
global_strategy_pool = None
global_multi_match_id_list = None
global_multi_max_seqlen_list = None

def compute_cost1(step_index):
    sorted_len = global_sorted_lens[step_index]
    cost_1, batch_indices = dynamic_strategy(
        global_strategy_pool,
        global_multi_match_id_list[-1],
        global_multi_max_seqlen_list[-1],
        0,
        sorted_len,
    )
    return (step_index, cost_1, batch_indices)

def compute_cost2(args):
    step_index, batch_indices = args
    sorted_len = global_sorted_lens[step_index]
    sorted_len_selected = sorted_len[batch_indices]
    cost_2, _ = batching_strategy(
        global_strategy_pool,
        global_multi_match_id_list[-1][0],
        sorted_len_selected,
        global_multi_max_seqlen_list[-1][0],
    )
    return (step_index, cost_2)

def train_dataset_provider(args):
    train_dataset = LLaMAJsonDataset(
        json_file=args.json_file,
        key=args.json_key,
        max_seq_len=args.max_seq_len,
        vocab_file=args.vocab_file,
        merge_file=args.merge_file
    )
    return train_dataset

def train_data_iterator(dataset, consumed_samples, global_batch_size=None, global_token_num=None):
    dataloader = build_data_loader(dataset, consumed_samples, global_batch_size=global_batch_size, global_token_num=global_token_num)
    train_data_iter = iter(dataloader)
    return train_data_iter
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
        "--max_seq_len", type=int, default=32768, help="maximum sequence length in the whole dataset"
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
        "--vocab_size", type=int, default=30522, help="total number of vocab"
    )
    parser.add_argument(
        "--ngpus", type=int, default=8, help="num of gpus"
    ) 
    args = parser.parse_args()
    args.multi_tp_pp_list = ast.literal_eval(args.multi_tp_pp_list)
    assert len(args.multi_tp_pp_list) >= 1, "there should be at least one strategy"
    
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
            aligned_max_seqlen = max_seqlen // 128 * 128
            max_seqlen_list.append(aligned_max_seqlen)
        multi_match_id_list.append(match_id_list)
        multi_max_seqlen_list.append(max_seqlen_list)
        print(f"Strategy {strategy_id}, match strategy id list: {match_id_list} and max seqlen list: {max_seqlen_list}")
        
    print('build dataset begin...')
    train_dataset = train_dataset_provider(args)
    print('build dataset end...')
    train_iter = train_data_iterator(train_dataset, 0, global_token_num=args.global_token_num)
    
    # single process
    '''
    for step in range(0, 100):
        global_batch = np.array(next(train_iter))
        sorted_batch, sorted_len = get_sorted_batch_and_len(global_batch, train_dataset.pad_id())
        print(f"{len(sorted_batch)} seqs sorted lens is {sorted_len}")
        cost_1, batch_indices = dynamic_strategy(strategy_pool, multi_match_id_list[-1], multi_max_seqlen_list[-1], 0, sorted_len)
        print("dispatching optimal objective for first dp:", cost_1)
        cost_2, _ = batching_strategy(strategy_pool, multi_match_id_list[-1][0], sorted_len[batch_indices], multi_max_seqlen_list[-1][0])
        print("packing optimal objective for first dp:", cost_2)
    '''
    
    # Set the start method to 'fork' to ensure child processes inherit memory
    multiprocessing.set_start_method('fork')

    # Initialize global variables before starting the Pool
    # Assuming strategy_pool, multi_match_id_list, multi_max_seqlen_list are defined
    global_strategy_pool = strategy_pool
    global_multi_match_id_list = multi_match_id_list
    global_multi_max_seqlen_list = multi_max_seqlen_list
    
    # multi process
    multi = 10
    for begin_step in range(0, 100, multi):
        global_sorted_lens = []
        global_batch_indices_map = {}
        steps_data = []
        print("Starting cost_1 computations...")
        # Collect data for multi steps
        for step in range(multi):
            global_batch = np.array(next(train_iter))
            sorted_batch, sorted_len = get_sorted_batch_and_len(global_batch, train_dataset.pad_id())
            print(f"{len(sorted_batch)} seqs sorted lens is {sorted_len}")
            global_sorted_lens.append(sorted_len)
            # Prepare arguments for cost_1 computation
            steps_data.append(step)  # Only pass the index

        # Start timing for cost_1 computations
        start_time = time.time()
        with multiprocessing.Pool(processes=multi) as pool:
            cost1_results = pool.map(compute_cost1, steps_data)
        end_time = time.time()
        total_time_cost1 = end_time - start_time

        # Create a local batch_indices_map in the parent process
        batch_indices_map = {}
        for step_index, cost_1, batch_indices in cost1_results:
            batch_indices_map[step_index] = batch_indices
            print(f"Dispatching optimal objective for step {step_index}: {cost_1}")

        print(f"Total time for cost_1 computations: {total_time_cost1} seconds")

        print("\nStarting cost_2 computations...")
        # Prepare arguments for compute_cost2
        compute_cost2_args = [(step_index, batch_indices_map[step_index]) for step_index in steps_data]

        # Start timing for cost_2 computations
        start_time = time.time()
        with multiprocessing.Pool(processes=multi) as pool:
            cost2_results = pool.map(compute_cost2, compute_cost2_args)
        end_time = time.time()
        total_time_cost2 = end_time - start_time

        # Output cost_2 values
        for step_index, cost_2 in cost2_results:
            print(f"Packing optimal objective for step {step_index}: {cost_2}")

        print(f"Total time for cost_2 computations: {total_time_cost2} seconds")