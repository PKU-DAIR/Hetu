import os
import json
import fcntl
import socket
import yaml
import hetu as ht
from queue import Queue
from hetu.nn.modules.parallel_utils import get_multi_ds_parallel_config, config2ds

GPUS_PER_NODE = 8

class GPUPos:
    def __init__(self, local_dp_id, global_dp_id, stage_id):
        self.local_dp_id = local_dp_id
        self.global_dp_id = global_dp_id
        self.stage_id = stage_id
        
    def __repr__(self):
        attrs = vars(self)
        attrs_str = ', '.join(f'{key} = {value}' for key, value in attrs.items())
        return f'{self.__class__.__name__}({attrs_str})'

def export_strategy_config(
    scheme_list,
    max_tokens_list,
    strategy_config_path='strategy_config/default_strategy_config.yaml'
):
    strategy_config = []
    for (tp, pp, dp), max_tokens in zip(scheme_list, max_tokens_list):
        strategy_config.append({
            'dp': dp,
            'tp': tp,
            'pp': pp,
            'max_tokens': max_tokens
        })

    data = {'strategy_config': strategy_config}

    with open(strategy_config_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def parse_strategy_config(strategy_config_path, split_scheme=False):
    with open(strategy_config_path, 'r') as file:
        data = yaml.safe_load(file)

    scheme_list = []
    max_tokens_list = []

    for strategy in data.get('strategy_config', []):
        tp = strategy.get('tp')
        pp = strategy.get('pp')
        dp = strategy.get('dp')
        max_tokens = strategy.get('max_tokens')
        
        if split_scheme:
            for _ in range(dp):
                scheme_list.append((tp, pp, 1))
                max_tokens_list.append(max_tokens)
        else:
            scheme_list.append((tp, pp, dp))
            max_tokens_list.append(max_tokens)

    return scheme_list, max_tokens_list

def write_with_lock(file_path, data):
    file_dir = os.path.dirname(file_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    with open(file_path, 'w') as f:
        # 获取文件锁
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(data, f, indent=4)
        finally:
            # 释放文件锁
            fcntl.flock(f, fcntl.LOCK_UN)

def distributed_init(ngpus, server_addr, server_port):
    hostname = socket.gethostname()
    os.environ['HETU_LOCAL_HOSTNAME'] = hostname
    ht.init_comm_group(ngpus, server_address=server_addr + ":" + server_port)

def assign_global_prop_to_all_variables(ds_parallel_config):
    zero = ds_parallel_config['zero']
    # assign zero to all variables
    config_queue = Queue()
    for value in ds_parallel_config.values():
        config_queue.put(value)
    while (not config_queue.empty()):
        config = config_queue.get()
        if type(config) == dict:
            if 'type' in config:
                if config['type'] == 'variable' and ('zero' not in config):
                    config['zero'] = zero
                if 'lora' in config:
                    config_queue.put(config['lora'])
            else:
                for value in config.values():
                    config_queue.put(value)
    return ds_parallel_config

def read_ds_parallel_config(ds_parallel_config, num_strategy=1):
    # read ds_parallel_config from json file
    print(f'load ds_parallel_config from: {ds_parallel_config}')
    config_paths = ds_parallel_config.split(',')
    assert len(config_paths) == num_strategy, \
        f"ds_parallel_config num should equal to num_strategy {num_strategy}"
    ds_parallel_configs = []
    for config_path in config_paths:
        with open(config_path, 'r') as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX)
                ds_parallel_config = json.load(f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        ds_parallel_config = assign_global_prop_to_all_variables(ds_parallel_config)
        ds_parallel_configs.append(ds_parallel_config)
    return ds_parallel_configs

def parse_multi_ds_parallel_config(ds_parallel_configs, module_name, _range=-1):
    ds_hierarchy = []
    dg_hierarchy = []
    multi_ds_parallel_config = get_multi_ds_parallel_config(ds_parallel_configs, module_name, _range)
    for ds_parallel_config in multi_ds_parallel_config:
        ds_union, dg_union = config2ds(ds_parallel_config)
        ds_hierarchy.append(ds_union)
        dg_hierarchy.append(dg_union)
    return ds_hierarchy, dg_hierarchy

def convert_strategy(scheme_list, ngpus, layers):
    """
    scheme_list: list of (tp, pp, dp) tuples
    """
    scheme_list = sorted(scheme_list, key=lambda x: -x[0])
    dp_list = [scheme[2] for scheme in scheme_list]
    replica_num = sum(dp_list)
    nnodes = ngpus // GPUS_PER_NODE
    used_gpus = {} # 记录每个node有多少GPU被用到了
    layers_tp_groups = [[] for _ in range(layers)] # 记录每个layer所在的所有的tp组
    gpu_pos = {} # 记录每个GPU的位置
    for i in range(nnodes):
        used_gpus[i] = 0
    # 先去判断每个GPU应该分给哪个策略
    # 这里的策略是保证tp在一个node中
    # 当有多个node可以存放该tp时则优先考虑占满某一个机器
    # 同时优先让pp跨机
    for global_dp_id, (tp, pp, dp) in enumerate(scheme_list):
        for local_dp_id in range(dp):
            stage_id = 0
            used_gpus_declined_order = sorted(used_gpus.items(), reverse=True)
            for _ in range(pp):
                # 记录当前的stage_id
                old_stage_id = stage_id
                for node_id, node_used_gpus in used_gpus_declined_order:
                    if node_used_gpus + tp <= GPUS_PER_NODE:
                        # 分配这tp个GPU的GPUPos
                        cur_gpus = range(node_id * GPUS_PER_NODE + node_used_gpus, node_id * GPUS_PER_NODE + node_used_gpus + tp)
                        cur_layers = range(layers // pp * stage_id, layers // pp * (stage_id + 1))
                        for gpu in cur_gpus:
                            gpu_pos[gpu] = GPUPos(local_dp_id, global_dp_id, stage_id)
                        for layer in cur_layers:
                            layers_tp_groups[layer].append(list(cur_gpus))
                        used_gpus[node_id] += tp
                        stage_id += 1
                        if stage_id == pp:
                            break
                    elif tp == 16:
                        if node_used_gpus == 0:
                            cur_node_id = node_id
                            next_node_id = -1
                            for i in range(cur_node_id + 1, nnodes):
                                if used_gpus[next_node_id] == 0:
                                    next_node_id = i
                                    break
                            assert next_node_id != -1, "can't find a node to store the tp GPUs when tp=16"
                            cur_gpus = range(cur_node_id * GPUS_PER_NODE, cur_node_id * GPUS_PER_NODE + 8)
                            next_gpus = range(next_node_id * GPUS_PER_NODE, next_node_id * GPUS_PER_NODE + 8)
                            cur_layers = range(layers // pp * stage_id, layers // pp * (stage_id + 1))
                            for gpu in cur_gpus:
                                gpu_pos[gpu] = GPUPos(local_dp_id, global_dp_id, stage_id)
                            for gpu in next_gpus:
                                gpu_pos[gpu] = GPUPos(local_dp_id, global_dp_id, stage_id)
                            for layer in cur_layers:
                                layers_tp_groups[layer].append(list(cur_gpus) + list(next_gpus))
                            used_gpus[cur_node_id] += 8
                            used_gpus[next_node_id] += 8
                            stage_id += 1
                            if stage_id == pp:
                                break
                        else:
                            continue
                # 如果stage_id没有增加，说明没有找到合适的node
                # 否则会从头开始找合适的node，适用于pp > nnodes的情况
                if old_stage_id == stage_id or stage_id == pp:
                    break
            assert stage_id == pp, f"current scheme_list {scheme_list} can't guarantee that tp GPUs are all in the same node"
    for layer_tp_groups in layers_tp_groups:
        assert len(layer_tp_groups) == replica_num, "length of tp group list should all be equal to dp degree"
    return layers_tp_groups, gpu_pos

def generate_lora_ds_parallel_config(
    ngpus,
    layers_tp_groups,
    ds_parallel_config_path,
    zero=False,
    recompute_granularity=None,
    recompute_method=None,
    recompute_num_layers=None,
    recompute_layer_idxs_list=None
):
    dp = len(layers_tp_groups[0])
    dp_union = [dp for _ in range(dp)]
    num_layers = len(layers_tp_groups)
    if dp == 1:
        zero = False
    tp_union_list = [[len(layer_tp_group) for layer_tp_group in layer_tp_groups]
                        for layer_tp_groups in layers_tp_groups]
    dg_union_list = layers_tp_groups
    
    strategy_groups = [set() for _ in range(dp)]
    for layer_tp_groups in layers_tp_groups:
        for i, layer_tp_group in enumerate(layer_tp_groups):
            strategy_groups[i] = strategy_groups[i].union(set(layer_tp_group))

    unique_layer_tp_groups = [set() for _ in range(dp)]
    pp_union = [0] * dp
    for layer_tp_groups in layers_tp_groups:
        for i, layer_tp_group in enumerate(layer_tp_groups):
            if tuple(layer_tp_group) not in unique_layer_tp_groups[i]:
                unique_layer_tp_groups[i].add(tuple(layer_tp_group))
                pp_union[i] += 1

    strategy_groups = [list(strategy_group) for strategy_group in strategy_groups]
    tp_pp_union_list = [len(strategy_group) * len(strategy_groups) for strategy_group in strategy_groups]
    # tp_pp_union_list = [sum([len(strategy_group) for strategy_group in strategy_groups])] * len(strategy_groups)
    
    if recompute_granularity is not None:
        if type(recompute_granularity) == list:
            if len(recompute_granularity) == 1:
                recompute_granularity = [recompute_granularity[0] for _ in range(dp)]
            else:
                assert len(recompute_granularity) == dp, \
                    f"recompute_granularity should have the same length as dp, but got {len(recompute_granularity)} vs. {dp}"
        elif type(recompute_granularity) == str:
            recompute_granularity = [recompute_granularity for _ in range(dp)]
        else:
            raise ValueError(f"recompute_granularity should be a string or a list of strings, but got {recompute_granularity}")
    else:
        recompute_granularity = [None for _ in range(dp)]
    
    for i, granularity in enumerate(recompute_granularity):
        if granularity == 'selective':
            assert recompute_method[i] is None, \
                f"recompute_method should be None for dp {i} when recompute_granularity is 'selective'"

    if recompute_method is not None:
        if type(recompute_method) == list:
            if len(recompute_method) == 1:
                recompute_method = [recompute_method[0] for _ in range(dp)]
            else:
                assert len(recompute_method) == dp, \
                    f"recompute_method should have the same length as dp, but got {len(recompute_method)} vs. {dp}"
        elif type(recompute_method) == str:
            recompute_method = [recompute_method for _ in range(dp)]
        else:
            raise ValueError(f"recompute_method should be a string or a list of strings, but got {recompute_method}")
    else:
        recompute_method = [None for _ in range(dp)]

    if recompute_num_layers is None:
        recompute_num_layers = [1 for _ in range(dp)]
    
    if type(recompute_num_layers) == list:
        if len(recompute_num_layers) == 1:
            recompute_num_layers = [recompute_num_layers[0] for _ in range(dp)]
        else:
            assert len(recompute_num_layers) == dp, \
                f"recompute_num_layers should have the same length as dp, but got {len(recompute_num_layers)} vs. {dp}"
    elif type(recompute_num_layers) == int:
        recompute_num_layers = [recompute_num_layers for _ in range(dp)]
    
    for i, recompute_num_layer in enumerate(recompute_num_layers):
        assert recompute_method[i] != 'block' or recompute_num_layer <= pp_union[i], \
            f"recompute_num_layer {recompute_num_layer} should be less than or equal to pp_union {pp_union[i]} for dp {i}"

    if recompute_layer_idxs_list is not None:
        if type(recompute_layer_idxs_list) == list:
            if len(recompute_layer_idxs_list) == 1:
                recompute_layer_idxs_list = [recompute_layer_idxs_list[0] for _ in range(dp)]
            else:
                assert len(recompute_layer_idxs_list) == dp, \
                    f"recompute_layer_idxs_list should have the same length as dp, but got {len(recompute_layer_idxs_list)} vs. {dp}"
            for i, recompute_layer_idxs in enumerate(recompute_layer_idxs_list):
                if len(recompute_layer_idxs) > 0:
                    if recompute_method[i] is None:
                        print(f"[WARNING] recompute_method will be ignored when recompute_layer_idxs is set for dp {i}, got method {recompute_method[i]} and layer_idxs {recompute_layer_idxs}")
        elif type(recompute_layer_idxs_list) == int:
            recompute_layer_idxs_list = [[recompute_layer_idxs_list] for _ in range(dp)]
        else:
            raise ValueError(f"recompute_layer_idxs_list should be an integer or a list of integers, but got {recompute_layer_idxs_list}")
    else:
        recompute_layer_idxs_list = [[] for _ in range(dp)]

    ds_parallel_config = {
        'zero': zero,
        'devices': list(range(ngpus)),
        'recompute_granularity': recompute_granularity,
        'recompute_layer_idxs_list': recompute_layer_idxs_list,
        'task_batch_idxs': {
            'split': {},
            'dup': tp_pp_union_list,
            'device_group_union': strategy_groups,
            'type': 'placeholder'
        },
        'input': {
            'split': {'0': dp_union},
            'dup': tp_union_list[0],
            'device_group_union': dg_union_list[0],
            'type': 'placeholder'
        },
        'gpt': {
            'wte': {
                'split': {'0': tp_union_list[0]},
                'dup': dp_union,
                'device_group_union': dg_union_list[0],
                'type': 'variable'
            },
            'wpe': {
                'split': {},
                'dup': [tp_union_list[0][i] * dp for i in range(dp)],
                'device_group_union': dg_union_list[0],
                'type': 'variable'
            },
            'blocks': {

            },
            'layernorm_final': {
                'split': {'0': [tp_union_list[-1][i] for i in range(dp)]},
                'dup': dp_union,
                'device_group_union': dg_union_list[-1],
                'type': 'variable'
            }
        },
        'lm_head': {
            'split': {'1': tp_union_list[-1]},
            'dup': dp_union,
            'device_group_union': dg_union_list[-1],
            'type': 'variable'
        },
        'label': {
            'split': {'0': dp_union},
            'dup': tp_union_list[-1],
            'device_group_union': dg_union_list[-1],
            'type': 'placeholder'
        }
    }
    
    for block_id in range(num_layers):
        blocks_json = ds_parallel_config['gpt']['blocks']
        block_recompute = []
        block_output_recompute = []
        for i in range(dp):
            if recompute_layer_idxs_list[i] == []:
                num_layers = recompute_num_layers[i]
                if recompute_method[i] == 'uniform':
                    block_recompute.append(True)
                    if (block_id + 1) % num_layers == 0:
                        block_output_recompute.append(False)
                    else:
                        block_output_recompute.append(True)
                elif recompute_method[i] == 'block':
                    pp_block_id = block_id % pp_union[i]
                    if pp_block_id < num_layers:
                        block_recompute.append(True)
                        if pp_block_id < num_layers - 1:
                            block_output_recompute.append(False)
                        else:
                            block_output_recompute.append(True)
                    else:
                        block_recompute.append(False)
                        block_output_recompute.append(False)
                elif recompute_method[i] is None:
                    recompute_layer_idxs_list[i] = range(num_layers)
                else:
                    raise ValueError(f"recompute_method should be 'uniform' or 'block', but got {recompute_method[i]}")
            else:
                if block_id in recompute_layer_idxs_list[i]:
                    block_recompute.append(True)
                    block_output_recompute.append(False)
                else:
                    block_recompute.append(False)
                    block_output_recompute.append(False)

        blocks_json[f'blocks{block_id}'] = {
            'range': [block_id,],
            'recompute': block_recompute,
            'cpu_offload': [False for _ in range(dp)],
            'output_recompute': block_output_recompute,
            'layernorm1': {
                'split': {'0': [tp_union_list[block_id][i] for i in range(dp)]},
                'dup': dp_union,
                'device_group_union': dg_union_list[block_id],
                'type': 'variable'
            },
            'attn': {
                'qkv': {
                    'split': {'1': tp_union_list[block_id]},
                    'dup': dp_union,
                    'device_group_union': dg_union_list[block_id],
                    'type': 'variable',
                    'lora': {
                        'lora_A': {
                            'split': {'1': tp_union_list[block_id]},
                            'dup': dp_union,
                            'device_group_union': dg_union_list[block_id],
                            'type': 'variable'
                        },
                        'lora_B': {
                            'split': {'1': tp_union_list[block_id]},
                            'dup': dp_union,
                            'device_group_union': dg_union_list[block_id],
                            'type': 'variable'
                        }
                    }
                },
                'dense': {
                    'split': {'0': tp_union_list[block_id]},
                    'dup': dp_union,
                    'device_group_union': dg_union_list[block_id],
                    'type': 'variable',
                    'lora': {
                        'lora_A': {
                            'split': {'0': tp_union_list[block_id]},
                            'dup': dp_union,
                            'device_group_union': dg_union_list[block_id],
                            'type': 'variable'
                        },
                        'lora_B': {
                            'split': {'1': tp_union_list[block_id]},
                            'dup': dp_union,
                            'device_group_union': dg_union_list[block_id],
                            'type': 'variable'
                        }
                    }
                }
            },
            'layernorm2': {
                'split': {'0': [tp_union_list[block_id][i] for i in range(dp)]},
                'dup': dp_union,
                'device_group_union': dg_union_list[block_id],
                'type': 'variable'
            },
            'mlp': {
                'dense_h_to_4h': {
                    'split': {'1': tp_union_list[block_id]},
                    'dup': dp_union,
                    'device_group_union': dg_union_list[block_id],
                    'type': 'variable',
                    'lora': {
                        'lora_A': {
                            'split': {'1': tp_union_list[block_id]},
                            'dup': dp_union,
                            'device_group_union': dg_union_list[block_id],
                            'type': 'variable'
                        },
                        'lora_B': {
                            'split': {'1': tp_union_list[block_id]},
                            'dup': dp_union,
                            'device_group_union': dg_union_list[block_id],
                            'type': 'variable'
                        }
                    }
                },
                'dense_4h_to_h': {
                    'split': {'0': tp_union_list[block_id]},
                    'dup': dp_union,
                    'device_group_union': dg_union_list[block_id],
                    'type': 'variable',
                    'lora': {
                        'lora_A': {
                            'split': {'0': tp_union_list[block_id]},
                            'dup': dp_union,
                            'device_group_union': dg_union_list[block_id],
                            'type': 'variable'
                        },
                        'lora_B': {
                            'split': {'1': tp_union_list[block_id]},
                            'dup': dp_union,
                            'device_group_union': dg_union_list[block_id],
                            'type': 'variable'
                        }
                    }
                }
            }
        }

    write_with_lock(ds_parallel_config_path, ds_parallel_config)
