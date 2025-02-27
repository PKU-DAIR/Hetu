import json
import fcntl
import os
from dataclasses import dataclass
from typing import List, Optional

GPUS_PER_NODE = 8

class GPUPos:
    def __init__(self, dp_id, stage_id):
        self.dp_id = dp_id
        self.stage_id = stage_id
        
    def __repr__(self):
        attrs = vars(self)
        attrs_str = ', '.join(f'{key} = {value}' for key, value in attrs.items())
        return f'{self.__class__.__name__}({attrs_str})'

@dataclass
class RecomputeConfig:
    recompute_granularity: List[Optional[str]] # dcp_size
    recompute_layer_idxs_list: List[List[int]] # dcp_size * [recompute_layer_1, recompute_layer_2, ..., recompute_layer_dcp_k]
    blocks_recompute: List[List[bool]] # total_layers * dcp_size
    blocks_output_recompute: List[List[bool]] # total_layers * dcp_size

def write_with_lock(file_path, data):
    with open(file_path, 'w') as f:
        # 获取文件锁
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            json.dump(data, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        finally:
            # 释放文件锁
            fcntl.flock(f, fcntl.LOCK_UN)
        
# 注意tp_pp_list默认按照tp从大到小的顺序
def convert_strategy(tp_pp_list, ngpus, layers):
    dp = len(tp_pp_list)
    # workaround: 单节点只使用一部分GPU
    if ngpus < GPUS_PER_NODE:
        # 直接按顺序分配即可
        layers_tp_groups = [] # 记录每个layer所在的所有的tp组
        gpu_pos = {} # 记录每个GPU的位置
        for _ in range(layers):
            layers_tp_groups.append([])
        base_gpu = 0
        for dp_id, tp_pp in enumerate(tp_pp_list):
            tp = tp_pp[0]
            pp = tp_pp[1]
            for stage_id in range(pp):
                cur_gpus = range(base_gpu, base_gpu + tp)
                cur_layers = range(layers // pp * stage_id, layers // pp * (stage_id + 1))
                for gpu in cur_gpus:
                    gpu_pos[gpu] = GPUPos(dp_id, stage_id)
                for layer in cur_layers:
                    layers_tp_groups[layer].append(list(cur_gpus))  
                base_gpu += tp 
        assert base_gpu == ngpus, "all gpus should be used eventually"
        for layer_tp_groups in layers_tp_groups:
            assert len(layer_tp_groups) == dp, "length of tp group list should all be equal to dp degree"
        return layers_tp_groups, gpu_pos
    # 多节点
    assert ngpus % GPUS_PER_NODE == 0, f"now only support using all gpus in a node (with {GPUS_PER_NODE} gpu each), but found ngpus = {ngpus}"
    nnodes = ngpus // GPUS_PER_NODE
    used_gpus = {} # 记录每个node有多少GPU被用到了
    layers_tp_groups = [] # 记录每个layer所在的所有的tp组
    gpu_pos = {} # 记录每个GPU的位置
    # 初始化
    for i in range(nnodes):
        used_gpus[i] = 0
    for _ in range(layers):
        layers_tp_groups.append([])
    # 先去判断每个GPU应该分给哪个策略
    # 这里的策略是保证tp在一个node中
    # 当有多个node可以存放该tp时则优先考虑占满某一个机器
    # 同时优先让pp跨机
    for dp_id, tp_pp in enumerate(tp_pp_list):
        tp = tp_pp[0]
        pp = tp_pp[1]
        stage_id = 0
        for round in range(pp):
            used_gpus_declined_order = sorted(
                used_gpus.items(), 
                key=lambda x: -x[1], 
            )
            if tp > GPUS_PER_NODE:
                assert tp % GPUS_PER_NODE == 0, "tp larger than node gpus num should be divided by it"
                rest_tp = tp
                acc_cur_gpus = []
                for node_id, node_used_gpus in used_gpus_declined_order:  
                    if node_used_gpus != 0:
                        continue
                    cur_gpus = range(node_id * GPUS_PER_NODE, (node_id + 1) * GPUS_PER_NODE)
                    for gpu in cur_gpus:
                        gpu_pos[gpu] = GPUPos(dp_id, stage_id)
                    acc_cur_gpus.extend(list(cur_gpus))
                    used_gpus[node_id] += GPUS_PER_NODE
                    rest_tp -= GPUS_PER_NODE
                    if rest_tp == 0:
                        break
                assert rest_tp == 0, f"cannot place tp {tp}"
                cur_layers = range(layers // pp * stage_id, layers // pp * (stage_id + 1))
                for layer in cur_layers:
                    layers_tp_groups[layer].append(acc_cur_gpus)
                stage_id += 1
            else:        
                for node_id, node_used_gpus in used_gpus_declined_order:     
                    if node_used_gpus + tp <= GPUS_PER_NODE:
                        # 分配这tp个GPU的GPUPos
                        cur_gpus = range(node_id * GPUS_PER_NODE + node_used_gpus, node_id * GPUS_PER_NODE + node_used_gpus + tp)
                        cur_layers = range(layers // pp * stage_id, layers // pp * (stage_id + 1))
                        for gpu in cur_gpus:
                            gpu_pos[gpu] = GPUPos(dp_id, stage_id)
                        for layer in cur_layers:
                            layers_tp_groups[layer].append(list(cur_gpus))
                        used_gpus[node_id] += tp
                        stage_id += 1
                        if stage_id == pp:
                            break
                if stage_id == pp:
                    break
        assert stage_id == pp, f"current tp_pp_list {tp_pp_list} can't guarantee that tp GPUs are all in the same node"
    for layer_tp_groups in layers_tp_groups:
        assert len(layer_tp_groups) == dp, "length of tp group list should all be equal to dp degree"
    return layers_tp_groups, gpu_pos

def generate_recompute_config(
    dcp_size: int, 
    total_layers: int,
    hetero_layers: List[List[int]], # dcp_size * [stage_layers_1, stage_layers_2, ..., stage_layers_dcp_k]
    recompute_granularity = None, 
    recompute_method = None, 
    recompute_num_layers = None, 
    recompute_layer_idxs_list = None
):
    # recompute_method
    if recompute_method is not None:
        if type(recompute_method) == list:
            if len(recompute_method) == 1:
                recompute_method = [recompute_method[0] for _ in range(dcp_size)]
            else:
                assert len(recompute_method) == dcp_size, \
                    f"recompute_method should have the same length as dcp, but got {len(recompute_method)} vs. {dcp_size}"
        elif type(recompute_method) == str:
            recompute_method = [recompute_method for _ in range(dcp_size)]
        else:
            raise ValueError(f"recompute_method should be a string or a list of strings, but got {recompute_method}")
    else:
        recompute_method = [None for _ in range(dcp_size)]
    # recompute_granularity
    if recompute_granularity is not None:
        if type(recompute_granularity) == list:
            if len(recompute_granularity) == 1:
                recompute_granularity = [recompute_granularity[0] for _ in range(dcp_size)]
            else:
                assert len(recompute_granularity) == dcp_size, \
                    f"recompute_granularity should have the same length as dcp, but got {len(recompute_granularity)} vs. {dcp_size}"
        elif type(recompute_granularity) == str:
            recompute_granularity = [recompute_granularity for _ in range(dcp_size)]
        else:
            raise ValueError(f"recompute_granularity should be a string or a list of strings, but got {recompute_granularity}")
    else:
        recompute_granularity = [None for _ in range(dcp_size)]
    for i, granularity in enumerate(recompute_granularity):
        if granularity == 'selective':
            assert recompute_method[i] is None, \
                f"recompute_method should be None for dcp {i} when recompute_granularity is 'selective'"
    # recompute_num_layers
    if recompute_num_layers is None:
        recompute_num_layers = [1 for _ in range(dcp_size)]
    if type(recompute_num_layers) == list:
        if len(recompute_num_layers) == 1:
            recompute_num_layers = [recompute_num_layers[0] for _ in range(dcp_size)]
        else:
            assert len(recompute_num_layers) == dcp_size, \
                f"recompute_num_layers should have the same length as dcp, but got {len(recompute_num_layers)} vs. {dp}"
    elif type(recompute_num_layers) == int:
        recompute_num_layers = [recompute_num_layers for _ in range(dcp_size)]
    for i, recompute_num_layer in enumerate(recompute_num_layers):
        assert recompute_method[i] != 'block' or recompute_num_layer <= len(hetero_layers[i]), \
            f"recompute_num_layer {recompute_num_layer} should be less than or equal to pp {len(hetero_layers[i])} for dcp {i}"
    # recompute_layer_idxs_list
    if recompute_layer_idxs_list is not None:
        if type(recompute_layer_idxs_list) == list:
            if len(recompute_layer_idxs_list) == 1:
                recompute_layer_idxs_list = [recompute_layer_idxs_list[0] for _ in range(dcp_size)]
            else:
                assert len(recompute_layer_idxs_list) == dcp_size, \
                    f"recompute_layer_idxs_list should have the same length as dcp, but got {len(recompute_layer_idxs_list)} vs. {dcp_size}"
            for i, recompute_layer_idxs in enumerate(recompute_layer_idxs_list):
                if len(recompute_layer_idxs) > 0:
                    if recompute_method[i] is not None:
                        print(f"[WARNING] recompute_method will be ignored when recompute_layer_idxs is set for dcp {i}, got method {recompute_method[i]} and layer_idxs {recompute_layer_idxs}")
        elif type(recompute_layer_idxs_list) == int:
            recompute_layer_idxs_list = [[recompute_layer_idxs_list] for _ in range(dcp_size)]
        else:
            raise ValueError(f"recompute_layer_idxs_list should be an integer or a list of integers, but got {type(recompute_layer_idxs_list)}: {recompute_layer_idxs_list}")
    else:
        recompute_layer_idxs_list = [[] for _ in range(dcp_size)]
    # blocks_recompute & blocks_output_recompute
    blocks_recompute = []
    blocks_output_recompute = []
    for block_id in range(total_layers):
        block_recompute = []
        block_output_recompute = []
        for i in range(dcp_size):
            if recompute_layer_idxs_list[i] == []:
                num_layers = recompute_num_layers[i]
                if recompute_method[i] == None:
                    block_recompute.append(False)
                    block_output_recompute.append(False)
                elif recompute_method[i] == 'uniform':
                    block_recompute.append(True)
                    if (block_id + 1) % num_layers == 0:
                        block_output_recompute.append(False)
                    else:
                        block_output_recompute.append(True)
                elif recompute_method[i] == 'block':
                    acc_layers = 0
                    pp_block_id = 0
                    for layers in hetero_layers[i]:
                        if acc_layers + layers > block_id:
                            pp_block_id = block_id - acc_layers
                            break
                        acc_layers += layers 
                    if pp_block_id < num_layers:
                        block_recompute.append(True)
                        if pp_block_id < num_layers - 1:
                            block_output_recompute.append(False)
                        else:
                            block_output_recompute.append(True)
                    else:
                        block_recompute.append(False)
                        block_output_recompute.append(False)
                else:
                    raise ValueError(f"recompute_method should be 'uniform' or 'block', but got {recompute_method[i]}")
            else:
                if block_id in recompute_layer_idxs_list[i]:
                    block_recompute.append(True)
                    block_output_recompute.append(False)
                else:
                    block_recompute.append(False)
                    block_output_recompute.append(False)
        blocks_recompute.append(block_recompute)
        blocks_output_recompute.append(block_output_recompute)
    # build recompute config
    return RecomputeConfig(
        recompute_granularity=recompute_granularity,
        recompute_layer_idxs_list=recompute_layer_idxs_list,
        blocks_recompute=blocks_recompute,
        blocks_output_recompute=blocks_output_recompute
    )

# 需要根据具体的模型去generate_ds_parallel_config
# 因此将这一部分放到了python/models目录下
# 不同模型（例如gpt和llama）调用不同的generate_ds_parallel_config
'''
def generate_ds_parallel_config(ngpus, layers_tp_groups, ds_parallel_config_path, zero=True):
    dp = len(layers_tp_groups[0])
    dp_union = [dp for _ in range(dp)]
    num_layers = len(layers_tp_groups)
    if dp == 1:
        zero = False
    tp_union_list = [[len(layer_tp_group) for layer_tp_group in layer_tp_groups] for layer_tp_groups in layers_tp_groups]
    dg_union_list = layers_tp_groups
    
    ds_parallel_config = {
        'zero': zero,
        'devices': list(range(ngpus)),
        'input': {
            'split': {'0': dp_union},
            'dup': tp_union_list[0],
            'device_group_union': dg_union_list[0],
            'type': 'placeholder'
        },
        'llama': {
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
            'rmsnorm_final': {
                'split': {'0': tp_union_list[-1]},
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
        blocks_json = ds_parallel_config['llama']['blocks']
        blocks_json[f'blocks{block_id}'] = {
            'range': [block_id,],
            'recompute': [False for _ in range(dp)],
            'rmsnorm1': {
                'split': {'0': tp_union_list[block_id]},
                'dup': dp_union,
                'device_group_union': dg_union_list[block_id],
                'type': 'variable'
            },
            'attn': {
                'qkv': {
                    'split': {'1': tp_union_list[block_id]},
                    'dup': dp_union,
                    'device_group_union': dg_union_list[block_id],
                    'type': 'variable'
                },
                'dense': {
                    'split': {'0': tp_union_list[block_id]},
                    'dup': dp_union,
                    'device_group_union': dg_union_list[block_id],
                    'type': 'variable'
                }
            },
            'rmsnorm2': {
                'split': {'0': tp_union_list[block_id]},
                'dup': dp_union,
                'device_group_union': dg_union_list[block_id],
                'type': 'variable'
            },
            'mlp': {
                'dense_h_to_4h': {
                    'split': {'1': tp_union_list[block_id]},
                    'dup': dp_union,
                    'device_group_union': dg_union_list[block_id],
                    'type': 'variable'
                },
                'dense_4h_to_h': {
                    'split': {'0': tp_union_list[block_id]},
                    'dup': dp_union,
                    'device_group_union': dg_union_list[block_id],
                    'type': 'variable'
                },
                'activation_func': {
                }
            }
        }
           
    write_with_lock(ds_parallel_config_path, ds_parallel_config)
'''

