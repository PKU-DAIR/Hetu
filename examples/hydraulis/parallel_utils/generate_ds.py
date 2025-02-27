import json
import fcntl
import os

GPUS_PER_NODE = 8

class GPUPos:
    def __init__(self, dp_id, stage_id):
        self.dp_id = dp_id
        self.stage_id = stage_id
        
    def __repr__(self):
        attrs = vars(self)
        attrs_str = ', '.join(f'{key} = {value}' for key, value in attrs.items())
        return f'{self.__class__.__name__}({attrs_str})'

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


