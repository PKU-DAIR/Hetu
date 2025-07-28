import json
import fcntl
import math


class GPUPos:
    def __init__(self, is_llm, dp_id, stage_id):
        self.is_vision = (is_llm == 0)
        self.is_llm = (is_llm == 1)
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
        finally:
            # 释放文件锁
            fcntl.flock(f, fcntl.LOCK_UN)



def convert_strategy_for_model(tp_pp_list, ngpus, layers, base_gpu, gpu_pos, is_llm):
    
    dp = len(tp_pp_list)

    layers_tp_groups = []
    
    for _ in range(layers):
        layers_tp_groups.append([])
    
    for dp_id, tp_pp in enumerate(tp_pp_list):
        tp = tp_pp[0]
        pp = tp_pp[1]
        for stage_id in range(pp):
            cur_gpus = range(base_gpu, base_gpu + tp)
            cur_layers = range(layers // pp * stage_id, layers // pp * (stage_id + 1))
            for gpu in cur_gpus:
                gpu_pos[gpu] = GPUPos(is_llm, dp_id, stage_id)
            for layer in cur_layers:
                layers_tp_groups[layer].append(list(cur_gpus))
            base_gpu += tp
        
    assert base_gpu <= ngpus, "The number of gpus don't match the number of gpus in the system"
    for layer_tp_groups in layers_tp_groups:
        assert len(layer_tp_groups) == dp, "The number of tp groups don't match the number of data parallelism"
    
    return layers_tp_groups, gpu_pos, base_gpu


def convert_strategy(vision_tp_pp_list, ngpus, layers):

    base_gpu = 0
    gpu_pos = {}
    vision_layers_tp_groups, gpu_pos, base_gpu = convert_strategy_for_model(vision_tp_pp_list, ngpus, layers, base_gpu, gpu_pos, False)
    print("base_gpu", base_gpu)

    assert base_gpu == ngpus, "The number of gpus don't match the number of gpus in the system"
    
    return vision_layers_tp_groups, gpu_pos


def generate_vit_ds_parallel_config(ngpus, vision_layers, vision_layer_tp_groups, ds_parallel_config_path, zero=False):


    vision_dp = len(vision_layer_tp_groups[0])
    # assert vision_dp == llm_dp, "The number of data parallelism in vision and llm don't match"

    vision_dp_union = [vision_dp for _ in range(vision_dp)]

    vision_layers = len(vision_layer_tp_groups)


    vision_zero = True

    vision_tp_union_list = [[len(layer_tp_group) for layer_tp_group in layer_tp_groups] for layer_tp_groups in vision_layer_tp_groups]
    
    vision_dg_union = vision_layer_tp_groups

    ds_parallel_config = {
        'vision_zero': vision_zero,
        'zero': zero,
        'devices': list(range(ngpus)),

        'image_input': {
            'split': {'0': vision_dp_union},
            'dup': vision_tp_union_list[0],
            'device_group_union': vision_dg_union[0],
            'type': 'placeholder'
        },

        'label': {
            'split': {'0': vision_dp_union},
            'dup': vision_tp_union_list[-1],
            'device_group_union': vision_dg_union[-1],
            'type': 'placeholder'
        },

        'vision': {
            'patch_embed': {
                "ln_1":{
                    'split': {},
                    'dup': [vision_tp_union_list[0][i] * vision_dp for i in range(vision_dp)],
                    'device_group_union': vision_dg_union[0],
                    'type': 'variable'
                },
                "proj":{
                    'split': {},
                    'dup': [vision_tp_union_list[0][i] * vision_dp for i in range(vision_dp)],
                    'device_group_union': vision_dg_union[0],
                    'type': 'variable'
                },
                "ln_2":{
                    'split': {},
                    'dup': [vision_tp_union_list[0][i] * vision_dp for i in range(vision_dp)],
                    'device_group_union': vision_dg_union[0],
                    'type': 'variable'
                },
            },
            'blocks': {

            
            },

        },
        'patch_merger': {
            "mlp_head": {
                'split': {'1': vision_tp_union_list[-1]},
                'dup': vision_dp_union,
                'device_group_union': vision_dg_union[-1],
                'type': 'variable'             
            },
        },

    }
    
    for block_id in range(vision_layers):
        blocks_json = ds_parallel_config["vision"]['blocks']
        blocks_json[f'blocks{block_id}'] = {
            'range': [block_id,],
            'recompute': [False for _ in range(vision_dp)],
            'layernorm1': {
                'split': {},
                'dup': [vision_tp_union_list[block_id][i] * vision_dp for i in range(vision_dp)],
                'device_group_union': vision_dg_union[block_id],
                'type': 'variable'
            },
            'attn': {
                'qkv': {
                    'split': {'1': vision_tp_union_list[block_id]},
                    'dup': vision_dp_union ,
                    'device_group_union': vision_dg_union[block_id],
                    'type': 'variable'
                },
                'dense': {
                    'split': {'0': vision_tp_union_list[block_id]},
                    'dup': vision_dp_union ,
                    'device_group_union': vision_dg_union[block_id],
                    'type': 'variable'
                }
            },
            'layernorm2': {
                'split': {},
                'dup': [vision_tp_union_list[block_id][i] * vision_dp for i in range(vision_dp)],
                'device_group_union': vision_dg_union[block_id],
                'type': 'variable'
            },
            'mlp': {
                'dense_h_to_4h': {
                    'split': {'1': vision_tp_union_list[block_id]},
                    'dup': vision_dp_union ,
                    'device_group_union': vision_dg_union[block_id],
                    'type': 'variable'
                },
                'dense_4h_to_h': {
                    'split': {'0': vision_tp_union_list[block_id]},
                    'dup': vision_dp_union ,
                    'device_group_union': vision_dg_union[block_id],
                    'type': 'variable'
                }
            }
        }

    
    write_with_lock(ds_parallel_config_path, ds_parallel_config)