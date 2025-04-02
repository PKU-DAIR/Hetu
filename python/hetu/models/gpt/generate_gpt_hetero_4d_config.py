import hydra
import json
import os
from hetu.utils.parallel import generate_recompute_config

def generate_gpt_hetero_4d_config(
    cp_list, 
    rank_to_device_mapping, 
    unused_rank, 
    hetero_layers, 
    accumulate_hetero_stages, 
    num_layers=32, 
    num_gpus=8, 
    dp=2, 
    tp=2, 
    zero=True,
    recompute_granularity=None,
    recompute_method=None,
    recompute_num_layers=None,
    recompute_layer_idxs_list=None
):
    
    if dp == 1:
        zero = False
    
    assert len(cp_list) == dp, "len of cp list should be equal to dp"
    dp_cp = sum(cp_list)
    # dp_union = [dp for _ in range(dp_cp)]
    # cp_union = [cp_list[i] for _ in range(cp_list[i]) for i in range(dp)]
    dp_cp_union = [dp_cp for _ in range(dp_cp)]
    
    tp_union_list = []
    dg_union_list = []
    for block_id in range(num_layers):
        hybrid_tp_degree = []
        hybrid_device_group = []
        for pipeline_id in range(dp_cp):
            device_group_num = 0
            cnt = 0
            for hetero_layer in hetero_layers[pipeline_id]:
                cnt += hetero_layer
                if block_id < cnt:
                    break
                device_group_num += 1
            ranks = range(device_group_num * tp + accumulate_hetero_stages[pipeline_id] * tp, 
                          (device_group_num + 1) * tp + accumulate_hetero_stages[pipeline_id] * tp)
            hybrid_tp_degree.append(len([rank for rank in ranks if rank not in unused_rank]))
            hybrid_device_group.append([rank_to_device_mapping[rank] for rank in ranks if rank not in unused_rank])
        tp_union_list.append(hybrid_tp_degree)
        dg_union_list.append(hybrid_device_group)
        
    recompute_config = generate_recompute_config(
        dp_cp,
        num_layers,
        hetero_layers,
        recompute_granularity=recompute_granularity,
        recompute_method=recompute_method,
        recompute_num_layers=recompute_num_layers,
        recompute_layer_idxs_list=recompute_layer_idxs_list
    )

    ds_parallel_config = {
        'zero': zero,
        'devices': list(range(num_gpus)),
        'recompute_granularity': recompute_config.recompute_granularity,
        'recompute_layer_idxs_list': recompute_config.recompute_layer_idxs_list,
        'input': {
            'split': {'0': dp_cp_union},
            'dup': tp_union_list[0],
            'device_group_union': dg_union_list[0],
            'type': 'placeholder'
        },
        'gpt': {
            'wte': {
                'split': {'0': tp_union_list[0]},
                'dup': dp_cp_union,
                'device_group_union': dg_union_list[0],
                'type': 'variable'
            },
            'blocks': {

            },
            'layernorm_final': {
                'split': {'0': tp_union_list[-1]},
                'dup': dp_cp_union,
                'device_group_union': dg_union_list[-1],
                'type': 'variable'
            }
        },
        'lm_head': {
            'split': {'1': tp_union_list[-1]},
            'dup': dp_cp_union,
            'device_group_union': dg_union_list[-1],
            'type': 'variable'
        },
        'label': {
            'split': {'0': dp_cp_union},
            'dup': tp_union_list[-1],
            'device_group_union': dg_union_list[-1],
            'type': 'placeholder'
        }
    }
    
    for block_id in range(num_layers):
        blocks_json = ds_parallel_config['gpt']['blocks']
        blocks_json[f'blocks{block_id}'] = {
            'range': [block_id,],
            'recompute': recompute_config.blocks_recompute[block_id],
            'output_recompute': recompute_config.blocks_output_recompute[block_id],
            'cpu_offload': [False for _ in range(dp_cp)],
            'layernorm1': {
                'split': {'0': tp_union_list[block_id]},
                'dup': dp_cp_union,
                'device_group_union': dg_union_list[block_id],
                'type': 'variable'
            },
            'attn': {
                'qkv': {
                    'split': {'1': tp_union_list[block_id]},
                    'dup': dp_cp_union,
                    'device_group_union': dg_union_list[block_id],
                    'type': 'variable'
                },
                'dense': {
                    'split': {'0': tp_union_list[block_id]},
                    'dup': dp_cp_union,
                    'device_group_union': dg_union_list[block_id],
                    'type': 'variable'
                }
            },
            'layernorm2': {
                'split': {'0': tp_union_list[block_id]},
                'dup': dp_cp_union,
                'device_group_union': dg_union_list[block_id],
                'type': 'variable'
            },
            'mlp': {
                'dense_h_to_4h': {
                    'split': {'1': tp_union_list[block_id]},
                    'dup': dp_cp_union,
                    'device_group_union': dg_union_list[block_id],
                    'type': 'variable'
                },
                'dense_4h_to_h': {
                    'split': {'0': tp_union_list[block_id]},
                    'dup': dp_cp_union,
                    'device_group_union': dg_union_list[block_id],
                    'type': 'variable'
                }
            }
        }
    return ds_parallel_config

@hydra.main(config_path='conf', config_name='config', version_base=None)
def main(config):
    config = config.parallel
    
    if config.cp_list is None:
        cp_list = [1 for _ in range(config.dp)]
    else:
        cp_list = config.cp_list
        assert len(cp_list) == config.dp, "len of cp list should be equal to dp"
    
    num_layers = config.num_layers
    hetero_layers = config.hetero_layers
    assert len(hetero_layers) == sum(cp_list), "number  of pipelines should be equal to dcp"
    accumulate_hetero_stages = [0,]
    for pipeline in hetero_layers:
        assert sum(pipeline) == num_layers, "sum of heterogenous layers of a single pipeline should be equal to the num of total layers"
        accumulate_hetero_stages.append(accumulate_hetero_stages[-1] + len(pipeline))
     
    if config.rank_to_device_mapping is None:
        rank_to_device_mapping = {}       
        for idx in range(config.num_gpus):
            rank_to_device_mapping[idx] = idx
    else:
        rank_to_device_mapping = config.rank_to_device_mapping
        
    if config.unused_rank is None:
        unused_rank = []
    else:
        unused_rank = config.unused_rank
     
    ds_parallel_config = generate_gpt_hetero_4d_config(
        cp_list, 
        rank_to_device_mapping, 
        unused_rank, 
        hetero_layers, 
        accumulate_hetero_stages, 
        num_layers, 
        config.num_gpus, 
        config.dp, 
        config.tp, 
        config.zero,
        config.recompute.recompute_granularity, 
        config.recompute.recompute_method, 
        config.recompute.recompute_num_layers, 
        config.recompute.recompute_layer_idxs_list
    )
    
    save_folder = config.ds_parallel_config_path
    file_name = config.ds_parallel_config_name
    os.makedirs(save_folder, exist_ok=True)
    with open(f'{save_folder}/{file_name}', 'w') as f:
        json.dump(ds_parallel_config, f, indent=4)

if __name__ == '__main__':
    main()