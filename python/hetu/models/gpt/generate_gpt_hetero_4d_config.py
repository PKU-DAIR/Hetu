import argparse
import json
import os
import ast

def generate_gpt_hetero_4d_config(cp_list, rank_to_device_mapping, unused_rank, hetero_layers, accumulate_hetero_stages, recompute_layers, num_layers=32, num_gpus=8, dp=2, tp=2, zero=True):
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

    ds_parallel_config = {
        'zero': zero,
        'devices': list(range(num_gpus)),
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
            'wpe': {
                'split': {},
                'dup': [tp_union_list[0][i] * dp_cp for i in range(dp_cp)],
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
            'recompute': [(True if block_id in recompute_layers[i] else False) for i in range(dp_cp)],
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
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_layers', type=int, default=32, help='size of gpt, 7b is 32 and 13b is 40.'
    )
    parser.add_argument(
        '--num_gpus', type=int, default=8, help='num of gpus.'
    )
    parser.add_argument(
        '--dp', type=int, default=2, help='dp.'
    )
    parser.add_argument(
        '--cp_list', type=str, default="", help='cp list.'
    )
    parser.add_argument(
        '--tp', type=int, default=2, help='tp.'
    )
    parser.add_argument(
        '--hetero_layers', type=str, help='heterogenous layers list.'
    )
    parser.add_argument(
        '--rank_to_device_mapping', type=str, default="", help='device to rank mapping.'
    )
    parser.add_argument(
        '--unused_rank', type=str, default="[]", help='unused rank list.'
    )
    parser.add_argument(
        '--zero', action='store_true', help='use zero or not.'
    )
    parser.add_argument(
        '--recompute_layers', type=str, default="", help='layers to recompute.'
    )
    parser.add_argument(
        '--file_name', type=str, default="", help="file path to save."
    )
    args = parser.parse_args()
    
    if args.cp_list == "":
        cp_list = [1 for _ in range(args.dp)]
    else:
        cp_list = ast.literal_eval(args.cp_list)
        assert len(cp_list) == args.dp, "len of cp list should be equal to dp"
    
    num_layers = args.num_layers
    hetero_layers = ast.literal_eval(args.hetero_layers)
    assert len(hetero_layers) == sum(cp_list), "number  of pipelines should be equal to dcp"
    accumulate_hetero_stages = [0,]
    for pipeline in hetero_layers:
        assert sum(pipeline) == num_layers, "sum of heterogenous layers of a single pipeline should be equal to the num of total layers"
        accumulate_hetero_stages.append(accumulate_hetero_stages[-1] + len(pipeline))
     
    if args.rank_to_device_mapping == "":
        rank_to_device_mapping = {}       
        for idx in range(args.num_gpus):
            rank_to_device_mapping[idx] = idx
    else:
        rank_to_device_mapping = ast.literal_eval(args.rank_to_device_mapping)
     
    if args.recompute_layers == "":   
        recompute_layers = [[] for _ in range(sum(cp_list))]
    else:
        recompute_layers = ast.literal_eval(args.recompute_layers)
        assert len(recompute_layers) == sum(cp_list), "recompute layers state should align to dcp num"  
        
    ds_parallel_config = generate_gpt_hetero_4d_config(cp_list, rank_to_device_mapping, ast.literal_eval(args.unused_rank), hetero_layers, accumulate_hetero_stages, recompute_layers, num_layers, args.num_gpus, args.dp, args.tp, args.zero)
    
    save_folder = './ds_parallel_config/gpt_hetero'
    if args.file_name == "":
        file_name = f'dcp{sum(cp_list)}_tp{args.tp}_pp{[len(pipeline) for pipeline in hetero_layers]}.json'
    else:
        file_name = args.file_name
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(f'{save_folder}/{file_name}', 'w') as f:
        json.dump(ds_parallel_config, f, indent=4)

