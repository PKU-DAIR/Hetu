import hydra
import json
import os
from hetu.utils.parallel import generate_recompute_config

def generate_gpt_4d_config(
    num_layers=32, 
    num_gpus=8, 
    dp=2, 
    cp=1, 
    tp=2, 
    pp=2, 
    zero=True,
    recompute_granularity=None,
    recompute_method=None,
    recompute_num_layers=None,
    recompute_layer_idxs_list=None
):
    
    if dp == 1:
        zero = False
    num_layers_per_stage = num_layers // pp
    num_devices_per_stage = num_gpus // pp
    device_groups = [list(range(stage_id * num_devices_per_stage, (stage_id + 1) * num_devices_per_stage)) for stage_id in range(pp)]
    
    recompute_config = generate_recompute_config(
        dp * cp,
        num_layers,
        [[num_layers_per_stage] * pp] * (dp * cp),
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
            'split': {'0': [dp * cp]},
            'dup': [tp],
            'device_group_union': [device_groups[0]],
            'type': 'placeholder'
        },
        'gpt': {
            'wte': {
                'split': {'0': [tp]},
                'dup': [dp * cp],
                'device_group_union': [device_groups[0]],
                'type': 'variable'
            },
<<<<<<< HEAD
            'wpe': {
                'split': {},
                'dup': [dp * cp * tp],
                'device_group_union': [device_groups[0]],
                'type': 'variable'
            },
            'blocks': {

            },
<<<<<<<< HEAD:python/hetu/models/gpt/generate_gpt_4d_config.py
            'layernorm_final': {
                'split': {'0': [tp]},
                'dup': [dp * cp],
========
            'rmsnorm_final': {
                'split': {},
                'dup': [dp * cp * tp],
>>>>>>>> upstream/main:examples/ampelos/ds_parallel_config/generate_gpt_3d_config.py
=======
            'blocks': {

            },
            'layernorm_final': {
                'split': {'0': [tp]},
                'dup': [dp * cp],
>>>>>>> upstream/main
                'device_group_union': [device_groups[-1]],
                'type': 'variable'
            }
        },
        'lm_head': {
            'split': {'1': [tp]},
            'dup': [dp * cp],
            'device_group_union': [device_groups[-1]],
            'type': 'variable'
        },
        'label': {
            'split': {'0': [dp * cp]},
            'dup': [tp],
            'device_group_union': [device_groups[-1]],
            'type': 'placeholder'
        }
    }

    for stage_id in range(pp):
        block_start_id = num_layers_per_stage * stage_id
        block_end_id = num_layers_per_stage * (stage_id + 1)
        for block_id in range(block_start_id, block_end_id):
            blocks_json = ds_parallel_config['llama']['blocks']
            blocks_json[f'blocks{block_id}'] = {
                'range': [block_id,],
<<<<<<< HEAD
                'recompute': [True if block_id in recompute_layers else False],
<<<<<<<< HEAD:python/hetu/models/gpt/generate_gpt_4d_config.py
                'layernorm1': {
                    'split': {'0': [tp]},
                    'dup': [dp * cp],
========
                'rmsnorm1': {
                    'split': {},
                    'dup': [dp * cp * tp],
>>>>>>>> upstream/main:examples/ampelos/ds_parallel_config/generate_gpt_3d_config.py
=======
                'recompute': recompute_config.blocks_recompute[block_id],
                'output_recompute': recompute_config.blocks_output_recompute[block_id],
                'cpu_offload': [False],
                'layernorm1': {
                    'split': {'0': [tp]},
                    'dup': [dp * cp],
>>>>>>> upstream/main
                    'device_group_union': [device_groups[stage_id]],
                    'type': 'variable'
                },
                'attn': {
                    'qkv': {
                        'split': {'1': [tp]},
                        'dup': [dp * cp],
                        'device_group_union': [device_groups[stage_id]],
                        'type': 'variable'
                    },
                    'dense': {
                        'split': {'0': [tp]},
                        'dup': [dp * cp],
                        'device_group_union': [device_groups[stage_id]],
                        'type': 'variable'
                    }
                },
<<<<<<< HEAD
<<<<<<<< HEAD:python/hetu/models/gpt/generate_gpt_4d_config.py
                'layernorm2': {
                    'split': {'0': [tp]},
                    'dup': [dp * cp],
========
                'rmsnorm2': {
                    'split': {},
                    'dup': [dp * cp * tp],
>>>>>>>> upstream/main:examples/ampelos/ds_parallel_config/generate_gpt_3d_config.py
=======
                'layernorm2': {
                    'split': {'0': [tp]},
                    'dup': [dp * cp],
>>>>>>> upstream/main
                    'device_group_union': [device_groups[stage_id]],
                    'type': 'variable'
                },
                'mlp': {
                    'dense_h_to_4h': {
                        'split': {'1': [tp]},
                        'dup': [dp * cp],
                        'device_group_union': [device_groups[stage_id]],
                        'type': 'variable'
                    },
                    'dense_4h_to_h': {
                        'split': {'0': [tp]},
                        'dup': [dp * cp],
                        'device_group_union': [device_groups[stage_id]],
                        'type': 'variable'
                    }
                }
            }
    return ds_parallel_config

<<<<<<< HEAD
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_layers', type=int, default=32, help='size of model, 7b is 32 and 13b is 40.'
    )
    parser.add_argument(
        '--num_gpus', type=int, default=8, help='num of gpus.'
    )
    parser.add_argument(
        '--dp', type=int, default=2, help='dp.'
    )
    parser.add_argument(
        '--cp', type=int, default=1, help='cp.'
    )
    parser.add_argument(
        '--tp', type=int, default=2, help='tp.'
    )
    parser.add_argument(
        '--pp', type=int, default=2, help='pp.'
    )
    parser.add_argument(
        '--zero', action='store_true', help='use zero or not.'
    )
    parser.add_argument(
        '--recompute_layers', type=str, default="[]", help='layers to recompute.'
    )
    args = parser.parse_args()
    num_layers = args.num_layers
        
    assert args.dp * args.cp * args.tp * args.pp == args.num_gpus, \
            f'dp * cp * tp * pp = {args.dp * args.cp * args.tp * args.pp} is not equal to num_gpus {args.num_gpus}!'
    
    ds_parallel_config = generate_gpt_4d_config(ast.literal_eval(args.recompute_layers), num_layers, args.num_gpus, args.dp, args.cp, args.tp, args.pp, args.zero)
    
<<<<<<<< HEAD:python/hetu/models/gpt/generate_gpt_4d_config.py
    save_folder = './ds_parallel_config/gpt_homo'
    file_name = f'dp{args.dp}_cp{args.cp}_tp{args.tp}_pp{args.pp}.json'
========
    # save_folder = './ds_parallel_config/homo'
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = cur_dir + "/homo"
    file_name = f'dcp{args.dp * args.cp}_tp{args.tp}_pp{args.pp}.json'
>>>>>>>> upstream/main:examples/ampelos/ds_parallel_config/generate_gpt_3d_config.py
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(f'{save_folder}/{file_name}', 'w') as f:
        json.dump(ds_parallel_config, f, indent=4)
    print(os.path.abspath(f'{save_folder}/{file_name}'))

=======
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(config):
    config = config.ds_parallel
    num_layers = config.num_layers
        
    assert config.dp * config.cp * config.tp * config.pp == config.num_gpus, \
            f'dp * cp * tp * pp = {config.dp * config.cp * config.tp * config.pp} is not equal to num_gpus {config.num_gpus}!'
    
    ds_parallel_config = generate_gpt_4d_config(
        num_layers, 
        config.num_gpus, 
        config.dp, 
        config.cp, 
        config.tp, 
        config.pp, 
        config.zero,
        config.recompute.recompute_granularity, 
        config.recompute.recompute_method, 
        config.recompute.recompute_num_layers, 
        config.recompute.recompute_layer_idxs
    )
    
    save_folder = config.ds_parallel_config_path
    file_name = config.ds_parallel_config_name
    os.makedirs(save_folder, exist_ok=True)
    with open(f'{save_folder}/{file_name}', 'w') as f:
        json.dump(ds_parallel_config, f, indent=4)

if __name__ == '__main__':
    main()
>>>>>>> upstream/main
