import hydra
from omegaconf import OmegaConf
import json
import os
from hetu.utils.parallel import generate_recompute_config

def generate_llama_4d_config(
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
        'llama': {
            'wte': {
                'split': {'0': [tp]},
                'dup': [dp * cp],
                'device_group_union': [device_groups[0]],
                'type': 'variable'
            },
            'blocks': {

            },
            'rmsnorm_final': {
                'split': {'0': [tp]},
                'dup': [dp * cp],
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
                'recompute': recompute_config.blocks_recompute[block_id],
                'output_recompute': recompute_config.blocks_output_recompute[block_id],
                'cpu_offload': [False],
                'rmsnorm1': {
                    'split': {'0': [tp]},
                    'dup': [dp * cp],
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
                'rmsnorm2': {
                    'split': {'0': [tp]},
                    'dup': [dp * cp],
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
                    },
                    'activation_func': {
                    }
                }
            }
    return ds_parallel_config

@hydra.main(config_path=None, config_name="config", version_base=None)
def main(config):
    config = OmegaConf.select(config, "ds_parallel")
    num_layers = config.num_layers
        
    assert config.dp * config.cp * config.tp * config.pp == config.num_gpus, \
            f'dp * cp * tp * pp = {config.dp * config.cp * config.tp * config.pp} is not equal to num_gpus {config.num_gpus}!'
    
    ds_parallel_config = generate_llama_4d_config(
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
        config.recompute.recompute_layer_idxs_list
    )
    
    save_folder = config.ds_parallel_config_path
    file_name = config.ds_parallel_config_name
    os.makedirs(save_folder, exist_ok=True)
    with open(f'{save_folder}/{file_name}', 'w') as f:
        json.dump(ds_parallel_config, f, indent=4)
    print(f"Generated llama 4D config file: {save_folder}/{file_name}")

if __name__ == '__main__':
    main()
