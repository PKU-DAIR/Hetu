import os
import gc
import json
import hetu
import logging
from hetu.utils.checkpoint import save_file
from hetu.models.utils import SAFE_WEIGHTS_NAME
from hetu.models.utils.common_utils import split_hetu_state_dict_into_shards

DEFAULT_DS = {
    'zero': False,
    'devices': [0],
    'input': {
        'split': {'0': [1]},
        'dup': [1],
        'device_group_union': [[0]],
        'type': 'placeholder'
    },
    'llama': {
        'wte': {
            'split': {'0': [1]},
            'dup': [1],
            'device_group_union': [[0]],
            'type': 'variable',
            'zero': False
        },
        'wpe': {
            'split': {},
            'dup': [1],
            'device_group_union': [[0]],
            'type': 'variable',
            'zero': False
        },
        'blocks': {
            'blocks0': {
                'range': [0],
                'recompute': [False],
                'rmsnorm1': {
                    'split': {'0': [1]},
                    'dup': [1],
                    'device_group_union': [[0]],
                    'type': 'variable',
                    'zero': False
                },
                'attn': {
                    'qkv': {
                        'split': {'1': [1]},
                        'dup': [1],
                        'device_group_union': [[0]],
                        'type': 'variable',
                        'zero': False
                    },
                    'dense': {
                        'split': {'0': [1]},
                        'dup': [1],
                        'device_group_union': [[0]],
                        'type': 'variable',
                        'zero': False
                    }
                },
                'rmsnorm2': {
                    'split': {'0': [1]},
                    'dup': [1],
                    'device_group_union': [[0]],
                    'type': 'variable',
                    'zero': False
                },
                'mlp': {
                    'dense_h_to_4h': {
                        'split': {'1': [1]},
                        'dup': [1],
                        'device_group_union': [[0]],
                        'type': 'variable',
                        'zero': False
                    },
                    'dense_4h_to_h': {
                        'split': {'0': [1]},
                        'dup': [1],
                        'device_group_union': [[0]],
                        'type': 'variable',
                        'zero': False
                    }
                }
            }
        },
        'rmsnorm_final': {
            'split': {'0': [1]},
            'dup': [1],
            'device_group_union': [[0]],
            'type': 'variable',
            'zero': False
        }
    },
    'lm_head': {
        'split': {'1': [1]},
        'dup': [1],
        'device_group_union': [[0]],
        'type': 'variable',
        'zero': False
    },
    'label': {
        'split': {'0': [1]},
        'dup': [1],
        'device_group_union': [[0]],
        'type': 'placeholder'
    }
}


def save_model(model_state_dict, output_path, precision, sharded_store):
    os.makedirs(output_path, exist_ok=True)

    try:
        def convert_precision(param, precision):
            dtype = eval(f"hetu.{precision}")
            param_dtype = param.dtype
            if param_dtype == dtype:
                return param
            else:
                return param.to(dtype)
        
        for name, param in model_state_dict.items():
            model_state_dict[name] = convert_precision(param, precision)
        
        if sharded_store:
            state_dict_split = split_hetu_state_dict_into_shards(model_state_dict)
            for filename, tensors in state_dict_split.filename_to_tensors.items():
                logging.info(f"tensors: {tensors}")
                logging.info(f"Saving to {filename}...")
                shard = {}
                try:
                    for tensor in tensors:
                        shard[tensor] = model_state_dict[tensor].contiguous()
                    save_path = os.path.join(output_path, filename)
                    save_file(
                        shard,
                        save_path,
                        metadata={"format": "ht"},
                    )
                    logging.info(f"{filename} saved to {save_path}")
                finally:
                    # release memory
                    shard.clear()
                    del shard

            if state_dict_split.is_sharded:
                index = {
                    "metadata": state_dict_split.metadata,
                    "weight_map": state_dict_split.tensor_to_filename,
                }
                index_path = os.path.join(output_path, "model.safetensors.index.json")
                logging.info(f"Saving index to {index_path}...")
                with open(index_path, "w", encoding="utf-8") as f:
                    content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                    f.write(content)
                logging.info(f"Index saved to {index_path}")
        else:
            save_file(
                model_state_dict,
                os.path.join(output_path, SAFE_WEIGHTS_NAME),
                metadata={"format": "ht"},
            )
    finally:
        # release memory
        model_state_dict.clear()
        gc.collect()
