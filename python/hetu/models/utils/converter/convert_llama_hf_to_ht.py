
r"""
Conversion script to convert Huggingface Llama checkpoints into Hetu checkpoint.
  Example to run this conversion script:
    python convert_hf_to_ht.py \
     --input_name_or_path <path_to_hf_checkpoints_folder> \
     --output_path <path_to_output_hetu_checkpoint>
     --precision bfloat16 \
     --sharded_store
"""

import gc
import logging
import argparse
import transformers
import hetu
import torch
from hetu.models.llama import LlamaLMHeadModel
from hetu.models.llama import LlamaConfig as HtLlamaConfig
from collections import OrderedDict
from hetu import setup_logging
from hetu.utils.parallel import distributed_init
from hetu.models.utils.converter.convert_utils import save_model, DEFAULT_DS

def get_args():
    parser = argparse.ArgumentParser(description="Convert Huggingface Llama checkpoints to Hetu checkpoints")
    parser.add_argument(
        "--input_name_or_path",
        type=str,
        required=True,
        help="Path to the Huggingface Llama checkpoint",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the Hetu checkpoint",
    )
    parser.add_argument(
        "--precision",
        type=str,
        required=True,
        choices=["float32", "float16", "bfloat16"],
        help="Precision of the model",
    )
    parser.add_argument(
        "--sharded_store",
        action="store_true",
        help="Whether to use sharded store",
    )
    parser.add_argument(
        "--test_loading",
        action="store_true",
        help="Whether to test loading the saved model",
    )
    parser.add_argument(
        "--server_addr",
        type=str,
        default='127.0.0.1',
        help="Server address for distributed training",
    )
    parser.add_argument(
        "--server_port",
        type=str,
        default='23457',
        help="Server port for distributed training",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs for distributed training",
    )
    args = parser.parse_args()
    return args

def convert_config(hf_config, precision):
    # TODO: consider RoPE related parameters
    hetu_config = HtLlamaConfig(
        vocab_size=hf_config['vocab_size'],
        hidden_size=hf_config['hidden_size'],
        intermediate_size=hf_config['intermediate_size'],
        num_hidden_layers=hf_config['num_hidden_layers'],
        num_attention_heads=hf_config['num_attention_heads'],
        num_key_value_heads=hf_config.get('num_key_value_heads', None),
        hidden_act='fast-swiglu',
        max_position_embeddings=hf_config['max_position_embeddings'],
        initializer_range=hf_config['initializer_range'],
        rms_norm_eps=hf_config['rms_norm_eps'],
        use_cache=True,
        rope_theta=hf_config.get('rope_theta', 10000.0),
        rope_scaling=hf_config.get('rope_scaling', None),
        attention_dropout=hf_config['attention_dropout'],
        head_dim=hf_config.get('head_dim', None),
        use_flash_attn=True,
        gated_linear_unit=True,
        model_dtype=precision,
    )
    return hetu_config    

def convert(input_name_or_path, output_path, precision, sharded_store=False, test_loading=False):
    logging.info(f"Converting Huggingface Llama checkpoint from {input_name_or_path}")
    # load the model from Huggingface
    model = transformers.AutoModel.from_pretrained(input_name_or_path)
    config = vars(model.config)
    
    logging.info(f"hf_config: {config}")
    logging.info("named parameters:")
    for name, _ in model.named_parameters():
        logging.info(f"- {name}")
    
    # convert hf config to hetu model config
    ht_config = convert_config(config, args.precision)
    
    # set precision
    if args.precision in ["32", "16"]:
        precision = int(float(args.precision))
    else:
        precision = args.precision
    
    # create hetu checkpoint state dict
    param_to_weights = lambda param: hetu.numpy_to_NDArray(param.float().numpy())
    checkpoint = OrderedDict()
    checkpoint['state_dict'] = OrderedDict()
    
    # read weights from hf model
    hidden_size = ht_config.hidden_size
    head_num = ht_config.num_attention_heads
    head_size = hidden_size // head_num
    num_layers = ht_config.num_hidden_layers
    num_key_value_heads = ht_config.num_key_value_heads
    
    hf_state_dict = model.state_dict()
    # 给hf_state_dict所有key加上前缀'model.'
    with hetu.graph("eager"):
        with hetu.context(eager_device="cpu"):
            hf_state_dict = {f'model.{k}': v for k, v in hf_state_dict.items()}
            embed_weight = hf_state_dict['model.embed_tokens.weight']
            checkpoint['state_dict'][f'model.wte.embedding_table'] = param_to_weights(embed_weight)

            for l in range(int(num_layers)):
                logging.info(f"converting layer {l}")
                # Attn
                old_tensor_shape = hf_state_dict[f'model.layers.{l}.self_attn.q_proj.weight'].size()
                new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]
                new_kv_tensor_shape = (num_key_value_heads, head_size) + old_tensor_shape[1:]
                q = hf_state_dict[f'model.layers.{l}.self_attn.q_proj.weight'].view(*new_q_tensor_shape)
                k = hf_state_dict[f'model.layers.{l}.self_attn.k_proj.weight'].view(*new_kv_tensor_shape)
                v = hf_state_dict[f'model.layers.{l}.self_attn.v_proj.weight'].view(*new_kv_tensor_shape)
                qkv_weights = torch.empty((0, head_size) + old_tensor_shape[1:])
                heads_per_group = head_num // num_key_value_heads
                for i in range(num_key_value_heads):
                    qkv_weights = torch.cat((qkv_weights, q[i * heads_per_group : (i + 1) * heads_per_group, :, :]), axis=0)
                    qkv_weights = torch.cat((qkv_weights, k[i : i + 1, :, :]), axis=0)
                    qkv_weights = torch.cat((qkv_weights, v[i : i + 1, :, :]), axis=0)
                qkv_weights = qkv_weights.reshape([head_size * (head_num + 2 * num_key_value_heads), hidden_size])
                checkpoint['state_dict'][f'model.h.{l}.attn.qkv_dense.weight'] = param_to_weights(qkv_weights)

                o_weight = hf_state_dict[f'model.layers.{l}.self_attn.o_proj.weight']
                checkpoint['state_dict'][f'model.h.{l}.attn.dense.weight'] = param_to_weights(o_weight)
                
                # MLP
                mlp_down_weight = hf_state_dict[f'model.layers.{l}.mlp.gate_proj.weight']
                mlp_gate_weight = hf_state_dict[f'model.layers.{l}.mlp.up_proj.weight']
                mlp_down_weight = torch.cat((mlp_down_weight, mlp_gate_weight), axis=0)
                checkpoint['state_dict'][f'model.h.{l}.mlp.dense_h_to_4h.weight'] = param_to_weights(mlp_down_weight)
                
                mlp_up_weight = hf_state_dict[f'model.layers.{l}.mlp.down_proj.weight'].t().contiguous()
                checkpoint['state_dict'][f'model.h.{l}.mlp.dense_4h_to_h.weight'] = param_to_weights(mlp_up_weight)
                
                # LayerNorm
                input_ln_weight = hf_state_dict[f'model.layers.{l}.input_layernorm.weight']
                checkpoint['state_dict'][f'model.h.{l}.ln_1.weight'] = param_to_weights(input_ln_weight)

                post_attn_ln_weight = hf_state_dict[f'model.layers.{l}.post_attention_layernorm.weight']
                checkpoint['state_dict'][f'model.h.{l}.ln_2.weight'] = param_to_weights(post_attn_ln_weight)
                
                logging.info(f"done layer {l}")
            
            # final ln
            final_ln_weight = hf_state_dict['model.norm.weight']
            checkpoint['state_dict']['model.ln_f.weight'] = param_to_weights(final_ln_weight)
        
            # lm_head
            if 'lm_head.weight' in hf_state_dict:
                if not config['tie_word_embeddings']:
                    lm_head_weight = hf_state_dict['lm_head.weight']
                    checkpoint['state_dict']['model.lm_head.weight'] = param_to_weights(lm_head_weight)
                else:
                    checkpoint['state_dict']['model.lm_head.weight'] = checkpoint['state_dict']['model.wte.embedding_table']
    
    # delete the model
    del model
    
    # load the checkpoint to hetu model and check missing and unexpected keys
    if test_loading:
        default_ds_parallel_config = DEFAULT_DS.copy()
        default_block_ds = default_ds_parallel_config["llama"]["blocks"]["blocks0"]
        default_ds_parallel_config["llama"]["blocks"] = {}
        for block_id in range(num_layers):
            block_ds = default_block_ds.copy()
            block_ds["range"] = [block_id]
            default_ds_parallel_config["llama"]["blocks"][f"blocks{block_id}"] = block_ds
        with hetu.graph("define_and_run", num_strategy=1):
            with hetu.autocast(hetu.bfloat16):
                model = LlamaLMHeadModel(ht_config, ds_parallel_configs=[default_ds_parallel_config])
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
        if missing_keys:
            logging.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys: {unexpected_keys}")
    
    # cast to target precision and save the model
    save_model(checkpoint['state_dict'], output_path, precision=precision, sharded_store=sharded_store)
    ht_config.save_pretrained(output_path)
    logging.info(f"Hetu model saved to {output_path}")

if __name__ == '__main__':
    setup_logging()
    try:
        args = get_args()
        if args.test_loading:
            assert args.server_addr and args.server_port, \
                "Server address and port must be provided for testing loading"
            assert args.num_gpus > 0, "Number of GPUs must be greater than 0 for testing loading"
            local_device, all_devices = distributed_init(args.num_gpus, args.server_addr, args.server_port)
        convert(args.input_name_or_path, args.output_path, args.precision, args.sharded_store, args.test_loading)
    finally:
        gc.collect()