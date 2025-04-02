import os
import hetu
import argparse
from hetu import LlamaLMHeadModel, setup_logging
from hetu.utils.parallel import distributed_init
from hetu.models.utils.converter.convert_utils import DEFAULT_DS

def get_args():
    parser = argparse.ArgumentParser(description="Convert Huggingface Llama checkpoints to Hetu checkpoints")
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
    parser.add_argument(
        "--num_layers",
        type=int,
        default=32,
        help="Number of layers in the model",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model checkpoint",
    )
    args = parser.parse_args()
    return args

def test_ckpt_load(model_path, num_layers):
    default_ds_parallel_config = DEFAULT_DS.copy()
    default_block_ds = default_ds_parallel_config["llama"]["blocks"]["blocks0"]
    default_ds_parallel_config["llama"]["blocks"] = {}
    for block_id in range(num_layers):
        block_ds = default_block_ds.copy()
        block_ds["range"] = [block_id]
        default_ds_parallel_config["llama"]["blocks"][f"blocks{block_id}"] = block_ds

    if not os.path.exists(model_path):
        raise ValueError(f"Model path {model_path} does not exist")
    print("load from original checkpoint...")
    # 把model_path的最后一个参数作为模型名
    model = LlamaLMHeadModel.from_pretrained(model_path, ds_parallel_configs=[default_ds_parallel_config], use_safetensors=True, model_dtype=hetu.bfloat16)
    model.save_pretrained(f"{model_path}-hetu", dtype=hetu.bfloat16)
    print("load from saved hetu checkpoint...")
    model = LlamaLMHeadModel.from_pretrained(f"{model_path}-hetu", ds_parallel_configs=[default_ds_parallel_config], use_safetensors=True, model_dtype=hetu.float32)

if __name__ == '__main__':
    setup_logging()
    args = get_args()
    assert args.server_addr and args.server_port, \
        "Server address and port must be provided for testing loading"
    assert args.num_gpus > 0, "Number of GPUs must be greater than 0 for testing loading"
    local_device, all_devices = distributed_init(args.num_gpus, args.server_addr, args.server_port)
    test_ckpt_load(args.model_path, args.num_layers)