import os
import time
import argparse
import hetu as ht
from hetu.models.llama.llama_model import LLaMALMHeadModel
from hetu.models.llama.llama_config import LLaMAConfig
from hetu.engine.data_utils.llama import LLaMAJsonDataset
from hetu.engine.distributed import distributed_init
from hetu.engine.parallel_config import read_ds_parallel_config, simple_check_blocks_range
from hetu.engine.wrapper import ModelWrapper, OptimizerWrapper, DatasetWrapper
from hetu.engine.trainer import HotSPaTrainer

def pretrain(args):
    # 1. config
    local_device, all_devices = distributed_init()
    ds_parallel_configs = read_ds_parallel_config(args)
    num_strategy = len(ds_parallel_configs)
    llama_config = LLaMAConfig(vocab_size=args.vocab_size, 
        n_positions=args.seq_len,
        n_ctx=args.seq_len,
        n_embd=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        n_layer=args.num_hidden_layers, 
        n_head=args.num_attention_heads, 
        seq_len=args.seq_len,
        resid_pdrop=args.dropout_prob,
        embd_pdrop=args.dropout_prob,
        attn_pdrop=args.dropout_prob,
        activation_function=args.hidden_act,
        use_flash_attn=args.use_flash_attn,
    )
    assert llama_config.use_flash_attn == True, "symbolic shape can only used when flash attn is on for now"
    simple_check_blocks_range(ds_parallel_configs, llama_config.num_hidden_layers, 'llama')

    # 2. wrapper
    model_wrapper = ModelWrapper(LLaMALMHeadModel, llama_config) 
    optimizer_wrapper = OptimizerWrapper(ht.AdamOptimizer)
    dataset_wrapper = DatasetWrapper(LLaMAJsonDataset)

    # 3. trainer
    trainer = HotSPaTrainer(model_wrapper, optimizer_wrapper, dataset_wrapper)

    # 4. build graph
    trainer.build(args, ds_parallel_configs)

    # 5. training
    trainer.train(args, ds_parallel_configs, local_device, all_devices)

    # 6. end
    print(f'{local_device}: train hetu ds parallel end...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_memory_experiment", action="store_true", help="run memory experiment."
    )
    parser.add_argument(
        "--hot_switch", action="store_true", help='enable parallelism hot switching.'
    )
    parser.add_argument(
        "--test_func", action="store_true", help='test functional for parallelism hot switching.'
    )
    parser.add_argument(
        '--gpu_id', type=int, default=0, help='Id of GPU to run.'
    )
    parser.add_argument(
        "--num_strategy", type=int, default=1, help="multi ds num"
    )
    parser.add_argument(
        "--ds_parallel_config", default="ds_parallel_config/dp2_tp2_pp2.json", type=str, 
        help="multi parallel strategy config file for hot switching"
    )
    parser.add_argument(
        "--bucket_sizes", default="32768 16384 4096 0", type=str, 
        help="multi bucket size for hot switching"
    )
    parser.add_argument(
        "--memory_file", default="", type=str, help="memory experiment result file"
    )
    parser.add_argument(
        "--global_batch_size", type=int, default=64, help="Training batch size global"
    )
    parser.add_argument(
        "--micro_batch_size", type=int, default=2, help="Training batch size each micro batch"
    )
    parser.add_argument(
        "--dataset", type=str, default='wikicorpus_en', help="Dataset used to train."
    )
    parser.add_argument(
        "--json_file", type=str, help='data json format file path'
    )
    parser.add_argument(
        "--json_key", type=str, help='json key for tokens'
    )
    parser.add_argument(
        "--vocab_file", type=str, help='llama vocab file path'
    )
    parser.add_argument(
        "--merge_file", type=str, help='llama merge file path'
    )
    parser.add_argument(
        "--vocab_size", type=int, default=30522, help="Total number of vocab"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
    )
    parser.add_argument(
        "--ffn_hidden_size", type=int, default=-1, help="FFN hidden size of transformer model",
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=12, help="Number of layers"
    )
    parser.add_argument(
        "-a",
        "--num_attention_heads",
        type=int,
        default=12,
        help="Number of attention heads",
    )
    parser.add_argument(
        "-s", "--seq_len", type=int, default=128, help="Maximum sequence len"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=4, help="Number of epochs"
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="Number of steps for each epoch",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate of adam"
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )
    parser.add_argument(
        "--hidden_act", type=str, default='gelu', help="Hidden activation to use."
    )
    parser.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout rate."
    )
    parser.add_argument(
        "--use_flash_attn", action="store_true", help="Use Flash Attention."
    )    
    parser.add_argument(
        "--bf16", action="store_true", help="Use bfloat16."
    )
    args = parser.parse_args()
    args.bucket_sizes = [int(s) for s in args.bucket_sizes.split()]
    pretrain(args)