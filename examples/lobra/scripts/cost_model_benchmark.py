import os
import argparse
import hetu as ht
import numpy as np
from trainer.utils import ModelWrapper, OptimizerWrapper, TrainerConfig
from model import LLaMAConfig, LLaMALMHeadModel, QKVFusedLLaMALMHeadModel
from profiler import Profiler
from utils import distributed_init, convert_strategy, generate_lora_ds_parallel_config, read_ds_parallel_config, write_to_csv, read_from_csv

def run_benchmark_llama(args, seq_len_range, profile_mbs, max_tokens):
    all_devices = ht.global_device_group()
    local_device = ht.local_device()
    gpu_id = all_devices.get_index(local_device)
    ngpus = args.tp
    layers_tp_groups, _ = convert_strategy([(args.tp, 1, 1)], ngpus, args.num_hidden_layers)
    config_file_path = f"ds_parallel_config/profile_cost_model.json"
    generate_lora_ds_parallel_config(ngpus, layers_tp_groups, config_file_path)
    ds_parallel_configs = read_ds_parallel_config(config_file_path)

    model_config = LLaMAConfig(
        vocab_size=args.vocab_size,
        ffn_hidden_size=args.ffn_hidden_size,
        n_embd=args.hidden_size,
        n_head=args.num_attention_heads,
        n_layer=args.num_layers,
        resid_pdrop=args.dropout_prob,
        embd_pdrop=args.dropout_prob,
        attn_pdrop=args.dropout_prob,
        use_flash_attn=args.use_flash_attn,
        sequence_parallel=args.sequence_parallel,
    )

    # wrapper
    trainer_config = TrainerConfig(args.trainer_config_path)
    if trainer_config.variant == 'fused':
        model_wrapper = ModelWrapper(QKVFusedLLaMALMHeadModel, model_config)
    elif trainer_config.variant == 'canonical':
        model_wrapper = ModelWrapper(LLaMALMHeadModel, model_config)
    else:
        raise ValueError(f'Unsupported variant: {trainer_config.variant}')
    optimizer_wrapper = OptimizerWrapper(ht.AdamOptimizer)
    
    # profiler
    args.default_seq_len = seq_len_range[0]
    args.default_mbs = profile_mbs[0]
    profiler = Profiler(
        args, 
        model_wrapper,
        optimizer_wrapper,
        trainer_config,
        ds_parallel_configs
    )
    
    # build graph
    profiler.build_model(args, ds_parallel_configs)
    
    # read from cache to avoid redundant profiling
    cache_dict = None
    if args.num_layers <= 3:
        rows = read_from_csv(args.profile_path)
        cache_dict = {(row['tp'], row['seq_len'], row['mbs']) : row['time'] for row in rows}
    else:
        rows = read_from_csv(args.validation_path)
        cache_dict = {(row['tp'], row['seq_len'], row['mbs']) : row['time'] for row in rows}
    
    # profile
    for seq_len in seq_len_range:
        for mbs in profile_mbs:
            if cache_dict is not None and (args.tp, seq_len, mbs) in cache_dict:
                continue
            if max_tokens != -1 and mbs * seq_len > max_tokens:
                continue
            print(f"profiling: (tp, seq_len, mbs) = ({args.tp}, {seq_len}, {mbs})")
            profiler.profile(mbs, seq_len)
            if gpu_id == 0:
                total_stream_time = profiler.total_stream_time
                total_time_entry = {
                    'tp': args.tp,
                    'seq_len': seq_len,
                    'mbs': mbs,
                    'time': np.mean(total_stream_time)
                }
                block_time = profiler.block_time
                block_time_entry = {
                    'tp': args.tp,
                    'seq_len': seq_len,
                    'mbs': mbs,
                    'time': np.mean(block_time)    
                }
                if args.num_layers <= 3:
                    write_to_csv(block_time_entry, args.profile_path)
                else:
                    write_to_csv(total_time_entry, args.validation_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout rate."
    )
    parser.add_argument(
        "--use_flash_attn", action="store_true", help="Use Flash Attention."
    )
    parser.add_argument(
        '--tp', type=int, default=1, help='tp degree'
    )
    parser.add_argument(
        '--sequence_parallel', action="store_true", help='Use Sequence Parallel'
    )
    parser.add_argument(
        "--bf16", action="store_true", help="Use bfloat16."
    )
    parser.add_argument(
        "--vocab_size", type=int, default=30522, help="Total number of vocab"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
    )
    parser.add_argument(
        "--ffn_hidden_size", type=int, default=768, help="FFN hidden size of llama model",
    )
    parser.add_argument(
        "--profile_steps", type=int, default=100, help="profile steps"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=10, help="warmup steps"
    )
    parser.add_argument(
        "-a",
        "--num_attention_heads",
        type=int,
        default=12,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--trainer_config_path", type=str, default='', help="Trainer config path of multi-task training."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate of adam"
    )
    parser.add_argument(
        "--validation_path", type=str, default='', help="validation path of profiler."
    )
    parser.add_argument(
        "--profile_path", type=str, default='', help="profile path of profiler."
    )
    parser.add_argument(
        "--profile_memory_path", type=str, default='', help="profile memory path of profiler."
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="num layers"
    )
    parser.add_argument(
        "--seq_len_range", type=str, default='', help="profile seq len range"
    )
    parser.add_argument(
        "--profile_mbs", type=str, default='', help="profile micro batch size"
    )
    parser.add_argument(
        "--server_addr", type=str, default='127.0.0.1', help="server's address"
    )
    parser.add_argument(
        "--server_port", type=str, default='23457', help="server's port"
    ) 
    args = parser.parse_args()
    if args.profile_mbs == '':
        profile_mbs = [1, 2, 4, 8, 16, 32]
    else:
        profile_mbs = list(map(int, args.profile_mbs.split(',')))
    if args.seq_len_range == '':
        seq_len_range = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    else:
        seq_len_range = list(map(int, args.seq_len_range.split(',')))

    # if memory profile exists, filter out seq_len that exceeds the memory limit
    max_tokens = -1
    if os.path.exists(args.profile_memory_path):
        rows = read_from_csv(args.profile_memory_path)
        memory_dict = {(row['tp'], row['pp']) : row['max_tokens'] for row in rows}
        for ds_config, tokens in memory_dict.items():
            if ds_config[0] == args.tp:
                max_tokens = max(max_tokens, tokens)
        seq_len_range = [seq_len for seq_len in seq_len_range if seq_len <= max_tokens]

    distributed_init(args.ngpus, args.server_addr, args.server_port)
    run_benchmark_llama(args, seq_len_range, profile_mbs, max_tokens)
        
