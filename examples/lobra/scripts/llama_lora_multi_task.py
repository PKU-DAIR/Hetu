import argparse
import hetu as ht
from utils import distributed_init
from model import LLaMAConfig, LLaMALMHeadModel, QKVFusedLLaMALMHeadModel, PackedLLaMALMHeadModel
from profiler.cost_model import CostModel
from data_utils import GPTJsonDataset
from utils import parse_strategy_config
from trainer import Trainer, TrainerConfig, StrategyConfig, DatasetWrapper, ModelWrapper, OptimizerWrapper

def finetune(args, max_tokens_list=None):
    local_device = ht.local_device()
    all_devices = ht.global_device_group()

    gpu_id = all_devices.get_index(local_device)
    trainer_config = TrainerConfig(args.trainer_config_path)
    assert args.train_task_num == trainer_config.train_task_num, \
        f"args.train_task_num should be equal to that in trainer_config, but got {args.train_task_num} v.s. {trainer_config.train_task_num}"
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
    cost_model = CostModel(
        model_config,
        args.profile_path,
        args.profile_memory_path,
        sequence_parallel=args.sequence_parallel
    )
    strategy_config = StrategyConfig(args.scheme_list, args.ngpus, args.num_layers, gpu_id)
    model_config.dp_symbol = ht.IntSymbol(1)

    # wrapper
    use_packing = False
    if trainer_config.variant == 'fused':
        model_wrapper = ModelWrapper(QKVFusedLLaMALMHeadModel, model_config)
    elif trainer_config.variant == 'packed':
        model_wrapper = ModelWrapper(PackedLLaMALMHeadModel, model_config)
        use_packing = True
    elif trainer_config.variant == 'canonical':
        model_wrapper = ModelWrapper(LLaMALMHeadModel, model_config)
    else:
        raise ValueError(f'Unsupported variant: {trainer_config.variant}')
    optimizer_wrapper = OptimizerWrapper(ht.AdamOptimizer)
    dataset_wrapper = DatasetWrapper(GPTJsonDataset)
    
    # trainer
    trainer = Trainer(
        args,
        dataset_wrapper,
        model_wrapper,
        optimizer_wrapper,
        trainer_config,
        cost_model,
        [strategy_config],
        max_tokens_list,
        use_packing=use_packing
    )
    
    # build graph
    trainer.build_model(args, strategy_config.ds_parallel_configs)
    # train
    if use_packing:
        trainer.packed_run(args)
    else:
        trainer.run(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy_config_path", default="", type=str, help="strategy config path"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=8192, help="max seq length of samples"
    )
    parser.add_argument(
        "--min_seq_length", type=int, default=256, help="min seq length of samples"
    )
    parser.add_argument(
        "--max_tokens", type=str, default="", help="max tokens of each strategy"
    )
    parser.add_argument(
        "--json_file", type=str, help='data json format file path'
    )
    parser.add_argument(
        "--json_key", type=str, help='json key for tokens'
    )
    parser.add_argument(
        "--vocab_file", type=str, help='gpt vocab file path'
    )
    parser.add_argument(
        "--merge_file", type=str, help='gpt merge file path'
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
        "--num_layers", type=int, default=12, help="Number of layers"
    )
    parser.add_argument(
        "--bucket_num", type=int, default=16, help="Number of dp bucket"
    )
    parser.add_argument(
        "--train_task_num", type=int, default=16, help="Number of train task"
    )
    parser.add_argument(
        "--ngpus", type=int, default=16, help="Number of GPUs"
    )
    parser.add_argument(
        "-a",
        "--num_attention_heads",
        type=int,
        default=12,
        help="Number of attention heads",
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
        "--sequence_parallel", action="store_true", help="Use Sequence Parallel."
    )    
    parser.add_argument(
        "--split_scheme", action="store_true", help="Use Sequence Parallel."
    )    
    parser.add_argument(
        "--bf16", action="store_true", help="Use bfloat16."
    )
    parser.add_argument(
        "--trainer_config_path", type=str, default='', help="Trainer config path of multi-task training."
    )
    parser.add_argument(
        "--scheme_list", type=str, default='', help="Scheme list of heterogeneous strategy"
    )
    parser.add_argument(
        "--profile_path", type=str, default='', help="profile path of profiler."
    )
    parser.add_argument(
        "--profile_memory_path", type=str, default='', help="memory profile path of profiler."
    )
    parser.add_argument(
        "--server_addr", type=str, default='127.0.0.1', help="server's address"
    )
    parser.add_argument(
        "--server_port", type=str, default='23457', help="server's port"
    ) 
    args = parser.parse_args()
    args.scheme_list, args.max_tokens = parse_strategy_config(args.strategy_config_path, args.split_scheme)
    distributed_init(args.ngpus, args.server_addr, args.server_port)
    finetune(args)
    print(f'train hetu ds parallel end...')