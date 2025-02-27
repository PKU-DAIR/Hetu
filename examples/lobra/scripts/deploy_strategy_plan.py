import os
import numpy as np
import argparse
from trainer.utils.dp_bucket import get_buckets_dp
from trainer.utils.wrapper import DatasetWrapper
from data_utils import GPTJsonDataset, Encoder
from model import LLaMAConfig
from profiler import CostModel
from types import SimpleNamespace
from utils import export_strategy_config
from trainer.planner import GroupStaticPlanner, BalanceStaticPlanner, PruneStaticPlanner
from trainer.trainer import TrainerConfig, DatasetContext

def deploy_strategy_plan(args):
    trainer_config = TrainerConfig(args.trainer_config_path)
    assert args.train_task_num == trainer_config.train_task_num, \
        f"args.train_task_num should be equal to that in trainer_config, but got {args.train_task_num} v.s. {trainer_config.train_task_num}"
    cost_model_config = LLaMAConfig(
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
        cost_model_config,
        args.profile_path,
        args.profile_memory_path,
        sequence_parallel=args.sequence_parallel
    )
    strategy_candidates = cost_model.get_strategy_candidates(args.num_layers)
    global_batch_size_list = trainer_config.get_global_batch_size_list()
    dataset_wrapper = DatasetWrapper(GPTJsonDataset)

    use_optimized_scheme_pool = True
    if os.environ.get('EXPR_SCHEME_PROPOSAL') == 'ON':
        use_optimized_scheme_pool = True
    elif os.environ.get('EXPR_SCHEME_PROPOSAL') == 'OFF':
        use_optimized_scheme_pool = False
    else:
        raise ValueError(f'EXPR_SCHEME_PROPOSAL should be ON or OFF, but got {os.environ.get("EXPR_SCHEME_PROPOSAL")}')

    strategy_deploy_pattern = os.environ.get('EXPR_DEPLOY_PATTERN')
    if strategy_deploy_pattern == 'GROUP':
        static_batch_planner = GroupStaticPlanner(cost_model, args.num_layers, trainer_config.train_task_num,
                                                  global_batch_size_list, args.ngpus, strategy_candidates,
                                                  use_optimized_scheme_pool=use_optimized_scheme_pool)
    elif strategy_deploy_pattern == 'BALANCE':
        static_batch_planner = BalanceStaticPlanner(cost_model, args.num_layers, trainer_config.train_task_num,
                                                    global_batch_size_list, args.ngpus, strategy_candidates,
                                                    use_optimized_scheme_pool=use_optimized_scheme_pool)
    elif strategy_deploy_pattern == 'PRUNE':
        static_batch_planner = PruneStaticPlanner(cost_model, args.num_layers, trainer_config.train_task_num,
                                                  global_batch_size_list, args.ngpus, strategy_candidates,
                                                  use_optimized_scheme_pool=use_optimized_scheme_pool)
    else:
        raise ValueError(f'Unsupported strategy deploy pattern: {strategy_deploy_pattern}')

    dataset_ctxs = []
    seq_len_distribution_list = []
    if os.environ.get('EXPR_CUSTOM_DISTRIBUTION') == 'ON':
        seq_len_distribution_list.append({256: 7, 512: 18, 1024: 33, 2048: 9, 4096: 1, 8192: 1})
        num = sum(seq_len_distribution_list[0].values())
        seq_len_distribution_list[0] = {key: value / num for key, value in seq_len_distribution_list[0].items()}
        static_batch_planner.global_batch_size_list = [num]
    elif os.environ.get('EXPR_CUSTOM_DISTRIBUTION') == 'OFF':
        encoder_args = {
            'key': 'text',
            'rank': 0,
            'make_vocab_size_divisible_by': 128,
            'tensor_model_parallel_size': 1,
            'vocab_extra_ids': 0,
            'tokenizer_type': 'GPT2BPETokenizer',
            'vocab_file': args.vocab_file,
            'merge_file': args.merge_file,
        }
        encoder_args = SimpleNamespace(**encoder_args)
        encoder = Encoder(encoder_args)
        train_dataset_pool = {}
        fine_grained_seq_len_num_distribution_list = []
        fine_grained_buckets_of_all_tasks = set()
        
        for i in range(trainer_config.train_task_num):
            task_config = trainer_config.task_configs[i]
            if train_dataset_pool.get((task_config.dataset_name, task_config.context_length)) is not None:
                train_dataset = train_dataset_pool[(task_config.dataset_name, task_config.context_length)]
            else:
                train_dataset = dataset_wrapper.create_dataset(
                    dataset_name=task_config.dataset_name,
                    key=task_config.json_key,
                    max_seq_len=task_config.context_length,
                    vocab_file=args.vocab_file,
                    merge_file=args.merge_file,
                    encoder=encoder)
                train_dataset_pool[(task_config.dataset_name, task_config.context_length)] = train_dataset
            dataset_ctx = DatasetContext(
                dataset=train_dataset,
                steps=task_config.steps,
                epochs=task_config.epochs)
            dataset_ctxs.append(dataset_ctx)

        bucket_limit = args.bucket_num
        alignment = 16
        if os.environ.get("BUCKET_PLAN") == "DYNAMIC":
            for i in range(trainer_config.train_task_num):
                train_dataset = dataset_ctxs[i].dataset
                fine_grained_buckets = train_dataset.get_aligned_buckets(alignment=alignment)
                fine_grained_buckets_of_all_tasks = fine_grained_buckets_of_all_tasks.union(fine_grained_buckets)
            fine_grained_buckets_of_all_tasks = sorted(list(fine_grained_buckets_of_all_tasks))
            max_seq_len = 0
            for i in range(trainer_config.train_task_num):
                train_dataset = dataset_ctxs[i].dataset
                fine_grained_seq_len_distribution = train_dataset.get_length_distribution(args.min_seq_length, args.max_seq_length, fine_grained_buckets_of_all_tasks)
                max_seq_len = max(max_seq_len, max(fine_grained_seq_len_distribution.keys()))
                sample_gbs = static_batch_planner.global_batch_size_list[i] * 100
                seq_len_num_distribution = {k: round(p * sample_gbs) for k, p in fine_grained_seq_len_distribution.items()}
                fine_grained_seq_len_num_distribution_list.append(seq_len_num_distribution)
            bucket_candidates = fine_grained_buckets_of_all_tasks
            merge_global_batch_seqlen_list = []
            has_max_seq_len = False # ensure max_seq_len is in the global batch
            for i in range(trainer_config.train_task_num):
                seq_len_num_distribution = fine_grained_seq_len_num_distribution_list[i]
                for k, v in seq_len_num_distribution.items():
                    if k == max_seq_len and v > 0:
                        has_max_seq_len = True
                    merge_global_batch_seqlen_list.extend([k] * v)
            if not has_max_seq_len:
                merge_global_batch_seqlen_list.append(max_seq_len)
            global_batch_seqlen_list = sorted(merge_global_batch_seqlen_list)
            dp_buckets = get_buckets_dp(np.array(global_batch_seqlen_list, dtype=np.int32), np.array(bucket_candidates, dtype=np.int32), bucket_limit)
            
            # adapt to new bucket
            for i in range(trainer_config.train_task_num):
                train_dataset = dataset_ctxs[i].dataset
                fine_grained_seq_len_distribution = train_dataset.get_length_distribution(args.min_seq_length, args.max_seq_length, dp_buckets)
                seq_len_distribution_list.append(fine_grained_seq_len_distribution)
        elif os.environ.get("BUCKET_PLAN") == "STATIC":
            if bucket_limit == 7:
                dp_buckets = [256, 512, 1024, 2048, 4096, 8192, 16384] # 7 bucket
            elif bucket_limit == 16:
                dp_buckets = [256, 512, 768, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192, 12288, 16384] # 16 buckets
            for i in range(trainer_config.train_task_num):
                train_dataset = dataset_ctxs[i].dataset
                seq_len_distribution = train_dataset.get_length_distribution(args.min_seq_length, args.max_seq_length, dp_buckets)
                seq_len_distribution_list.append(seq_len_distribution)
        elif os.environ.get("BUCKET_PLAN") == "PROFILE" and \
             os.environ.get("EXPR_EFFECTIVENESS") == "ON":
            pass
        else:
            raise ValueError(f'Invalid BUCKET_PLAN: {os.environ.get("BUCKET_PLAN")}')
    else:
        raise ValueError(f'EXPR_CUSTOM_DISTRIBUTION should be ON or OFF, but got {os.environ.get("EXPR_CUSTOM_DISTRIBUTION")}')

    # experiment only
    if os.environ.get("BUCKET_PLAN") == "PROFILE" and \
       os.environ.get("EXPR_EFFECTIVENESS") == "ON":
        with open("effectiveness.txt", 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 4):
                seq_len_distribution_list = []
                seq_len_distribution = eval(lines[i])
                for v in seq_len_distribution.values():
                    seq_len_distribution_list.append(v)
                with open('effectiveness_static.txt', 'a') as ff:
                    ff.write(f'step: {i // 4}\n')
                static_batch_planner.schedule(seq_len_distribution_list)
    else:
        ds_parallel_config, _ = static_batch_planner.schedule(seq_len_distribution_list)
        export_strategy_config(ds_parallel_config['scheme_list'],
                               ds_parallel_config['max_tokens_list'],
                               args.strategy_config_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_seq_length", type=int, default=8192, help="max seq length of samples"
    )
    parser.add_argument(
        "--min_seq_length", type=int, default=256, help="min seq length of samples"
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
        "--dropout_prob", type=float, default=0.1, help="Dropout rate."
    )
    parser.add_argument(
        "--use_flash_attn", action="store_true", help="Use Flash Attention."
    )
    parser.add_argument(
        "--sequence_parallel", action="store_true", help='Use Sequence Parallel'
    )
    parser.add_argument(
        "--ngpus", type=int, default=8, help="gpu num"
    )
    parser.add_argument(
        "--bucket_num", type=int, default=16, help="bucket num"
    )
    parser.add_argument(
        "--train_task_num", type=int, default=16, help="Number of train task"
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
        "--strategy_config_path", type=str, default='', help="Strategy config path."
    )
    parser.add_argument(
        "--profile_path", type=str, default='', help="profile path of profiler."
    )
    parser.add_argument(
        "--profile_memory_path", type=str, default='', help="max tokens path of profiler."
    )
    args = parser.parse_args()
    deploy_strategy_plan(args)