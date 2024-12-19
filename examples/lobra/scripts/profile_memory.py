import os
import argparse
import subprocess
from tqdm import tqdm
from utils import write_to_csv, read_from_csv

def run_profile(
    args,
    tp,
    pp,
    seq_len
):
    num_micro_batches = pp * 2
    dp = 2 if tp * pp * 2 <= args.num_gpus_limit else 1
    cmd = f"bash scripts/run_benchmark.sh \
            {args.num_layers} {args.hidden_size} {args.num_attention_heads} {args.train_task_num} \
            {seq_len} 1 {num_micro_batches} \
            {dp} {tp} {pp} {args.server_addr} {args.server_port} \
            {args.host_file} {args.env_file} \
            null {args.trainer_config_path} profile_memory"
    try:
        subprocess.run(cmd, shell=True, check=True)
        return 0
    except Exception as e:
        print(e)
        return -1

def profile_memory(args, seq_len_range):
    # for specific strategy
    if args.num_gpus_limit == -1:
        print(f"profile max tokens for specific strategy (tp, pp) = ({args.tp}, {args.pp})")
        max_tokens = 0
        for seq_len in seq_len_range:
            if run_profile(args, args.tp, args.pp, seq_len) == 0:
                max_tokens = seq_len
            else:
                break
        print(f"strategy (tp, pp) = ({args.tp}, {args.pp}) found max tokens = {max_tokens}")
        max_tokens_entry = {
            'tp': args.tp,
            'pp': args.pp,
            'max_tokens': max_tokens
        }
        write_to_csv(max_tokens_entry, args.profile_memory_path)
    else:
        cache_dict = None
        if os.path.exists(args.profile_memory_path):
            rows = read_from_csv(args.profile_memory_path)
            cache_dict = {(row['tp'], row['pp']) : row['max_tokens'] for row in rows}
        # pp_candidates 为 args.num_layers 的因数
        pp_candidates = [i for i in range(1, args.num_layers + 1) if args.num_layers % i == 0]
        tp_candidates = [1, 2, 4, 8]
        pbar = tqdm(total=len(pp_candidates) * len(tp_candidates))
        for pp in pp_candidates:
            for tp in tp_candidates:
                if tp * pp > args.num_gpus_limit or (cache_dict is not None and (tp, pp) in cache_dict.keys()):
                    pbar.update(1)
                    continue
                max_tokens = 0
                for seq_len in seq_len_range:
                    if run_profile(args, tp, pp, seq_len) == 0:
                        max_tokens = seq_len
                    else:
                        break
                print(f"strategy (tp, pp, sp) = ({tp}, {pp}) found max tokens = {max_tokens}")
                max_tokens_entry = {
                    'tp': tp,
                    'pp': pp,
                    'max_tokens': max_tokens
                }
                write_to_csv(max_tokens_entry, args.profile_memory_path)
                pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tp", type=int, default=1, help="tp degree"
    )
    parser.add_argument(
        "--pp", type=int, default=1, help="pp degree"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
    )
    parser.add_argument(
        "--train_task_num", type=int, default=1, help="Number of layers"
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
        "--profile_memory_path", type=str, default='', help="save path of max tokens."
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="num layers"
    )
    parser.add_argument(
        "--seq_len_limit", type=str, default='', help="seq length limit"
    )
    parser.add_argument(
        "--num_gpus_limit", type=int, default=-1, help="num gpus limit"
    )
    parser.add_argument(
        "--server_addr", type=str, default='127.0.0.1', help="server's address"
    )
    parser.add_argument(
        "--server_port", type=str, default='23457', help="server's port"
    )
    parser.add_argument(
        "--host_file", type=str, default='scripts/hostfile.yaml', help="hostfile path"
    )
    parser.add_argument(
        "--env_file", type=str, default='scripts/env.sh', help="env path"
    )
    args = parser.parse_args()
    candidate_seq_len_range = [1024, 2048, 4096, 8192, 16384]
    candidate_seq_len_range = [seq_len for seq_len in candidate_seq_len_range if seq_len <= int(args.seq_len_limit)]
    profile_memory(args, candidate_seq_len_range)
