import os
import time
import signal
import argparse
import hetu as ht
import numpy as np
from tqdm import tqdm
from trainer.utils import ModelWrapper, OptimizerWrapper, TrainerConfig
from model import LLamaConfig, LLamaLMHeadModel, QKVFusedLLamaLMHeadModel
from utils import distributed_init, read_ds_parallel_config, parse_multi_ds_parallel_config, convert_strategy, generate_lora_ds_parallel_config, write_to_csv
from trainer.batch_scheduler import make_micro_batches
from peft.lora import MultiLoraModel

def llama_benchmark(args):
    all_devices = ht.global_device_group()
    local_device = ht.local_device()
    gpu_id = all_devices.get_index(local_device)
    ngpus = args.dp * args.tp * args.pp
    layers_tp_groups, _ = convert_strategy([(args.tp, args.pp, args.dp)], ngpus, args.num_hidden_layers)
    config_file_path = f"ds_parallel_config/benchmark.json"
    generate_lora_ds_parallel_config(ngpus, layers_tp_groups, config_file_path)
    ds_parallel_configs = read_ds_parallel_config(config_file_path)

    model_config = LLamaConfig(
        vocab_size=args.vocab_size,
        ffn_hidden_size=args.ffn_hidden_size,
        n_embd=args.hidden_size,
        n_head=args.num_attention_heads,
        n_layer=args.num_layers,
        resid_pdrop=args.dropout_prob,
        embd_pdrop=args.dropout_prob,
        attn_pdrop=args.dropout_prob,
        use_flash_attn=args.use_flash_attn,
        sequence_parallel=args.sequence_parallel
    )

    # wrapper
    trainer_config = TrainerConfig(args.trainer_config_path)
    assert args.train_task_num == trainer_config.train_task_num, \
        f"args.train_task_num should be equal to that in trainer_config, but got {args.train_task_num} v.s. {trainer_config.train_task_num}"
    if trainer_config.variant == 'fused':
        pretrained_model_wrapper = ModelWrapper(QKVFusedLLamaLMHeadModel, model_config)
    elif trainer_config.variant == 'canonical':
        pretrained_model_wrapper = ModelWrapper(LLamaLMHeadModel, model_config)
    else:
        raise ValueError(f'Unsupported variant: {trainer_config.variant}')
    optimizer_wrapper = OptimizerWrapper(ht.AdamOptimizer)
    
    # profiler
    args.default_seq_len = args.seq_length
    args.default_mbs = args.micro_batch_size
    
    # build model
    with ht.graph("define_and_run", num_strategy=1):
        if args.bf16:
            precision = "ht.bfloat16"
        else:
            precision = "ht.float32"
        precision = eval(precision)
        with ht.autocast(precision):
            input_ds_hierarchy, input_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'input')
            label_ds_hierarchy, label_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'label')
            task_ds_hierarchy, task_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'task_batch_idxs')
            
            default_seq_len = args.seq_length
            default_mbs_times_dp = args.micro_batch_size * args.dp
            default_batch_offset = 0
            default_batch_size = args.micro_batch_size
            pretrained_model_wrapper.model_config.dp_symbol = ht.IntSymbol(args.dp)
    
            input_ids = ht.parallel_placeholder(ht.int64, global_shape=[default_mbs_times_dp, default_seq_len], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='input_ids')
            masked_lm_labels = ht.parallel_placeholder(ht.int64, global_shape=[default_mbs_times_dp, default_seq_len], ds_hierarchy=label_ds_hierarchy, device_group_hierarchy=label_dg_hierarchy, name='masked_lm_labels')
            pretrained_model_wrapper.model_config.task_batch_idxs = []
            for i in range(trainer_config.train_task_num):
                pretrained_model_wrapper.model_config.task_batch_idxs.append(
                    ht.parallel_placeholder(
                        ht.int64,
                        global_shape=[default_batch_offset, default_batch_size, default_batch_size],
                        ds_hierarchy=task_ds_hierarchy,
                        device_group_hierarchy=task_dg_hierarchy,
                        name='task_batch_idxs_task{}'.format(i),
                        is_cpu=True
                    )
                )
            pretrained_model_wrapper.model_config.train_task_num = trainer_config.train_task_num
            
            pretrained_model = pretrained_model_wrapper.create_model(ds_parallel_configs=ds_parallel_configs)
            peft_configs = [task_config.lora_config for task_config in trainer_config.task_configs]
            model = MultiLoraModel(
                model=pretrained_model,
                peft_configs=peft_configs,
                config=pretrained_model_wrapper.model_config
            )
            loss = model(
                input_ids=input_ids,
                labels=masked_lm_labels,
            )
            opt = optimizer_wrapper.create_optimizer(lr=args.lr)
            train_op = opt.minimize(loss)
    
    micro_batch_num = args.num_micro_batches
    mbs = args.micro_batch_size
    max_tokens = args.seq_length * mbs
    pretrained_model.config.dp_symbol.set_data(args.dp)

    e2e_time = []
    _profile_option = False if args.save_path == "null" else True
    pbar = None
    if gpu_id == 0:
        pbar = tqdm(total=args.profile_steps) if _profile_option else None
    if args.warmup_steps > 0 and gpu_id == 0:
        print(f"Warmup takes {args.warmup_steps} steps...")

    input_ids_list = []
    masked_lm_labels_list = []
    task_batch_idxs_list = [[] for _ in range(trainer_config.train_task_num)]
    test_micro_batches = make_micro_batches(mbs, micro_batch_num, args.seq_length, trainer_config.train_task_num)
    for test_micro_batch in test_micro_batches:
        batch_data = np.array(test_micro_batch.batch_data).reshape(mbs, -1)
        labels = batch_data[:, 1:]
        tokens = batch_data[:, :-1]
        for train_task_idx in range(trainer_config.train_task_num):
            task_batch_idxs_i = np.zeros([test_micro_batch.batch_offset_list[train_task_idx], test_micro_batch.batch_size_list[train_task_idx], test_micro_batch.batch_size], dtype=np.int64)
            task_batch_idxs_list[train_task_idx].append(task_batch_idxs_i)
        input_ids_list.append(tokens.astype(np.int64))
        masked_lm_labels_list.append(labels.astype(np.int64))
    feed_dict = {
        input_ids: input_ids_list,
        masked_lm_labels: masked_lm_labels_list,
    }
    for i in range(trainer_config.train_task_num):
        feed_dict[pretrained_model_wrapper.model_config.task_batch_idxs][i] = task_batch_idxs_list[i]

    # run
    for step in range(args.profile_steps + args.warmup_steps):
        print(f"step: {step}")
        try:
            with ht.autocast(precision):
                start_time = time.time()
                results = train_op.graph.run(
                            loss,
                            [loss, train_op],
                            feed_dict=feed_dict,
                            num_micro_batches=micro_batch_num,
                            cur_strategy_id=0,
                            run_level=ht.run_level("update"),
                            grad_scale=1.0)
                end_time = time.time()
        except RuntimeError as e:
            print(e)
            os.killpg(0, signal.SIGTERM)
        print(f"e2e_time = {end_time - start_time:.3f}s")
        if _profile_option and gpu_id == 0 and step >= args.warmup_steps:
            e2e_time.append(end_time - start_time)
            pbar.update(1)
    
    # record
    if _profile_option and gpu_id == 0:
        pbar.close()
        record_entry = {
            'dp': args.dp,
            'tp': args.tp,
            'pp': args.pp,
            'train_task_num': args.train_task_num,
            'mbs': args.micro_batch_size,
            'seq_len': args.seq_length,
            'max_tokens': max_tokens,
            'num_micro_batches': args.num_micro_batches,
            'e2e_time': np.mean(e2e_time),
            'throughput_per_gpu': (micro_batch_num * max_tokens) / (np.mean(e2e_time) * args.tp * args.pp)
        }
        write_to_csv(record_entry, args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
        '--dp', type=int, default=1, help='dp degree'
    )
    parser.add_argument(
        '--tp', type=int, default=1, help='tp degree'
    )
    parser.add_argument(
        '--pp', type=int, default=1, help='pp degree'
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
        "--lr", type=float, default=1e-5, help="Learning rate of adam"
    )
    parser.add_argument(
        "--save_path", type=str, default='', help="save path of max tokens."
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="num layers"
    )
    parser.add_argument(
        "--seq_length", type=int, default=1024, help="profile seq length"
    )
    parser.add_argument(
        "--micro_batch_size", type=int, default=2, help="micro batch size"
    )
    parser.add_argument(
        "--num_micro_batches", type=int, default=16, help="num micro batches"
    )
    parser.add_argument(
        "--profile_steps", type=int, default=100, help="profile steps"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=10, help="warmup steps"
    )
    parser.add_argument(
        "--server_addr", type=str, default='127.0.0.1', help="server's address"
    )
    parser.add_argument(
        "--server_port", type=str, default='23457', help="server's port"
    )
    args = parser.parse_args()
    distributed_init(args.ngpus, args.server_addr, args.server_port)
    llama_benchmark(args)