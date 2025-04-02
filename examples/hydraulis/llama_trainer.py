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
from hetu.engine.trainer import HotSPaTrainer, MalleusTrainer

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
    trainer = MalleusTrainer(model_wrapper, optimizer_wrapper, dataset_wrapper)

    # 4. build graph
    trainer.build(args, ds_parallel_configs)

    # 5. training
    trainer.train(args, ds_parallel_configs, local_device, all_devices)

    # 6. end
    print(f'{local_device}: train hetu ds parallel end...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fake_seqlens", type=str, default="[]", help="seqlen list of fake data"
    )
    parser.add_argument(
        "--batching_method", type=int, default=4, help="batching method"
    )
    parser.add_argument(
        "--strategy_pool", type=str, default="./strategy/strategy_pool.json", help="json path to the strategy pool"
    )
    parser.add_argument(
        "--multi_tp_pp_list", type=str, default="[]", help="multi hetero dp strategy list"
    )
    parser.add_argument(
        "--global_batch_size", type=int, default=64, help="global training batch size"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=4096, help="maximum sequence length in the whole dataset"
    )
    parser.add_argument(
        "--data_path", type=str, nargs='+', help=' The blend string, consisting of either a single dataset or a flattened sequential sequence of weight-dataset pairs'
    )
    parser.add_argument(
        "--data_cache_path", type=str, help='Where all re-useable dataset indices are to be cached'
    )
    parser.add_argument(
        "--tokenizer_type", type=str, help='tokenizer type'
    )
    parser.add_argument(
        "--split", type=str, help=' The split string, a comma separated weighting for the dataset splits when drawing samples from a single distribution'
    )
    parser.add_argument(
        "--vocab_file", type=str, help='gpt vocab file path'
    )
    parser.add_argument(
        "--merge_file", type=str, help='gpt merge file path'
    )
    parser.add_argument(
        "--vocab_size", type=int, default=30522, help="total number of vocab"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="hidden size of transformer model",
    )
    parser.add_argument(
        "--ffn_hidden_size", type=int, default=-1, help="ffn hidden size of transformer model",
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=12, help="number of layers"
    )
    parser.add_argument(
        "--num_attention_heads", type=int, default=32, help="number of attention heads",
    )
    parser.add_argument(
        "--epochs", type=int, default=4, help="number of epochs"
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="number of steps for each epoch",
    )
    parser.add_argument(
        "--lr_warmup_init", type=float, default=0, help="Initial value for learning rate warmup"
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="learning rate of adam"
    )
    parser.add_argument(
        "--min_lr", type=float, default=0, help="Minumum value for learning rate"
    )
    parser.add_argument(
        "--lr_decay_style", type=str, default='linear', choices=['constant', 'linear', 'cosine', 'inverse-square-root'],
    )
    parser.add_argument(
        "--lr_warmup_iters", type=int, default=0, help='number of iterations to warm up learning rate over'
    )
    parser.add_argument(
        "--lr_decay_iters", type=int, default=None, help="number of iterations to decay learning rate over"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="weight_decay of adam"
    )
    parser.add_argument(
        "--start_weight_decay", type=float, default=0.01, help="Initial weight decay coefficient for L2 regularization."
    )
    parser.add_argument(
        "--end_weight_decay", type=float, default=0.01, help="End of run weight decay coefficient for L2 regularization."
    )
    parser.add_argument('--weight_decay_incr_style', type=str, default='constant',
                        choices=['constant', 'linear', 'cosine'],
                        help='Weight decay increment function.'
    )
    parser.add_argument(
        "--hidden_act", type=str, default='gelu', help="hidden activation to use."
    )
    parser.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout rate."
    )
    parser.add_argument(
        "--use_flash_attn", action="store_true", help="use Flash Attention."
    )    
    parser.add_argument(
        "--bf16", action="store_true", help="use bfloat16."
    )
    parser.add_argument(
        "--server_addr", type=str, default='127.0.0.1', help="server's address"
    )
    parser.add_argument(
        "--server_port", type=str, default='23457', help="server's port"
    ) 
    parser.add_argument(
        "--ngpus", type=int, default=8, help="num of gpus"
    ) 
    parser.add_argument(
        "--seed", type=int, default=12345, help="num of gpus"
    ) 
    args = parser.parse_args()
    args.bucket_sizes = [int(s) for s in args.bucket_sizes.split()]
    pretrain(args)