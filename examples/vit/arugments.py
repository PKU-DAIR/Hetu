
import argparse
parser = argparse.ArgumentParser()


def _add_model_args(parser):
    group = parser.add_argument_group(title='model')
    # LLM
    group.add_argument("--vocab_size", type=int, default=30522, help="total number of vocab")
    # group.add_argument("--hidden_size", type=int, default=768, help="hidden size of transformer model")
    # group.add_argument("--ffn_hidden_size", type=int, default=-1, help="ffn hidden size of transformer model")
    # group.add_argument("--num_hidden_layers", type=int, default=12, help="number of layers")
    # group.add_argument("--num_attention_heads", type=int, default=32, help="number of attention heads")
    group.add_argument("--hidden_act", type=str, default='gelu', help="hidden activation to use")
    group.add_argument("--dropout_prob", type=float, default=0.1, help="Dropout rate")
    group.add_argument("--use_flash_attn", action="store_true", help="use Flash Attention")

    # Vision
    group.add_argument("--vision_embed_dim", type=int, default=768, help="embed dim")
    group.add_argument("--vision_num_heads", type=int, default=12, help="num heads")
    group.add_argument("--vision_mlp_dim", type=int, default=3072, help="mlp dim")
    group.add_argument("--vision_num_layers", type=int, default=12, help="num layers")
    group.add_argument("--vision_dropout", type=float, default=0.1, help="dropout")
    group.add_argument("--patch_size", type=int, default=16, help="patch size")
    group.add_argument("--in_channels", type=int, default=3, help="in channels")
    group.add_argument("--max_position_embeddings", type=int, default=4096, help="max position embeddings")
    
    return parser


def _add_data_args(parser):
    group = parser.add_argument_group(title='data')
    group.add_argument("--data_path", type=str, help="The data path")
    group.add_argument("--data_cache_path", type=str, help="Where all re-useable dataset indices are to be cached")
    group.add_argument("--tokenizer_type", type=str, default="GPT2BPETokenizer", help="tokenizer type")
    group.add_argument("--split", type=str, default="98,1,1", help="The split string, a comma-separated weighting for the dataset splits when drawing samples from a single distribution")
    # group.add_argument("--vocab_file", type=str, help="gpt vocab file path")
    # group.add_argument("--merge_file", type=str, help="gpt merge file path")
    group.add_argument("--max_seq_len", type=int, default=4096, help="maximum sequence length in the whole dataset")
    group.add_argument("--fake_seqlens", type=str, default="[]", help="seqlen list of fake data")
    group.add_argument("--image_size", type=int, default=224, help="image size")
    group.add_argument("--num_classes", type=int, default=10, help="number of classes")
    return parser


def _add_training_args(parser):
    group = parser.add_argument_group(title='training')
    group.add_argument("--global_batch_size", type=int, default=64, help="global training batch size")
    group.add_argument("--micro_batch_size", type=int, default=2, help="micro training batch size")
    group.add_argument("--batching_method", type=int, default=4, help="batching method")
    group.add_argument("--epochs", type=int, default=4, help="number of epochs")
    group.add_argument("--steps", type=int, default=20, help="number of steps for each epoch")
    group.add_argument("--lr", type=float, default=1e-5, help="learning rate of adam")
    group.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    group.add_argument("--seed", type=int, default=12345, help="random seed for reproducibility")
    group.add_argument("--bf16", action="store_true", help="use bfloat16")
    group.add_argument("--torch_profile", type=int, default=0, help="use pytorch profiler")
    return parser

def _add_distributed_args(parser):
    group = parser.add_argument_group(title='distributed')
    group.add_argument("--llm_multi_tp_pp_list", type=str, default="[]", help="llm multi hetero dp strategy list")
    group.add_argument("--vision_multi_tp_pp_list", type=str, default="[]", help="vision multi hetero dp strategy list")
    group.add_argument("--ngpus", type=int, default=8, help="number of gpus")
    group.add_argument("--server_addr", type=str, default="127.0.0.1", help="server's address")
    group.add_argument("--server_port", type=str, default="23457", help="server's port")
    group.add_argument("--strategy_pool", type=str, default="./strategy/strategy_pool.json", help="json path to the strategy pool")
    return parser

def add_all_args(parser):
    parser = _add_model_args(parser)
    parser = _add_data_args(parser)
    parser = _add_training_args(parser)
    parser = _add_distributed_args(parser)
    return parser