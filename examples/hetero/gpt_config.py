import hetu
class GPTConfig(object):
    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        ffn_hidden_size=-1,
        n_layer=12,
        n_head=12,
        n_inner=None,
        seq_len=128,
        activation_function="relu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=False,
        bos_token_id=50256,
        eos_token_id=50256,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        global_batch_size = 100,
        num_micro_batches = 1,
        dp = 1,
        use_flash_attn = False,
        lora_dtype="fp32",
        dqtype="fp32",
    ):
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.seq_len = seq_len
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.global_batch_size = global_batch_size
        self.num_micro_batches = num_micro_batches
        self.dp = dp
        self.hidden_size = self.n_embd
        if ffn_hidden_size == -1:
            self.ffn_hidden_size = 4*self.n_embd
        else:
            self.ffn_hidden_size = ffn_hidden_size
        self.max_position_embeddings = self.n_positions
        self.num_attention_heads = self.n_head
        self.num_hidden_layers = self.n_layer

        self.add_cross_attention = False
        self.use_return_dict = False
        self.output_attentions = False
        self.output_hidden_states= False
        self.use_flash_attn = use_flash_attn
        if lora_dtype == "fp16":
            self.lora_dtype = hetu.float16
        elif lora_dtype == "bf16":
            self.lora_dtype = hetu.bfloat16
        elif lora_dtype == "fp4":
            self.lora_dtype = hetu.float4
        elif lora_dtype == "nf4":
            self.lora_dtype = hetu.nfloat4
        else:
            self.lora_dtype = hetu.float32
        
        if dqtype == "fp16":
            self.dqtype = hetu.float16
        elif dqtype == "bf16":
            self.dqtype = hetu.bfloat16
        else:
            self.dqtype = hetu.float32