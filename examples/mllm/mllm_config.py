import hetu

class LLaMAConfig(object):
    def __init__(
        self,
        vocab_size=50257,
        hidden_size=768,
        ffn_hidden_size=-1,
        n_layer=12,
        n_head=12,
        n_inner=None,
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
        use_flash_attn = False,
        lora_dtype="fp32",
        dqtype="fp32",
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
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
        if ffn_hidden_size == -1:
            self.ffn_hidden_size = 4 * self.hidden_size
        else:
            self.ffn_hidden_size = ffn_hidden_size
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


class VisionConfig(object):
    def __init__(
        self,
        in_channels = 3,
        patch_size = 14,
        spatial_merge_size = 2,
        temporal_patch_size = 2,
        num_hidden_layers = 24,
        num_attention_heads = 16,
        embed_dim = 1024,
        use_flash_attn = True,
        add_bias_linear = False,
        add_qkv_bias = False,
        hidden_size = 1024,
        hidden_dropout = 0.0,
        attention_dropout = 0.0,
        mlp_dim = 4096,
        acitvation_func = "relu",
        dqtype="fp32",
    ):
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.embed_dim = embed_dim
        self.use_flash_attn = use_flash_attn
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.hidden_size = hidden_size
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.mlp_dim = mlp_dim 
        self.acitvation_func = acitvation_func
        if dqtype == "fp16":
            self.dqtype = hetu.float16
        elif dqtype == "bf16":
            self.dqtype = hetu.bfloat16
        else:
            self.dqtype = hetu.float32

class MLLMConfig(object):
    def __init__(
        self,
        VisionConfig,
        LLamaConfig
    ):
        self.VisionConfig = VisionConfig
        self.LLamaConfig = LLamaConfig
