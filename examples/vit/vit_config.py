import hetu

class VisionConfig(object):
    def __init__(
        self,
        in_channels = 3,
        image_size = 224,
        patch_size = 14,
        spatial_merge_size = 2,
        temporal_patch_size = 2,
        num_hidden_layers = 24,
        num_attention_heads = 16,
        embed_dim = 1024,
        num_classes = 10,
        micro_batch_size = 2,
        use_flash_attn = True,
        add_bias_linear = False,
        add_qkv_bias = False,
        max_position_embeddings = 4096,
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
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.num_hidden_layers = num_hidden_layers
        self.micro_batch_size = micro_batch_size
        self.num_attention_heads = num_attention_heads
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_flash_attn = use_flash_attn
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.max_position_embeddings = max_position_embeddings
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
