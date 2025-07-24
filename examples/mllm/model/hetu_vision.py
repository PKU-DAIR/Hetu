import hetu as ht
import numpy as np
import torch
from hetu.nn.modules.parallel_multi_ds import parallel_data_provider, parallel_multi_data_provider, get_multi_ds_parallel_config


class PatchEmbed(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, name="patch_embed"):
        super(PatchEmbed, self).__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        # self.temporal_patch_size = 1
        self.in_channels = config.in_channels
        self.embed_dim = config.embed_dim

        kernel_size = self.patch_size
        self.proj = ht.nn.HtParallelConv2d(in_channels = self.in_channels * self.temporal_patch_size, 
                                   out_channels = self.embed_dim, 
                                   kernel_size = kernel_size, 
                                   stride=kernel_size, 
                                   padding=0,
                                   bias=False,
                                   multi_ds_parallel_config = get_multi_ds_parallel_config(ds_parallel_configs, 'proj'),
                                   name="proj")

    def  forward(self, x):
        print(f"x shape is {x.shape}")
        print(f"x global shape is {x.global_shape}")
        print(f"self.in_channels is {self.in_channels}")
        print(f"self.patch_size is {self.patch_size}")
        print(f"self.temporal_patch_size is {self.temporal_patch_size}")
        assert(x.shape[1] == self.in_channels * self.patch_size * self.patch_size * self.temporal_patch_size)
        target_type = self.proj.weight.dtype
        x = ht.reshape(x, [-1, self.in_channels * self.temporal_patch_size, self.patch_size, self.patch_size])
        print(f"x shape is {x.shape}")
        print("type", target_type)
        print("device", x.device)
        ht.data_transfer(target_type, x, x.device)
        x = self.proj(x)
        x = ht.reshape(x, [-1, self.embed_dim])
        return x

class PatchMerger(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, name="patch_merger"):
        super(PatchMerger, self).__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.hidden_size = config.hidden_size
        self.ds_parallel_configs = ds_parallel_configs
        self.ln_q = ht.nn.HtMultiParallelLayerNorm(self.embed_dim, get_multi_ds_parallel_config(ds_parallel_configs, 'ln_q'), sequence_parallel=True, name=f'ln_q_patch_merger')


        self.linear1 = ht.nn.HtMultiColumnParallelLinear(
            self.embed_dim,
            self.embed_dim,
            get_multi_ds_parallel_config(ds_parallel_configs, 'dense_embed_dim_to_embed_dim'),
            bias=self.add_bias,
            gather_output=False,
            name=f'colp_{name}'
            # skip_bias_add=True
        )

        self.linear2 = ht.nn.HtMultiRowParallelLinear(
            self.embed_dim,
            self.hidden_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'dense_embed_dim_to_hidden_size'),
            sequence_parallel=False,
            bias=self.add_bias,
            name=f'rowp_{name}'
            # init_method=output_layer_init_method
        )

    def forward(self, x):
        x = self.ln_q(x)
        x = self.linear1(x)
        x = ht.relu(x)
        x = self.linear2(x)
        return x

class VisionAttention(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, layer_idx, name='attn'):
        super().__init__()

        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        self.use_flash_attn = config.use_flash_attn
        # self.add_bias = True
        self.add_bias = False

        self.masked_value = -1e4

        self.embed_dim = config.embed_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        # Layer-wise attention scaling, reordering, and upcasting
        self.layer_idx = layer_idx

        self.qkv_dense = ht.nn.HtMultiColumnParallelLinear(
            self.embed_dim,
            3 * self.embed_dim,
            get_multi_ds_parallel_config(ds_parallel_configs, 'qkv', layer_idx),
            bias=self.add_bias,
            gather_output=False,
            name=f'colp_{name}'
        )

        self.dense = ht.nn.HtMultiRowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            get_multi_ds_parallel_config(ds_parallel_configs, 'dense', layer_idx),
            sequence_parallel=True,
            bias=self.add_bias,
            name=f'rowp_{name}'
        )

        # self.attn_dropout = ht.nn.Dropout(config.attn_pdrop)
        # self.resid_dropout = ht.nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        # column parallel, [micro_batch_size * seq_len, 3 * embed_dim]
        qkv = self.qkv_dense(hidden_states)
        
        assert self.use_flash_attn, "currently only support flash attn"

        attn_output = ht.parallel_attn(
            qkv,             
            self.head_dim, 
            1, # group_query_ratio = q heads / k(v) heads, 1 means MHA and >1 means GQA
            self.config.multi_seq_lens_symbol, 
            self.config.multi_cp_group_symbol,
            True,
            self.config.cu_seqlens_list[self.layer_idx],
            self.config.cu_seqlens_list[self.layer_idx],
            self.config.max_seqlen_symbol,
            self.config.max_seqlen_symbol
        )[0]
        
        # row parallel, shape = [mbs * seq_len, num_heads * head_dim]
        attn_output = self.dense(attn_output)
        # dropout
        # attn_output = self.resid_dropout(attn_output)
        return attn_output


class VisionParallelMLP(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, layer_idx, name='mlp'):
        super(VisionParallelMLP, self).__init__()
        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        # self.add_bias = True
        self.add_bias = False
        self.swiglu = True 

        self.dense_h_to_4h = ht.nn.HtMultiColumnParallelLinear(
            config.embed_dim,
            config.mlp_dim,
            get_multi_ds_parallel_config(ds_parallel_configs, 'dense_h_to_4h', layer_idx),
            bias=self.add_bias,
            gather_output=False,
            name=f'colp_{name}'
            # skip_bias_add=True
        )

        self.dense_4h_to_h = ht.nn.HtMultiRowParallelLinear(
            config.mlp_dim,
            config.embed_dim,
            get_multi_ds_parallel_config(ds_parallel_configs, 'dense_4h_to_h', layer_idx),
            sequence_parallel=True,
            bias=self.add_bias,
            name=f'rowp_{name}'
            # init_method=output_layer_init_method
        )

        # self.dropout = ht.nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        # [b * seq_len, h] -> [b * seq_len, 4h]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = ht.relu(intermediate_parallel)

        # [b * seq_len, 4h] -> [b * seq_len, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output
    


class VisionBlock(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, layer_idx):
        super().__init__()
        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        self.layer_idx = layer_idx
        self.embed_dim = config.embed_dim

        # sequence parallel: layernorm前做reduce-scatter(这一部分由row prallel的reduce-scatter完成); layernorm后做allgather
        self.rmsnorm_1 = ht.nn.HtMultiParallelLayerNorm(self.embed_dim, get_multi_ds_parallel_config(ds_parallel_configs, 'layernorm1', layer_idx), sequence_parallel=True, name=f'rmsnorm1_block{layer_idx}')
        self.attn = VisionAttention(config, get_multi_ds_parallel_config(ds_parallel_configs, "attn", layer_idx), layer_idx=layer_idx, name=f'attn_block{layer_idx}')
        self.rmsnorm_2 = ht.nn.HtMultiParallelLayerNorm(self.embed_dim, get_multi_ds_parallel_config(ds_parallel_configs, 'layernorm2', layer_idx), sequence_parallel=True, name=f'rmsnorm2_block{layer_idx}')
        self.mlp = VisionParallelMLP(config, get_multi_ds_parallel_config(ds_parallel_configs, "mlp", layer_idx), layer_idx=layer_idx, name=f'mlp_block{layer_idx}')

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        residual = hidden_states
        
        hidden_states = self.rmsnorm_1(hidden_states)
        attn_output = self.attn(
            hidden_states, # [b, seq_len, embed_dim]
            attention_mask=attention_mask # [b, 1, 1, seq_len]
        )
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.rmsnorm_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states
    

class VisionModel(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs):
        super(VisionModel, self).__init__()
        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        self.dtype = ht.float32

        self.embed_dim = config.embed_dim
        self.patch_embed = PatchEmbed(config, get_multi_ds_parallel_config(ds_parallel_configs, 'patch_embed'))
        # self.wpe = ht.nn.HtMultiParallelEmbedding(config.max_position_embeddings, self.embed_dim, get_multi_ds_parallel_config(ds_parallel_configs, 'wpe'), name='wpe')

        # self.drop = ht.nn.Dropout(config.embd_pdrop)
        blocks = []
        for i in range(config.num_hidden_layers):
            blocks.append(VisionBlock(config, get_multi_ds_parallel_config(ds_parallel_configs, f'blocks{i}'), layer_idx=i))
        self.h = ht.nn.ModuleList(blocks)
        self.patch_merger = PatchMerger(config, get_multi_ds_parallel_config(ds_parallel_configs, 'patch_merger'))

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask=None,
        token_type_ids=None,
    ):
        # input_ids: [b * seq_len]        
        # token_type_ids: [b * seq_len]
        if token_type_ids is not None:
            assert token_type_ids.global_shape == input_ids.global_shape \
                and token_type_ids.distributed_states.check_equal(input_ids.distributed_states), \
                'token_type_ids global_shape and distributed_states should be equal to input_ids'

        # embeddding: [b * seq_len, embed_dim]
        inputs_embeds = self.patch_embed(input_ids) # [b * seq_len, embed_dim]
        # position_embeds = self.wpe(position_ids) # [b * seq_len, embed_dim]
        # hidden_states = inputs_embeds + position_embeds # [b * seq_len, embed_dim]
        hidden_states = inputs_embeds
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids) # [b * seq_len, embed_dim]
            hidden_states = hidden_states + token_type_embeds
        # dropout
        # hidden_states = self.drop(hidden_states)
        
        # for sequence parallel
        # todo: this is pretty hacky, find a better way
        sequence_parallel = True
        if sequence_parallel:
            ds_hierarchy_input = hidden_states.ds_hierarchy
            ds_hierarchy_output = []
            for ds_union_input in ds_hierarchy_input:
                ds_list_split0 = []
                for ds_input in ds_union_input.ds_list:
                    ds_split0 = ht.DistributedStates(ds_input.device_num, {0: ds_input.device_num}, [0])
                    assert ds_union_input.hetero_dim == -3 or ds_union_input.hetero_dim == 0, \
                        "Workaround: sequence_parallel assume input only hetero on split0"
                    assert ds_input.device_num == ds_input.get_dim(0) * ds_input.get_dim(-1), \
                        "Workaround: sequence_parallel assume input only split in dimension 0 for dp"
                    ds_list_split0.append(ds_split0)
                ds_hierarchy_output.append(ht.DistributedStatesUnion(ds_list_split0, 0 if ds_union_input.hetero_dim != -3 else -3))
            # [b * seq_len // tp, embed_dim]
            hidden_states = ht.comm(hidden_states, ds_hierarchy_output, name="workaround_sp_scatter")

        # 12 x multihead self-attn
        for i, block in enumerate(self.h):
            hidden_states = block(
                hidden_states, # [b * seq_len, embed_dim]
                attention_mask=attention_mask # [b, 1, 1, seq_len]
            )
            # hetero需要显示地插入通信算子
            if i != len(self.h) - 1:
                next_block = self.h[i + 1]
                if next_block.rmsnorm_1.sequence_parallel:
                    hidden_states = ht.comm(hidden_states, next_block.rmsnorm_1.ds_union_map['split0'], next_block.rmsnorm_1.device_group_unions, name=f"pipeline_layer_{i}_comm")
                else:
                    hidden_states = ht.comm(hidden_states, next_block.attn.qkv_dense.ds_union_map['split0_dup'], next_block.rmsnorm_1.device_group_unions, name=f"pipeline_layer_{i}_comm")
        # layernorm
        hidden_states = self.patch_merger(hidden_states)
        return hidden_states