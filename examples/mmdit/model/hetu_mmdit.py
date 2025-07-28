import hetu as ht
import numpy as np
import torch
import math

from hetu.nn.modules.parallel_multi_ds import parallel_data_provider, parallel_multi_data_provider, get_multi_ds_parallel_config
from model.hetu_vae import Encoder

freqs_global = None

def generate_freqs(dim, max_period, dtype):
    assert dim % 2 == 0
    half = dim // 2
    freqs = np.exp(-math.log(max_period) * np.arange(start = 0, stop = half, dtype = dtype) / half).reshape(1, -1)
    return freqs

def modulate(x, shift, scale, add_one = True, is_residual = False):
    scale = scale.reshape([scale.global_shape[0], -1, scale.global_shape[1]])
    if not is_residual:
        shift = shift.reshape([shift.global_shape[0], -1, shift.global_shape[1]])
    if add_one:
        return x * (1 + scale) + shift
    else:
        return x * scale + shift


class PatchEmbed(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, name="patch_embed"):
        super(PatchEmbed, self).__init__()
        self.config = config
        self.patch_size = config.patch_size
        # self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.block_out_channels[-1]
        self.embed_dim = config.hidden_size

        kernel_size = self.patch_size
        self.proj = ht.nn.HtParallelConv3d(in_channels = self.in_channels, 
                                   out_channels = self.embed_dim, 
                                   kernel_size = kernel_size, 
                                   stride=kernel_size, 
                                   padding=0,
                                   bias=False,
                                   multi_ds_parallel_config = get_multi_ds_parallel_config(ds_parallel_configs, 'proj'),
                                   name="proj")

    def  forward(self, x):
        x = self.proj(x) # [b, c, h, w] -> [b, c, h // p, w // p]
        x = x.reshape([x.global_shape[0], x.global_shape[1], -1]) # [b, c, h // p * w // p]
        x = ht.transpose(x, [0, 2, 1]) # [b, h // p * w // p, c]   
        
        print("xxxx", x.shape, x.ds_hierarchy)

        return x
        


class TimestepEmbedder(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, name='timestep_embedder'):
        super(TimestepEmbedder, self).__init__()
        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        self.embed_dim = config.hidden_size
        self.frequency_embedding_size = config.frequency_embedding_size

        self.add_bias = False

        self.linear1 = ht.nn.HtMultiColumnParallelLinear(
            self.frequency_embedding_size,
            self.embed_dim,
            get_multi_ds_parallel_config(ds_parallel_configs, 'linear1'),
            bias=self.add_bias,
            gather_output=False,
            name=f'colp_{name}'
            # skip_bias_add=True
        )

        self.linear2 = ht.nn.HtMultiRowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            get_multi_ds_parallel_config(ds_parallel_configs, 'linear2'),
            sequence_parallel=False,
            bias=self.add_bias,
            name=f'rowp_{name}'
            # init_method=output_layer_init_method
        )

    def timestep_embedding(self, t, dim, max_period = 10000):
        global freqs_global
        if freqs_global == None:
            freqs = generate_freqs(dim * 2, max_period, np.float32)
            # device_group = self.dense.device_group
            # device_index = get_device_index(device_group)
            # ds = self.dense.ds_map['dup']
            freqs_global = ht.from_numpy_parallel(freqs, self.linear1.ds_union_map['dup_split0'], device_group_hierarchy=self.linear1.device_group_unions, requires_grad=False, name='freqs')
        args = t * freqs_global
        return args

    def forward(self, timestep):
        t_freq = self.timestep_embedding(timestep, self.frequency_embedding_size)
        t_emb = self.linear1(t_freq)
        t_emb = ht.silu(t_emb)
        t_emb = self.linear2(t_emb)
        return t_emb


# class LabelEmbedder(ht.nn.Module):
    
#     def __init__(self, config, ds_parallel_configs, name='label_embedder'):
#         super(LabelEmbedder, self).__init__()
#         self.config = config
#         self.ds_parallel_configs = ds_parallel_configs
#         self.embed_dim = config.hidden_size
#         self.label_embedding = ht.nn.HtMultiParallelEmbedding(config.num_classes, self.embed_dim, get_multi_ds_parallel_config(ds_parallel_configs, 'wpe'), name='wpe')

#     def token_drop(self, labels):
#         raise NotImplementedError("token_drop is not implemented")

#     def forward(self, labels):
#         embeddings = self.label_embedding(labels)
#         return embeddings

class FinalLayer(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs):
        super(FinalLayer, self).__init__()
        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        self.embed_dim = config.hidden_size

        self.final_norm = ht.nn.HtMultiParallelLayerNorm(self.embed_dim, get_multi_ds_parallel_config(ds_parallel_configs, 'layernorm_final'), sequence_parallel=False, name='rmsnorm_final')

        self.final_linear = ht.nn.HtMultiColumnParallelLinear(
            config.hidden_size,
            config.hidden_size * 2,
            get_multi_ds_parallel_config(ds_parallel_configs, 'final_linear'),
            bias=False,
            gather_output=True,
            name='final_linear'
        )

    def forward(self, x, c):
        adaLN_modulation_output = ht.silu(self.final_linear(c))
        shift, scale = ht.split(adaLN_modulation_output, 2, dim=1)
        x = self.final_norm(x)
        x = modulate(x, shift, scale)
        # x = self.final_linear(x)
        return x


class TextEmbedder(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, name='text_embedder'):
        super(TextEmbedder, self).__init__()
        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        self.embed_dim = config.hidden_size
        # self.pooled_projection_dim = config.pooled_projection_dim
        self.pooled_projection_dim = config.hidden_size

        self.add_bias = True

        self.linear1 = ht.nn.HtMultiColumnParallelLinear(
            self.pooled_projection_dim,
            self.embed_dim,
            get_multi_ds_parallel_config(ds_parallel_configs, 'linear1'),
            bias=self.add_bias,
            gather_output=False,
            name=f'colp_{name}'
            # skip_bias_add=True
        )

        self.linear2 = ht.nn.HtMultiRowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            get_multi_ds_parallel_config(ds_parallel_configs, 'linear2'),
            sequence_parallel=False,
            bias=self.add_bias,
            name=f'rowp_{name}'
            # init_method=output_layer_init_method
        )

    def forward(self, hidden_state):
        hidden_state = self.linear1(hidden_state)
        hidden_state = ht.silu(hidden_state)
        hidden_state = self.linear2(hidden_state)
        return hidden_state


# class CombinedTimestepTextProjEmbeddings(ht.nn.Module):
#     def __init__(self, config, ds_parallel_configs):
#         super().__init__()

#         self.time_proj = TimestepEmbedder(config, get_multi_ds_parallel_config(ds_parallel_configs, 'time_proj'))
#         self.text_embedder = TextEmbedder(config, get_multi_ds_parallel_config(ds_parallel_configs, 'text_embedder'))

#     def forward(self, timestep, pooled_projection):
#         timesteps_emb = self.time_proj(timestep)
#         pooled_projections = self.text_embedder(pooled_projection)

#         conditioning = timesteps_emb + pooled_projections

#         return conditioning



# self-attn
class MMDiTAttention(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, layer_idx, is_last_layer=False, name='attn'):
        super().__init__()

        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        self.use_flash_attn = config.use_flash_attn
        # self.add_bias = True
        self.add_bias = False
        self.lasy_layer = is_last_layer

        # max_positions = config.max_position_embeddings
        # self.bias = np.tril(np.ones((max_positions, max_positions), dtype=np.int64).reshape(
        #             1, 1, max_positions, max_positions))
        self.masked_value = -1e4

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

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
            sequence_parallel=False,
            bias=self.add_bias,
            name=f'rowp_{name}'
        )

        self.qkv_dense_context = ht.nn.HtMultiColumnParallelLinear(
            self.embed_dim,
            3 * self.embed_dim,
            get_multi_ds_parallel_config(ds_parallel_configs, 'qkv', layer_idx),
            bias=self.add_bias,
            gather_output=False,
            name=f'colp_{name}_context'
        )
        
        if not self.lasy_layer:
            self.dense_context = ht.nn.HtMultiRowParallelLinear(
                self.embed_dim,
                self.embed_dim,
                get_multi_ds_parallel_config(ds_parallel_configs, 'dense', layer_idx),
                sequence_parallel=False,
                bias=self.add_bias,
                name=f'rowp_{name}_context'
            )

        print("over")


        # self.attn_dropout = ht.nn.Dropout(config.attn_pdrop)
        # self.resid_dropout = ht.nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states
    ):
        # column parallel, [micro_batch_size * seq_len, 3 * embed_dim]

        # print("hidden_states", hidden_states.shape, hidden_states.ds_hierarchy)
        # print("encoder hidden states", encoder_hidden_states.shape, encoder_hidden_states.ds_hierarchy)

        hidden_states_batch_size, hidden_states_seq_len, hidden_states_embed_dim = hidden_states.global_shape
        encoder_hidden_states_batch_size, encoder_hidden_states_seq_len, encoder_hidden_states_embed_dim = encoder_hidden_states.global_shape

        hidden_states = hidden_states.reshape([ht.IntSymbol(-1), ht.IntSymbol(hidden_states_embed_dim)])
        qkv = self.qkv_dense(hidden_states)
        qkv = qkv.reshape([ht.IntSymbol(-1), self.config.vision_seq_len_symbol, ht.IntSymbol(self.num_heads), ht.IntSymbol(3 * self.head_dim)])
        query, key, value = ht.split(qkv, 3, qkv.ndim - 1)


        encoder_hidden_states = encoder_hidden_states.reshape([ht.IntSymbol(-1), ht.IntSymbol(encoder_hidden_states_embed_dim)])
        qkv_context = self.qkv_dense_context(encoder_hidden_states)
        qkv_context = qkv_context.reshape([ht.IntSymbol(-1), self.config.text_seq_len_symbol, ht.IntSymbol(self.num_heads), ht.IntSymbol(3 * self.head_dim)])
        query_context, key_context, value_context = ht.split(qkv_context, 3, qkv_context.ndim - 1)

        print("query shape ", query.shape, query.ds_hierarchy)
        print("query_context shape ", query_context.shape, query_context.ds_hierarchy)
        print("key shape ", key.shape, key.ds_hierarchy)
        query_combined = ht.concat(query, query_context, axis=1)
        key_combined = ht.concat(key, key_context, axis=1)
        value_combined = ht.concat(value, value_context, axis=1)


        attn_output = ht.attn(query_combined, key_combined, value_combined, 0.0, -1, False, False)[0]

        
        attn_output_hidden_satates = ht.slice(attn_output, [0, 0, 0, 0], [attn_output.shape[0], self.config.vision_seq_len_symbol.data, attn_output.shape[-2], attn_output.shape[-1]])
        attn_output_context = ht.slice(attn_output, [0, self.config.vision_seq_len_symbol.data, 0, 0], [attn_output.shape[0], self.config.text_seq_len_symbol.data, attn_output.shape[-2], attn_output.shape[-1]])


        attn_output_hidden_satates = attn_output_hidden_satates.reshape([ht.IntSymbol(-1), ht.IntSymbol(self.num_heads * self.head_dim)])
        attn_output = self.dense(attn_output_hidden_satates)
        attn_output = attn_output.reshape([ht.IntSymbol(hidden_states_batch_size), ht.IntSymbol(hidden_states_seq_len), ht.IntSymbol(self.num_heads * self.head_dim)])


        if not self.lasy_layer:
            attn_output_context = ht.reshape(attn_output_context, [ht.IntSymbol(-1), ht.IntSymbol(self.num_heads * self.head_dim)])
            attn_output_context = self.dense_context(attn_output_context)
            attn_output_context = attn_output_context.reshape([ht.IntSymbol(encoder_hidden_states_batch_size), ht.IntSymbol(encoder_hidden_states_seq_len), ht.IntSymbol(self.num_heads * self.head_dim)])
        else:
            attn_output_context = attn_output_context.reshape([ht.IntSymbol(encoder_hidden_states_batch_size), ht.IntSymbol(encoder_hidden_states_seq_len), ht.IntSymbol(self.num_heads * self.head_dim)])

        # dropout
        # attn_output = self.resid_dropout(attn_output)
        return attn_output, attn_output_context



class ParallelMLP(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, layer_idx, name='mlp'):
        super(ParallelMLP, self).__init__()
        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        # self.add_bias = True
        self.add_bias = False
        
        self.swiglu = True 
        ffn_hidden_size = config.ffn_hidden_size # 2.7h
        if self.swiglu:
            ffn_hidden_size *= 2 # for swiglu: h -> 2 * 2.7h

        self.dense_h_to_4h = ht.nn.HtMultiColumnParallelLinear(
            config.hidden_size,
            ffn_hidden_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'dense_h_to_4h', layer_idx),
            bias=self.add_bias,
            gather_output=False,
            name=f'colp_{name}'
            # skip_bias_add=True
        )

        # self.bias_gelu_fusion = bias_gelu_fusion
        # self.activation_func = ht.nn.NewGeLU()

        self.dense_4h_to_h = ht.nn.HtMultiRowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'dense_4h_to_h', layer_idx),
            sequence_parallel=False,
            bias=self.add_bias,
            name=f'rowp_{name}'
            # init_method=output_layer_init_method
        )

        self.dropout = ht.nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        # [b * seq_len, h] -> [b * seq_len, 4h]
        batch_size, seq_len, hidden_size = hidden_states.global_shape
        hidden_states = hidden_states.reshape([-1, hidden_states.global_shape[-1]])
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        # intermediate_parallel = self.activation_func(intermediate_parallel)
        intermediate_parallel = ht.swiglu(intermediate_parallel)

        # [b * seq_len, 4h] -> [b * seq_len, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        output = output.reshape([batch_size, seq_len, hidden_size])
        # output = self.dropout(output)
        return output

class MMDiTMLP(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, layer_idx, name='mlp'):
        super(MMDiTMLP, self).__init__()
        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        self.parallel_mlp = ParallelMLP(config, ds_parallel_configs, layer_idx, name)

    def forward(self, hidden_states):
        # origin_shape = hidden_states.global_shape # [b * seq_len, hidden_size]
        # assert len(origin_shape) == 2, "sequence_parallel: all is 2 dim matmul"
        hidden_states = self.parallel_mlp(hidden_states)
        return hidden_states


class MMDitBlock(ht.nn.Module):

    def __init__(self, config, ds_parallel_configs, layer_idx, is_last_layer):   
        super(MMDitBlock, self).__init__()
        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.is_last_layer = is_last_layer

        # sequence parallel: layernorm前做reduce-scatter(这一部分由row prallel的reduce-scatter完成); layernorm后做allgather
        self.rmsnorm_1 = ht.nn.HtMultiParallelLayerNorm(self.hidden_size, get_multi_ds_parallel_config(ds_parallel_configs, 'layernorm1', layer_idx), sequence_parallel=False, name=f'rmsnorm1_block{layer_idx}')
        self.attn = MMDiTAttention(config, get_multi_ds_parallel_config(ds_parallel_configs, "attn", layer_idx), layer_idx=layer_idx, is_last_layer=is_last_layer, name=f'attn_block{layer_idx}')
        self.rmsnorm_2 = ht.nn.HtMultiParallelLayerNorm(self.hidden_size, get_multi_ds_parallel_config(ds_parallel_configs, 'layernorm2', layer_idx), sequence_parallel=False, name=f'rmsnorm2_block{layer_idx}')
        self.mlp = MMDiTMLP(config, get_multi_ds_parallel_config(ds_parallel_configs, "mlp", layer_idx), layer_idx=layer_idx, name=f'mlp_block{layer_idx}')


        self.rmsnorm_1_context = ht.nn.HtMultiParallelLayerNorm(self.hidden_size, get_multi_ds_parallel_config(ds_parallel_configs, 'layernorm1', layer_idx), sequence_parallel=False, name=f'rmsnorm1_context_block{layer_idx}')
        if not self.is_last_layer:
            self.rmsnorm_2_context = ht.nn.HtMultiParallelLayerNorm(self.hidden_size, get_multi_ds_parallel_config(ds_parallel_configs, 'layernorm2', layer_idx), sequence_parallel=False, name=f'rmsnorm2_context_block{layer_idx}')
            self.mlp_context = MMDiTMLP(config, get_multi_ds_parallel_config(ds_parallel_configs, "mlp", layer_idx), layer_idx=layer_idx, name=f'mlp_context_block{layer_idx}')

        self.adaLN_modulation_linear = ht.nn.HtMultiColumnParallelLinear(
            self.hidden_size,
            self.hidden_size * 6,
            get_multi_ds_parallel_config(ds_parallel_configs, 'adaLN_modulation_linear'),
            bias=self.add_bias,
            gather_output=True,
            name=f'adaLN_modulation_linear'
            # skip_bias_add=True
        )

        if self.is_last_layer:
            self.adaLN_modulation_linear_context = ht.nn.HtMultiColumnParallelLinear(
                self.hidden_size,
                self.hidden_size * 2,
                get_multi_ds_parallel_config(ds_parallel_configs, 'adaLN_modulation_linear'),
                bias=self.add_bias,
                gather_output=True,
                name=f'adaLN_modulation_linear_context'
                # skip_bias_add=True
            )            
        else:
            self.adaLN_modulation_linear_context = ht.nn.HtMultiColumnParallelLinear(
                self.hidden_size,
                self.hidden_size * 6,
                get_multi_ds_parallel_config(ds_parallel_configs, 'adaLN_modulation_linear'),
                bias=self.add_bias,
                gather_output=True,
                name=f'adaLN_modulation_linear_context'
                # skip_bias_add=True
            )
    

    
    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb
    ):
        
        print("hidden_states block", hidden_states.shape, hidden_states.ds_hierarchy)
        adaLN_modulation_output = ht.silu(self.adaLN_modulation_linear(temb))
        
        print("adaLN_modulation_output", adaLN_modulation_output.shape)
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ht.split(adaLN_modulation_output, num_chunks=6, dim=1)

        adaLN_modulation_output_context = ht.silu(self.adaLN_modulation_linear_context(temb))
        if not self.is_last_layer:
            shift_msa_context, scale_msa_context, gate_msa_context, shift_mlp_context, scale_mlp_context, gate_mlp_context = ht.split(adaLN_modulation_output_context, num_chunks=6, dim=1)
        else:
            shift_msa_context, scale_msa_context,  = ht.split(adaLN_modulation_output_context, num_chunks=2, dim=1)


        residual = hidden_states
        print("before rmsnorm_1 hidden_states shape ", hidden_states.shape, hidden_states.ds_hierarchy)
        hidden_states = self.rmsnorm_1(hidden_states)
        print("hidden_states shape ", hidden_states.shape, hidden_states.ds_hierarchy)

        residual_context = encoder_hidden_states
        encoder_hidden_states = self.rmsnorm_1_context(encoder_hidden_states)

        print("ggg ")
        attn_output, attn_output_context = self.attn(
            modulate(hidden_states, shift_msa, scale_msa), # [b, seq_len, hidden_size]
            modulate(encoder_hidden_states, shift_msa_context, scale_msa_context) # [b, seq_len, hidden_size]
        )

        print("2222")


        print("hidden_states shape ", hidden_states.shape, hidden_states.ds_hierarchy)
        print("gate_msa shape ", gate_msa.shape, gate_msa.ds_hierarchy)
        print("residual shape ", residual.shape, residual.ds_hierarchy)

        hidden_states = modulate(attn_output, residual, gate_msa, False, True)
        
        print("3333")

        if not self.is_last_layer:
            encoder_hidden_states = modulate(attn_output_context, residual_context, gate_msa_context, False, True)
        else:
            encoder_hidden_states = attn_output_context

        # gate_msa = gate_msa.reshape([gate_msa.global_shape[0], -1, gate_msa.global_shape[1]])
        # attn_output = attn_output.reshape([gate_msa.global_shape[0], -1, attn_output.global_shape[1]])
        # hidden_states = gate_msa * attn_output
        # hidden_states = hidden_states.reshape([-1, hidden_states.global_shape[2]])
        # hidden_states = residual + hidden_states

        # gate_msa_context = gate_msa_context.reshape([gate_msa_context.global_shape[0], -1, gate_msa_context.global_shape[1]])
        # attn_output_context = attn_output_context.reshape([gate_msa_context.global_shape[0], -1, attn_output_context.global_shape[1]])
        # encoder_hidden_states = gate_msa_context * attn_output_context
        # encoder_hidden_states = encoder_hidden_states.reshape([-1, encoder_hidden_states.global_shape[2]])
        # encoder_hidden_states = residual_context + encoder_hidden_states
        
        residual = hidden_states
        hidden_states = self.rmsnorm_2(hidden_states)
        mlp_output = self.mlp(
            modulate(hidden_states, shift_mlp, scale_mlp)
        )
        hidden_states = modulate(mlp_output, residual, gate_mlp, False, True)



        if not self.is_last_layer:
            residual_context = encoder_hidden_states
            encoder_hidden_states = self.rmsnorm_2_context(encoder_hidden_states)
            mlp_output_context = self.mlp_context(
                modulate(encoder_hidden_states, shift_mlp_context, scale_mlp_context)
            )
            encoder_hidden_states = modulate(mlp_output_context, residual_context, gate_mlp_context, False, True)
        else:
            encoder_hidden_states = encoder_hidden_states


        return hidden_states, encoder_hidden_states



class MMDiTModel(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs):
        super(MMDiTModel, self).__init__()
        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        self.dtype = ht.float32

        self.wte = ht.nn.HtMultiVocabParallelEmbedding(config.vocab_size, config.hidden_size, get_multi_ds_parallel_config(ds_parallel_configs, 'wte'), name='wte')
        self.pos_embed = PatchEmbed(config, get_multi_ds_parallel_config(ds_parallel_configs, 'patch_embed'))
        # self.time_text_embedder = CombinedTimestepTextProjEmbeddings(config, get_multi_ds_parallel_config(ds_parallel_configs, 'time_text_embedder'))
        self.time_embedder = TimestepEmbedder(config, get_multi_ds_parallel_config(ds_parallel_configs, 'time_embedder'))

        blocks = []
        for i in range(config.num_hidden_layers):
            is_last_layer = i == config.num_hidden_layers - 1
            blocks.append(MMDitBlock(config, get_multi_ds_parallel_config(ds_parallel_configs, f'blocks{i}'), layer_idx=i, is_last_layer=is_last_layer))
        self.h = ht.nn.ModuleList(blocks)

        self.final_layer = FinalLayer(config, get_multi_ds_parallel_config(ds_parallel_configs, 'final_layer'))       


    def unpatchify(self,  x):
        pass
    
    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        time_steps):


        
        hidden_states = self.pos_embed(hidden_states)
        temb = self.time_embedder(time_steps)
        encoder_hidden_states = self.wte(encoder_hidden_states)

        print("hidden_states shape ", hidden_states.shape, hidden_states.ds_hierarchy)
        print("encoder_hidden_states shape ", encoder_hidden_states.shape, encoder_hidden_states.ds_hierarchy)


        sequence_parallel = False
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
            encoder_hidden_states = ht.comm(encoder_hidden_states, ds_hierarchy_output, name="workaround_sp_scatter")
        

        for i, block in enumerate(self.h):
            print("llm block id ", i)
            hidden_states, encoder_hidden_states = block(
                hidden_states, encoder_hidden_states, temb
            )
            # hetero需要显示地插入通信算子
            if i != len(self.h) - 1:
                next_block = self.h[i + 1]
                if next_block.rmsnorm_1.sequence_parallel:
                    hidden_states = ht.comm(hidden_states, next_block.rmsnorm_1.ds_union_map['split0'], next_block.rmsnorm_1.device_group_unions, name=f"pipeline_layer_{i}_comm")
                    encoder_hidden_states = ht.comm(encoder_hidden_states, next_block.rmsnorm_1.ds_union_map['split0'], next_block.rmsnorm_1.device_group_unions, name=f"pipeline_layer_{i}_comm")
                else:
                    hidden_states = ht.comm(hidden_states, next_block.attn.qkv_dense.ds_union_map['split0_dup'], next_block.rmsnorm_1.device_group_unions, name=f"pipeline_layer_{i}_comm")
                    encoder_hidden_states = ht.comm(encoder_hidden_states, next_block.attn.qkv_dense.ds_union_map['split0_dup'], next_block.rmsnorm_1.device_group_unions, name=f"pipeline_layer_{i}_comm")
        print("before final layer hidden_states shape ", hidden_states.shape, hidden_states.ds_hierarchy)
        hidden_states = self.final_layer(hidden_states, temb)
        print("after final layer hidden_states shape ", hidden_states.shape, hidden_states.ds_hierarchy)
        # x = self.unpatchify(x)
        return hidden_states



class MMDitMHeadModel(ht.nn.Module):

    def __init__(self, config, ds_parallel_configs):
        super(MMDitMHeadModel, self).__init__()
        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        
        self.vae = Encoder(config, get_multi_ds_parallel_config(ds_parallel_configs, 'vae'))
        self.transformer = MMDiTModel(config, get_multi_ds_parallel_config(ds_parallel_configs, 'mmdit'))
        
        # share embedding table
        # we manually add comm op here
        # because we don't know if it is a P2P or a BatchedIsendIrecv in hetero settings
        # self.lm_head.weight = ht.comm(self.transformer.wte.embedding_table, self.lm_head.ds_union_map['dup_split0'], self.lm_head.device_group_unions, name="share_weight_comm") 
    
    def forward(
        self,
        image_inputs,
        text_ids,
        time_steps,
        sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod,
        labels,
        position_ids=None,
        # attention_mask=None,
        # token_type_ids=None,
    ):
        image_inputs = self.vae(image_inputs)

        print("sqrt_alphas_cumprod", sqrt_alphas_cumprod.global_shape, sqrt_alphas_cumprod.ds_hierarchy)
        print("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod.global_shape, sqrt_one_minus_alphas_cumprod.ds_hierarchy)
        print("labels", labels.global_shape, labels.ds_hierarchy)
        print("image_inputs", image_inputs.global_shape, image_inputs.ds_hierarchy)

        image_inputs = sqrt_alphas_cumprod * image_inputs + sqrt_one_minus_alphas_cumprod * labels # labels 是 noise

        # [b * seq_len, n_embd]
        hidden_states = self.transformer(
            image_inputs,
            text_ids,
            time_steps
        )

        loss = None
        if labels is not None:
            loss = ht.vocab_parallel_cross_entropy(hidden_states,
                labels, ignored_index = -1, reduction = "mean")    


        return loss


