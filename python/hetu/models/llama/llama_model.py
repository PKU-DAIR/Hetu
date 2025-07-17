import math
import hetu as ht
import numpy as np

from hetu.models.llama.llama_config import LlamaConfig
from hetu.models.utils.model_utils import PreTrainedModel
from hetu.utils.parallel import get_multi_ds_parallel_config
from hetu.data import IGNORE_INDEX

class RotaryEmbedding(ht.nn.Module):
    def __init__(
        self,
        rotary_dim,
        seq_len_interpolation_factor: float = None,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        use_cpu_initialization=True,
        name='rotary_embedding',
    ):
        assert use_cpu_initialization == True, "Currently only support cpu initialization by numpy"
        super().__init__()
        
        self.name = name
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.inv_freq = 1.0 / (
            rotary_base ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim)
        )
        
        if rope_scaling:
            self.inv_freq = self._apply_scaling(self.inv_freq, factor=rope_scaling_factor)
    
    def _apply_scaling(
        self,
        freqs,
        factor=8,
        low_freq_factor=1,
        high_freq_factor=4,
        original_max_position_embeddings=8192,
    ):
        # This implementation is adapted from:
        # https://github.com/huggingface/transformers/blob/2a5a6ad18aa22e98429bb5ecb880660328030ea0/src/transformers/modeling_rope_utils.py#L303-L343

        factor = factor  # `8` in the original implementation
        low_freq_factor = low_freq_factor  # `1` in the original implementation
        high_freq_factor = high_freq_factor  # `4` in the original implementation
        old_context_len = original_max_position_embeddings  # `8192` in the original implementation

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * math.pi / freqs
        # wavelen < high_freq_wavelen: do nothing
        # wavelen > low_freq_wavelen: divide by factor
        inv_freq_llama = np.where(wavelen > low_freq_wavelen, freqs / factor, freqs)
        # otherwise: interpolate between the two, using a smooth factor
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smoothed_inv_freq = (
            1 - smooth_factor
        ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        inv_freq_llama = np.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

        return inv_freq_llama
    
    def get_cos_sin(self, max_seq_len: int, offset: int = 0):
        seq = np.arange(max_seq_len, dtype=self.inv_freq.dtype) + offset
        
        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        freqs = np.outer(seq, self.inv_freq)
        freqs = np.concatenate((freqs, freqs), axis=-1)
        cos = np.cos(freqs).astype(self.inv_freq.dtype)
        sin = np.sin(freqs).astype(self.inv_freq.dtype)
        return cos, sin

    def _load_from_state_dict(self, state_dict, local_device, prefix, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # ignore inv_freq in state_dict
        state_dict.pop(prefix + 'inv_freq', None)
        super()._load_from_state_dict(state_dict, local_device, prefix, strict,
                                      missing_keys, unexpected_keys, error_msgs)

# self-attn
class LlamaAttention(ht.nn.Module):
    def __init__(self, config: LlamaConfig, ds_parallel_configs, layer_idx, name='attn'):
        super().__init__()

        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        self.layer_idx = layer_idx
        
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.use_flash_attn = config.use_flash_attn
        
        if config.num_attention_heads % config.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads must be divisible by num_key_value_heads (got {config.num_attention_heads} vs. {config.num_key_value_heads})"
            )

        # deprecated
        '''
        self.masked_value = -1e4
        self.scale_attn_weights = self.head_dim**-0.5
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        '''
        
        self.rotary_embedding = RotaryEmbedding(
            self.head_dim,
            seq_len_interpolation_factor=config.rope_scaling,
            rope_scaling=config.rope_scaling,
            use_cpu_initialization=True,
            name=f'rotary_embedding_{name}',
        )

        if config.use_packed_qkv:
            self.qkv_dense = ht.nn.HtMultiColumnParallelLinear(
                config.hidden_size,
                (config.num_attention_heads + 2 * config.num_key_value_heads) * self.head_dim,
                get_multi_ds_parallel_config(ds_parallel_configs, 'qkv', layer_idx),
                bias=config.attention_bias,
                gather_output=False,
                name=f'colp_{name}'
            )
        else:
            self.q_dense = ht.nn.HtMultiColumnParallelLinear(
                config.hidden_size,
                config.num_attention_heads * self.head_dim,
                get_multi_ds_parallel_config(ds_parallel_configs, 'qkv', layer_idx),
                bias=config.attention_bias,
                gather_output=False,
                name=f'colp_{name}_q'
            )

            self.k_dense = ht.nn.HtMultiColumnParallelLinear(
                config.hidden_size,
                config.num_key_value_heads * self.head_dim,
                get_multi_ds_parallel_config(ds_parallel_configs, 'qkv', layer_idx),
                bias=config.attention_bias,
                gather_output=False,
                name=f'colp_{name}_k'
            )

            self.v_dense = ht.nn.HtMultiColumnParallelLinear(
                config.hidden_size,
                config.num_key_value_heads * self.head_dim,
                get_multi_ds_parallel_config(ds_parallel_configs, 'qkv', layer_idx),
                bias=config.attention_bias,
                gather_output=False,
                name=f'colp_{name}_v'
            )

        self.dense = ht.nn.HtMultiRowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'dense', layer_idx),
            sequence_parallel=True,
            bias=config.attention_bias,
            name=f'rowp_{name}'
        )

    def _attn(self, query, key_t, value, attention_mask=None):
        raise NotImplementedError("Not supported for hetero dp")
        '''
        # q*k^T, shape=[micro_batch_size, num_heads, seq_len, seq_len]
        attn_weights = ht.bmm(query, key_t)
        micro_batch_size, num_heads, seq_len, seq_len = attn_weights.global_shape

        # scale
        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.global_shape[-1]) ** 0.5)
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        # mask
        device_index = get_device_index(self.qkv_dense.device_groups[0])
        # todo: move causal_mask outside and turn to a placeholder
        causal_mask = ht.from_numpy_parallel(parallel_multi_data_provider(
                                               np.tile(self.bias[:, :, :seq_len, :seq_len], 
                                                 (micro_batch_size, num_heads, 1, 1)),
                                               attn_weights.multi_distributed_states,
                                               self.qkv_dense.device_groups),
                                             attn_weights.multi_distributed_states, requires_grad=False,
                                             device_groups=self.qkv_dense.device_groups, name='causal_mask')
        
        # todo: move mask outside and turn to a placeholder
        mask = ht.from_numpy_parallel(parallel_multi_data_provider(
                                        np.full(attn_weights.global_shape, self.masked_value, dtype=np.float32),
                                        attn_weights.multi_distributed_states, 
                                        self.qkv_dense.device_groups), 
                                      attn_weights.multi_distributed_states, requires_grad=False,
                                      device_groups=self.qkv_dense.device_groups, name='mask')        
        attn_weights = ht.where(causal_mask, attn_weights, mask)
        if attention_mask is not None:
            # attn_weights: shape=[micro_batch_size, num_heads, seq_len, seq_len]
            # attention_mask: shape=[micro_batch_size, 1, 1, seq_len], 注意ds的设置
            # 被mask的<pad>位置上值为-1e4, 没有被mask的位置上值为0
            # todo: +-*/允许对应维度一个为n一个为1的情况下, n被切分
            # print(f'attn_weights global_shape={attn_weights.global_shape}, attention_mask.global_shape={attention_mask.global_shape}')
            # print(f'attn_weights shape={attn_weights.shape}, attention_mask.shape={attention_mask.shape}')
            attn_weights = attn_weights + attention_mask
        # softmax
        attn_weights = ht.softmax(attn_weights, 3)
        # dropout
        # attn_weights = self.attn_dropout(attn_weights)
        # weight sum, shape=[micro_batch_size, num_heads, seq_len, head_dim]
        attn_output = ht.bmm(attn_weights, value)

        return attn_output, attn_weights
        '''

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        # column parallel, [micro_batch_size*seq_len, 3*embed_dim]
        if self.config.use_packed_qkv:
            qkv = self.qkv_dense(hidden_states)

            # apply relative positional encoding (rotary embedding)
            # TODO: 支持动态seq_len
            # 注意此处传入generate_cos_sin的seq_len值要设置为不小于max_seqlen
            cos_np, sin_np = self.rotary_embedding.get_cos_sin(self.config.max_seqlen_symbol.data)
            ds_hierarchy = [
                ht.DistributedStatesUnion([ht.DistributedStates(ds.device_num, {-1: ds.device_num}, [-1]) for ds in ds_union.ds_list], ds_union.hetero_dim)
                    for ds_union in self.dense.ds_union_map['dup']
            ]
            dg_hierarchy = self.qkv_dense.device_group_unions
            sin_global = ht.from_numpy_parallel(sin_np, ds_hierarchy, device_group_hierarchy=dg_hierarchy, requires_grad=False, name=f'sin_attn')
            cos_global = ht.from_numpy_parallel(cos_np, ds_hierarchy, device_group_hierarchy=dg_hierarchy, requires_grad=False, name=f'cos_attn')
            qkv = ht.rotary(
                qkv, cos_global, sin_global,
                self.head_dim,
                self.num_key_value_groups, # group_query_ratio = q heads / k(v) heads, 1 means MHA and >1 means GQA
                self.config.multi_seq_lens_symbol,
                self.config.multi_cp_group_symbol,
                self.config.packing,
                self.config.cu_seqlens_list[self.layer_idx],
                self.config.cu_seqlens_list[self.layer_idx],
                self.config.max_seqlen_symbol,
                self.config.max_seqlen_symbol,
                inplace=False
            )

            assert self.use_flash_attn, "currently only support flash attn"
            # TODO: support packing api
            attn_output = ht.parallel_attn(
                qkv,             
                self.head_dim, 
                self.num_key_value_groups, # group_query_ratio = q heads / k(v) heads, 1 means MHA and >1 means GQA
                self.config.multi_seq_lens_symbol, 
                self.config.multi_cp_group_symbol,
                self.config.packing,
                self.config.cu_seqlens_list[self.layer_idx],
                self.config.cu_seqlens_list[self.layer_idx],
                self.config.max_seqlen_symbol,
                self.config.max_seqlen_symbol,
                self.attention_dropout
            )[0]
        else:
            q = self.q_dense(hidden_states)
            k = self.k_dense(hidden_states)
            v = self.v_dense(hidden_states)

            #TODO: add rotary.

            assert self.use_flash_attn, "currently only support flash attn"
            # TODO: support packing api
            attn_output = ht.flash_attn(
                q, k, v,             
                self.head_dim, 
                self.num_key_value_groups, # group_query_ratio = q heads / k(v) heads, 1 means MHA and >1 means GQA
                self.config.multi_seq_lens_symbol, 
                self.config.multi_cp_group_symbol,
                self.config.packing,
                None,
                None,
                None,
                None,
            )[0]
        
        # row parallel, shape=[micro_batch_size*seq_len, num_heads*head_dim]
        attn_output = self.dense(attn_output)
        return attn_output

class LlamaMLP(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, layer_idx, name='mlp'):
        super().__init__()
        
        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        
        intermediate_size = config.intermediate_size
        
        if config.gated_linear_unit:
            intermediate_size *= 2 # for swiglu: h -> 2 * 2.7*h

        self.dense_h_to_4h = ht.nn.HtMultiColumnParallelLinear(
            config.hidden_size,
            intermediate_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'dense_h_to_4h', layer_idx),
            bias=config.mlp_bias,
            gather_output=False,
            name=f'colp_{name}'
        )

        self.dense_4h_to_h = ht.nn.HtMultiRowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'dense_4h_to_h', layer_idx),
            sequence_parallel=True,
            bias=config.mlp_bias,
            name=f'rowp_{name}'
        )

    def forward(self, hidden_states):
        origin_shape = hidden_states.global_shape # [b * seq_len, hidden_size]
        assert len(origin_shape) == 2, "sequence parallel: all is 2 dim matmul"
        # [b*seq_len, h] -> [b*seq_len, 4h]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        # intermediate_parallel = self.activation_func(intermediate_parallel)
        with ht.recompute(multi_recompute = [
            [False] if ds_parallel_config['recompute_granularity'] is None else
            [
                True if dp_recompute_granularity == 'selective' and self.layer_idx in recompute_layer_idxs else False
                for dp_recompute_granularity, recompute_layer_idxs in zip(ds_parallel_config['recompute_granularity'], ds_parallel_config['recompute_layer_idxs_list'])
            ]
            for ds_parallel_config in self.ds_parallel_configs
        ]):
            intermediate_parallel = ht.swiglu(intermediate_parallel)

        # [b*seq_len, 4h] -> [b*seq_len, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output

class LlamaBlock(ht.nn.Module):
    def __init__(self, config: LlamaConfig, ds_parallel_configs, layer_idx):
        super().__init__()
        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        self.layer_idx = layer_idx
        hidden_size = config.hidden_size

        # sequence parallel: layernorm前做reduce-scatter(这一部分由row prallel的reduce-scatter完成); layernorm后做allgather
        self.ln_1 = ht.nn.HtMultiParallelRMSNorm(hidden_size, get_multi_ds_parallel_config(ds_parallel_configs, 'rmsnorm1', layer_idx), sequence_parallel=True, eps=config.rms_norm_eps, name=f'rmsnorm1_block{layer_idx}')
        self.attn = LlamaAttention(config, get_multi_ds_parallel_config(ds_parallel_configs, "attn", layer_idx), layer_idx=layer_idx, name=f'attn_block{layer_idx}')
        self.ln_2 = ht.nn.HtMultiParallelRMSNorm(hidden_size, get_multi_ds_parallel_config(ds_parallel_configs, 'rmsnorm2', layer_idx), sequence_parallel=True, eps=config.rms_norm_eps, name=f'rmsnorm2_block{layer_idx}')
        self.mlp = LlamaMLP(config, get_multi_ds_parallel_config(ds_parallel_configs, "mlp", layer_idx), layer_idx=layer_idx, name=f'mlp_block{layer_idx}')

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(
            hidden_states, # [b * seq_len, hidden_size]
            attention_mask=attention_mask, # [b, 1, 1, seq_len]
        )
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states

class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)

class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig, ds_parallel_configs):
        super().__init__(config)
        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        
        self.sequence_parallel = config.sequence_parallel

        self.wte = ht.nn.HtMultiVocabParallelEmbedding(config.vocab_size, config.hidden_size, get_multi_ds_parallel_config(ds_parallel_configs, 'wte'), name='wte')
        self.h = ht.nn.ModuleList(
            [LlamaBlock(config, get_multi_ds_parallel_config(ds_parallel_configs, f'blocks{i}'), layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.ln_f = ht.nn.HtMultiParallelRMSNorm(config.hidden_size, get_multi_ds_parallel_config(ds_parallel_configs, 'rmsnorm_final'), sequence_parallel=True, eps=config.rms_norm_eps, name='rmsnorm_final')

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask=None,
        token_type_ids=None,
    ):
        # all seqs in the same micro batch are packed into one seq
        # input_ids: [b * seq_len]        
        # token_type_ids: [b * seq_len]
        if token_type_ids is not None:
            assert token_type_ids.global_shape == input_ids.global_shape \
                and token_type_ids.distributed_states.check_equal(input_ids.distributed_states), \
                'token_type_ids global_shape and distributed_states should be equal to input_ids'

        # embeddding: [b * seq_len, embed_dim]
        inputs_embeds = self.wte(input_ids)
        hidden_states = inputs_embeds
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids) # [b * seq_len, embed_dim]
            hidden_states = hidden_states + token_type_embeds
        
        # for sequence parallel
        if self.config.sequence_parallel:
            if hidden_states.check_ds_hierarchy_equal(self.h[0].ln_1.ds_union_map['split0']):
                hidden_states = hidden_states
            else:
                hidden_states = ht.comm(hidden_states, self.h[0].ln_1.ds_union_map['split0'])

        # multihead self-attn
        for i, block in enumerate(self.h):
            hidden_states = block(
                hidden_states, # [b * seq_len, embed_dim]
                attention_mask=attention_mask, # [b, 1, 1, seq_len]
            )
            
            # for hetero pipeline
            if i != len(self.h) - 1:
                next_block = self.h[i + 1]
                if next_block.ln_1.sequence_parallel:
                    hidden_states = ht.comm(hidden_states, next_block.ln_1.ds_union_map['split0'], next_block.ln_1.device_group_unions, name=f"pipeline_layer_{i}_comm")
                else:
                    hidden_states = ht.comm(hidden_states, next_block.attn.qkv_dense.ds_union_map['split0_dup'], next_block.ln_1.device_group_unions, name=f"pipeline_layer_{i}_comm")
        # layernorm
        hidden_states = self.ln_f(hidden_states)
        return hidden_states

class LlamaLMHeadModel(LlamaPreTrainedModel):

    def __init__(self, config: LlamaConfig, ds_parallel_configs):
        super().__init__(config)
        self.config = config
        self.ds_parallel_configs = ds_parallel_configs

        self.model = LlamaModel(config, get_multi_ds_parallel_config(ds_parallel_configs, 'llama'))
        self.lm_head = ht.nn.HtMultiColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'lm_head'),
            bias=False,
            gather_output=False,
            name='lm_head'
        )

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None
    ):
        # [b * seq_len, n_embd]
        hidden_states = self.model(
            input_ids,
            position_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # column parallel, [b * seq_len, n_embd] -> [b * seq_len, vocab_size], and splited in vocab dimension
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # TODO: we can't use `sum` reduction for hetero dp
            # loss = ht.vocab_parallel_cross_entropy(lm_logits,
            #     labels, ignored_index = IGNORE_INDEX, reduction = "sum")
            loss_unreduce = ht.vocab_parallel_cross_entropy(lm_logits,
                labels, ignored_index = IGNORE_INDEX, reduction = "none")
            # .reshape([-1])
            loss = ht.sum(loss_unreduce)

        return loss
