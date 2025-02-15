import hetu as ht
import numpy as np
from hetu.nn.modules.parallel_utils import get_multi_ds_parallel_config, parallel_multi_data_provider, get_device_index

sin_global = None
cos_global = None

def generate_cos_sin(seqlen, rotary_dim, dtype):
    assert rotary_dim % 2 == 0
    angle = np.random.rand(seqlen * 2, rotary_dim // 2) * 2 * np.pi
    cos = np.cos(angle).astype(dtype)
    sin = np.sin(angle).astype(dtype)
    return cos, sin

# self-attn
class LLamaAttention(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, layer_idx, name='attn'):
        super().__init__()

        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        self.name = name
        self.use_flash_attn = config.use_flash_attn
        self.add_bias = False

        # max_positions = config.max_position_embeddings
        # self.bias = np.tril(np.ones((max_positions, max_positions), dtype=np.int64).reshape(
        #             1, 1, max_positions, max_positions))
        self.masked_value = -1e4

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_groups = config.num_query_groups
        self.head_dim = self.embed_dim // self.num_heads
        self.query_projection_size = self.embed_dim
        self.kv_projection_size = self.num_groups * self.head_dim
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

        self.q_proj = ht.nn.HtMultiColumnParallelLinear(
            self.embed_dim,
            self.query_projection_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'qkv', layer_idx),
            bias=self.add_bias,
            gather_output=False,
            name=f'colp_base_q_{name}'
        )

        self.k_proj = ht.nn.HtMultiColumnParallelLinear(
            self.embed_dim,
            self.kv_projection_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'qkv', layer_idx),
            bias=self.add_bias,
            gather_output=False,
            name=f'colp_base_k_{name}'
        )
        
        self.v_proj = ht.nn.HtMultiColumnParallelLinear(
            self.embed_dim,
            self.kv_projection_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'qkv', layer_idx),
            bias=self.add_bias,
            gather_output=False,
            name=f'colp_base_v_{name}'
        )
        
        self.o_proj = ht.nn.HtMultiRowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            get_multi_ds_parallel_config(ds_parallel_configs, 'dense', layer_idx),
            sequence_parallel=config.sequence_parallel,
            bias=self.add_bias,
            name=f'rowp_base_o_{name}'
        )

    def forward(
        self,
        hidden_states,
        mbs_times_dp_symbol,
        seq_len_symbol,
    ):
        # embed_dim = hidden_states.global_shape[-1]
        # mbs_times_dp_symbol = hidden_states.symbolic_shape[0] * self.config.dp_symbol
        # seq_len_symbol = hidden_states.symbolic_shape[1]
        # # [micro_batch_size*seq_len, embed_dim]
        # hidden_states = hidden_states.reshape([self.config.mbs_times_dp_symbol * self.config.seq_len_symbol, ht.IntSymbol(embed_dim)], name=f'reshape_{self.name}')
        # print(f'hidden_states.global_shape={hidden_states.global_shape}, hidden_states.shape={hidden_states.shape}, hidden_states.distributed_states={hidden_states.distributed_states}')        
        # column parallel, [micro_batch_size*seq_len, 3*embed_dim]
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        query = q.reshape([mbs_times_dp_symbol, seq_len_symbol, ht.IntSymbol(self.num_heads), ht.IntSymbol(self.head_dim)], name=f'reshape_q_{self.name}')
        key = k.reshape([mbs_times_dp_symbol, seq_len_symbol, ht.IntSymbol(self.num_groups), ht.IntSymbol(self.head_dim)], name=f'reshape_k_{self.name}')
        value = v.reshape([mbs_times_dp_symbol, seq_len_symbol, ht.IntSymbol(self.num_groups), ht.IntSymbol(self.head_dim)], name=f'reshape_v_{self.name}')

        # apply relative positional encoding (rotary embedding)
        def apply_rotary_pos_emb(x, _name='q'):
            global sin_global, cos_global
            if sin_global is None or cos_global is None:
                cos_np, sin_np = generate_cos_sin(seq_len_symbol, int(0.5*self.head_dim), np.float32)
                device_groups = self.o_proj.device_groups
                multi_ds = self.o_proj.ds_map['dup']
                sin_global = ht.from_numpy_parallel(parallel_multi_data_provider(sin_np, multi_ds, device_groups),
                                                    multi_ds, device_groups=device_groups, requires_grad=False, name=f'sin{_name}')
                cos_global = ht.from_numpy_parallel(parallel_multi_data_provider(cos_np, multi_ds, device_groups),
                                                    multi_ds, device_groups=device_groups, requires_grad=False, name=f'cos{_name}')
            out = ht.rotary(x, cos_global, sin_global, inplace=True)
            return out

        #query = apply_rotary_pos_emb(query, _name='q')
        #key = apply_rotary_pos_emb(key, _name='k')
        assert self.use_flash_attn, "currently only support flash attn"
        attn_output = ht.attn(query, key, value, 0, -1, True, name=f'attn_{self.name}')[0]
        
        # [mbs_times_dp*seq_len, num_heads*head_dim]
        attn_output = attn_output.reshape([mbs_times_dp_symbol * seq_len_symbol, ht.IntSymbol(self.num_heads * self.head_dim)], name=f'reshape_{self.name}')
        # row parallel, shape=[mbs_times_dp*seq_len, num_heads*head_dim]
        attn_output = self.o_proj(attn_output)
        # [micro_batch_size, seq_len, num_heads*head_dim]
        # attn_output = attn_output.reshape([self.config.mbs_times_dp_symbol, self.config.seq_len_symbol, ht.IntSymbol(self.num_heads * self.head_dim)], name=f'reshape_{self.name}')
        # dropout
        # attn_output = self.resid_dropout(attn_output)

        # [micro_batch_size, seq_len, num_heads*head_dim]
        return attn_output

class ParallelMLP(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, layer_idx, name='mlp'):
        super(ParallelMLP, self).__init__()
        self.ds_parallel_configs = ds_parallel_configs
        self.config = config
        self.name = name
        self.add_bias = False

        self.swiglu = True
        ffn_hidden_size = config.ffn_hidden_size # 2.7*h
        if self.swiglu:
            ffn_hidden_size *= 2 # for swiglu: h -> 2 * 2.7*h

        self.dense_h_to_4h = ht.nn.HtMultiColumnParallelLinear(
            config.hidden_size,
            ffn_hidden_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'dense_h_to_4h', layer_idx),
            bias=self.add_bias,
            gather_output=False,
            name=f'colp_base_h_to_4h_{name}'
            # skip_bias_add=True
        )

        self.dense_4h_to_h = ht.nn.HtMultiRowParallelLinear(
            #ffn_hidden_size,
            config.ffn_hidden_size,
            config.hidden_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'dense_4h_to_h', layer_idx),
            sequence_parallel=config.sequence_parallel,
            bias=self.add_bias,
            name=f'rowp_base_4h_to_h_{name}'
            # init_method=output_layer_init_method
        )

        self.dropout = ht.nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        # [b*seq_len, h] -> [b*seq_len, 2* 2.7* h]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        # fused kernel: x1*sigmoid(x1)*x2
        with ht.recompute(multi_recompute = [
            [False] if ds_parallel_config['recompute_granularity'] is None else
            [
                True if dp_recompute_granularity == 'selective' and self.layer_idx in recompute_layer_idxs else False
                for dp_recompute_granularity, recompute_layer_idxs in zip(ds_parallel_config['recompute_granularity'], ds_parallel_config['recompute_layer_idxs_list'])
            ]
            for ds_parallel_config in self.ds_parallel_configs
        ]):
            intermediate_parallel = ht.swiglu(intermediate_parallel, name=f'swiglu_{self.name}')

        # [b*seq_len, 4h] -> [b*seq_len, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output

class LLamaMLP(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, layer_idx, name='mlp'):
        super(LLamaMLP, self).__init__()
        self.ds_parallel_configs = ds_parallel_configs
        self.config = config
        self.name = name
        self.parallel_mlp = ParallelMLP(config, ds_parallel_configs, layer_idx, name)

    def forward(self, hidden_states):
        origin_shape = hidden_states.global_shape # [b * seq_len, hidden_size]
        assert len(origin_shape) == 2, "sequence parallel: all is 2 dim matmul"
        hidden_states = self.parallel_mlp(hidden_states)
        return hidden_states

class LLamaBlock(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, layer_idx):
        super().__init__()
        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        self.name = f'LLamaBlock{layer_idx}'
        hidden_size = config.hidden_size

        self.rmsnorm_1 = ht.nn.HtMultiParallelRMSNorm(
            hidden_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'layernorm1', layer_idx),
            sequence_parallel=config.sequence_parallel,
            recompute_allgather=True,
            name=f'rmsnorm1_{self.name}'
        )
        
        self.attn = LLamaAttention(
            config,
            ds_parallel_configs,
            layer_idx=layer_idx,
            name=f'LLamaAttn{layer_idx}_{self.name}'
        )
        
        self.rmsnorm_2 = ht.nn.HtMultiParallelRMSNorm(
            hidden_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'layernorm2', layer_idx),
            sequence_parallel=config.sequence_parallel,
            recompute_allgather=True,
            name=f'rmsnorm2_{self.name}'
        )
        
        self.mlp = LLamaMLP(
            config,
            ds_parallel_configs,
            layer_idx=layer_idx,
            name=f'LLamaMLP{layer_idx}_{self.name}'
        )

    def forward(
        self,
        hidden_states,
        mbs_times_dp_symbol,
        seq_len_symbol,
        attention_mask=None,
    ):
        residual = hidden_states
        hidden_states = self.rmsnorm_1(hidden_states)
        attn_output = self.attn(
            hidden_states, # [bsz*seq_len, hidden_size]
            mbs_times_dp_symbol,
            seq_len_symbol,
        )
        # residual connection
        # hidden_states = attn_output + residual
        hidden_states = ht.add(attn_output, residual, name=f'add_{self.name}')

        residual = hidden_states
        hidden_states = self.rmsnorm_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        # hidden_states =  residual + feed_forward_hidden_states
        hidden_states = ht.add(residual, feed_forward_hidden_states, name=f'add_{self.name}')

        return hidden_states


class LLamaModel(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs):
        super(LLamaModel, self).__init__()
        self.config = config

        self.embed_dim = config.hidden_size
        self.wte = ht.nn.HtMultiVocabParallelEmbedding(
            config.vocab_size,
            self.embed_dim,
            get_multi_ds_parallel_config(ds_parallel_configs, 'wte'),
            name='wte'
        )

        blocks = []
        for i in range(config.num_hidden_layers):
            blocks.append(LLamaBlock(config, ds_parallel_configs, layer_idx=i))
        self.h = ht.nn.ModuleList(blocks)
        # the final rmsnorm doesn't need to recompute allgather
        self.rmsnorm_f = ht.nn.HtMultiParallelRMSNorm(
            config.hidden_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'layernorm_final'),
            sequence_parallel=config.sequence_parallel,
            recompute_allgather=False,
            name='rmsnorm_final'
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
    ):
        # input_ids: [b, seq_len]
        # token_type_ids: [b, seq_len]
        if token_type_ids is not None:
            assert token_type_ids.global_shape == input_ids.global_shape \
                and token_type_ids.distributed_states.check_equal(input_ids.distributed_states), \
                'token_type_ids global_shape and distributed_states should be equal to input_ids'

        # attention_mask: [b, 1, 1, seq_len]
        if attention_mask is not None:
            assert attention_mask.global_shape == input_ids.global_shape \
                and attention_mask.distributed_states.check_equal(attention_mask.distributed_states), \
                'attention_mask global_shape and distributed_states should be equal to input_ids!'
            mbs_times_dp_symbol = input_ids.symbolic_shape[0] * self.config.dp_symbol
            attention_mask = attention_mask.reshape([mbs_times_dp_symbol, ht.IntSymbol(1), ht.IntSymbol(1), input_ids.symbolic_shape[1]])
            # 原attention_mask: 1为使用的值, 0为mask的值
            # attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0 # 0为使用的值, -10000为mask的值

        # embeddding: [b, seq_len, embed_dim]
        inputs_embeds = self.wte(input_ids) # [b, seq_len, embed_dim]
        hidden_states = inputs_embeds # [b, seq_len, embed_dim]
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids) # [b, seq_len, embed_dim]
            hidden_states = hidden_states + token_type_embeds

        mbs_times_dp_symbol = hidden_states.symbolic_shape[0] * self.config.dp_symbol
        seq_len_symbol = hidden_states.symbolic_shape[1]
        embed_dim = hidden_states.global_shape[-1]
        hidden_states = hidden_states.reshape([mbs_times_dp_symbol * seq_len_symbol, ht.IntSymbol(embed_dim)])
        
        # for sp
        if self.config.sequence_parallel:
            if hidden_states.check_ds_hierarchy_equal(self.h[0].rmsnorm_1.ds_union_map['split0']):
                hidden_states = hidden_states
            else:
                hidden_states = ht.comm(hidden_states, self.h[0].rmsnorm_1.ds_union_map['split0'])

        # 12 x multihead self-attn
        for block in self.h:
            hidden_states = block(
                hidden_states, # [b*seq_len, embed_dim]
                mbs_times_dp_symbol,
                seq_len_symbol,
                attention_mask=attention_mask, # [b, 1, 1, seq_len]
            )
        # layernorm
        hidden_states = self.rmsnorm_f(hidden_states)
        hidden_states = hidden_states.reshape([mbs_times_dp_symbol, seq_len_symbol, ht.IntSymbol(embed_dim)])
        return hidden_states

# the ds_parallel_config use the same as gpt
class LLamaLMHeadModel(ht.nn.Module):

    def __init__(self, config, ds_parallel_configs):
        super(LLamaLMHeadModel, self).__init__()
        self.transformer = LLamaModel(config, ds_parallel_configs)
        self.lm_head = ht.nn.HtMultiColumnParallelLinear(
            config.n_embd,
            config.vocab_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'lm_head'),
            bias=False,
            gather_output=False,
            name='lm_head'
        )
        self.config = config
    
    def forward(
        self,
        input_ids=None,
        position_ids=None, # align with gpt
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        # cu_seqlens_list=None
    ):
        # [b, seq_len, n_embd]
        hidden_states = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        # if not hidden_states.check_multi_ds_equal(self.lm_head.ds_map['split0_dup']):
        #     hidden_states = ht.comm(hidden_states, self.lm_head.ds_map['split0_dup'])
        # [b, s, h] -> [b, s-1, h]
        shift_hidden_states = ht.slice(hidden_states, [ht.IntSymbol(0), ht.IntSymbol(0), ht.IntSymbol(0)], [hidden_states.symbolic_shape[0], hidden_states.symbolic_shape[1] - 1, hidden_states.symbolic_shape[2]])
        hidden_size = hidden_states.global_shape[-1]
        mbs_times_dp_symbol = hidden_states.symbolic_shape[0] * self.config.dp_symbol
        seq_len_symbol = hidden_states.symbolic_shape[1]
        # [b*(s-1), h]
        shift_hidden_states = shift_hidden_states.reshape([mbs_times_dp_symbol * (seq_len_symbol - 1), ht.IntSymbol(hidden_size)])
        # column parallel, [b*(s-1), h]->[b*(s-1), vocab_size], and splited in vocab dimension
        shift_lm_logits = self.lm_head(shift_hidden_states)

        total_loss = None
        if labels is not None:
            if self.config.train_task_num == 1:
                shift_labels = ht.slice(labels, [ht.IntSymbol(0), ht.IntSymbol(1)], \
                                        [labels.symbolic_shape[0], labels.symbolic_shape[1] - 1], name=f'shift_labels')
                loss = ht.vocab_parallel_cross_entropy(shift_lm_logits, shift_labels,
                                                       ignored_index = -1, reduction = "mean", name=f"vocab_cross_entropy_task0")
                total_loss = loss
            else:
                shift_labels = ht.slice(labels, [ht.IntSymbol(0), ht.IntSymbol(1)], \
                                        [labels.symbolic_shape[0], labels.symbolic_shape[1] - 1], name=f'slice_shift_labels')
                shift_labels_of_tasks = ht.split(
                    shift_labels,
                    self.config.task_batch_idxs,
                    dim=0,
                    name=f'split_task_shift_labels'
                )
                shift_lm_logits_of_tasks = ht.split(
                    shift_lm_logits,
                    self.config.task_batch_idxs,
                    dim=0,
                    name='split_task_shift_lm_logits'
                )
                for i in range(self.config.train_task_num):
                    loss = ht.vocab_parallel_cross_entropy(shift_lm_logits_of_tasks[i],  
                        shift_labels_of_tasks[i], ignored_index = -1, reduction = "mean", name=f"vocab_cross_entropy_task{i}")
                    if total_loss is None:
                        total_loss = loss
                    else:
                        total_loss = ht.add(total_loss, loss, name=f'loss_add_task_{i}')
        return total_loss
