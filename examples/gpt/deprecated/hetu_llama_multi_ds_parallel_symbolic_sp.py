import hetu as ht
import numpy as np
import torch

from hetu.nn.modules.parallel_multi_ds import parallel_data_provider, parallel_multi_data_provider, get_multi_ds_parallel_config

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
        self.use_flash_attn = config.use_flash_attn
        # self.add_bias = True
        self.add_bias = False

        max_positions = config.max_position_embeddings
        self.bias = np.tril(np.ones((max_positions, max_positions), dtype=np.int64).reshape(
                    1, 1, max_positions, max_positions))
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
            sequence_parallel=True,
            bias=self.add_bias,
            name=f'rowp_{name}'
        )

        self.attn_dropout = ht.nn.Dropout(config.attn_pdrop)
        self.resid_dropout = ht.nn.Dropout(config.resid_pdrop)


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
        qkv = self.qkv_dense(hidden_states)
        # print(f'qkv.global_shape={qkv.global_shape}, qkv.shape={qkv.shape}, qkv.distributed_states={qkv.distributed_states}')        
        # [micro_batch_size, seq_len, num_heads, 3*head_dim]
        qkv = qkv.reshape([self.config.mbs_times_dp_symbol, self.config.seq_len_symbol, ht.IntSymbol(self.num_heads), ht.IntSymbol(3 * self.head_dim)])
        # print(f'qkv_{self.layer_idx} multi ds = {qkv.multi_distributed_states}') # ensure multi ds for reshape was right 
        # q,k,v shape=[micro_batch_size, seq_len, num_heads, head_dim]
        query, key, value = ht.split(qkv, 3, qkv.ndim - 1)

        # apply relative positional encoding (rotary embedding)
        # TODO: 支持动态seq_len
        def apply_rotary_pos_emb(x, _name='q'):
            cos_np, sin_np = generate_cos_sin(self.config.seq_len_symbol.data, int(0.5 * self.head_dim), np.float32)
            device_group_hierarchy = self.qkv_dense.device_group_unions
            ds_hierarchy = self.dense.ds_union_map['dup']
            # 去除zero
            ds_hierarchy = [
                ht.DistributedStatesUnion([ht.DistributedStates(ds.device_num, {-1: ds.device_num}, [-1]) for ds in ds_union.ds_list], ds_union.hetero_dim)
                    for ds_union in ds_hierarchy
            ]
            sin_global = ht.from_numpy_parallel(sin_np, ds_hierarchy, device_group_hierarchy=device_group_hierarchy, requires_grad=False, name=f'sin_{_name}')
            cos_global = ht.from_numpy_parallel(cos_np, ds_hierarchy, device_group_hierarchy=device_group_hierarchy, requires_grad=False, name=f'cos_{_name}')
            out = ht.rotary(x, cos_global, sin_global, inplace=True)
            return out
        
        # query = apply_rotary_pos_emb(query, _name='q')
        # key = apply_rotary_pos_emb(key, _name='k')
        
        if self.use_flash_attn:
            attn_output = ht.attn(query, key, value, 0, -1, True)[0]
        else:
            # [micro_batch_size, num_heads, seq_len, head_dim]
            query = query.transpose([0, 2, 1, 3], name="AttentionOp_query")
            value = value.transpose([0, 2, 1, 3], name="AttentionOp_value")
            # [micro_batch_size, num_heads, head_dim, seq_len]
            key_t = key.transpose([0, 2, 3, 1], name="AttentionOp_key") # k^T

            # self-attn, shape=[micro_batch_size, num_heads, seq_len, head_dim]
            attn_output, attn_weights = self._attn(query, key_t, value, attention_mask)

            # [micro_batch_size, seq_len, num_heads, head_dim]
            attn_output = attn_output.transpose([0, 2, 1, 3])
        
        # [micro_batch_size*seq_len, num_heads*head_dim]
        attn_output = attn_output.reshape([self.config.mbs_times_dp_symbol * self.config.seq_len_symbol, ht.IntSymbol(self.num_heads * self.head_dim)])
        # row parallel, shape=[micro_batch_size*seq_len, num_heads*head_dim]
        attn_output = self.dense(attn_output)
        # dropout
        # attn_output = self.resid_dropout(attn_output)
        return attn_output



class ParallelMLP(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, layer_idx, name='mlp'):
        super(ParallelMLP, self).__init__()
        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        # self.add_bias = True
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
            name=f'colp_{name}'
            # skip_bias_add=True
        )

        # self.bias_gelu_fusion = bias_gelu_fusion
        # self.activation_func = ht.nn.NewGeLU()

        self.dense_4h_to_h = ht.nn.HtMultiRowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'dense_4h_to_h', layer_idx),
            sequence_parallel=True,
            bias=self.add_bias,
            name=f'rowp_{name}'
            # init_method=output_layer_init_method
        )

        self.dropout = ht.nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        # [b*seq_len, h] -> [b*seq_len, 4h]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        # intermediate_parallel = self.activation_func(intermediate_parallel)
        intermediate_parallel = ht.swiglu(intermediate_parallel)

        # [b*seq_len, 4h] -> [b*seq_len, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        # output = self.dropout(output)
        return output

class LLamaMLP(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, layer_idx, name='mlp'):
        super(LLamaMLP, self).__init__()
        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        self.parallel_mlp = ParallelMLP(config, ds_parallel_configs, layer_idx, name)

    def forward(self, hidden_states):
        origin_shape = hidden_states.global_shape # [b, seq_len, hidden_size]
        assert len(origin_shape) == 2, "sp: all is 2 dim matmul"
        '''
        if len(origin_shape) != 2: # shape adaptor
            hidden_states = hidden_states.reshape([self.config.mbs_times_dp_symbol * self.config.seq_len_symbol, ht.IntSymbol(origin_shape[-1])])
        '''
        hidden_states = self.parallel_mlp(hidden_states)
        '''
        if len(origin_shape) != 2: # shape adaptor
            # two undetermined dim, we therefore should use symbolic shape here
            hidden_states = hidden_states.reshape([self.config.mbs_times_dp_symbol, self.config.seq_len_symbol, ht.IntSymbol(origin_shape[-1])])
        '''
        return hidden_states

class LLamaBlock(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, layer_idx):
        super().__init__()
        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        self.layer_idx = layer_idx
        hidden_size = config.hidden_size

        # sequence parallel: layernorm前做reduce-scatter(这一部分由row prallel的reduce-scatter完成); layernorm后做allgather
        self.rmsnorm_1 = ht.nn.HtMultiParallelLayerNorm(hidden_size, get_multi_ds_parallel_config(ds_parallel_configs, 'layernorm1', layer_idx), sequence_parallel=True, name=f'rmsnorm1_block{layer_idx}')
        self.attn = LLamaAttention(config, get_multi_ds_parallel_config(ds_parallel_configs, "attn", layer_idx), layer_idx=layer_idx, name=f'attn_block{layer_idx}')
        self.rmsnorm_2 = ht.nn.HtMultiParallelLayerNorm(hidden_size, get_multi_ds_parallel_config(ds_parallel_configs, 'layernorm2', layer_idx), sequence_parallel=True, name=f'rmsnorm2_block{layer_idx}')
        self.mlp = LLamaMLP(config, get_multi_ds_parallel_config(ds_parallel_configs, "mlp", layer_idx), layer_idx=layer_idx, name=f'mlp_block{layer_idx}')

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        residual = hidden_states
        hidden_states = self.rmsnorm_1(hidden_states)
        attn_output = self.attn(
            hidden_states, # [b, seq_len, hidden_size]
            attention_mask=attention_mask, # [b, 1, 1, seq_len]
        )
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.rmsnorm_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        # hidden_states =  feed_forward_hidden_states + residual
        hidden_states =  residual + feed_forward_hidden_states

        return hidden_states


class LLamaModel(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs):
        super(LLamaModel, self).__init__()
        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        self.dtype = ht.float32

        self.embed_dim = config.hidden_size
        self.wte = ht.nn.HtMultiVocabParallelEmbedding(config.vocab_size, self.embed_dim, get_multi_ds_parallel_config(ds_parallel_configs, 'wte'), name='wte')
        # self.wpe = ht.nn.HtMultiParallelEmbedding(config.max_position_embeddings, self.embed_dim, get_multi_ds_parallel_config(ds_parallel_configs, 'wpe'), name='wpe')

        self.drop = ht.nn.Dropout(config.embd_pdrop)
        blocks = []
        for i in range(config.num_hidden_layers):
            blocks.append(LLamaBlock(config, get_multi_ds_parallel_config(ds_parallel_configs, f'blocks{i}'), layer_idx=i))
            # for _, block_config in ds_parallel_config['blocks'].items():
            #     if i >= block_config['range'][0] and i <= block_config['range'][1]:
            #         blocks.append(GPTBlock(config, block_config, layer_idx=i))
            #         break
        self.h = ht.nn.ModuleList(blocks)
        self.rmsnorm_f = ht.nn.HtMultiParallelLayerNorm(self.embed_dim, get_multi_ds_parallel_config(ds_parallel_configs, 'layernorm_final'), sequence_parallel=True, name='rmsnorm_final')

    def forward(
        self,
        input_ids,
        position_ids,
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
            attention_mask = attention_mask.reshape([self.config.mbs_times_dp_symbol, ht.IntSymbol(1), ht.IntSymbol(1), self.config.seq_len_symbol])
            # 原attention_mask: 1为使用的值, 0为mask的值
            # attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0 # 0为使用的值, -10000为mask的值

        # embeddding: [b, seq_len, embed_dim]
        inputs_embeds = self.wte(input_ids) # [b, seq_len, embed_dim]
        # position_embeds = self.wpe(position_ids) # [b, seq_len, embed_dim]
        # todo: fix backward grad tensor reduce bug for add(extension dims)
        # hidden_states = inputs_embeds + position_embeds # [b, seq_len, embed_dim]
        hidden_states = inputs_embeds
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids) # [b, seq_len, embed_dim]
            hidden_states = hidden_states + token_type_embeds
        # dropout
        # hidden_states = self.drop(hidden_states)
        
        # [bsz, seq_len, embed_dim] -> [bsz * seq_len, embed_dim]
        _, _, embed_dim = hidden_states.global_shape
        hidden_states = hidden_states.reshape([self.config.mbs_times_dp_symbol * self.config.seq_len_symbol, ht.IntSymbol(embed_dim)])
        # for sequence parallel
        # todo: this is pretty hacky, find a better way
        sp = True
        if sp:
            ds_hierarchy_input = hidden_states.ds_hierarchy
            ds_hierarchy_output = []
            for ds_union_input in ds_hierarchy_input:
                ds_list_split0 = []
                for ds_input in ds_union_input.ds_list:
                    ds_split0 = ht.DistributedStates(ds_input.device_num, {0: ds_input.device_num}, [0])
                    assert ds_union_input.hetero_dim == -3 or ds_union_input.hetero_dim == 0, \
                        "Workaround: sp assume input only hetero on split0"
                    assert ds_input.device_num == ds_input.get_dim(0) * ds_input.get_dim(-1), \
                        "Workaround: sp assume input only split in dimension 0 for dp"
                    ds_list_split0.append(ds_split0)
                ds_hierarchy_output.append(ht.DistributedStatesUnion(ds_list_split0, 0 if ds_union_input.hetero_dim != -3 else -3))
            # [b * seq_len // tp, embed_dim]
            hidden_states = ht.comm(hidden_states, ds_hierarchy_output, name="workaround_sp_comm")

        # 12 x multihead self-attn
        for i, block in enumerate(self.h):
            hidden_states = block(
                hidden_states, # [b, seq_len, embed_dim]
                attention_mask=attention_mask, # [b, 1, 1, seq_len]
            )
            # hetero需要显示地插入通信算子
            if i != len(self.h) - 1:
                next_block = self.h[i + 1]
                if next_block.rmsnorm_1.sp:
                    hidden_states = ht.comm(hidden_states, next_block.rmsnorm_1.ds_union_map['split0'], next_block.rmsnorm_1.device_group_unions, name=f"pipeline_layer_{i}_comm")
                else:
                    hidden_states = ht.comm(hidden_states, next_block.attn.qkv_dense.ds_union_map['split0_dup'], next_block.rmsnorm_1.device_group_unions, name=f"pipeline_layer_{i}_comm")
        # layernorm
        hidden_states = self.rmsnorm_f(hidden_states)
        return hidden_states

class LLamaLMHeadModel(ht.nn.Module):

    def __init__(self, config, ds_parallel_configs):
        super(LLamaLMHeadModel, self).__init__()
        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        self.transformer = LLamaModel(config, get_multi_ds_parallel_config(ds_parallel_configs, 'gpt'))
        self.lm_head = ht.nn.HtMultiColumnParallelLinear(
            config.n_embd,
            config.vocab_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'lm_head'),
            bias=False,
            gather_output=False,
            name='lm_head'
        )
        # share embedding table
        # we manually add comm op here
        # because we don't know if it is a P2P or a BatchedIsendIrecv in hetero settings
        # self.lm_head.weight = ht.comm(self.transformer.wte.embedding_table, self.lm_head.ds_union_map['dup_split0'], self.lm_head.device_group_unions, name="share_weight_comm") 
        self.config = config
    
    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None
    ):
        # [b, seq_len, n_embd]
        hidden_states = self.transformer(
            input_ids,
            position_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        # need allgather here: [b*s//tp, h] -> [b*s, h]
        if not hidden_states.check_ds_hierarchy_equal(self.lm_head.ds_union_map['split0_dup']):
            hidden_states = ht.comm(hidden_states, self.lm_head.ds_union_map['split0_dup'])
        # [b*s, h] -> [b, s, h]
        _, n_embd = hidden_states.global_shape
        hidden_states = hidden_states.reshape([self.config.mbs_times_dp_symbol, self.config.seq_len_symbol, ht.IntSymbol(n_embd)])
        # [b, seq_len, n_embd] -> [b, seq_len-1, n_embd]
        shift_hidden_states = ht.slice(hidden_states, [ht.IntSymbol(0), ht.IntSymbol(0), ht.IntSymbol(0)], [hidden_states.symbolic_shape[0], hidden_states.symbolic_shape[1] - 1, hidden_states.symbolic_shape[2]])
        # [b*(seq_len-1), n_embd]
        shift_hidden_states = shift_hidden_states.reshape([self.config.mbs_times_dp_symbol * (self.config.seq_len_symbol - 1), ht.IntSymbol(n_embd)])
        # column parallel, [b*(seq_len-1), n_embd]->[b*(seq_len-1), vocab_size], and splited in vocab dimension
        shift_lm_logits = self.lm_head(shift_hidden_states)

        loss = None
        if labels is not None:
            # lm_logits: [b, seq_len-1, vocab_size], labels: [b, seq_len-1]
            # todo: slice op input local shape, should change into global shape
            # print(f'before slice, shift_logits.shape: {lm_logits.global_shape}, {lm_logits.shape}; shift_labels.shape: {labels.global_shape}, {labels.shape}')
            shift_labels = ht.slice(labels, [ht.IntSymbol(0), ht.IntSymbol(1)], [labels.symbolic_shape[0], labels.symbolic_shape[1] - 1])
            # print(f'after slice, shift_logits.shape: {shift_logits.global_shape}, shift_labels.shape: {shift_labels.global_shape}')
            # softmax cross_entropy loss = sum(-log(softmax(vocab[label])))
            # because of ignored_index, so cannot use auto distributed reduce for mean
            # need sum over distributed tensor, and divide the not ignored_index num after by hand
            # print(shift_lm_logits.distributed_states)
            loss = ht.vocab_parallel_cross_entropy(shift_lm_logits,
                shift_labels, ignored_index = -1, reduction = "mean")

        # output = (shift_lm_logits,)
        # output = ((loss,) + output) if loss is not None else output
        return loss # ((loss), (shift_lm_logits))