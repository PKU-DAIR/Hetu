import hetu as ht
import numpy as np
import torch
from queue import Queue

from hetu.nn.modules.parallel_multi_ds import parallel_data_provider, parallel_multi_data_provider, get_device_index

def get_multi_ds_parallel_config(ds_parallel_configs, module_name, _range=-1):
    multi_ds_parallel_config = []
    for ds_parallel_config in ds_parallel_configs:
        config_queue = Queue()
        config_queue.put(ds_parallel_config)
        while (not config_queue.empty()):
            config = config_queue.get()
            if module_name in config:
                multi_ds_parallel_config.append(config[module_name])
                break
            else:
                for value in config.values():
                    if type(value) == dict:
                        if "range" in value and (_range < value["range"][0] or _range > value["range"][-1]):
                            continue
                        config_queue.put(value)
    assert len(multi_ds_parallel_config) == len(ds_parallel_configs), 'ds_parallel_configs parse error!'
    return multi_ds_parallel_config
  
# self-attn
class GPTAttention(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, layer_idx, name='attn'):
        super().__init__()

        self.config = config
        self.use_flash_attn = config.use_flash_attn
        self.add_bias = True

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
            bias=self.add_bias,
            name=f'rowp_{name}'
        )

        self.attn_dropout = ht.nn.Dropout(config.attn_pdrop)
        self.resid_dropout = ht.nn.Dropout(config.resid_pdrop)


    def _attn(self, query, key_t, value, attention_mask=None):
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
        # from_numpy_parallel need multi ds, now provide the data based on multi ds[0]
        # causal_mask = ht.from_numpy_parallel(parallel_data_provider(np.tile(self.bias[:, :, :seq_len, :seq_len], 
        #                                                                     (micro_batch_size, num_heads, 1, 1)),
        #                                                             attn_weights.distributed_states, device_index),
        #                                      attn_weights.multi_distributed_states, requires_grad=False,
        #                                      device_groups=self.qkv_dense.device_groups, name='causal_mask')
        # now support multi ds & multi provided data
        causal_mask = ht.from_numpy_parallel(parallel_multi_data_provider(
                                              np.tile(self.bias[:, :, :seq_len, :seq_len], 
                                                      (micro_batch_size, num_heads, 1, 1)),
                                              attn_weights.multi_distributed_states,
                                              self.qkv_dense.device_groups),
                                             attn_weights.multi_distributed_states, requires_grad=False,
                                             device_groups=self.qkv_dense.device_groups, name='causal_mask')
        
        # mask = ht.from_numpy_parallel(parallel_data_provider(np.full(attn_weights.global_shape, self.masked_value, dtype=np.float32),
        #                                                      attn_weights.distributed_states, device_index), 
        #                               attn_weights.multi_distributed_states, requires_grad=False,
        #                               device_groups=self.qkv_dense.device_groups, name='mask')
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

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        micro_batch_size, seq_len, embed_dim = hidden_states.global_shape
        assert micro_batch_size == self.config.global_batch_size // self.config.num_micro_batches and embed_dim == self.embed_dim, \
            f"GPTAttention input shape should be equal to ({micro_batch_size, seq_len, embed_dim})"
        
        # [micro_batch_size*seq_len, embed_dim]
        hidden_states = hidden_states.reshape([-1, embed_dim])
        # print(f'hidden_states.global_shape={hidden_states.global_shape}, hidden_states.shape={hidden_states.shape}, hidden_states.distributed_states={hidden_states.distributed_states}')        
        # column parallel, [micro_batch_size*seq_len, 3*embed_dim]
        qkv = self.qkv_dense(hidden_states)
        # print(f'qkv.global_shape={qkv.global_shape}, qkv.shape={qkv.shape}, qkv.distributed_states={qkv.distributed_states}')        
        # [micro_batch_size, seq_len, num_heads, 3*head_dim]
        qkv = qkv.reshape([micro_batch_size, -1, self.num_heads, 3 * self.head_dim])
        # print(f'qkv_{self.layer_idx} multi ds = {qkv.multi_distributed_states}') # ensure multi ds for reshape was right 
        # q,k,v shape=[micro_batch_size, seq_len, num_heads, head_dim]
        query, key, value = ht.split(qkv, 3, qkv.ndim - 1)

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
        attn_output = attn_output.reshape([-1, self.num_heads * self.head_dim])
        # row parallel, shape=[micro_batch_size*seq_len, num_heads*head_dim]
        attn_output = self.dense(attn_output)
        # [micro_batch_size, seq_len, num_heads*head_dim]
        attn_output = attn_output.reshape([micro_batch_size, -1, self.num_heads * self.head_dim])
        # dropout
        # attn_output = self.resid_dropout(attn_output)

        # [micro_batch_size, seq_len, num_heads*head_dim]
        return attn_output



class ParallelMLP(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, layer_idx, name='mlp'):
        super(ParallelMLP, self).__init__()
        self.config = config
        self.add_bias = True

        self.dense_h_to_4h = ht.nn.HtMultiColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'dense_h_to_4h', layer_idx),
            bias=self.add_bias,
            gather_output=False,
            name=f'colp_{name}'
            # skip_bias_add=True
        )

        # self.bias_gelu_fusion = bias_gelu_fusion
        self.activation_func = ht.nn.NewGeLU() # should be gelu

        self.dense_4h_to_h = ht.nn.HtMultiRowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'dense_4h_to_h', layer_idx),
            bias=self.add_bias,
            name=f'rowp_{name}'
            # init_method=output_layer_init_method
        )

        self.dropout = ht.nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        # [b*seq_len, h] -> [b*seq_len, 4h]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)

        # [b*seq_len, 4h] -> [b*seq_len, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        # output = self.dropout(output)
        return output

class GPTMLP(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, layer_idx, name='mlp'):
        super(GPTMLP, self).__init__()
        self.config = config
        self.parallel_mlp = ParallelMLP(config, ds_parallel_configs, layer_idx, name)

    def forward(self, hidden_states):
        origin_shape = hidden_states.global_shape # [b, seq_len, hidden_size]
        if len(origin_shape) != 2: # shape adaptor
            hidden_states = hidden_states.reshape([-1, origin_shape[-1]])
        hidden_states = self.parallel_mlp(hidden_states)
        if len(origin_shape) != 2: # shape adaptor
            hidden_states = hidden_states.reshape(origin_shape)
        return hidden_states

class GPTBlock(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs, layer_idx):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size

        self.ln_1 = ht.nn.HtMultiParallelLayerNorm(hidden_size, get_multi_ds_parallel_config(ds_parallel_configs, 'layernorm1', layer_idx), eps=config.layer_norm_epsilon, name=f'ln1_block{layer_idx}')
        self.attn = GPTAttention(config, ds_parallel_configs, layer_idx=layer_idx, name=f'attn_block{layer_idx}')
        self.ln_2 = ht.nn.HtMultiParallelLayerNorm(hidden_size, get_multi_ds_parallel_config(ds_parallel_configs, 'layernorm2', layer_idx), eps=config.layer_norm_epsilon, name=f'ln2_block{layer_idx}')
        self.mlp = GPTMLP(config, ds_parallel_configs, layer_idx=layer_idx, name=f'mlp_block{layer_idx}')

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(
            hidden_states, # [b, seq_len, hidden_size]
            attention_mask=attention_mask, # [b, 1, 1, seq_len]
        )
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        # hidden_states =  feed_forward_hidden_states + residual
        hidden_states =  residual + feed_forward_hidden_states

        return hidden_states


class GPTModel(ht.nn.Module):
    def __init__(self, config, ds_parallel_configs):
        super(GPTModel, self).__init__()
        self.config = config
        self.dtype = ht.float32

        self.embed_dim = config.hidden_size
        self.wte = ht.nn.HtMultiVocabParallelEmbedding(config.vocab_size, self.embed_dim, get_multi_ds_parallel_config(ds_parallel_configs, 'wte'), name='wte')
        self.wpe = ht.nn.HtMultiParallelEmbedding(config.max_position_embeddings, self.embed_dim, get_multi_ds_parallel_config(ds_parallel_configs, 'wpe'), name='wpe')

        self.drop = ht.nn.Dropout(config.embd_pdrop)
        blocks = []
        for i in range(config.num_hidden_layers):
          blocks.append(GPTBlock(config, ds_parallel_configs, layer_idx=i))
            # for _, block_config in ds_parallel_config['blocks'].items():
            #     if i >= block_config['range'][0] and i <= block_config['range'][1]:
            #         blocks.append(GPTBlock(config, block_config, layer_idx=i))
            #         break
        self.h = ht.nn.ModuleList(blocks)
        self.ln_f = ht.nn.HtMultiParallelLayerNorm(self.embed_dim, get_multi_ds_parallel_config(ds_parallel_configs, 'layernorm_final'), eps=config.layer_norm_epsilon, name='ln_final')

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
    ):
        # input_ids: [b, seq_len]
        micro_batch_size, seq_len = input_ids.global_shape
        assert self.config.global_batch_size // self.config.num_micro_batches == micro_batch_size, \
            'input_ids parallel batch_size should be equal to config'
        
        # token_type_ids: [b, seq_len]
        if token_type_ids is not None:
            assert token_type_ids.global_shape == input_ids.global_shape \
                and token_type_ids.distributed_states.check_equal(input_ids.distributed_states), \
                'token_type_ids global_shape and distributed_states should be equal to input_ids'

        # position_ids: [1, seq_len]
        position_ids = np.arange(0, seq_len, dtype=np.int64) # pos: [0, 1, 2, ..., seq_len-1]
        position_ids = np.tile(position_ids, [micro_batch_size, 1]) # shape: [b, seq_len]
        # position_ids = ht.from_numpy(position_ids)
        device_index = get_device_index(self.wpe.device_groups[0])
        # position_ids = ht.from_numpy_parallel(parallel_data_provider(position_ids, input_ids.distributed_states, device_index), 
        #                                       input_ids.multi_distributed_states, device_groups=self.wpe.device_groups, name='position_ids')
        position_ids = ht.from_numpy_parallel(parallel_multi_data_provider(position_ids, 
                                                                           input_ids.multi_distributed_states, 
                                                                           self.wpe.device_groups), 
                                              input_ids.multi_distributed_states, 
                                              device_groups=self.wpe.device_groups, 
                                              name='position_ids')

        # attention_mask: [b, 1, 1, seq_len]
        if attention_mask is not None:
            assert attention_mask.global_shape == input_ids.global_shape \
                and attention_mask.distributed_states.check_equal(attention_mask.distributed_states), \
                'attention_mask global_shape and distributed_states should be equal to input_ids!'
            attention_mask = attention_mask.reshape([micro_batch_size, 1, 1, -1])
            # 原attention_mask: 1为使用的值, 0为mask的值
            # attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0 # 0为使用的值, -10000为mask的值

        # embeddding: [b, seq_len, embed_dim]
        inputs_embeds = self.wte(input_ids) # [b, seq_len, embed_dim]
        position_embeds = self.wpe(position_ids) # [b, seq_len, embed_dim]
        # todo: fix backward grad tensor reduce bug for add(extension dims)
        hidden_states = inputs_embeds + position_embeds # [b, seq_len, embed_dim]
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids) # [b, seq_len, embed_dim]
            hidden_states = hidden_states + token_type_embeds
        # dropout
        # hidden_states = self.drop(hidden_states)

        # 12 x multihead self-attn
        for i, block in enumerate(self.h):
            hidden_states = block(
                hidden_states, # [b, seq_len, embed_dim]
                attention_mask=attention_mask, # [b, 1, 1, seq_len]
            )
        # layernorm
        hidden_states = self.ln_f(hidden_states)
        return hidden_states

class GPTLMHeadModel(ht.nn.Module):

    def __init__(self, config, ds_parallel_configs):
        super(GPTLMHeadModel, self).__init__()
        self.transformer = GPTModel(config, ds_parallel_configs)
        self.lm_head = ht.nn.HtMultiColumnParallelLinear(
            config.n_embd,
            config.vocab_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'lm_head'),
            bias=False,
            gather_output=False,
            name='lm_head'
        )
        self.lm_head.weight = self.transformer.wte.embedding_table # share embedding table
        self.config = config
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None
    ):
        # [b, seq_len, n_embd]
        hidden_states = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # [b, seq_len, n_embd] -> [b, seq_len-1, n_embd]
        shift_hidden_states = ht.slice(hidden_states, [0,0,0], [hidden_states.shape[0], hidden_states.shape[1] - 1, hidden_states.shape[2]])
        _,  _, n_embd = hidden_states.global_shape
        # [b*(seq_len-1), n_embd]
        shift_hidden_states = shift_hidden_states.reshape([-1, n_embd])
        # column parallel, [b*(seq_len-1), n_embd]->[b*(seq_len-1), vocab_size], and splited in vocab dimension
        shift_lm_logits = self.lm_head(shift_hidden_states)

        loss = None
        if labels is not None:
            # lm_logits: [b, seq_len-1, vocab_size], labels: [b, seq_len-1]
            # todo: slice op input local shape, should change into global shape
            # print(f'before slice, shift_logits.shape: {lm_logits.global_shape}, {lm_logits.shape}; shift_labels.shape: {labels.global_shape}, {labels.shape}')
            shift_labels = ht.slice(labels, [0,1], [labels.shape[0], labels.shape[1] - 1])
            # print(f'after slice, shift_logits.shape: {shift_logits.global_shape}, shift_labels.shape: {shift_labels.global_shape}')
            # softmax cross_entropy loss = sum(-log(softmax(vocab[label])))
            # because of ignored_index, so cannot use auto distributed reduce for mean
            # need sum over distributed tensor, and divide the not ignored_index num after by hand
            # print(shift_lm_logits.distributed_states)
            loss = ht.vocab_parallel_cross_entropy(shift_lm_logits,
                shift_labels, ignored_index = -1, reduction = "mean")

        output = (shift_lm_logits,)
        output = ((loss,) + output) if loss is not None else output
        return output # ((loss), (shift_lm_logits))