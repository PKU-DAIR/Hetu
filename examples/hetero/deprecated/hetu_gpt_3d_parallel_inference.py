import hetu as ht
import numpy as np
import torch

from hetu.nn.modules.parallel import parallel_data_provider, get_device_index


# self-attn
class GPTAttention(ht.nn.Module):
    def __init__(self, config, device_group, layer_idx=None):
        super().__init__()

        self.config = config
        self.device_group = device_group
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

        self.qkv_dense = ht.nn.ColumnParallelLinear(
            self.embed_dim,
            3 * self.embed_dim,
            self.device_group,
            dp=config.dp,
            bias=self.add_bias,
            gather_output=False,
        )

        self.dense = ht.nn.RowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            self.device_group,
            dp = config.dp,
            bias=self.add_bias,
        )

        # self.attn_dropout = ht.nn.Dropout(config.attn_pdrop)
        # self.resid_dropout = ht.nn.Dropout(config.resid_pdrop)


    def _attn(self, query, key_t, value, attention_mask=None, causal_mask=None):
        # q*k^T, shape=[micro_batch_size, num_heads, query_length, key_length]
        attn_weights = ht.bmm(query, key_t)
        micro_batch_size, num_heads, query_length, key_length = attn_weights.global_shape

        # scale
        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.global_shape[-1]) ** 0.5)
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        # mask
        device_index = get_device_index(self.device_group)
        if causal_mask is None:
            causal_mask = ht.from_numpy_parallel(parallel_data_provider(np.tile(self.bias[:, :, key_length - query_length : key_length, :key_length], 
                                                                                (micro_batch_size, num_heads, 1, 1)),
                                                                        attn_weights.distributed_states, device_index),
                                                attn_weights.distributed_states, device_group=self.device_group)
        # seems have some bugs after using from_numpy_parallel (the program may get stuck when exiting)?
        mask = ht.from_numpy_parallel(parallel_data_provider(np.full(attn_weights.global_shape, self.masked_value, dtype=np.float32),
                                                             attn_weights.distributed_states, device_index), 
                                      attn_weights.distributed_states, device_group=self.device_group)
        attn_weights = ht.where(causal_mask, attn_weights, mask)
        if attention_mask is not None:
            # attn_weights: shape=[micro_batch_size, num_heads, query_length, key_length]
            # attention_mask: shape=[micro_batch_size, 1, 1, key_length], 注意ds的设置
            # 被mask的<pad>位置上值为-1e4, 没有被mask的位置上值为0
            # todo: +-*/允许对应维度一个为n一个为1的情况下, n被切分
            attn_weights = attn_weights + attention_mask
        # softmax
        attn_weights = ht.softmax(attn_weights, 3)
        # dropout
        # attn_weights = self.attn_dropout(attn_weights)
        # weight sum, shape=[micro_batch_size, num_heads, query_length, head_dim]
        attn_output = ht.bmm(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        causal_mask=None,
        use_cache=None,
    ):
        micro_batch_size, seq_len, embed_dim = hidden_states.global_shape
        assert micro_batch_size == self.config.global_batch_size // self.config.num_micro_batches and embed_dim == self.embed_dim, \
            f"GPTAttention input shape should be equal to ({micro_batch_size, seq_len, embed_dim})"
            
        assert seq_len == 1 if layer_past != None else True, \
            f"GPTAttention input shape should be equal to ({micro_batch_size, 1, embed_dim}) rather than ({micro_batch_size, seq_len, embed_dim})"

        # [micro_batch_size*seq_len, embed_dim]
        hidden_states = hidden_states.reshape([micro_batch_size * seq_len, embed_dim], padding_axis=0)
        # column parallel, [micro_batch_size*seq_len, 3*embed_dim]
        qkv = self.qkv_dense(hidden_states)
        # [micro_batch_size, seq_len, num_heads, 3*head_dim]
        qkv = qkv.reshape([micro_batch_size, seq_len, self.num_heads, 3 * self.head_dim], padding_axis=1)
        # q,k,v shape=[micro_batch_size, seq_len, num_heads, head_dim]
        query, key, value = ht.split(qkv, 3, qkv.ndim - 1, padding_axis=1)

        # [micro_batch_size, num_heads, seq_len, head_dim]
        query = query.transpose([0, 2, 1, 3])
        value = value.transpose([0, 2, 1, 3])
        # [micro_batch_size, num_heads, head_dim, seq_len]
        key_t = key.transpose([0, 2, 3, 1]) # k^T

        if layer_past is not None:
            past_key_t, past_value = layer_past
            key_t = ht.dynamic_concat([past_key_t, key_t], axis=3)
            value = ht.dynamic_concat([past_value, value], axis=2)

        if use_cache is True:
            present = (key_t, value)
        else:
            present = None 

        # self-attn, shape=[micro_batch_size, num_heads, seq_len, head_dim]
        attn_output, attn_weights = self._attn(query, key_t, value, attention_mask, causal_mask)

        # [micro_batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose([0, 2, 1, 3])
        # [micro_batch_size*seq_len, num_heads*head_dim]
        attn_output = attn_output.reshape([micro_batch_size * seq_len, self.num_heads * self.head_dim], padding_axis=0)
        # row parallel, shape=[micro_batch_size*seq_len, num_heads*head_dim]
        attn_output = self.dense(attn_output)
        # [micro_batch_size, seq_len, num_heads*head_dim]
        attn_output = attn_output.reshape([micro_batch_size, seq_len, self.num_heads * self.head_dim], padding_axis=1)
        # dropout
        # attn_output = self.resid_dropout(attn_output)
        
        outputs = (attn_output, present)

        # [micro_batch_size, seq_len, num_heads*head_dim]
        return outputs # a, present



class ParallelMLP(ht.nn.Module):
    def __init__(self, config, device_group):
        super(ParallelMLP, self).__init__()
        self.config = config
        self.device_group = device_group
        self.add_bias = True

        self.dense_h_to_4h = ht.nn.ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            device_group,
            dp=config.dp,
            bias=self.add_bias,
            gather_output=False,
            # skip_bias_add=True
        )

        # self.bias_gelu_fusion = bias_gelu_fusion
        self.activation_func = ht.nn.NewGeLU()

        self.dense_4h_to_h = ht.nn.RowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            device_group,
            dp=config.dp,
            bias=self.add_bias,
            # init_method=output_layer_init_method
        )

        # self.dropout = ht.nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        # [b*seq_len, h] -> [b*seq_len, 4h]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)

        # [b*seq_len, 4h] -> [b*seq_len, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        # output = self.dropout(output)
        return output

class GPTMLP(ht.nn.Module):
    def __init__(self, config, device_group):
        super(GPTMLP, self).__init__()
        self.config = config
        self.device_group = device_group
        self.parallel_mlp = ParallelMLP(config, self.device_group)

    def forward(self, hidden_states):
        origin_shape = hidden_states.global_shape # [b, seq_len, hidden_size]
        if len(origin_shape) != 2: # shape adaptor
            hidden_states = hidden_states.reshape([origin_shape[0] * origin_shape[1], origin_shape[2]], padding_axis=0)
        hidden_states = self.parallel_mlp(hidden_states)
        if len(origin_shape) != 2: # shape adaptor
            hidden_states = hidden_states.reshape(origin_shape, padding_axis=1)
        return hidden_states

class GPTBlock(ht.nn.Module):
    def __init__(self, config, device_group, layer_idx=None):
        super().__init__()
        self.config = config
        self.device_group = device_group
        
        hidden_size = config.hidden_size

        self.ln_1 = ht.nn.ParallelLayerNorm(hidden_size, device_group, eps=config.layer_norm_epsilon)
        self.attn = GPTAttention(config, device_group, layer_idx=layer_idx)
        self.ln_2 = ht.nn.ParallelLayerNorm(hidden_size, device_group, eps=config.layer_norm_epsilon)
        self.mlp = GPTMLP(config, device_group)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        causal_mask=None,
        use_cache=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        outputs = self.attn(
            hidden_states, # train: [b, cur_len(padding to max_len), hidden_size], inference: [b, 1, hidden_size]
            layer_past=layer_past, # train: None, inference: [b, num_heads, cur_len - 1(padding to max_len), hidden_size] * 2
            attention_mask=attention_mask, # [b, 1, 1, cur_len(padding to max_len)]
            causal_mask=causal_mask, # train: None, inference: [b, num_heads, 1, cur_len(padding to max_len)]
            use_cache=use_cache, # train: False, inference: True
        )
        attn_output = outputs[0]
        present = outputs[1]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states
        
        return hidden_states, present


class GPTModel(ht.nn.Module):
    def __init__(self, config, device_groups):
        super(GPTModel, self).__init__()
        self.config = config
        self.device_groups = device_groups
        self.dtype = ht.float32

        self.embed_dim = config.hidden_size
        # self.wte = ht.nn.ParallelEmbedding(config.vocab_size, self.embed_dim, device_group)
        self.wte = ht.nn.VocabParallelEmbedding(config.vocab_size, self.embed_dim, device_groups[0], dp=config.dp)
        self.wpe = ht.nn.ParallelEmbedding(config.max_position_embeddings, self.embed_dim, device_groups[0])

        # self.drop = ht.nn.Dropout(config.embd_pdrop)
        self.h = ht.nn.ModuleList([
            GPTBlock(config, device_groups[i // (config.num_hidden_layers // 2)], layer_idx=i)
            for i in range(config.num_hidden_layers)])
        self.ln_f = ht.nn.ParallelLayerNorm(self.embed_dim, device_groups[1], eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids,
        position_ids=None,
        past_key_values=None,
        attention_mask=None,
        causal_mask=None,
        token_type_ids=None,
        use_cache=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        # input_ids: [b, seq_len]
        micro_batch_size, seq_len = input_ids.global_shape
        assert self.config.global_batch_size // self.config.num_micro_batches == micro_batch_size, \
            'input_ids parallel micro_batch_size should be equal to config'
        
        # token_type_ids: [b, seq_len]
        if token_type_ids is not None:
            assert token_type_ids.global_shape == input_ids.global_shape \
                and token_type_ids.distributed_states.check_equal(input_ids.distributed_states), \
                'token_type_ids global_shape and distributed_states should be equal to input_ids'

        if past_key_values is None:
            # after_padding_length = seq_len
            past_key_values = tuple([None] * len(self.h))
            if position_ids is None:
                # position_ids: [1, seq_len]
                position_ids = np.arange(0, seq_len, dtype=np.int64) # pos: [0, 1, 2, ..., seq_len-1]
                position_ids = np.tile(position_ids, [micro_batch_size, 1]) # shape: [b, seq_len]
                # position_ids = ht.from_numpy(position_ids)
                device_index = get_device_index(self.device_groups[0])
                position_ids = ht.from_numpy_parallel(parallel_data_provider(position_ids, input_ids.distributed_states, device_index), 
                                                        input_ids.distributed_states, device_group=self.device_groups[0])
        else:
            assert position_ids != None
            # after_padding_length = past_key_values[0][1].global_shape[-2] # i.e. the sequence length of past_value after padding

        # attention_mask: [b, 1, 1, after_padding_length]
        if attention_mask is not None:
            assert attention_mask.distributed_states.check_equal(input_ids.distributed_states), \
                'attention_mask distributed_states should be equal to input_ids!'
            attention_mask = attention_mask.reshape([micro_batch_size, 1, 1, attention_mask.global_shape[-1]], padding_axis=-1)
            # 原attention_mask: 1为使用的值, 0为mask的值
            # attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0 # 0为使用的值, -10000为mask的值

        # embedding: [b, seq_len, embed_dim]
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
        presents = () if use_cache else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            outputs = block(
                hidden_states, 
                layer_past=layer_past, 
                attention_mask=attention_mask, 
                causal_mask=causal_mask[i // (self.config.num_hidden_layers // 2)],
                use_cache=use_cache, 
            )
            hidden_states = outputs[0]
            if use_cache:
                presents = presents + (outputs[1],)
        # layernorm
        hidden_states = self.ln_f(hidden_states)
        return hidden_states, presents

class GPTLMHeadModel(ht.nn.Module):

    def __init__(self, config, device_groups):
        super(GPTLMHeadModel, self).__init__()
        self.transformer = GPTModel(config, device_groups)
        self.lm_head = ht.nn.ColumnParallelLinear(
            config.n_embd,
            config.vocab_size,
            device_groups[1],
            dp=config.dp,
            bias=False,
            gather_output=True, # last dimension(vocab_size) need to do softmax, so cannot be splited
        )
        self.config = config
        self.device_groups = device_groups
    
    def forward(
        self,
        input_ids=None,
        position_ids=None,
        past_key_values=None,
        attention_mask=None,
        causal_mask=None,
        token_type_ids=None,
        labels=None,
        use_cache=None,
    ):
        # [b, seq_len, n_embd]
        transformer_outputs = self.transformer(
            input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            token_type_ids=token_type_ids,
            use_cache=use_cache,
        )
        hidden_states = transformer_outputs[0]
        new_past_key_values = transformer_outputs[1]
        # [b, seq_len, n_embd]
        micro_batch_size, seq_len, n_embd = hidden_states.global_shape
        # [b*seq_len, n_embd]
        hidden_states = hidden_states.reshape([micro_batch_size * seq_len, n_embd], padding_axis=0)
        # column parallel, [b*seq_len, n_embd]->[b*seq_len, vocab_size]
        lm_logits = self.lm_head(hidden_states)
        # [b, seq_len, vocab_size]
        lm_logits = lm_logits.reshape([micro_batch_size, seq_len, self.config.vocab_size], padding_axis=1)
        loss = None
        if labels is not None:
            # lm_logits: [b, seq_len-1, vocab_size], labels: [b, seq_len-1]
            # todo: slice op input local shape, should change into global shape
            # print(f'before slice, shift_logits.shape: {lm_logits.global_shape}, shift_labels.shape: {labels.global_shape}')
            shift_logits = ht.slice(lm_logits, [0,0,0], [lm_logits.shape[0], lm_logits.shape[1] - 1, lm_logits.shape[2]])
            shift_labels = ht.slice(labels, [0,1], [labels.shape[0], labels.shape[1] - 1])
            # print(f'after slice, shift_logits.shape: {shift_logits.global_shape}, shift_labels.shape: {shift_labels.global_shape}')
            # softmax cross_entropy loss = sum(-log(softmax(vocab[label])))
            # because of ignored_index, so cannot use auto distributed reduce for mean
            # need sum over distributed tensor, and divide the not ignored_index num after by hand
            loss = ht.softmax_cross_entropy_sparse(shift_logits,  
                   shift_labels, ignored_index = -1, reduction = "mean")
        output = (lm_logits,)
        output = ((loss,) + output) if loss is not None else output
        output = ((new_past_key_values,) + output) if new_past_key_values is not None else output
        return output # ((new_past_key_values), (loss), lm_logits)




