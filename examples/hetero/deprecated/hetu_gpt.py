import hetu as ht
import numpy as np
import torch

class Conv1D(ht.nn.Module):
    
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        # self.weight = ht.normal([nx, nf], requires_grad=True)
        self.weight = ht.randn([nx, nf], 0., 0.02, requires_grad = True)
        self.bias = ht.zeros([nf], requires_grad=True)
        # ht.nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.shape[:-1] + [self.nf,]
        x1 = x.reshape([-1, x.shape[-1]])
        x = ht.matmul(x1, self.weight) + self.bias
        x = x.reshape(size_out) 
        return x

class GPTAttention(ht.nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()

        max_positions = config.max_position_embeddings
        # self.register_buffer(
        #     "bias",
        #     ht.from_numpy(np.tril(np.ones((max_positions, max_positions), dtype=np.uint8).reshape(
        #             1, 1, max_positions, max_positions)))
        # )
        # self.bias = ht.from_numpy(np.tril(np.ones((max_positions, max_positions), dtype=np.uint8).reshape(
        #             1, 1, max_positions, max_positions))).to_variable(False)
        self.bias = np.tril(np.ones((max_positions, max_positions), dtype=np.int64).reshape(
                    1, 1, max_positions, max_positions))
        # self.register_buffer("masked_bias", ht.from_numpy(np.array([1e-4])))
        self.masked_bias = ht.from_numpy(np.array([-1e4], dtype = np.float32))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = Dropout(config.attn_pdrop)
        self.resid_dropout = Dropout(config.resid_pdrop)
        self.config = config

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = ht.bmm(query, key.transpose([0,1,3,2]))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.shape[-1]) ** 0.5)
    
        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.shape[-2], key.shape[-2]
            # causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            u1 = np.tile(self.bias[:, :, key_length - query_length : key_length, :key_length], (self.config.batch_size,12,1,1))
            causal_mask = ht.from_numpy(u1)
            # causal_mask = self.bias.slice([0, 0, key_length - query_length, 0],
            #     [self.bias.shape[0], self.bias.shape[1], query_length, key_length])
            # print(causal_mask.shape, " ", attn_weights.shape, " ", self.masked_bias.shape)
            # attn_weights = ht.where(causal_mask, attn_weights, self.masked_bias.broadcast(attn_weights))
            # attn_weights = ht.where(causal_mask, attn_weights, attn_weights)
            mask = self.masked_bias.broadcast(attn_weights)
            attn_weights = ht.where(causal_mask, attn_weights, mask)
            # attn_weights = attn_weights

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask
        attn_weights = ht.softmax(attn_weights, 3)
        # attn_weights = ht.sigmoid(attn_weights)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights#.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = ht.bmm(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.shape[:-1] + [num_heads, attn_head_size]
        tensor = tensor.reshape(new_shape)
        return tensor.transpose([0, 2, 1, 3])  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.transpose([0, 2, 1, 3])
        new_shape = tensor.shape[:-2] + [num_heads * attn_head_size]
        out = tensor.reshape(new_shape)
        return out

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            # query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
            tmp = self.c_attn(hidden_states)
            out = tmp.split(3, 2)
            query = out[0]
            key = out[1]
            value = out[2]
            # query = tmp.split([2], [0], [3])
            # key = tmp.split([2], [1], [3])
            # value = tmp.split([2], [2], [3])
            
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = ht.concat([past_key, key], dim=-2)
            value = ht.concat([past_value, value], dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

      
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

class GPTMLP(ht.nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = None
        if config.activation_function == "relu":
            self.act = ht.relu
        elif config.activation_function == 'tanh':
            self.act = ht.tanh
        self.dropout = Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class GPTBlock(ht.nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPTAttention(config, layer_idx=layer_idx)
        self.ln_2 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPTAttention(config, is_cross_attention=True)
            self.ln_cross_attn = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPTMLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        residual = hidden_states

        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs # hidden_states, present, (attentions, cross_attentions)


class GPTModel(ht.nn.Module):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config):
        super(GPTModel, self).__init__()

        self.embed_dim = config.hidden_size

        self.wte = Embedding(config.vocab_size, self.embed_dim)
        print("EMBED1:",config.vocab_size)
        self.wpe = Embedding(config.max_position_embeddings, self.embed_dim)
        print("EMBED2:",config.max_position_embeddings,)

        self.drop = Dropout(config.embd_pdrop)
        self.h = ht.nn.ModuleList([GPTBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.config = config
        self.dtype = ht.float32

    def get_head_mask(self, head_mask, num_hidden_layers: int, is_attention_chunked: bool = False):
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
            # print(input_ids, " ", input_ids.trainable)
            input_ids = input_ids.reshape([-1, input_shape[-1]])
            # print(input_ids," ", input_ids.trainable)
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape([-1, input_shape[-1]])
        if position_ids is not None:
            position_ids = position_ids.reshape([-1, input_shape[-1]])


        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            # position_ids = ht.from_numpy(np.arange(past_length, input_shape[-1] + past_length, dtype=np.int64)).to_variable(False)
            # print("P1",position_ids)
            # out_shape = position_ids.shape[1:]
            # position_ids = position_ids.reshape(out_shape).reshape([-1, input_shape[-1]])
            # position_ids = position_ids.to_variable(False)
            # print("P2",position_ids)
            position_ids = np.arange(past_length, input_shape[-1] + past_length, dtype=np.int64)
            out_shape = (1,) + position_ids.shape
            position_ids = position_ids.reshape(out_shape).reshape(-1, input_shape[-1])
            # position_ids = ht.from_numpy(position_ids).to_variable(False)
            position_ids = ht.from_numpy(position_ids)

        # GPTAttention mask.
        if attention_mask is not None:
            batch_size = self.config.batch_size
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            print(attention_mask.shape)
            attention_mask = attention_mask.reshape([-1, attention_mask.shape[1]])
            print(attention_mask.shape)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            # attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.reshape([attention_mask.shape[0], 1, 1, attention_mask.shape[1]])

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask#.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = ht.ones(encoder_hidden_shape)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)
        hid = hidden_states


        output_shape = input_shape + [hidden_states.shape[-1]]

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            # if self.model_parallel:
            #     torch.cuda.set_device(hidden_states.device)
            #     # Ensure layer_past is on same device as hidden_states (might not be correct)
            #     if layer_past is not None:
            #         layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
            #     # Ensure that attention_mask is always on the same device as hidden_states
            #     if attention_mask is not None:
            #         attention_mask = attention_mask.to(hidden_states.device)
            #     if isinstance(head_mask, torch.Tensor):
            #         head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # if self.gradient_checkpointing and self.training:

            #     if use_cache:
            #         use_cache = False

            #     def create_custom_forward(module):
            #         def custom_forward(*inputs):
            #             # None for past_key_value
            #             return module(*inputs, use_cache, output_attentions)

            #         return custom_forward

            #     outputs = torch.utils.checkpoint.checkpoint(
            #         create_custom_forward(block),
            #         hidden_states,
            #         None,
            #         attention_mask,
            #         head_mask[i],
            #         encoder_hidden_states,
            #         encoder_attention_mask,
            #     )
            # else:
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]

            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
        
        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.reshape(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        

        return tuple(
            v
            for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
            if v is not None
        )

class GPTLMHeadModel(ht.nn.Module):

    def __init__(self, config):
        super(GPTLMHeadModel, self).__init__()
        self.transformer = GPTModel(config)
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)
        self.config = config
    
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            # shift_logits = lm_logits[..., :-1, :].contiguous()
            # shift_labels = labels[..., 1:].contiguous()
            # shift_logits = lm_logits.slice([0, 0], [lm_logits.shape[0], lm_logits.shape[1]])
            # shift_labels = labels.slice([0], [labels.shape[0]])
            shift_logits = lm_logits
            shift_labels = labels
            shift_logits = ht.slice(shift_logits, [0,0,0],[shift_logits.shape[0], shift_logits.shape[1] - 1, shift_logits.shape[2]])
            shift_labels = ht.slice(shift_labels, [0,1],[shift_labels.shape[0], shift_labels.shape[1]-1])
            print("shift shape: ", shift_logits.shape, " ", shift_labels.shape)
            # Flatten the tokens
            # loss_fct = CrossEntropyLoss(ignore_index=-1)
            # loss = ht.softmax_cross_entropy_sparse(shift_logits.reshape([-1, shift_logits.shape[-1]]), 
            #         shift_labels.reshape([-1]), ignored_index = -1)
            loss = ht.softmax_cross_entropy_sparse(shift_logits,  
                   shift_labels, ignored_index = -1, reduction = "sum")

        output = (lm_logits,) + transformer_outputs[1:]
        output = ((loss,) + output) if loss is not None else output
        return output

class Dropout(object):
    def __init__(self, dropout_prob=None):
        self.dropout_prob = dropout_prob

    def __call__(self, input_tensor):
        if self.dropout_prob is None or self.dropout_prob == 0.0:
            return input_tensor
        # output = ht.dropout(input_tensor, 1.0 - self.dropout_prob)
        output = input_tensor
        return output

class LayerNorm(object):
    def __init__(self, hidden_size, eps=1e-12):
        self.eps = eps
        self.hidden_size = hidden_size
        self.scale = ht.ones([hidden_size,], requires_grad=True)
        self.bias = ht.zeros([hidden_size,], requires_grad=True)

    def __call__(self, x):
        u = x.mean([-1], keepdims=[True])
        s = (x - u)
        s = s * s
        s = s.mean([-1], keepdims=[True])
        x = (x - u) / ht.sqrt(s + self.eps)
        x = self.scale * x + self.bias
        return x

class Embedding(object):
    def __init__(self, num_embeddings, embedding_dim, embedding_name=None, initializer=ht.nn.init.normal_):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = ht.nn.functional.xavier_normal_([num_embeddings, embedding_dim], requires_grad=True)
        # ht.nn.Parameter(ht.empty([num_embeddings, embedding_dim], trainable=True))
        # ht.nn.init.xavier_normal_(self.weight)

    def to(self, dtype):
        self.weight = self.weight.to(datatype=dtype)

    def clone(self, dtype):
        model = Embedding(self.num_embeddings, self.embedding_dim)
        model.weight = self.weight.to(datatype=dtype)
        return model
    
    def __call__(self, input_tensor):
        return ht.embedding_lookup(self.weight, input_tensor)

class Linear(object):
    def __init__(self, in_features, out_features, bias=True, activation=None, kernel_initializer=ht.nn.init.xavier_normal_, bias_initializer=ht.nn.init.zeros_, input_shape=None):
        self.bias_flag = bias
        self.activation = activation
        #self.weights = kernel_initializer(name='dense_weights', shape=(in_features, out_features))
        self.weights = ht.nn.functional.xavier_normal_([out_features, in_features], requires_grad=True)
        # self.weights = ht.nn.Parameter(ht.empty([out_features, in_features], trainable=True))
        # ht.nn.init.xavier_normal_(self.weights)
        if self.bias_flag:
            self.bias = ht.zeros([out_features,], requires_grad=True)
            # self.bias = ht.nn.Parameter(ht.zeros([out_features,], trainable=True))
            # ht.nn.init.zeros_(self.bias)
        self.input_shape = input_shape
        self.in_features = in_features
        self.out_features = out_features
        if self.input_shape is not None and self.input_shape[-1] != in_features:
            print("Specified in_features is not equal to input_shape[-1].")
            assert (False)

    def __call__(self, input_tensor):
        self.input_shape = input_tensor.shape
        if self.input_shape is not None and len(self.input_shape) != 2:
            input_tensor = ht.reshape(
                input_tensor, [-1, self.in_features])
        #outputs = ht.matmul_op(input_tensor, self.weights)
        outputs = ht.matmul(input_tensor, self.weights, trans_b=True)
        if self.bias_flag:
            outputs = outputs + ht.broadcast(self.bias, outputs)
        if self.activation is not None:
            outputs = self.activation(outputs)
        if self.input_shape is not None and len(self.input_shape) != 2:
            outputs = ht.reshape(
                outputs, self.input_shape[:-1]+[self.out_features])
        return outputs