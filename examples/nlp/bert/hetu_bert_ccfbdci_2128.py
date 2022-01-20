import hetu as ht
import numpy as np

'''
Bert Module Architecture & Input/Output Tensor Size

BertModel Inputs: 
    input_ids: [batch_size, seq_len], word token indices in the vocabulary

BertModel Outputs:
    sequence_output: [batch_size, seq_len, hidden_size] (from BertEncoder)
    pooled_output: [batch_size, hidden_size] (from BertPooler)

BertModel:
    --[batch_size, seq_len]--
    BertEmbeddings:
        Embedding(word/position/token_type)
        LayerNorm
        Dropout
    --[batch_size, seq_len, hidden_size]--

    --[batch_size, seq_len, hidden_size]--
    BertEncoder:
        BertLayer(num_hidden_layers):
            BertAttention:
                BertSelfAttention
                --[batch_size, seq_len, hidden_size]--
                BertSelfOutput:
                    Linear
                    Dropout
                    Add & LayerNorm

            --[batch_size, seq_len, hidden_size]--
            BertIntermediate:
                Linear + Act(gule)
            --[batch_size, seq_len, intermediate_size]--
            BertOutput:
                Linear
                Dropout
                Add & LayerNorm
    --[batch_size, seq_len, hidden_size]--

    --[batch_size, seq_len, hidden_size]--
    BertPooler:
        (Slice, select [cls])
        --[batch_size, hidden_size]--
        Linear + Act(Tanh)
    --[batch_size, hidden_size]--

Bert
'''


'''
BertEmbeddings:
--------------------------------------------------------------------------------------------------'''
class BertEmbeddings(object):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        self.seq_len = config.max_position_embeddings
        self.batch_size = config.batch_size

        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size, "word_embeddings")
        self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size, 'position_embeddings')
        self.token_type_embeddings = Embedding(config.type_vocab_size, config.hidden_size, 'token_type_embeddings')

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = Dropout(config.hidden_dropout_prob, inplace = True)

    def __call__(self, input_ids, token_type_ids):
        '''
        inputs:
            input_ids: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len]

        outputs:
            embeddings: [batch_size, seq_len, hidden_size]
        '''
        seq_length= self.seq_len
        batch_size = self.batch_size
        position_ids = ht.Variable('position_ids', value=np.arange(seq_length).reshape((1,-1)).repeat(batch_size,axis=0), dtype=np.long, trainable=False, ctx=input_ids.ctx)


        '''Embedding Size
        inputs_id:[batch_size, seq_len], embedding_table:[vocab_size, hidden_size] 
        position_ids:[batch_size, seq_len], embedding_table:[seq_len, hidden_size]
        token_type_ids:[batch_size, seq_len], embedding_tabel:[type_vocab_size, hidden_size]
            --> embeddings: [batch_size, seq_len, hidden_size]
        '''
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
'''-----------------------------------------------------------------------------------------------'''

class BertEncoder_prev(object):
    def __init__(self, config, layer_id_start = None, layer_num = None):
        self.output_hidden_states = config.output_hidden_states
        if layer_id_start is not None and layer_num is not None:
            self.layer = [BertLayer(config) for _ in range(layer_id_start, layer_id_start+layer_num-1)]
            self.layer.append(BertAttention(config))
        else:
            self.layer = [BertLayer(config) for _ in range(config.num_hidden_layers)]
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size

    def __call__(self, hidden_states, attention_mask=None):
        '''
        inputs:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, num_heads, seq_len, seq_len]
        outputs:
            hidden_states: [batch_size, seq_len, hidden_size]
            all_hidden_states: optional, num_hidden_layers * [batch_size, seq_len, hidden_size]
        '''

        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)


        return hidden_states  # last-layer hidden state

class BertEncoder_last(object):
    def __init__(self, config, layer_id_start = None, layer_num = None):
        self.output_hidden_states = config.output_hidden_states
        if layer_id_start is not None and layer_num is not None:
            self.layer = [BertIntermediateOutput(config)]
            self.layer += [BertLayer(config) for _ in range(layer_id_start, layer_id_start+layer_num)]
        else:
            self.layer = [BertLayer(config) for _ in range(config.num_hidden_layers)]
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size

    def __call__(self, hidden_states, attention_mask=None):
        '''
        inputs:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, num_heads, seq_len, seq_len]
        outputs:
            hidden_states: [batch_size, seq_len, hidden_size]
            all_hidden_states: optional, num_hidden_layers * [batch_size, seq_len, hidden_size]
        '''

        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)


        return hidden_states  # last-layer hidden state


'''
BertEncoder & BertLayer:
--------------------------------------------------------------------------------------------------'''
class BertEncoder(object):
    def __init__(self, config, layer_id_start = None, layer_num = None):
        self.output_hidden_states = config.output_hidden_states
        if layer_id_start is not None and layer_num is not None:
            self.layer = [BertLayer(config) for _ in range(layer_id_start, layer_id_start+layer_num)]

        else:
            self.layer = [BertLayer(config) for _ in range(config.num_hidden_layers)]
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size

    def __call__(self, hidden_states, attention_mask=None):
        '''
        inputs:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, num_heads, seq_len, seq_len]
        outputs:
            hidden_states: [batch_size, seq_len, hidden_size]
            all_hidden_states: optional, num_hidden_layers * [batch_size, seq_len, hidden_size]
        '''

        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)

        return hidden_states  # last-layer hidden state

class BertIntermediateOutput(object):
    def __init__(self, config):
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
    def __call__(self, attention_output, attention_mask):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertLayer(object):
    def __init__(self, config, sequantial = False):
        self.attention = BertAttention(config, sequantial)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def __call__(self, hidden_states, attention_mask):
        '''
        inputs:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, num_heads, seq_len, seq_len]
        outputs:
            layer_output: [batch_size, seq_len, hidden_size]
        '''
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
'''-----------------------------------------------------------------------------------------------'''


'''
BertAttention & BertSelfAttention & BertSelfOutput
--------------------------------------------------------------------------------------------------'''
class BertAttention(object):
    def __init__(self, config, sequantial = False):
        if sequantial:
            self.self = BertSelfAttention_sequential(config)
        else:
            self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def __call__(self, input_tensor, attention_mask):
        '''
        inputs:
            input_tensor: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, num_heads, seq_len, seq_len]
        outputs:
            attention_output: [batch_size, seq_len, hidden_size]
        '''
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertSelfAttention(object):
    def __init__(self, config):
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size #all_head_size == hidden_size
        self.hidden_size = config.hidden_size
        self.seq_len = config.max_position_embeddings
        self.batch_size = config.batch_size

        linear_input_shape = [self.batch_size, self.seq_len, self.hidden_size]
        self.query = Linear(config.hidden_size, self.all_head_size, input_shape=linear_input_shape)
        self.key = Linear(config.hidden_size, self.all_head_size, input_shape=linear_input_shape)
        self.value = Linear(config.hidden_size, self.all_head_size, input_shape=linear_input_shape)

        # self.dropout = Dropout(config.attention_probs_dropout_prob, inplace = False)
        self.softmaxdropout = lambda x : ht.softmaxdropout_op(x, config.attention_probs_dropout_prob)

    def transpose_for_scores(self, input_tensor):
        output_tensor = ht.array_reshape_op(
            input_tensor, [self.batch_size, self.seq_len, self.num_attention_heads, self.attention_head_size])
        output_tensor = ht.transpose_op(output_tensor, [0, 2, 1, 3])
        return output_tensor

    def transpose_key_for_scores(self, input_tensor):
        output_tensor = ht.array_reshape_op(
            input_tensor, [self.batch_size, self.seq_len, self.num_attention_heads, self.attention_head_size])
        output_tensor = ht.transpose_op(output_tensor, [0, 2, 3, 1])
        return output_tensor

    def __call__(self, hidden_states, attention_mask):
        '''
        inputs:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, 1, 1, seq_len]
        outputs:
            context_layer: [batch_size, seq_len, hidden_size]
        '''

        # linear transformation
        mixed_query_layer = self.query(hidden_states) # [batch_size, seq_len, hidden_size]
        mixed_key_layer = self.key(hidden_states) # [batch_size, seq_len, hidden_size]
        mixed_value_layer = self.value(hidden_states) # [batch_size, seq_len, hidden_size]

        # transpose
        query_layer = self.transpose_for_scores(mixed_query_layer) # [batch_size, num_heads, seq_len, head_size]
        key_layer = self.transpose_key_for_scores(mixed_key_layer) # [batch_size, num_heads, head_size, seq_len]
        value_layer = self.transpose_for_scores(mixed_value_layer) # [batch_size, num_heads, seq_len, head_size]

        # score
        key_layer_scaled = key_layer * (1.0 / np.sqrt(float(self.attention_head_size)))
        attention_scores = ht.batch_matmul_op(query_layer, key_layer_scaled) # [batch_size, num_heads, seq_len, seq_len]

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + ht.broadcastto_op(attention_mask, attention_scores)  # [batch_size, num_heads, seq_len, seq_len]

        # # Normalize the attention scores to probabilities.
        # attention_probs = ht.softmax_op(attention_scores, recompute = False)

        # # This is actually dropping out entire tokens to attend to, which might
        # # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        attention_probs = self.softmaxdropout(attention_scores)

        context_layer = ht.batch_matmul_op(attention_probs, value_layer) # [batch_size, num_heads, seq_len, head_size]
        context_layer = ht.transpose_op(context_layer, [0, 2, 1, 3]) # [batch_size, seq_len, num_heads, head_size]
        context_layer = ht.array_reshape_op(context_layer, [-1, self.seq_len, self.all_head_size]) # [batch_size, seq_len, hidden_size]
        return context_layer

class BertSelfAttention_sequential(object):
    def __init__(self, config):
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size #all_head_size == hidden_size
        self.hidden_size = config.hidden_size
        self.seq_len = config.max_position_embeddings
        self.batch_size = config.batch_size

        linear_input_shape = [self.batch_size, self.seq_len, self.hidden_size]
        self.querys = [Linear(config.hidden_size, self.attention_head_size, input_shape=linear_input_shape, bias = False) for i in range(config.num_attention_heads)]
        self.keys = [Linear(config.hidden_size, self.attention_head_size, input_shape=linear_input_shape, bias = False) for i in range(config.num_attention_heads)]
        self.values = [Linear(config.hidden_size, self.attention_head_size, input_shape=linear_input_shape, bias = False) for i in range(config.num_attention_heads)]

        self.dropout = Dropout(config.attention_probs_dropout_prob)
        # self.softmaxdropout = lambda x : ht.softmaxdropout_op(x, config.attention_probs_dropout_prob)

    def __call__(self, hidden_states, attention_mask):
        '''
        inputs:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, 1, 1, seq_len]
        outputs:
            context_layer: [batch_size, seq_len, hidden_size]
        '''

        attention_mask = ht.array_reshape_op(attention_mask, [self.batch_size, 1, self.seq_len])

        context_layers = []

        for i in range(self.num_attention_heads):
            # linear transformation
            mixed_query_layer = self.querys[i](hidden_states) # [batch_size, seq_len, head_size]
            mixed_key_layer = self.keys[i](hidden_states) # [batch_size, seq_len, head_size]
            mixed_value_layer = self.values[i](hidden_states) # [batch_size, seq_len, head_size]

            key_layer = ht.transpose_op(mixed_key_layer, [0, 2, 1]) # [batch_size, head_size, seq_len]
            query_layer = mixed_query_layer
            value_layer = mixed_value_layer

            # score
            key_layer_scaled = key_layer * (1.0 / np.sqrt(float(self.attention_head_size)))
            attention_scores = ht.batch_matmul_op(query_layer, key_layer_scaled) # [batch_size, seq_len, seq_len]

            attention_scores = attention_scores + ht.broadcastto_op(attention_mask, attention_scores)  # [batch_size, seq_len, seq_len]

            attention_probs = ht.softmax_op(attention_scores, recompute = False)

            attention_probs = self.dropout(attention_probs)

            context_layer = ht.batch_matmul_op(attention_probs, value_layer) # [batch_size, seq_len, head_size]
            context_layer = ht.array_reshape_op(context_layer, [self.batch_size, self.seq_len, 1, self.attention_head_size]) # [batch_size, seq_len, 1, head_size]
            context_layers.append(context_layer)

        context_layer = ht.concatenate_op(context_layers, axis = 2) # [batch_size, seq_len, num_heads, head_size]
        context_layer = ht.array_reshape_op(context_layer, [-1, self.seq_len, self.all_head_size]) # [batch_size, seq_len, hidden_size]
        return context_layer

class BertSelfOutput(object):
    def __init__(self, config):
        linear_input_shape = [config.batch_size, config.max_position_embeddings, config.hidden_size]
        self.dense = Linear(config.hidden_size, config.hidden_size, input_shape=linear_input_shape)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = Dropout(config.hidden_dropout_prob, inplace = True)
        # self.dropoutresidual = lambda x, y: ht.dropoutresidual_op(x, y, config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor):
        '''
        inputs:
            hidden_states: [batch_size, seq_len, hidden_size]
            input_tensor: [batch_size, seq_len, hidden_size]
        outputs:
            hidden_states: [batch_size, seq_len, hidden_size]
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # hidden_states = self.dropoutresidual(hidden_states, input_tensor)
        # hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
'''-----------------------------------------------------------------------------------------------'''


'''
BertIntermediate & BertOutput （2-layer FeedForward)
--------------------------------------------------------------------------------------------------'''
class BertIntermediate(object):
    def __init__(self, config):
        if config.hidden_act == "relu":
            self.intermediate_act_fn = ht.relu_op
        elif config.hidden_act == "gelu":
            self.intermediate_act_fn = ht.gelu_op
            print("Gelu activation is not implemented yet.")
            assert(False)
        linear_input_shape = [config.batch_size, config.max_position_embeddings, config.hidden_size]
        # mini_seq = 64
        # linear_input_shape = [config.batch_size, mini_seq, config.hidden_size]

        #self.dense = Linear(config.hidden_size, config.intermediate_size, activation = self.intermediate_act_fn, input_shape=linear_input_shape)
        # self.dense = Linear(config.hidden_size, config.intermediate_size, input_shape=linear_input_shape)

        self.dense = LinearAct(config.hidden_size, config.intermediate_size, activation = config.hidden_act, input_shape=linear_input_shape)


    def __call__(self, hidden_states):
        '''
        inputs:
            hidden_states: [batch_size, seq_len, hidden_size]
        outputs:
            hidden_states: [batch_size, seq_len, intermediate_size]
        '''
        hidden_states = self.dense(hidden_states)
        return hidden_states

class BertOutput(object):
    def __init__(self, config):
        linear_input_shape = [config.batch_size, config.max_position_embeddings, config.intermediate_size]
        # mini_seq = 64
        # linear_input_shape = [config.batch_size, mini_seq, config.intermediate_size]
        self.dense = Linear(config.intermediate_size, config.hidden_size, input_shape=linear_input_shape)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = Dropout(config.hidden_dropout_prob, inplace = True)
        # self.dropoutresidual = lambda x, y: ht.dropoutresidual_op(x, y, config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor):
        '''
        inputs:
            hidden_states: [batch_size, seq_len, intermediate_size]
        outputs:
            hidden_states: [batch_size, seq_len, hidden_size]
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # hidden_states = self.dropoutresidual(hidden_states, input_tensor)
        # hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
'''-----------------------------------------------------------------------------------------------'''


'''
BertPooler
--------------------------------------------------------------------------------------------------'''
class BertPooler(object):
    def __init__(self, config):
        self.dense = Linear(config.hidden_size, config.hidden_size, activation = ht.tanh_op)
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
    def __call__(self, hidden_states):
        '''
        inputs:
            hidden_states: [batch_size, seq_len, hidden_size]
        outputs:
            pooled_output: [batch_size, hidden_size]
        '''
        first_token_tensor = ht.slice_op(hidden_states,(0,0,0),(self.batch_size,1,self.hidden_size))
        first_token_tensor = ht.array_reshape_op(first_token_tensor, [self.batch_size, self.hidden_size])
        pooled_output = self.dense(first_token_tensor)
        return pooled_output
'''-----------------------------------------------------------------------------------------------'''

'''
Bert Downstream Heads
--------------------------------------------------------------------------------------------------'''
class BertPredictionHeadTransform(object):
    def __init__(self, config):
        if config.hidden_act == "relu":
            self.hidden_act = ht.relu_op
        elif config.hidden_act == "gelu":
            self.hidden_act = ht.gelu_op
            print("Gelu activation is not implemented yet.")
            assert(False)
        linear_input_shape = [config.batch_size, config.max_position_embeddings, config.hidden_size]
        # self.dense_act = Linear(config.hidden_size, config.hidden_size, activation=self.hidden_act, input_shape=linear_input_shape)
        self.dense_act = LinearAct(config.hidden_size, config.hidden_size, activation=config.hidden_act, input_shape=linear_input_shape)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def __call__(self, hidden_states):
        '''
        inputs:
            hidden_states: [batch_size, seq_len, hidden_size]
        outputs:
            hidden_states: [batch_size, seq_len, hidden_size]
        '''
        hidden_states = self.dense_act(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(object):
    def __init__(self, config, bert_model_embedding_weights):
        '''
        bert_model_embedding_weights: [vocab_size, hidden_size]
        '''
        self.transform = BertPredictionHeadTransform(config)

        linear_input_shape = [config.batch_size, config.max_position_embeddings, config.hidden_size]
        self.decoder = Linear(config.hidden_size, config.vocab_size, bias_initializer=ht.init.zeros,input_shape=linear_input_shape)
        #self.decoder.weights = ht.transpose_op(bert_model_embedding_weights)
        self.decoder.weights = bert_model_embedding_weights

    def __call__(self, hidden_states):
        '''
        inputs:
            hidden_states: [batch_size, seq_len, hidden_size]
        outputs:
            hidden_states: [batch_size, seq_len, vocab_size]
        '''
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(object):
    def __init__(self, config, bert_model_embedding_weights):
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def __call__(self, sequence_output):
        '''
        inputs:
            sequence_output: [batch_size, seq_len, hidden_size]
        outputs:
            prediction_scores: [batch_size, seq_len, vocab_size]
        '''
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(object):
    def __init__(self, config):
        self.seq_relationship = Linear(config.hidden_size, 2)

    def __call__(self, pooled_output):
        '''
        inputs:
            pooled_output: [batch_size, hidden_size]
        outputs:
            seq_relationship_score: [batch_size, 2]
        '''
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(object):
    def __init__(self, config, bert_model_embedding_weights):
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = Linear(config.hidden_size, 2)

    def __call__(self, sequence_output, pooled_output):
        '''
        inputs:
            sequence_output: [batch_size, seq_len, hidden_size]
            pooled_output: [batch_size, hidden_size]
        outputs:
            prediction_scores: [batch_size, seq_len, vocab_size]
            seq_relationship_score: [batch_size, 2]
        '''
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

'''-----------------------------------------------------------------------------------------------'''


'''
BertModel:
--------------------------------------------------------------------------------------------------'''
class BertModel(object):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.batch_size=config.batch_size
        self.seq_len=config.max_position_embeddings

    def __call__(self, input_ids, token_type_ids, attention_mask):
        extended_attention_mask = ht.array_reshape_op(attention_mask, [self.batch_size, 1, 1, self.seq_len])
        extended_attention_mask = (extended_attention_mask+(-1.0)) * 10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        sequence_output = self.encoder(embedding_output, extended_attention_mask)
        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output

'''-----------------------------------------------------------------------------------------------'''


'''
Pipeline parallel (2 devices):
--------------------------------------------------------------------------------------------------'''

class BertModel_pp(object):
    def __init__(self, config):
        self.layer_num_per_device = int(config.num_hidden_layers/2)

        with ht.context(ht.gpu(0)):
            self.embeddings = BertEmbeddings(config)

        self.encoders = []
        # for i in range(2):
        #     with ht.context(ht.gpu(i)):
        #         self.encoders.append(BertEncoder(config, i*self.layer_num_per_device, self.layer_num_per_device))


        with ht.context(ht.gpu(0)):
            self.encoders.append(BertEncoder_prev(config, 0, self.layer_num_per_device))

        with ht.context(ht.gpu(1)):
            self.encoders.append(BertEncoder_last(config, self.layer_num_per_device, self.layer_num_per_device))

        with ht.context(ht.gpu(1)):
            self.pooler = BertPooler(config)

        self.batch_size=config.batch_size
        self.seq_len=config.max_position_embeddings

    def __call__(self, input_ids, token_type_ids, attention_mask):
        with ht.context(ht.gpu(0)):
            extended_attention_mask = ht.array_reshape_op(attention_mask, [self.batch_size, 1, 1, self.seq_len])
            # extended_attention_mask = ht.array_reshape_op(attention_mask, [self.batch_size, 1, self.seq_len])

            extended_attention_mask = (extended_attention_mask+(-1.0)) * 10000.0

            embedding_output = self.embeddings(input_ids, token_type_ids)
            sequence_output = embedding_output

        for i in range(2):
            with ht.context(ht.gpu(i)):
                sequence_output = self.encoders[i](sequence_output, extended_attention_mask)

        with ht.context(ht.gpu(1)):
            pooled_output = self.pooler(sequence_output)

            return sequence_output, pooled_output

class BertForPreTraining_mp(object):
    def __init__(self, config):
        self.bert = BertModel_pp(config)

        with ht.context(ht.gpu(0)):
            self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)

        self.vocab_size=config.vocab_size

    def __call__(self, input_ids, token_type_ids, attention_mask, masked_lm_labels=None, next_sentence_label=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)

        with ht.context(ht.gpu(0)):
            prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

            return_op = [prediction_scores, seq_relationship_score]
            if masked_lm_labels is not None and next_sentence_label is not None:
                masked_lm_loss = ht.softmaxcrossentropy_sparse_op(prediction_scores, masked_lm_labels,ignored_index=-1, grad_inplace = True)
                next_sentence_loss = ht.softmaxcrossentropy_sparse_op(seq_relationship_score, next_sentence_label,ignored_index=-1, grad_inplace = True)

                return_op += [masked_lm_loss, next_sentence_loss]
            return return_op
'''-----------------------------------------------------------------------------------------------'''



'''
BertForPreTraining:
--------------------------------------------------------------------------------------------------'''
class BertForPreTraining(object):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: optional next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)

        self.vocab_size=config.vocab_size

    def __call__(self, input_ids, token_type_ids, attention_mask, masked_lm_labels=None, next_sentence_label=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        return_op = [prediction_scores, seq_relationship_score]
        if masked_lm_labels is not None and next_sentence_label is not None:
            '''
            masked_lm_labels: [batch_size, seq_len]
            prediction_scores: [batch_size, seq_len, vocab_size]
            next_sentence_label: [batch_size]
            seq_relationship_score: [batch_size, 2]

            masked_lm_loss: [batch_size*seq_len]
            next_sentence_loss: [batch_size]
            '''

            masked_lm_loss = ht.softmaxcrossentropy_sparse_op(prediction_scores, masked_lm_labels, ignored_index=-1)
            next_sentence_loss = ht.softmaxcrossentropy_sparse_op(seq_relationship_score, next_sentence_label, ignored_index=-1)

            return_op += [masked_lm_loss, next_sentence_loss]
        return return_op


class BertForMaskedLM(object):
    """BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `masked_lm_labels` is  not `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForMaskedLM(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.vocab_size=config.vocab_size

    def __call__(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask)
        prediction_scores = self.cls(sequence_output)

        return_op = [prediction_scores]
        if masked_lm_labels is not None:
            '''
            masked_lm_labels: [batch_size, seq_len]
            prediction_scores: [batch_size, seq_len, vocab_size]

            masked_lm_loss: [batch_size*seq_len]
            '''
            masked_lm_loss = ht.softmaxcrossentropy_sparse_op(prediction_scores, masked_lm_labels, ignored_index=-1)
            return_op += [masked_lm_loss]

        return return_op


class BertForNextSentencePrediction(object):
    """BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `next_sentence_label` is not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `next_sentence_label` is `None`:
            Outputs the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForNextSentencePrediction(config)
    seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)

    def __call__(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        seq_relationship_score = self.cls(pooled_output)

        return_op = [seq_relationship_score]
        if next_sentence_label is not None:
            '''
            next_sentence_label: [batch_size]
            seq_relationship_score: [batch_size, 2]

            next_sentence_loss: [batch_size]
            '''
            next_sentence_loss = ht.softmaxcrossentropy_sparse_op(seq_relationship_score, next_sentence_label, ignored_index=-1)
            return_op += [next_sentence_loss]

        return return_op

'''-----------------------------------------------------------------------------------------------'''


'''
Downstream tasks:
--------------------------------------------------------------------------------------------------'''
class BertForSequenceClassification(object):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.classifier = Linear(config.hidden_size, num_labels)

    def __call__(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output) # [batch_size, num_labels]

        if labels is not None:
            loss = ht.softmaxcrossentropy_sparse_op(logits, labels, ignored_index = -1)
            return loss, logits
        else:
            return logits
'''-----------------------------------------------------------------------------------------------'''



'''
Bert Layer utils (Embedding & BerLayerNorm & Dropout & Linear)
--------------------------------------------------------------------------------------------------'''
class Embedding(object):
    def __init__(self, num_embeddings, embedding_dim, embedding_name=None, initializer=ht.init.xavier_normal):
        self.weight = initializer(name=embedding_name, shape=(num_embeddings, embedding_dim))
    def __call__(self, input_tensor):
        return ht.embedding_lookup_op(self.weight, input_tensor)

class BertLayerNorm(object):
    def __init__(self, hidden_size, eps=1e-12):
        self.eps=eps
        self.scale = ht.init.ones(name='layer_norm_scale', shape=(hidden_size, ))
        self.bias = ht.init.zeros(name='layer_norm_bias', shape=(hidden_size, ))
    def __call__(self, input_tensor):
        return ht.layer_normalization_op(input_tensor, self.scale, self.bias, eps=self.eps)

class Dropout(object):
    def __init__(self, dropout_prob=None, inplace = False):
        self.dropout_prob = dropout_prob
        self.inplace = inplace
    def __call__(self, input_tensor):
        if self.dropout_prob is None or self.dropout_prob == 0.0:
            return input_tensor
        output = ht.dropout_op(input_tensor, 1.0 - self.dropout_prob, inplace = self.inplace)
        return output

class LinearAct(object):
    def __init__(self, in_features, out_features, activation, kernel_initializer=ht.init.xavier_normal, bias_initializer=ht.init.zeros, input_shape=None):
        self.activation = activation
        self.weights = kernel_initializer(name='dense_weights', shape=(out_features, in_features))
        self.bias = bias_initializer(name='dense_bias', shape=(out_features,))
        self.input_shape=input_shape
        self.in_features = in_features
        self.out_features = out_features
        if self.input_shape is not None and self.input_shape[-1]!=in_features:
            print("Specified in_features is not equal to input_shape[-1].")
            assert(False)
    def __call__(self, input_tensor):
        if self.input_shape is not None and len(self.input_shape)!=2:
            input_tensor = ht.array_reshape_op(input_tensor, [-1, self.in_features])

        outputs = ht.linearAct_op(input_tensor, self.weights, self.bias, trans_B = True, act = self.activation)

        if self.input_shape is not None and len(self.input_shape)!=2:
            outputs = ht.array_reshape_op(outputs, self.input_shape[:-1]+[self.out_features])
        return outputs

        

class Linear(object):
    def __init__(self, in_features, out_features, bias=True, activation=None, kernel_initializer=ht.init.xavier_normal, bias_initializer=ht.init.zeros, input_shape=None):
        self.bias_flag = bias
        self.activation = activation
        #self.weights = kernel_initializer(name='dense_weights', shape=(in_features, out_features))
        self.weights = kernel_initializer(name='dense_weights', shape=(out_features, in_features))
        if self.bias_flag:
            self.bias = bias_initializer(name='dense_bias', shape=(out_features,))
        self.input_shape=input_shape
        self.in_features = in_features
        self.out_features = out_features
        if self.input_shape is not None and self.input_shape[-1]!=in_features:
            print("Specified in_features is not equal to input_shape[-1].")
            assert(False)
    def __call__(self, input_tensor):
        if self.input_shape is not None and len(self.input_shape)!=2:
            input_tensor = ht.array_reshape_op(input_tensor, [-1, self.in_features])
        #outputs = ht.matmul_op(input_tensor, self.weights)

        # outputs = ht.matmul_op(input_tensor, self.weights, trans_B = True)
        # if self.bias_flag:
        #     outputs = outputs + ht.broadcastto_op(self.bias, outputs)

        if self.bias_flag:
            outputs = ht.linear_op(input_tensor, self.weights, self.bias, trans_B = True)
        else:
            outputs = ht.matmul_op(input_tensor, self.weights, trans_B = True)

        if self.activation is not None:
            outputs = self.activation(outputs)
        if self.input_shape is not None and len(self.input_shape)!=2:
            outputs = ht.array_reshape_op(outputs, self.input_shape[:-1]+[self.out_features])
        return outputs
'''-----------------------------------------------------------------------------------------------'''


'''
BERT Config:
--------------------------------------------------------------------------------------------------'''
class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="relu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 output_hidden_states=False,
                 batch_size=100,
                 ):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.output_hidden_states = output_hidden_states
        self.batch_size = batch_size


'''-----------------------------------------------------------------------------------------------'''
