
import hetu as ht
import numpy as np
import torch
from hetu.nn.modules.parallel_multi_ds import parallel_data_provider, parallel_multi_data_provider, get_multi_ds_parallel_config

from model.hetu_llama import LLamaModel
from model.hetu_vision import VisionModel
    
class MLLMModel(ht.nn.Module):

    def __init__(self, vision_config, llm_config, ds_parallel_configs):
        super(MLLMModel, self).__init__()
        self.vision_config = vision_config
        self.llm_config = llm_config
        self.ds_parallel_configs = ds_parallel_configs
        self.vision = VisionModel(vision_config,  get_multi_ds_parallel_config(ds_parallel_configs, 'vision'))
        self.llm = LLamaModel(llm_config, get_multi_ds_parallel_config(ds_parallel_configs, 'llama'))
        self.lm_head = ht.nn.HtMultiColumnParallelLinear(
            llm_config.hidden_size,
            llm_config.vocab_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'lm_head'),
            bias=False,
            gather_output=False,
            name='lm_head'
        )
        # share embedding table
        # we manually add comm op here
        # because we don't know if it is a P2P or a BatchedIsendIrecv in hetero settings
        # self.lm_head.weight = ht.comm(self.transformer.wte.embedding_table, self.lm_head.ds_union_map['dup_split0'], self.lm_head.device_group_unions, name="share_weight_comm") 
    
    def forward(
        self,
        image_inputs=None,
        text_ids=None,
        position_ids=None,
        image_mask=None,
        video_mask=None,
        attention_mask=None,
        token_type_ids=None,
        loss_mask=None,
        labels=None
    ):
        
        vision_hidden_states = self.vision(image_inputs, position_ids, attention_mask, token_type_ids)
        # print("over vision forward")

        text_embedding = self.llm.wte(text_ids)
        
        print('vision_hidden_states.ds_hierarchy: ', vision_hidden_states.ds_hierarchy)
        print("self.vision.h[self.vision_config.num_hidden_layers - 1].rmsnorm_1.device_group_unions", self.vision.h[self.vision_config.num_hidden_layers - 1].rmsnorm_1.device_group_unions)
        print("text_embedding.ds_hierarchy: ", text_embedding.ds_hierarchy)
        print("self.llm.h[0].rmsnorm_1.device_group_unions: ", self.llm.h[0].rmsnorm_1.device_group_unions)
        vision_hidden_states = ht.comm(vision_hidden_states, vision_hidden_states.ds_hierarchy, self.vision.h[self.vision_config.num_hidden_layers - 1].rmsnorm_1.device_group_unions, text_embedding.ds_hierarchy, self.llm.h[0].rmsnorm_1.device_group_unions, 
                                        self.llm_config.packing_slice_list, True, True, name="vision_hidden_states_comm")

        # image_mask_repeat = ht.repeat(image_mask, text_embedding.shape)
        llm_inputs_embedding = ht.masked_scatter(text_embedding, image_mask, vision_hidden_states)
        # llm_inputs_embedding = vision_hidden_states + text_embedding

        # [b * seq_len, n_embd]
        hidden_states = self.llm(
            input_ids = None,
            inputs_embeds = llm_inputs_embedding,
            position_ids = position_ids,
            attention_mask = attention_mask,
            token_type_ids=token_type_ids,
        )
        
        '''
        # need allgather here: [b * s // tp, h] -> [b * s, h]
        if not hidden_states.check_ds_hierarchy_equal(self.lm_head.ds_union_map['split0_dup']):
            hidden_states = ht.comm(hidden_states, self.lm_head.ds_union_map['split0_dup'])
        '''
        
        # column parallel, [b * seq_len, n_embd] -> [b * seq_len, vocab_size], and splited in vocab dimension
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # print(f"lm_logits shape {lm_logits.shape}, labels shape {labels.shape}")
            loss_unreduced = ht.vocab_parallel_cross_entropy(lm_logits, labels, ignored_index = -1, reduction = "none").reshape([-1])
            loss_sum = ht.sum(ht.mul(loss_unreduced, loss_mask))
            loss_valid_tokens = ht.sum(loss_mask)
            loss = ht.div(loss_sum, loss_valid_tokens)
            

        return loss
