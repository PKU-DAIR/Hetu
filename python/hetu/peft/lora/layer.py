import os
import numpy as np
import hetu
from queue import Queue
from typing import Dict, Optional, Tuple, List
from hetu.nn.modules.module import Module
import hetu.nn.modules.parallel_multi_ds as parallel_multi_ds
from hetu.utils.parallel import get_multi_ds_parallel_config

class HtMultiRowParallelLinear(parallel_multi_ds.HtMultiRowParallelLinear):
    def forward(self, input_p):
        if input_p.check_ds_hierarchy_equal(self.ds_union_map['split01']):
            tensor_split01 = input_p
        else:
            tensor_split01 = hetu.comm(input_p, self.ds_union_map['split01']) # exists src_ds == dst_ds case, just ignore it in comm_op
        tensor_split0_partial = hetu.linear(tensor_split01, self.weight, trans_a=False, trans_b=False, device_group_hierarchy=self.device_group_unions, name=self.name)
        # allreduce for lora
        output = hetu.comm(tensor_split0_partial, self.ds_union_map['split0_dup']) # allreduce   
        output = output + self.bias if self.bias is not None else output
        return output

class HtMultiColumnParallelLinear(parallel_multi_ds.HtMultiColumnParallelLinear):
    pass

class LoraLayer():
    def __init__(self, base_layer: Module, multi_ds_parallel_config, name) -> None:
        self.base_layer = base_layer
        self.ds_parallel_configs = multi_ds_parallel_config
        self.rank = 0
        self.lora_alpha = 1
        self.lora_dropout: Module = Module()
        self.lora_A: Module = Module()
        self.lora_B: Module = Module()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.name = name
    
    def update_layer(self, rank, lora_alpha, lora_dropout, use_rslora):
        if rank <= 0:
            raise ValueError(f"`rank` should be a positive integer value but the value passed is {rank}")
        self.rank = rank
        self.lora_alpha = lora_alpha
        if use_rslora:
            self.scaling = lora_alpha / (rank ** 0.5)
        else:
            self.scaling = lora_alpha / rank

        if lora_dropout > 0.0:
            self.lora_dropout = hetu.nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = hetu.nn.Identity()
        
        lora_a_multi_ds_parallel_configs = get_multi_ds_parallel_config(self.ds_parallel_configs, 'lora_A')
        lora_b_multi_ds_parallel_configs = get_multi_ds_parallel_config(self.ds_parallel_configs, 'lora_B')

        if isinstance(self.base_layer, parallel_multi_ds.HtMultiRowParallelLinear):
            self.lora_A = HtMultiRowParallelLinear(self.in_features, self.rank, lora_a_multi_ds_parallel_configs,
                                                   bias=False, init_method='he_uniform_',
                                                   dtype=self.base_layer.dtype, name=f'lora_A_{self.name}')
            self.lora_B = HtMultiColumnParallelLinear(self.rank, self.out_features, lora_b_multi_ds_parallel_configs,
                                                      bias=False, gather_output=True, init_method='zeros_',
                                                      dtype=self.base_layer.dtype, name=f'lora_B_{self.name}')
        elif isinstance(self.base_layer, parallel_multi_ds.HtMultiColumnParallelLinear):
            self.lora_A = HtMultiColumnParallelLinear(self.in_features, self.rank, lora_a_multi_ds_parallel_configs,
                                                      bias=False, gather_output=True, init_method='he_uniform_',
                                                      dtype=self.base_layer.dtype, name=f'lora_A_{self.name}')
            self.lora_B = HtMultiColumnParallelLinear(self.rank, self.out_features, lora_b_multi_ds_parallel_configs,
                                                      bias=False, gather_output=False, init_method='zeros_',
                                                      dtype=self.base_layer.dtype, name=f'lora_B_{self.name}')

class MultiLoraLayers():
    def __init__(self, base_layer: Module, multi_ds_parallel_config, name) -> None:
        self.base_layer = base_layer
        self.ds_parallel_configs = multi_ds_parallel_config
        self.lora_layers: Dict[int, LoraLayer] = {}
        self.name = name
    
    def update_layers(self, ranks, lora_alphas, lora_dropouts, use_rsloras, task_indices):
        for i, task_indice in enumerate(task_indices):
            self.lora_layers[task_indice] = LoraLayer(self.base_layer, self.ds_parallel_configs, f'{self.name}_task{task_indice}')
            self.lora_layers[task_indice].update_layer(ranks[i], lora_alphas[i], lora_dropouts[i], use_rsloras[i])

class HtLoRAMultiColumnParallelLinear(Module, LoraLayer):
    def __init__(
        self,
        base_layer,
        multi_ds_parallel_config,
        rank: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        use_rslora: bool = False,
        name='colp_lora'
    ):
        super(HtLoRAMultiColumnParallelLinear, self).__init__()
        lora_name = name.replace('base', 'lora')
        LoraLayer.__init__(self, base_layer, multi_ds_parallel_config, lora_name)
        self.update_layer(rank, lora_alpha, lora_dropout, use_rslora)
    
    def forward(self, input_p):
        base_result = self.base_layer(input_p)
        lora_result = hetu.mul(self.lora_B(self.lora_A(input_p)), self.scaling, name=f'mul_{self.name}')
        if lora_result.check_ds_hierarchy_equal(base_result.ds_hierarchy):
            lora_comm_result = lora_result
        else:
            lora_comm_result = hetu.comm(lora_result, base_result.ds_hierarchy, name=f'comm_{self.name}')
        output = hetu.add(base_result, lora_comm_result, name=f'sync_add_{self.name}')
        return output

class HtLoRAMultiRowParallelLinear(Module, LoraLayer):
    def __init__(
        self,
        base_layer,
        multi_ds_parallel_config,
        rank: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        use_rslora: bool = False,
        name='rowp_lora'
    ):
        super(HtLoRAMultiRowParallelLinear, self).__init__()
        lora_name = name.replace('base', 'lora')
        LoraLayer.__init__(self, base_layer, multi_ds_parallel_config, lora_name)
        self.update_layer(rank, lora_alpha, lora_dropout, use_rslora)
    
    def forward(self, input_p):
        base_result = self.base_layer(input_p)
        lora_result = hetu.mul(self.lora_B(self.lora_A(input_p)), self.scaling, name=f'mul_{self.name}')
        if lora_result.check_ds_hierarchy_equal(base_result.ds_hierarchy):
            lora_comm_result = lora_result
        else:
            lora_comm_result = hetu.comm(lora_result, base_result.ds_hierarchy, name=f'comm_{self.name}')
        output = hetu.add(base_result, lora_comm_result, name=f'sync_add_{self.name}')
        return output

class HtMultiLoRAMultiColumnParallelLinear(Module, MultiLoraLayers):
    def __init__(
        self,
        base_layer,
        multi_ds_parallel_config,
        config,
        ranks: List[int] = [0],
        lora_alphas: List[int] = [1],
        lora_dropouts: List[float] = [0.0],
        use_rsloras: List[bool] = [False],
        task_indices: List[int] = [0],
        name='colp_lora'
    ):
        super(HtMultiLoRAMultiColumnParallelLinear, self).__init__()
        lora_name = name.replace('base', 'lora')
        self.config = config
        MultiLoraLayers.__init__(self, base_layer, multi_ds_parallel_config, lora_name)
        self.update_layers(ranks, lora_alphas, lora_dropouts, use_rsloras, task_indices)
    
    def forward(self, input_p):
        if input_p.check_ds_hierarchy_equal(self.base_layer.ds_union_map['split0_dup']):
            tensor_split0_dup = input_p
        else:
            tensor_split0_dup = hetu.comm(input_p, self.base_layer.ds_union_map['split0_dup'])
            print(f"warning: column parallel linear need extra communication for \
                    adapt input tensor distributed_states {input_p.ds_hierarchy} into {self.base_layer.ds_union_map['split0_dup']}!")

        base_result = self.base_layer(tensor_split0_dup)
        if self.config.train_task_num == 1:
            lora_result = self.lora_layers[0].lora_B(self.lora_layers[0].lora_A(tensor_split0_dup))
            lora_comm_result = hetu.mul(lora_result, self.lora_layers[0].scaling, name=f'mul_{self.name}_task0')
            if lora_result.check_ds_hierarchy_equal(base_result.ds_hierarchy):
                lora_comm_result = lora_result
            else:
                lora_comm_result = hetu.comm(lora_result, base_result.ds_hierarchy, name=f'comm_{self.name}_task0')
            base_result = hetu.index_add_(base_result, lora_comm_result, self.config.task_batch_idxs[0], dim=0, name=f'index_add_{self.name}_task0')
        else: 
            # TODO: 改成支持packing
            task_tensor_split0_dup_list = hetu.split(tensor_split0_dup, self.config.task_batch_idxs, dim=0, name=f'split_task_{self.name}')
            for i in range(self.config.train_task_num):
                lora_result = self.lora_layers[i].lora_B(self.lora_layers[i].lora_A(task_tensor_split0_dup_list[i]))
                lora_comm_result = hetu.mul(lora_result, self.lora_layers[i].scaling, name=f'mul_{self.name}_task{i}')
                if lora_result.check_ds_hierarchy_equal(base_result.ds_hierarchy):
                    lora_comm_result = lora_result
                else:
                    lora_comm_result = hetu.comm(lora_result, base_result.ds_hierarchy, name=f'comm_{self.name}_task{i}')
                base_result = hetu.index_add_(base_result, lora_comm_result, self.config.task_batch_idxs[i], dim=0, name=f'index_add_{self.name}_task{i}')
        return base_result

class HtMultiLoRAMultiRowParallelLinear(Module, MultiLoraLayers):
    def __init__(
        self,
        base_layer,
        multi_ds_parallel_config,
        config,
        ranks: List[int] = [0],
        lora_alphas: List[int] = [1],
        lora_dropouts: List[float] = [0.0],
        use_rsloras: List[bool] = [False],
        task_indices: List[int] = [0],
        name='rowp_lora'
    ):
        super(HtMultiLoRAMultiRowParallelLinear, self).__init__()
        lora_name = name.replace('base', 'lora')
        self.config = config
        MultiLoraLayers.__init__(self, base_layer, multi_ds_parallel_config, lora_name)
        self.update_layers(ranks, lora_alphas, lora_dropouts, use_rsloras, task_indices)
    
    def forward(self, input_p):
        if input_p.check_ds_hierarchy_equal(self.base_layer.ds_union_map['split01']):
            tensor_split0_dup = input_p
        else:
            tensor_split0_dup = hetu.comm(input_p, self.base_layer.ds_union_map['split01'])
            print(f"warning: row parallel linear need extra communication for \
                    adapt input tensor distributed_states {input_p.ds_hierarchy} into {self.base_layer.ds_union_map['split01']}!")
        base_result = self.base_layer(tensor_split0_dup)
        if self.config.train_task_num == 1:
            lora_result = self.lora_layers[0].lora_B(self.lora_layers[0].lora_A(tensor_split0_dup))
            lora_comm_result = hetu.mul(lora_result, self.lora_layers[0].scaling, name=f'mul_{self.name}_task0')
            if lora_result.check_ds_hierarchy_equal(base_result.ds_hierarchy):
                lora_comm_result = lora_result
            else:
                lora_comm_result = hetu.comm(lora_result, base_result.ds_hierarchy, name=f'comm_{self.name}_task0')
            base_result = hetu.index_add_(base_result, lora_comm_result, self.config.task_batch_idxs[0], dim=0, name=f'index_add_{self.name}_task0')
        else:
            task_tensor_split0_dup_list = hetu.split(tensor_split0_dup, self.config.task_batch_idxs, dim=0, name=f'split_task_{self.name}')
            for i in range(self.config.train_task_num):
                lora_result = self.lora_layers[i].lora_B(self.lora_layers[i].lora_A(task_tensor_split0_dup_list[i]))
                lora_comm_result = hetu.mul(lora_result, self.lora_layers[i].scaling, name=f'mul_{self.name}_task{i}')
                if lora_result.check_ds_hierarchy_equal(base_result.ds_hierarchy):
                    lora_comm_result = lora_result
                else:
                    lora_comm_result = hetu.comm(lora_result, base_result.ds_hierarchy, name=f'comm_{self.name}_task{i}')
                base_result = hetu.index_add_(base_result, lora_comm_result, self.config.task_batch_idxs[i], dim=0, name=f'index_add_{self.name}_task{i}')
        return base_result

def dispatch_lora_layer(target, **kwargs) -> Optional[Module]:
    new_module = None
    ds_parallel_configs = get_multi_ds_parallel_config(target.ds_parallel_configs, 'lora')
    
    if isinstance(target, parallel_multi_ds.HtMultiRowParallelLinear):
        new_module = HtLoRAMultiRowParallelLinear(target, ds_parallel_configs, name=target.name, **kwargs)
    elif isinstance(target, parallel_multi_ds.HtMultiColumnParallelLinear):
        new_module = HtLoRAMultiColumnParallelLinear(target, ds_parallel_configs, name=target.name, **kwargs)
    else:
        print(f"Not Supported for module {target}")
    
    return new_module

def dispatch_multi_lora_layers(target, config, **kwargs) -> Optional[Module]:
    new_module = None
    ds_parallel_configs = get_multi_ds_parallel_config(target.ds_parallel_configs, 'lora')
    
    if isinstance(target, parallel_multi_ds.HtMultiRowParallelLinear):
        new_module = HtMultiLoRAMultiRowParallelLinear(target, ds_parallel_configs, config, name=target.name, **kwargs)
    elif isinstance(target, parallel_multi_ds.HtMultiColumnParallelLinear):
        new_module = HtMultiLoRAMultiColumnParallelLinear(target, ds_parallel_configs, config, name=target.name, **kwargs)
    else:
        print(f"Not Supported for module {target}")
    
    return new_module