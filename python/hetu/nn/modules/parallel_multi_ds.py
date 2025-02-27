import hetu
import numpy as np
from queue import Queue
from .module import Module
import numbers
from hetu.utils.parallel import get_local_index, config2ds, get_multi_recompute_from

__all__ = [
    'HtMultiColumnParallelLinear', 
    'HtMultiRowParallelLinear', 
    'HtMultiParallelEmbedding',
    'HtMultiVocabParallelEmbedding',
    'HtMultiParallelLayerNorm',
    'HtMultiParallelRMSNorm',
]

class HtMultiParallelRMSNorm(Module):
    def __init__(self, normalized_shape, multi_ds_parallel_config, sequence_parallel=False, recompute_allgather=False, dtype=hetu.float32, name='rmsnorm'):
        super(HtMultiParallelRMSNorm, self).__init__()
        self.ds_parallel_configs = multi_ds_parallel_config
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = [normalized_shape]  # type: ignore[assignment]
        self.normalized_shape = list(normalized_shape)  # type: ignore[arg-type]
        self.sequence_parallel = sequence_parallel
        self.recompute_allgather = recompute_allgather
        self.name = name
        self.layer_idx = int(name.split('Block')[1]) if 'Block' in name else -1
        self.ds_union_map = {'dup': [], 'split0': [], 'split0_dup': []}
        self.device_index = []
        self.device_group_unions = []
        for ds_parallel_config in multi_ds_parallel_config:
            ds_union_dup_split0, device_group_union = config2ds(ds_parallel_config)
            self.device_group_unions.append(device_group_union)
            hetero_size = len(device_group_union)
            dcp_union = [ds_union_dup_split0.get(i).get_dim(-1) for i in range(hetero_size)]
            tp_union = [ds_union_dup_split0.get(i).get_dim(0) for i in range(hetero_size)]
            hetero_dim = ds_union_dup_split0.hetero_dim
            assert hetero_dim == -1 or hetero_dim == -3, "ParallelRMSNorm only support hetero on dup"
            assert np.array_equal(np.array(dcp_union) * np.array(tp_union) / hetero_size, np.array([device_group.num_devices for device_group in device_group_union]) 
                , f'ParallelRMSNorm get wrong ds_parallel_config: {ds_parallel_config}!')
            device_group_index, device_index = get_local_index(device_group_union)
            self.device_index.append(device_index)
            ds_list_dup = [hetu.DistributedStates(device_group_union[i].num_devices * hetero_size, {-1: device_group_union[i].num_devices * hetero_size}, [-1])
                for i in range(hetero_size)] # for sp data
            ds_union_dup = hetu.DistributedStatesUnion(ds_list_dup, -1 if hetero_dim != -3 else -3)
            ds_list_split0 = [hetu.DistributedStates(device_group_union[i].num_devices * hetero_size, {0: device_group_union[i].num_devices * hetero_size}, [0])
                for i in range(hetero_size)] # for sp data
            ds_union_split0 = hetu.DistributedStatesUnion(ds_list_split0, 0 if hetero_dim != -3 else -3)
            ds_list_split0_dup = [hetu.DistributedStates(device_group_union[i].num_devices * hetero_size, {-1: tp_union[i], 0: dcp_union[i]}, [0, -1])
                for i in range(hetero_size)] # for data
            ds_union_split0_dup = hetu.DistributedStatesUnion(ds_list_split0_dup, 0 if hetero_dim != -3 else -3)
            self.ds_union_map['dup'].append(ds_union_dup)
            self.ds_union_map['split0'].append(ds_union_split0)
            self.ds_union_map['split0_dup'].append(ds_union_split0_dup)
            
        self.weight = hetu.parallel_parameter(eval(f'hetu.ones_initializer()'), 
                                              self.normalized_shape, self.ds_union_map['dup'], 
                                              self.device_index, dtype=dtype, requires_grad=True, 
                                              device_group_hierarchy=self.device_group_unions, name=f'{name}_weight')

    def forward(self, input_p):
        if self.sequence_parallel:
            if input_p.check_ds_hierarchy_equal(self.ds_union_map['split0']):
                input_p = input_p
            else:
                input_p = hetu.comm(input_p, self.ds_union_map['split0'])
            with hetu.recompute(multi_recompute = get_multi_recompute_from(self.ds_parallel_configs, self.layer_idx)):
                output_rms_split0 = hetu.fused_rmsnorm(input_p, self.weight, self.normalized_shape, \
                                                    device_group_hierarchy=self.device_group_unions, name=self.name + '_sp')[0]
            # handle allgather recompute manually
            if output_rms_split0.check_ds_hierarchy_equal(self.ds_union_map['split0_dup']):
                output_rms = output_rms_split0
            else:
                if self.recompute_allgather:
                    with hetu.recompute(multi_recompute=[[True] * len(self.ds_parallel_configs)]):
                        output_rms = hetu.comm(output_rms_split0, self.ds_union_map['split0_dup'])
                else:
                    output_rms = hetu.comm(output_rms_split0, self.ds_union_map['split0_dup'])
        else:
            if input_p.check_ds_hierarchy_equal(self.ds_union_map['split0_dup']):
                input_p = input_p
            else:
                input_p = hetu.comm(input_p, self.ds_union_map['split0_dup'])
            with hetu.recompute(multi_recompute = get_multi_recompute_from(self.ds_parallel_configs, self.layer_idx)):
                output_rms = hetu.fused_rmsnorm(input_p, self.weight, self.normalized_shape, \
                                                device_group_hierarchy=self.device_group_unions, name=self.name)[0]
        return output_rms

class HtMultiParallelLayerNorm(Module):
    def __init__(self, normalized_shape, multi_ds_parallel_config, sequence_parallel=False, recompute_allgather=False, eps=1e-5, dtype=hetu.float32, name='ln'):
        super(HtMultiParallelLayerNorm, self).__init__()
        self.ds_parallel_configs = multi_ds_parallel_config
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = [normalized_shape]  # type: ignore[assignment]
        self.normalized_shape = list(normalized_shape)  # type: ignore[arg-type]
        self.sequence_parallel = sequence_parallel
        self.recompute_allgather = recompute_allgather
        self.eps = eps
        self.name = name
        self.layer_idx = int(name.split('Block')[1]) if 'Block' in name else -1
        self.ds_union_map = {'dup': [], 'split0': [], 'split0_dup': []}
        self.device_index = []
        self.device_group_unions = []
        for ds_parallel_config in multi_ds_parallel_config:
            ds_union_dup_split0, device_group_union = config2ds(ds_parallel_config)
            self.device_group_unions.append(device_group_union)
            hetero_size = len(device_group_union)
            dcp_union = [ds_union_dup_split0.get(i).get_dim(-1) for i in range(hetero_size)]
            tp_union = [ds_union_dup_split0.get(i).get_dim(0) for i in range(hetero_size)]
            hetero_dim = ds_union_dup_split0.hetero_dim
            assert hetero_dim == -1 or hetero_dim == -3, "ParallelLayerNorm only support hetero on dup"
            assert np.array_equal(np.array(dcp_union) * np.array(tp_union) / hetero_size, np.array([device_group.num_devices for device_group in device_group_union]) 
                , f'ParallelLayerNorm get wrong ds_parallel_config: {ds_parallel_config}!')
            device_group_index, device_index = get_local_index(device_group_union)
            self.device_index.append(device_index)
            ds_list_dup = [hetu.DistributedStates(device_group_union[i].num_devices * hetero_size, {-1: device_group_union[i].num_devices * hetero_size}, [-1])
                for i in range(hetero_size)] # for sp data
            ds_union_dup = hetu.DistributedStatesUnion(ds_list_dup, -1 if hetero_dim != -3 else -3)
            ds_list_split0 = [hetu.DistributedStates(device_group_union[i].num_devices * hetero_size, {0: device_group_union[i].num_devices * hetero_size}, [0])
                for i in range(hetero_size)] # for sp data
            ds_union_split0 = hetu.DistributedStatesUnion(ds_list_split0, 0 if hetero_dim != -3 else -3)
            ds_list_split0_dup = [hetu.DistributedStates(device_group_union[i].num_devices * hetero_size, {-1: tp_union[i], 0: dcp_union[i]}, [0, -1])
                for i in range(hetero_size)] # for data
            ds_union_split0_dup = hetu.DistributedStatesUnion(ds_list_split0_dup, 0 if hetero_dim != -3 else -3)
            self.ds_union_map['dup'].append(ds_union_dup)
            self.ds_union_map['split0'].append(ds_union_split0)
            self.ds_union_map['split0_dup'].append(ds_union_split0_dup)
            
        self.weight = hetu.parallel_parameter(eval(f'hetu.ones_initializer()'), 
                                              self.normalized_shape, self.ds_union_map['dup'], 
                                              self.device_index, dtype=dtype, requires_grad=True, 
                                              device_group_hierarchy=self.device_group_unions, name=f'{name}_weight')
        self.bias = hetu.parallel_parameter(eval(f'hetu.zeros_initializer()'), 
                                              self.normalized_shape, self.ds_union_map['dup'], 
                                              self.device_index, dtype=dtype, requires_grad=True, 
                                              device_group_hierarchy=self.device_group_unions, name=f'{name}_bias')

    def forward(self, input_p):
        if self.sequence_parallel:
            if input_p.check_ds_hierarchy_equal(self.ds_union_map['split0']):
                input_p = input_p
            else:
                input_p = hetu.comm(input_p, self.ds_union_map['split0'])
            with hetu.recompute(multi_recompute = get_multi_recompute_from(self.ds_parallel_configs, self.layer_idx)):
                output_ln_split0 = hetu.fused_layernorm(input_p, self.weight, self.bias, self.normalized_shape, self.eps, \
                                                        device_group_hierarchy=self.device_group_unions, name=self.name + '_sp')[0]
            # handle allgather recompute manually
            if output_ln_split0.check_ds_hierarchy_equal(self.ds_union_map['split0_dup']):
                output_ln = output_ln_split0
            else:
                if self.recompute_allgather:
                    with hetu.recompute(multi_recompute=[[True] * len(self.ds_parallel_configs)]):
                        output_ln = hetu.comm(output_ln_split0, self.ds_union_map['split0_dup'])
                else:
                    output_ln = hetu.comm(output_ln_split0, self.ds_union_map['split0_dup'])
        else:
            if input_p.check_ds_hierarchy_equal(self.ds_union_map['split0_dup']):
                input_p = input_p
            else:
                input_p = hetu.comm(input_p, self.ds_union_map['split0_dup'])
            with hetu.recompute(multi_recompute = get_multi_recompute_from(self.ds_parallel_configs, self.layer_idx)):
                output_ln = hetu.fused_layernorm(input_p, self.weight, self.bias, self.normalized_shape, self.eps, \
                                                device_group_hierarchy=self.device_group_unions, name=self.name)[0]
        return output_ln

class HtMultiParallelEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim, multi_ds_parallel_config, 
                 init_method='xavier_normal_', dtype=hetu.float32, name='embedding'):
        super(HtMultiParallelEmbedding, self).__init__()
        self.ds_parallel_configs = multi_ds_parallel_config
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.name = name
        self.ds_union_map = {'dup': []}
        self.device_index = []
        self.device_group_unions = []
        for ds_parallel_config in multi_ds_parallel_config:
            ds_union_dup, device_group_union = config2ds(ds_parallel_config)
            self.device_group_unions.append(device_group_union)
            device_group_index, device_index = get_local_index(device_group_union)
            self.device_index.append(device_index)
            self.ds_union_map['dup'].append(ds_union_dup)
        
        # embedding_table should not be splited in any dimension!
        self.embedding_table = hetu.parallel_parameter(eval(f'hetu.{init_method}initializer()'), 
                                                       [num_embeddings, embedding_dim], self.ds_union_map['dup'], 
                                                       self.device_index, dtype=dtype, requires_grad=True, 
                                                       device_group_hierarchy=self.device_group_unions, name=f'{name}_table')
    
    def forward(self, input_p):
        return hetu.embedding_lookup(self.embedding_table, input_p, device_group_hierarchy=self.device_group_unions, name=self.name)
    
class HtMultiVocabParallelEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim, multi_ds_parallel_config, 
                init_method='xavier_normal_', dtype=hetu.float32, name='vocab_embedding'):
        super(HtMultiVocabParallelEmbedding, self).__init__()
        self.ds_parallel_configs = multi_ds_parallel_config
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.name = name

        self.ds_union_map = {'split0_dup': [], 'dup_split0': []}
        self.device_index = []
        self.device_group_unions = []
        self.vocab_start_index = []
        for ds_parallel_config in multi_ds_parallel_config:
            ds_union_dup_split0, device_group_union = config2ds(ds_parallel_config) # for embedding table
            self.device_group_unions.append(device_group_union)
            hetero_size = len(device_group_union)
            dcp_union = [ds_union_dup_split0.get(i).get_dim(-1) for i in range(hetero_size)]
            tp_union = [ds_union_dup_split0.get(i).get_dim(0) for i in range(hetero_size)]
            hetero_dim = ds_union_dup_split0.hetero_dim
            assert hetero_dim == -1 or hetero_dim == -3, "VocabParallelEmbedding only support hetero on dup"
            assert np.array_equal(np.array(dcp_union) * np.array(tp_union) / hetero_size, np.array([device_group.num_devices for device_group in device_group_union]) 
                , f'VocabParallelEmbedding get wrong ds_parallel_config: {ds_parallel_config}!')
            device_group_index, device_index = get_local_index(device_group_union)
            self.device_index.append(device_index)
            ds_list_split0_dup = [hetu.DistributedStates(device_group_union[i].num_devices * hetero_size, {-1: tp_union[i], 0: dcp_union[i]}, [0, -1])
                for i in range(hetero_size)] # for data
            ds_union_split0_dup = hetu.DistributedStatesUnion(ds_list_split0_dup, 0 if hetero_dim != -3 else -3)
            self.ds_union_map['split0_dup'].append(ds_union_split0_dup)
            self.ds_union_map['dup_split0'].append(ds_union_dup_split0)
            
            dup_group_idx = ds_union_dup_split0.get_local(device_group_index).get_dup_group_index(device_index)
            vocab_start_index = num_embeddings // tp_union[device_group_index] * dup_group_idx
            self.vocab_start_index.append(vocab_start_index)

        # embedding_table was splited in vocab dimension
        self.embedding_table = hetu.parallel_parameter(eval(f'hetu.{init_method}initializer()'), 
                                                       [num_embeddings, embedding_dim], self.ds_union_map['dup_split0'], 
                                                       self.device_index, dtype=dtype, requires_grad=True, 
                                                       device_group_hierarchy=self.device_group_unions, name=f'{name}_table')
    
    def forward(self, input_p):
        if input_p.check_ds_hierarchy_equal(self.ds_union_map['split0_dup']):
            tensor_split0_dup = input_p
        else:
            # sequence parallel: need use buffer for allgather to save activation
            tensor_split0_dup = hetu.comm(input_p, self.ds_union_map['split0_dup'])
            print(f"warning: vocab parallel embedding need extra communication for \
                    adapt input tensor ds hierarchy {input_p.ds_hierarchy} into {self.ds_union_map['split0_dup']}!")

        # walkaround: do offset inside embedding lookup op 
        # input_offset = tensor_split0_dup - self.vocab_start_index[0] # should do in embedding_lookup op for multi ds?
        lookup_split0_partial = hetu.embedding_lookup(self.embedding_table, tensor_split0_dup, self.vocab_start_index, 
                                                      device_group_hierarchy=self.device_group_unions, name=self.name+"_"+tensor_split0_dup.name)
        if lookup_split0_partial.check_ds_hierarchy_equal(self.ds_union_map['split0_dup']): # pure dp
            output = lookup_split0_partial
        else:
            output = hetu.comm(lookup_split0_partial, self.ds_union_map['split0_dup'])
        return output

class HtMultiColumnParallelLinear(Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    """
    def __init__(self, in_features, out_features, multi_ds_parallel_config,
                 bias=True, gather_output=True, init_method='xavier_normal_', 
                 dtype=hetu.float32, name='colp'):
        super(HtMultiColumnParallelLinear, self).__init__()
        self.ds_parallel_configs = multi_ds_parallel_config
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.name = name

        self.ds_union_map = {'dup_split0': [], 'split0_dup': []}
        self.device_index = []
        self.device_group_unions = []
        for ds_parallel_config in multi_ds_parallel_config:
            ds_union_dup_split1, device_group_union = config2ds(ds_parallel_config)
            self.device_group_unions.append(device_group_union)
            zero = ds_parallel_config['zero']
            hetero_size = len(device_group_union)
            dcp_union = [ds_union_dup_split1.get(i).get_dim(-1) for i in range(hetero_size)]
            tp_union = [ds_union_dup_split1.get(i).get_dim(1) for i in range(hetero_size)]
            hetero_dim = ds_union_dup_split1.hetero_dim
            assert hetero_dim == -1 or hetero_dim == -3, "ColumnParallelLinear only support hetero on dup"
            assert np.array_equal(np.array(dcp_union) * np.array(tp_union) / hetero_size, np.array([device_group.num_devices for device_group in device_group_union])
                , f'ColumnParallelLinear get wrong ds_parallel_config: {ds_parallel_config}!')        
            device_group_index, device_index = get_local_index(device_group_union)
            self.device_index.append(device_index)
            # assume num_devices=8, there exists 4 cases: dp=1 tp=8, dp=2 tp=4, dp=4 tp=2, dp=8 tp=1
            # when dp=1 tp=8, weights: ds_dup_split0->ds_split0, data: ds_split0_dup->ds_dup
            # when dp=8 tp=1, weights: ds_dup_split0->ds_dup, data: ds_split0_dup->ds_split0
            ds_list_dup_split0 = [hetu.DistributedStates(device_group_union[i].num_devices * hetero_size, {-1: dcp_union[i], 0: tp_union[i]}, [-1, 0], zero)
                for i in range(hetero_size)] # for weights with trans_b
            ds_list_split0_dup = [hetu.DistributedStates(device_group_union[i].num_devices * hetero_size, {-1: tp_union[i], 0: dcp_union[i]}, [0, -1])
                for i in range(hetero_size)] # for data
            ds_union_dup_split0 = hetu.DistributedStatesUnion(ds_list_dup_split0, -1 if hetero_dim != -3 else -3) # for weights with trans_b
            ds_union_split0_dup = hetu.DistributedStatesUnion(ds_list_split0_dup, 0 if hetero_dim != -3 else -3) # for data
            self.ds_union_map['dup_split0'].append(ds_union_dup_split0)
            self.ds_union_map['split0_dup'].append(ds_union_split0_dup)
        
        self.weight = hetu.parallel_parameter(eval(f'hetu.{init_method}initializer()'), 
                                              [out_features, in_features], 
                                              self.ds_union_map['dup_split0'], self.device_index, 
                                              dtype=dtype, requires_grad=True, 
                                              device_group_hierarchy=self.device_group_unions, name=f'{name}_weight')
        if bias:
            self.bias = hetu.parallel_parameter(hetu.zeros_initializer(), [out_features], 
                                                self.ds_union_map['dup_split0'], self.device_index,
                                                dtype=dtype, requires_grad=True, 
                                                device_group_hierarchy=self.device_group_unions, name=f'{name}_bias')
        else:
            self.bias = None
      
    def forward(self, input_p):
        if input_p.check_ds_hierarchy_equal(self.ds_union_map['split0_dup']):
            tensor_split0_dup = input_p
        else:
            tensor_split0_dup = hetu.comm(input_p, self.ds_union_map['split0_dup'])
            '''
            print(f"sp: column parallel linear need extra communication for \
                    adapt input tensor ds hierarchy {input_p.ds_hierarchy} into {self.ds_union_map['split0_dup']}!")
            '''
        
        if self.bias != None:
            tensor_split01 = hetu.linear(tensor_split0_dup, self.weight, self.bias, trans_a=False, trans_b=True, device_group_hierarchy=self.device_group_unions, name=f'linaer_{self.name}')
        else:
            tensor_split01 = hetu.linear(tensor_split0_dup, self.weight, trans_a=False, trans_b=True, device_group_hierarchy=self.device_group_unions, name=f'linear_{self.name}')
        
        if not self.gather_output:
            output = tensor_split01
        else:
            if tensor_split01.check_ds_hierarchy_equal(self.ds_union_map['split0_dup']): # pure dp
                output = tensor_split01
            else:
                output = hetu.comm(tensor_split01, self.ds_union_map['split0_dup'])

        return output
    
# process: x->split1, w->split0 => y->partial => y->dup    
class HtMultiRowParallelLinear(Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    """
    def __init__(self, in_features, out_features, 
                 multi_ds_parallel_config, sequence_parallel=False, bias=True, 
                 init_method='xavier_normal_', 
                 dtype=hetu.float32, name='rowp'):
        super(HtMultiRowParallelLinear, self).__init__()
        self.ds_parallel_configs = multi_ds_parallel_config
        self.in_features = in_features
        self.out_features = out_features
        self.name = name
        self.sequence_parallel = sequence_parallel
        
        self.ds_union_map = {'dup_split0': [], 'dup_split1': [], 'dup': [], 'split0': [], 'split01': [], 'split0_dup': []}
        self.device_index = []
        self.device_group_unions = []
        for ds_parallel_config in multi_ds_parallel_config:
            ds_union_dup_split0, device_group_union = config2ds(ds_parallel_config)
            self.device_group_unions.append(device_group_union)
            zero = ds_parallel_config['zero']
            hetero_size = len(device_group_union)
            dcp_union = [ds_union_dup_split0.get(i).get_dim(-1) for i in range(hetero_size)]
            tp_union = [ds_union_dup_split0.get(i).get_dim(0) for i in range(hetero_size)]
            hetero_dim = ds_union_dup_split0.hetero_dim
            assert hetero_dim == -1 or hetero_dim == -3, "RowParallelLinear only support hetero on dup"
            assert np.array_equal(np.array(dcp_union) * np.array(tp_union) / hetero_size, np.array([device_group.num_devices for device_group in device_group_union])
                , f'RowParallelLinear get wrong ds_parallel_config: {ds_parallel_config}!')        
            device_group_index, device_index = get_local_index(device_group_union)
            self.device_index.append(device_index)
            # assume num_devices=8, there exists 4 cases: dp=1 tp=8, dp=2 tp=4, dp=4 tp=2, dp=8 tp=1
            ds_list_dup_split1 = [hetu.DistributedStates(device_group_union[i].num_devices * hetero_size, {-1: dcp_union[i], 1: tp_union[i]}, [-1, 1], zero)
                for i in range(hetero_size)] # for weight with trans_b
            ds_list_dup = [hetu.DistributedStates(device_group_union[i].num_devices * hetero_size, {-1: dcp_union[i] * tp_union[i]}, [-1], zero)
                for i in range(hetero_size)] # for bias
            ds_list_split0 = [hetu.DistributedStates(device_group_union[i].num_devices * hetero_size, {0: dcp_union[i] * tp_union[i]}, [0])
                for i in range(hetero_size)] # for sp data
            ds_list_split01 = [hetu.DistributedStates(device_group_union[i].num_devices * hetero_size, {0: dcp_union[i], 1: tp_union[i]}, [0, 1])
                for i in range(hetero_size)] # for data split in dimension 1
            ds_list_split0_dup = [hetu.DistributedStates(device_group_union[i].num_devices * hetero_size, {0: dcp_union[i], -1: tp_union[i]}, [0, -1])
                for i in range(hetero_size)] # for data reduce partial to dup
            ds_union_dup_split1 = hetu.DistributedStatesUnion(ds_list_dup_split1, -1 if hetero_dim != -3 else -3) # for weight with trans_b
            ds_union_dup = hetu.DistributedStatesUnion(ds_list_dup, -1 if hetero_dim != -3 else -3) # for bias
            ds_union_split0 = hetu.DistributedStatesUnion(ds_list_split0, 0 if hetero_dim != -3 else -3) # for sp data
            ds_union_split01 = hetu.DistributedStatesUnion(ds_list_split01, 0 if hetero_dim != -3 else -3) # for data split in dimension 1
            ds_union_split0_dup = hetu.DistributedStatesUnion(ds_list_split0_dup, 0 if hetero_dim != -3 else -3) # for data reduce partial to dup
            self.ds_union_map['dup_split0'].append(ds_union_dup_split0)
            self.ds_union_map['dup_split1'].append(ds_union_dup_split1)
            self.ds_union_map['dup'].append(ds_union_dup)
            self.ds_union_map['split0'].append(ds_union_split0)
            self.ds_union_map['split01'].append(ds_union_split01)
            self.ds_union_map['split0_dup'].append(ds_union_split0_dup)
            
        self.weight = hetu.parallel_parameter(eval(f'hetu.{init_method}initializer()'), 
                                              [in_features, out_features], 
                                              self.ds_union_map['dup_split0'], self.device_index, 
                                              dtype=dtype, requires_grad=True, 
                                              device_group_hierarchy=self.device_group_unions, name=f'{name}_weight')        
        if bias:
            self.bias = hetu.parallel_parameter(hetu.zeros_initializer(), [out_features], 
                                                self.ds_union_map['dup'], self.device_index,
                                                dtype=dtype, requires_grad=True, 
                                                device_group_hierarchy=self.device_group_unions, name=f'{name}_bias')
        else:
            self.bias = None

    def forward(self, input_p):
        if input_p.check_ds_hierarchy_equal(self.ds_union_map['split01']):
            tensor_split01 = input_p
        else:
            tensor_split01 = hetu.comm(input_p, self.ds_union_map['split01']) # exists src_ds == dst_ds case, just ignore it in comm_op
        
        tensor_split0_partial = hetu.linear(tensor_split01, self.weight, trans_a=False, trans_b=False, device_group_hierarchy=self.device_group_unions, name=f'linear_{self.name}')
         
        if self.sequence_parallel:
            output = hetu.comm(tensor_split0_partial, self.ds_union_map['split0']) # reduce-scatter
        else:
            output = hetu.comm(tensor_split0_partial, self.ds_union_map['split0_dup']) # allreduce   
        
        output = output + self.bias if self.bias is not None else output
        
        return output
