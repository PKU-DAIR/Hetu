import hetu
from .module import Module
import numbers

__all__ = [
    'HtColumnParallelLinear', 
    'HtRowParallelLinear', 
    'HtParallelEmbedding',
    'HtVocabParallelEmbedding',
    'HtParallelLayerNorm',
    'HtParallelRMSNorm',
]

def parallel_data_provider(global_data, ds, device_index):
    order, states = ds.order, ds.states
    local_map = hetu.map_to_local_data(ds, device_index)
    local_data = global_data.copy()
    for dim in order:
        if dim < 0:
            continue
        splits = states[dim]
        split_index = local_map[dim]
        start = int(split_index * (global_data.shape[dim] / splits))
        stop = min(int((split_index + 1) * (global_data.shape[dim] / splits)), global_data.shape[dim])
        local_data = local_data.take(range(start, stop), axis=dim)
    return local_data

def get_device_index(device_group):
    local_device = hetu.local_device()
    if device_group.contains(local_device):
        device_index = device_group.get_index(local_device)
    else: # for pipeline parallel other stages
        device_index = -1 # only map placement group, will not map placement and do instantiate
    return device_index

# walkaround: just give order by type(placeholder/varibale), may not include all cases
def config2ds(config):
    num_devices = len(config['device_group'])
    zero = False
    split = {}
    for key, value in config['split'].items():
        split[int(key)] = value
    states = {-1: config['dup'], **split}
    if config['type'] == 'placeholder':
        order = sorted(split.keys()) + [-1]
    elif config['type'] == 'variable':
        order = [-1] + sorted(split.keys())
        assert 'zero' in config, f"variable config must have zero!"
        zero = config['zero']
    else:
        raise RuntimeError(f"unsupported type {config['type']}!")
    ds = hetu.DistributedStates(num_devices, states, order, zero)
    
    all_devices = hetu.global_device_group()
    device_group = hetu.DeviceGroup([all_devices.get(device_id) for device_id in config['device_group']])
    return ds, device_group

class HtParallelRMSNorm(Module):
    def __init__(self, normalized_shape, ds_parallel_config, sequence_parallel=False, dtype=hetu.float32, name='rmsnorm'):
        super(HtParallelRMSNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = [normalized_shape]  # type: ignore[assignment]
        self.normalized_shape = list(normalized_shape)  # type: ignore[arg-type]
        self.sequence_parallel = sequence_parallel
        self.name = name
        ds, self.device_group = config2ds(ds_parallel_config)
        device_index = get_device_index(self.device_group)
        # self.sp_size = ds_parallel_config['sp'] # equal to tp?
        self.ds_split0 = hetu.DistributedStates(ds.device_num, {0: ds.device_num}, [0], False) # for activation, no zero
        self.weight = hetu.parallel_parameter(eval(f'hetu.ones_initializer()'), 
                                              self.normalized_shape, ds, device_index, 
                                              dtype=dtype, requires_grad=True, 
                                              device_group=self.device_group, name=f'{name}_weight')

    def forward(self, input_p):
        # [bsz*seq_len, hidden_size]
        if self.sequence_parallel:
            ds_input = input_p.distributed_states
            assert ds_input.check_equal(self.ds_split0), \
                'for sequence parallel, layernorm need input fully sharded in dimension 0!'
            # do sequence parallel layernorm: [bsz*seq_len // tp, hidden_size]
            #print(f'in sp, rmsnorm input shape = {input_p.shape}')
            output_rms = hetu.rms_norm(input_p, None, self.weight, None, is_rms_norm=True, \
                                       device_group=self.device_group, name=self.name+'_sp')[0]
            # allgather will be auto done in later column parallel
        else:
            # [bsz*seq_len, hidden_size]
            #print(f'in nosp, rmsnorm input shape = {input_p.shape}')
            output_rms = hetu.rms_norm(input_p, None, self.weight, None, is_rms_norm=True, \
                                       device_group=self.device_group, name=self.name)[0]
        return output_rms

class HtParallelLayerNorm(Module):
    def __init__(self, normalized_shape, ds_parallel_config, sequence_parallel=False, eps=1e-5, 
                 dtype=hetu.float32, name='ln', ori_model = None):
        super(HtParallelLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = [normalized_shape]  # type: ignore[assignment]
        self.normalized_shape = list(normalized_shape)  # type: ignore[arg-type]
        self.sequence_parallel = sequence_parallel
        self.eps = eps
        self.name = name
        self.ds_parallel_config = ds_parallel_config
        self.dtype = dtype
        ds, self.device_group = config2ds(ds_parallel_config)
        device_index = get_device_index(self.device_group)
        self.ds_split0 = hetu.DistributedStates(ds.device_num, {0: ds.device_num}, [0], False) # for activation, no zero
        if ori_model is None:
            self.weight = hetu.parallel_parameter(eval(f'hetu.ones_initializer()'), 
                                                self.normalized_shape, ds, device_index, 
                                                dtype=dtype, requires_grad=True, 
                                                device_group=self.device_group, name=f'{name}_weight')
            self.bias = hetu.parallel_parameter(eval(f'hetu.zeros_initializer()'), 
                                                self.normalized_shape, ds, device_index, 
                                                dtype=dtype, requires_grad=True, 
                                                device_group=self.device_group, name=f'{name}_bias')
        else:
            self.weight = ori_model.weight
            self.bias = ori_model.bias

    def forward(self, input_p):
        # [bsz*seq_len, hidden_size]
        if self.sequence_parallel:
            ds_input = input_p.distributed_states
            assert ds_input.check_equal(self.ds_split0), \
                'for sequence parallel, layernorm need input fully sharded in dimension 0!'
            # do sequence parallel layernorm: [bsz*seq_len // tp, hidden_size]
            #print(f'in sp, ln input shape = {input_p.shape}')
            output_ln = hetu.fused_layernorm(input_p, self.weight, self.bias, self.normalized_shape, \
                                             self.eps, device_group=self.device_group, name=self.name+'_sp')[0]
            # allgather will be auto done in later column parallel
        else:
            # [bsz*seq_len, hidden_size]
            #print(f'in nosp, ln input shape = {input_p.shape}')
            output_ln = hetu.fused_layernorm(input_p, self.weight, self.bias, self.normalized_shape, \
                                             self.eps, device_group=self.device_group, name=self.name)[0]
        return output_ln

class HtParallelEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim, ds_parallel_config, init_method='xavier_normal_', 
                 dtype=hetu.float32, name='embedding', ori_model = None):
        super(HtParallelEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.ds_parallel_config = ds_parallel_config
        self.dtype = dtype
        
        ds, self.device_group = config2ds(ds_parallel_config)
        self.name = name
        device_index = get_device_index(self.device_group)
        # embedding_table should not be splited in any dimension!
        if ori_model is None:
            self.embedding_table = hetu.parallel_parameter(eval(f'hetu.{init_method}initializer()'), 
                                                        [num_embeddings, embedding_dim], ds, device_index, 
                                                        dtype=dtype, requires_grad=True, 
                                                        device_group=self.device_group, name=f'{name}_table')
        else:
            self.embedding_table = ori_model.embedding_table
    
    def forward(self, input_p):
        return hetu.embedding_lookup(self.embedding_table, input_p, device_group=self.device_group, name=self.name)
    
class HtVocabParallelEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim, ds_parallel_config, init_method='xavier_normal_', 
                 dtype=hetu.float32, name='vocab_embedding', ori_model = None):
        super(HtVocabParallelEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.name = name
        self.ds_parallel_config = ds_parallel_config
        self.dtype = dtype
        ds_dup_split0, self.device_group = config2ds(ds_parallel_config) # for embedding table
        dp, tp, num_devices = ds_parallel_config['dup'], ds_parallel_config['split'].get('0', 1), len(ds_parallel_config['device_group'])
        assert dp * tp == num_devices, f'VocabParallelEmbedding get wrong ds_parallel_config: {ds_parallel_config}!'
        device_index = get_device_index(self.device_group)
        ds_split0_dup = hetu.DistributedStates(num_devices, {-1: tp, 0: dp}, [0, -1]) # for data
        ds_dup = hetu.DistributedStates(num_devices, {-1: dp * tp}, [-1])
        self.ds_map = {'split0_dup': ds_split0_dup, 'dup_split0': ds_dup_split0, 'dup': ds_dup}

        dup_group_idx = ds_dup_split0.get_dup_group_index(device_index)
        self.vocab_start_index = num_embeddings // tp * dup_group_idx

        # embedding_table was splited in vocab dimension
        if ori_model is None:
            self.embedding_table = hetu.parallel_parameter(eval(f'hetu.{init_method}initializer()'), 
                                                        [num_embeddings, embedding_dim], ds_dup_split0, device_index, 
                                                        dtype=dtype, requires_grad=True, 
                                                        device_group=self.device_group, name=f'{name}_table')
        else:
            self.embedding_table = ori_model.embedding_table
    
    def forward(self, input_p):
        if input_p.distributed_states.check_equal(self.ds_map['split0_dup']):
            tensor_split0_dup = input_p
        else:
            tensor_split0_dup = hetu.comm(input_p, self.ds_map['split0_dup'])
            #print(f"warning: vocab parallel embedding need extra communication for \
            #        adapt input tensor distributed_states into {self.ds_map['split0_dup']}!")

        input_offset = tensor_split0_dup - self.vocab_start_index
        lookup_split0_partial = hetu.embedding_lookup(self.embedding_table, input_offset, device_group=self.device_group, name=self.name)
        if lookup_split0_partial.distributed_states.check_equal(self.ds_map['split0_dup']): # pure dp
            output = lookup_split0_partial
        else:
            output = hetu.comm(lookup_split0_partial, self.ds_map['split0_dup'])
        return output

class HtColumnParallelLinear(Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    """
    def __init__(self, in_features, out_features, ds_parallel_config,
                 bias=True, gather_output=True, init_method='xavier_normal_', 
                 dtype=hetu.float32, name='colp', ori_model = None):
        super(HtColumnParallelLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.name = name
        self.ds_parallel_config = ds_parallel_config
        self.dtype = dtype

        ds_dup_split1, self.device_group = config2ds(ds_parallel_config)
        dp, tp, num_devices, zero = ds_parallel_config['dup'], \
                                    ds_parallel_config['split'].get('1', 1), \
                                    len(ds_parallel_config['device_group']), \
                                    ds_parallel_config['zero']
        assert dp * tp == num_devices, f'ColumnParallelLinear get wrong ds_parallel_config: {ds_parallel_config}!'        
        device_index = get_device_index(self.device_group)
        # assume num_devices=8, there exists 4 cases: dp=1 tp=8, dp=2 tp=4, dp=4 tp=2, dp=8 tp=1
        # when dp=1 tp=8, weights: ds_dup_split0->ds_split0, data: ds_split0_dup->ds_dup
        # when dp=8 tp=1, weights: ds_dup_split0->ds_dup, data: ds_split0_dup->ds_split0
        ds_dup_split0 = hetu.DistributedStates(num_devices, {-1: dp, 0: tp}, [-1, 0], zero) # for weights with trans_b
        ds_split0_dup = hetu.DistributedStates(num_devices, {-1: tp, 0: dp}, [0, -1]) # for data
        ds_dup = hetu.DistributedStates(num_devices, {-1: dp * tp}, [-1], zero)
        self.ds_map = {'dup_split0': ds_dup_split0, 'split0_dup': ds_split0_dup, 'dup': ds_dup}
        if ori_model is None:
            self.weight = hetu.parallel_parameter(eval(f'hetu.{init_method}initializer()'), 
                                                [out_features, in_features], 
                                                ds_dup_split0, device_index, 
                                                dtype=dtype, requires_grad=True, 
                                                device_group=self.device_group, name=f'{name}_weight')
        else:
            self.weight = ori_model.weight
        if bias:
            if ori_model is None:
                self.bias = hetu.parallel_parameter(hetu.zeros_initializer(), [out_features], 
                                                    ds_dup_split0, device_index,
                                                    dtype=dtype, requires_grad=True, 
                                                    device_group=self.device_group, name=f'{name}_bias')
            else:
                self.bias = ori_model.bias
        else:
            self.bias = None
      
    def forward(self, input_p):
        if input_p.distributed_states.check_equal(self.ds_map['split0_dup']):
            tensor_split0_dup = input_p
        else: # sequence parallel: need use buffer for allgather for save activation
            tensor_split0_dup = hetu.comm(input_p, self.ds_map['split0_dup'])
            # print(f"warning: column parallel linear need extra communication for \
            #         adapt input tensor ds {input_p.distributed_states} into {self.ds_map['split0_dup']}!")
        
        #tensor_split01 = hetu.linear(tensor_split0_dup, self.weight, self.bias, trans_b=True, device_group=self.device_group, name=self.name)
        tensor_split01 = hetu.linear(tensor_split0_dup, self.weight, trans_b=True, device_group=self.device_group, name=self.name)
        if not self.gather_output:
            output = tensor_split01
        else:
            if tensor_split01.distributed_states.check_equal(self.ds_map['split0_dup']): # pure dp
                output = tensor_split01
            else:
                output = hetu.comm(tensor_split01, self.ds_map['split0_dup'])

        return output
    
# process: x->split1, w->split0 => y->partial => y->dup    
class HtRowParallelLinear(Module):
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
                 ds_parallel_config, sequence_parallel=False, bias=True, 
                 init_method='xavier_normal_', 
                 dtype=hetu.float32, name='rowp', ori_model = None):
        super(HtRowParallelLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.name = name
        self.ds_parallel_config = ds_parallel_config
        self.dtype = dtype
        self.sequence_parallel = sequence_parallel

        ds_dup_split0, self.device_group = config2ds(ds_parallel_config)
        dp, tp, num_devices, zero = ds_parallel_config['dup'], \
                                    ds_parallel_config['split'].get('0', 1), \
                                    len(ds_parallel_config['device_group']), \
                                    ds_parallel_config['zero']
        assert dp * tp == num_devices, f'RowParallelLinear get wrong ds_parallel_config: {ds_parallel_config}!'        
        device_index = get_device_index(self.device_group)
        # assume num_devices=8, there exists 4 cases: dp=1 tp=8, dp=2 tp=4, dp=4 tp=2, dp=8 tp=1
        ds_dup_split1 = hetu.DistributedStates(num_devices, {-1: dp, 1: tp}, [-1, 1], zero) # for weight with trans_b
        ds_dup = hetu.DistributedStates(num_devices, {-1: num_devices}, [-1], zero) # for bias
        ds_split01 = hetu.DistributedStates(num_devices, {0: dp, 1: tp}, [0, 1]) # for data split in dimension 1
        ds_split0_dup = hetu.DistributedStates(num_devices, {-1: tp, 0: dp}, [0, -1]) # for data reduce partial to dup
        ds_split0 = hetu.DistributedStates(num_devices, {0: num_devices}, [0]) # for sequence parallel
        self.ds_map = {'dup_split1': ds_dup_split1, 'dup': ds_dup, 'split01': ds_split01, 'split0_dup': ds_split0_dup, 'split0': ds_split0}

        if ori_model is None:
            self.weight = hetu.parallel_parameter(eval(f'hetu.{init_method}initializer()'), 
                                                [out_features, in_features], 
                                                ds_dup_split1, device_index, 
                                                dtype=dtype, requires_grad=True, 
                                                device_group=self.device_group, name=f'{name}_weight')      
        else:
            self.weight = ori_model.weight  
        if bias:
            if ori_model is None:
                self.bias = hetu.parallel_parameter(hetu.zeros_initializer(), [out_features], 
                                                    ds_dup, device_index,
                                                    dtype=dtype, requires_grad=True, 
                                                    device_group=self.device_group, name=f'{name}_bias')
            else:
                self.bias = ori_model.bias
        else:
            self.bias = None

    def forward(self, input_p):
        if input_p.distributed_states.check_equal(self.ds_map['split01']):
            tensor_split01 = input_p
        else:
            tensor_split01 = hetu.comm(input_p, self.ds_map['split01'])

        tensor_split0_partial = hetu.linear(tensor_split01, self.weight, trans_b=True, device_group=self.device_group, name=self.name)
        if tensor_split0_partial.distributed_states.check_equal(self.ds_map['split0_dup']): # pure dp
            output = tensor_split0_partial
        else:
            if self.sequence_parallel:
                output = hetu.comm(tensor_split0_partial, self.ds_map['split0']) # reduce-scatter
            else:
                output = hetu.comm(tensor_split0_partial, self.ds_map['split0_dup']) # allreduce
        output = output + self.bias if self.bias is not None else output

        return output
