import hetu
from queue import Queue

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
    assert len(multi_ds_parallel_config) == len(ds_parallel_configs), \
        f'ds_parallel_configs parse error, cannot find {module_name}' if _range == -1 else \
        f'ds_parallel_configs parse error, cannot find {module_name} with range id {_range}'   
    return multi_ds_parallel_config

def get_multi_recompute_from(ds_parallel_configs, layer_idx):
    multi_recompute_configs = []
    for ds_parallel_config in ds_parallel_configs:
        if ds_parallel_config['recompute_granularity'] is None:
            multi_recompute_configs.append([False])
        else:
            dp_recompute_configs = []
            for dp_recompute_granularity, recompute_layer_idxs in \
                zip(ds_parallel_config['recompute_granularity'], \
                    ds_parallel_config['recompute_layer_idxs_list']):
                if dp_recompute_granularity == 'selective' and layer_idx in recompute_layer_idxs:
                    dp_recompute_configs.append(True)
                else:
                    dp_recompute_configs.append(False)
            multi_recompute_configs.append(dp_recompute_configs)
    return multi_recompute_configs

def parallel_data_provider(global_data, ds_union, device_group_index, device_index):
    ds = ds_union.get_local(device_group_index)
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

def parallel_multi_data_provider(global_data, ds_unions, device_group_unions):
    multi_local_data = []
    for i in range(len(ds_unions)):
        ds_union = ds_unions[i]
        device_group_union = device_group_unions[i]
        device_group_index, device_index = get_local_index(device_group_union)
        ds = ds_union.get_local(device_group_index)
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
        multi_local_data.append(local_data)
    return multi_local_data

def get_local_index(device_group_union):
    local_device = hetu.local_device()
    device_group_index = -1
    device_index = -1
    for device_group_index, device_group in enumerate(device_group_union):
        if device_group.contains(local_device):
            device_index = device_group.get_index(local_device)
            break
    if device_group_index == len(device_group_union): # for pipeline parallel other stages
        device_group_index = -1
        device_index = -1 # only map placement group, will not map placement and do instantiate
    return device_group_index, device_index

def get_device_index(device_group):
    local_device = hetu.local_device()
    if device_group.contains(local_device):
        device_index = device_group.get_index(local_device)
    else: # for pipeline parallel other stages
        device_index = -1 # only map placement group, will not map placement and do instantiate
    return device_index

# walkaround: just give order by type(placeholder/varibale), may not include all cases
def config2ds(config):
    ds_list = []
    dg_list = []
    if config['type'] == 'placeholder':
        hetero_dim = 0
    elif config['type'] == 'variable':
        hetero_dim = -1
    else:
        raise RuntimeError(f"unsupported type {config['type']}!")   
    hetero_sum = len(config['device_group_union'])
    if hetero_sum == 1:
        hetero_dim = -3
    for hetero_num in range(hetero_sum):
        dummy_num_devices = len(config['device_group_union'][hetero_num]) * hetero_sum
        zero = False
        split = {}
        for key, value in config['split'].items():
            assert len(value) == hetero_sum, "hetero sum mismatches"
            split[int(key)] = value[hetero_num]
        assert len(config['dup']) == hetero_sum, "hetero sum mismatches"
        states = {-1: config['dup'][hetero_num], **split}
        if config['type'] == 'placeholder':
            order = sorted(split.keys()) + [-1]
        elif config['type'] == 'variable':
            order = [-1] + sorted(split.keys())
            assert 'zero' in config, f"variable config must have zero!"
            zero = config['zero']
        else:
            raise RuntimeError(f"unsupported type {config['type']}!")
        ds = hetu.DistributedStates(dummy_num_devices, states, order, zero)
        all_devices = hetu.global_device_group()
        dg = hetu.DeviceGroup([all_devices.get(device_id) for device_id in config['device_group_union'][hetero_num]])
        ds_list.append(ds)
        dg_list.append(dg)
    return hetu.DistributedStatesUnion(ds_list, hetero_dim), dg_list
