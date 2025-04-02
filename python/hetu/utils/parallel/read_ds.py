import json
import fcntl
import hetu as ht
from queue import Queue

def read_with_lock(file_path):
    """
    Read and parse a JSON file with file locking to ensure thread safety.
    
    Args:
        file_path (str): Path to the JSON file to be read
    
    Returns:
        dict: The parsed JSON content
    """
    with open(file_path, 'r') as f:
        # 获取文件锁
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            return json.load(f)
        finally:
            # 释放文件锁
            fcntl.flock(f, fcntl.LOCK_UN)

# walkaround: just give order by type(placeholder/varibale), may not include all cases
def config2ds(config):
    """
    Convert configuration dictionary to distributed states and device groups.
    
    Args:
        config (dict): Configuration dictionary with type, split, dup, and device_group_union
    
    Returns:
        tuple: A tuple containing (DistributedStatesUnion, DeviceGroupList)
    
    Raises:
        RuntimeError: If the configuration type is not supported
    """
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
        ds = ht.DistributedStates(dummy_num_devices, states, order, zero)
        all_devices = ht.global_device_group()
        dg = ht.DeviceGroup([all_devices.get(device_id) for device_id in config['device_group_union'][hetero_num]])
        ds_list.append(ds)
        dg_list.append(dg)
    return ht.DistributedStatesUnion(ds_list, hetero_dim), dg_list

def config_spread_zero(ds_parallel_config):
    """
    Spread zero configuration to all variable type configs in the configuration.
    
    Args:
        ds_parallel_config (dict): The distributed parallel configuration
    
    Returns:
        dict: The updated configuration with zero spread to all variables
    """
    zero = ds_parallel_config['zero']
    # assign zero to all variables
    config_queue = Queue()
    for value in ds_parallel_config.values():
        config_queue.put(value)
    while (not config_queue.empty()):
        config = config_queue.get()
        if type(config) == dict:
            if 'type' in config:
                if config['type'] == 'variable' and 'zero' not in config:
                    config['zero'] = zero
            else:
                for value in config.values():
                    config_queue.put(value)
    return ds_parallel_config

def read_ds_parallel_config(ds_parallel_config, num_strategy=None):
    """
    Read distributed parallel configurations from specified file paths.
    
    Args:
        ds_parallel_config (str): Comma-separated paths to config files
        num_strategy (int, optional): Expected number of strategies
    
    Returns:
        list: List of loaded and processed configuration dictionaries
    
    Raises:
        AssertionError: If number of config paths doesn't match num_strategy
    """
    print(f'load ds_parallel_config from: {ds_parallel_config}')
    config_paths = ds_parallel_config.split(',')
    if num_strategy is not None:
        assert len(config_paths) == num_strategy, \
            f'ds_parallel_config num should equal to num_strategy {num_strategy}'
    else:
        num_strategy = len(config_paths)
    ds_parallel_configs = []
    for config_path in config_paths:
        ds_parallel_config = read_with_lock(config_path)
        ds_parallel_configs.append(config_spread_zero(ds_parallel_config))
    return ds_parallel_configs

def get_multi_ds_parallel_config(ds_parallel_configs, module_name, _range=-1):
    """
    Extract multiple configurations from distributed parallel configs.
    
    Args:
        ds_parallel_configs (list): List of distributed parallel configuration dictionaries
        module_name (str): The module name to search for in configurations
        _range (int, optional): Range index for filtering configurations
    
    Returns:
        list: List of given configuration dictionaries
    
    Raises:
        AssertionError: If module with specified name/range isn't found in all configurations
    """
    multi_ds_parallel_config = []
    # 有些global属性需要一直继承到每个module内
    global_attributes = ['recompute_granularity', 'recompute_layer_idxs_list']
    for ds_parallel_config in ds_parallel_configs:
        config_queue = Queue()
        config_queue.put(ds_parallel_config)
        # print(ds_parallel_config)
        while (not config_queue.empty()):
            config = config_queue.get()
            if module_name in config:
                module_config = config[module_name]
                for global_attribute in global_attributes:
                    if global_attribute in ds_parallel_config:
                        module_config[global_attribute] = ds_parallel_config[global_attribute]
                multi_ds_parallel_config.append(module_config)
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
    """
    Get recomputation configurations for a specific layer from distributed parallel configs.
    
    Args:
        ds_parallel_configs (list): List of distributed parallel configuration dictionaries
        layer_idx (int): The layer index to check for recomputation
    
    Returns:
        list: List of recomputation configurations (booleans) for each strategy
    """
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

def parse_multi_ds_parallel_config(ds_parallel_configs, module_name, _range=-1):
    """
    Parse multiple distributed parallel configurations into distributed states and device groups.
    
    Args:
        ds_parallel_configs (list): List of distributed parallel configuration dictionaries
        module_name (str): The module name to parse configurations for
        _range (int, optional): Range index for filtering configurations
    
    Returns:
        tuple: A tuple containing (DistributedStatesHierarchy, DeviceGroupHierarchy)
    """
    ds_hierarchy = []
    dg_hierarchy = []
    multi_ds_parallel_config = get_multi_ds_parallel_config(ds_parallel_configs, module_name, _range)
    for ds_parallel_config in multi_ds_parallel_config:
        ds_union, dg_union = config2ds(ds_parallel_config)
        ds_hierarchy.append(ds_union)
        dg_hierarchy.append(dg_union)
    return ds_hierarchy, dg_hierarchy
