import json
import fcntl
from queue import Queue
from hetu.utils.parallel import config2ds

def read_with_lock(file_path):
    with open(file_path, 'r') as f:
        # 获取文件锁
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            return json.load(f)
        finally:
            # 释放文件锁
            fcntl.flock(f, fcntl.LOCK_UN)

def read_ds_parallel_config(ds_parallel_config, num_strategy):
    # read ds_parallel_config from json file
    print(f'load ds_parallel_config from: {ds_parallel_config}')
    config_paths = ds_parallel_config.split(',')
    assert len(config_paths) == num_strategy, \
        f'ds_parallel_config num should equal to num_strategy {num_strategy}'
    ds_parallel_configs = []
    for config_path in config_paths:
        ds_parallel_config = read_with_lock(config_path)
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
        ds_parallel_configs.append(ds_parallel_config)
    return ds_parallel_configs

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

def parse_multi_ds_parallel_config(ds_parallel_configs, module_name, _range=-1):
    ds_hierarchy = []
    dg_hierarchy = []
    multi_ds_parallel_config = get_multi_ds_parallel_config(ds_parallel_configs, module_name, _range)
    for ds_parallel_config in multi_ds_parallel_config:
        ds_union, dg_union = config2ds(ds_parallel_config)
        ds_hierarchy.append(ds_union)
        dg_hierarchy.append(dg_union)
    return ds_hierarchy, dg_hierarchy