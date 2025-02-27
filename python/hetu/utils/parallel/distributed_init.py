import os
import socket
import hetu as ht
from hetu.rpc.kv_store import KeyValueStoreClient, ProducerConsumer

def distributed_init(ngpus, server_addr, server_port, need_kv_store=False):
    if 'HETU_LOCAL_HOSTNAME' not in os.environ:
        # 通过socket获取主机名并设置环境变量
        hostname = socket.gethostname()
        os.environ['HETU_LOCAL_HOSTNAME'] = hostname
    else:
        print(f"Environment variable 'HETU_LOCAL_HOSTNAME' already set: {os.environ['HETU_LOCAL_HOSTNAME']}")
    ht.init_comm_group(ngpus, server_address = server_addr + ":" + server_port)
    local_device = ht.local_device()
    all_devices = ht.global_device_group()
    if need_kv_store:
        kv_store_client = KeyValueStoreClient(address = server_addr + ":" + server_port)
        return local_device, all_devices, kv_store_client
    return local_device, all_devices

def parallel_data_provider(global_data, ds_union, device_group_index, device_index):
    ds = ds_union.get_local(device_group_index)
    order, states = ds.order, ds.states
    local_map = ht.map_to_local_data(ds, device_index)
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
        local_map = ht.map_to_local_data(ds, device_index)
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
    local_device = ht.local_device()
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
    local_device = ht.local_device()
    if device_group.contains(local_device):
        device_index = device_group.get_index(local_device)
    else: # for pipeline parallel other stages
        device_index = -1 # only map placement group, will not map placement and do instantiate
    return device_index