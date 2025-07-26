import os
import socket
import logging
import numpy as np
import hetu as ht
from typing import Union, Tuple
from hetu.rpc.kv_store import KeyValueStoreClient

def distributed_init(
    ngpus: int,
    server_addr: str = "127.0.0.1",
    server_port: str = "23457",
    need_kv_store: bool = False,
) -> Union[Tuple[ht.device, ht.DeviceGroup], Tuple[ht.device, ht.DeviceGroup, KeyValueStoreClient]]:
    """
    Initialize distributed communication for Hetu.
    
    Args:
        ngpus: Number of GPUs to use.
        server_addr: Address of the server for communication.
        server_port: Port of the server for communication.
        need_kv_store: Whether to initialize a key-value store client.
        
    Returns:
        If need_kv_store is False:
            Tuple of local device and global device group.
        If need_kv_store is True:
            Tuple of local device, global device group, and key-value store client.
    """
    if 'HETU_LOCAL_HOSTNAME' not in os.environ:
        # Get hostname through socket and set as environment variable
        hostname = socket.gethostname()
        os.environ['HETU_LOCAL_HOSTNAME'] = hostname
    else:
        logging.info(f"Environment variable 'HETU_LOCAL_HOSTNAME' already set: {os.environ['HETU_LOCAL_HOSTNAME']}")
    ht.init_comm_group(ngpus, server_address = server_addr + ":" + server_port)
    local_device = ht.local_device()
    all_devices = ht.global_device_group()
    if need_kv_store:
        kv_store_client = KeyValueStoreClient(address = server_addr + ":" + server_port)
        return local_device, all_devices, kv_store_client
    return local_device, all_devices

def parallel_data_provider(
    global_data: np.ndarray,
    ds_union,
    device_group_index: int,
    device_index: int,
):
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

def get_dg_from_union(device, dg_union):
    for i, dg in enumerate(dg_union):
        if dg.contains(device):
            return i, dg
    return None, None
