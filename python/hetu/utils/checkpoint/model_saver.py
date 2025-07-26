import multiprocessing.shared_memory
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import hetu
import numpy as np
import time
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
import shutil
import fnmatch

from safetensors import deserialize, safe_open, serialize, serialize_file
from collections import OrderedDict, deque
from .ht_safetensors import get_states_union, load_file, _tobytes
import grpc
import re
import csv
import pickle
import paramiko
import logging
logging.getLogger("paramiko").setLevel(logging.WARNING)
import scp
from enum import Enum
import fsspec
import io
import struct
import json
import signal

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
generated_path = os.path.join(current_directory, "rpc")
if generated_path not in sys.path:
    sys.path.append(generated_path)
from hetu.rpc import heturpc_pb2
from hetu.rpc import heturpc_pb2_grpc

WEIGHTS_NAME = 'hetu_pytorch_model'
WEIGHTS_FORMAT = '.safetensors'
TEMP_SPLITS = 32
SPLIT_DIMS = 2
TIMEOUT = 5.0
UIDLEN = 8
OVERLAP_CPU_AND_IO = True
USE_COVER_STEP = True
TMP_STEP = "step9999999"
PREVIOUS_STEP = "step9999998"
SAVING_BARRIER_VALUE = 79
HDFS_HOST = "hdfs://"
HDFS_USER = "admin"

class SAVER_DST(Enum):
    SINGLE_DISK = 0
    MULTI_DISK = 1       
    HDFS = 2
    OTHERS = 3

SHARE_DISK = False



base_step = 0

def test():
    return 0

def hdtype2size(hdtype):
    if (hdtype == hetu.bfloat16 or hdtype == hetu.float16):
        return 2
    if (hdtype == hetu.float32 or hdtype == hetu.int32):
        return 4
    if hdtype == hetu.int64:
        return 8

def hdtype2ndtype(hdtype):
    if (hdtype == hetu.bfloat16 or hdtype == hetu.float16):
        return np.float16
    if hdtype == hetu.float32:
        return np.float32
    if hdtype == hetu.int32:
        return np.int32
    if hdtype == hetu.int64:
        return np.int64

def sttype2ndtype(sttype):
    if sttype == "F16":
        return np.float16
    if sttype == "F32":
        return np.float32
    if sttype == "I32":
        return np.int32
    if sttype == "I64":
        return np.int64

def full_tensor(key):
    if key[-4:] == "step":
        return True
    if "rmsnorm" in key:
        return True
    return False

def save_file(
    tensors: Dict[str, np.ndarray],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
    param_state_dict: Dict = None,
    local_device = None,
):
    if not os.path.exists(filename):
        open(filename, "w").close()
    use_split = True
    save_st = time.time()
    if use_split:
        #TODO currently all tensors split in dim 0
        split_dim = 0
        flattened = {}
        for k, v in param_state_dict.items():
            # print(f"key:{k}, value:{tensors[k].flatten()[:10]}")
            if k[-4:] == "step":
                flattened[k] = {"dtype": tensors[k].dtype.name, 
                                "shape": tensors[k].shape, 
                                "data": _tobytes(tensors[k])}
                continue
            states_union = v['states_union']
            du = v['device_union']
            splits = TEMP_SPLITS
            for idx, dg in enumerate(du):
                if local_device.global_index in dg:
                    states = states_union[idx]
            if split_dim in states.keys():
                splits = splits // states[split_dim]
            partial_splits = TEMP_SPLITS 
            if k[-4:] == "mean" or k[-8:] == "variance":
                partial_splits = partial_splits // len(du)
                # splits = splits // len(du)
            start_idx = 0
            tidx = 0
            for idx, dg in enumerate(du):
                if local_device.global_index in dg:
                    tidx = dg.index(local_device.global_index)
                    start_idx = start_idx + (tidx * splits) % partial_splits
                    end_idx = start_idx + splits
                    splits_idxs = [i for i in range(start_idx, end_idx)]
                    break
                if (partial_splits < TEMP_SPLITS):
                    start_idx += partial_splits
                
            tensor_splits = np.split(tensors[k], splits, axis=0)
            for idx, global_idx in enumerate(splits_idxs):
                key = k + "_split_" + str(global_idx)
                flattened[key] = {"dtype": tensor_splits[idx].dtype.name, 
                                  "shape": tensor_splits[idx].shape, 
                                  "data": _tobytes(tensor_splits[idx])}
    else:
        flattened = {k: {"dtype": v.dtype.name, "shape": v.shape, "data": _tobytes(v)} for k, v in tensors.items()}
    save_ed = time.time()
    print('Flattened_Time = %.4f'%(save_ed - save_st))
    serialize_file(flattened, filename, metadata=metadata)
    save_ed = time.time()
    print('Safetensors_Save_Time = %.4f'%(save_ed - save_st))


def save_file_hdfs(
    tensors: Dict[str, np.ndarray],
    filename: Union[str, os.PathLike],
    fs,
    metadata: Optional[Dict[str, str]] = None,
    param_state_dict: Dict = None,
    local_device = None,
):
    # if not fs.exists(filename):
    #     open(filename, "w").close()
    use_split = True
    save_st = time.time()
    if use_split:
        #TODO currently all tensors split in dim 0
        split_dim = 0
        flattened = OrderedDict()
        splits_idx_keys = OrderedDict()
        for i in range(-1, TEMP_SPLITS):
            splits_idx_keys[i] = []
        for k, v in param_state_dict.items():
            states_union = v['states_union']
            du = v['device_union']
            full = v['full']

            # Tensors without ZERO3, don't split

            # print(f"key:{k}, full:{full}, is_full:{full_tensor(k)}")

            if full:
                full_k = k + "_full"
                splits_idx_keys[-1].append(full_k)
                flattened[full_k] = {"dtype": tensors[k].dtype.name, 
                                     "shape": tensors[k].shape, 
                                     "data": _tobytes(tensors[k])}
                continue

            splits = TEMP_SPLITS

            for idx, dg in enumerate(du):
                if local_device.global_index in dg:
                    states = states_union[idx]
            if split_dim in states.keys():
                splits = splits // states[split_dim]
            partial_splits = TEMP_SPLITS 
            # if k[-4:] == "mean" or k[-8:] == "variance":
            #     partial_splits = partial_splits // len(du)
            partial_splits = partial_splits // len(du)
                # splits = splits // len(du)
            start_idx = 0
            tidx = 0
            for idx, dg in enumerate(du):
                if local_device.global_index in dg:
                    tidx = dg.index(local_device.global_index)
                    start_idx = start_idx + (tidx * splits) % partial_splits
                    end_idx = start_idx + splits
                    splits_idxs = [i for i in range(start_idx, end_idx)]
                    break
                if (partial_splits < TEMP_SPLITS):
                    start_idx += partial_splits
                
            tensor_splits = np.split(tensors[k], splits, axis=0)
            for idx, global_idx in enumerate(splits_idxs):
                key = k + "_split_" + str(global_idx)
                # print(f"split_{global_idx}_append:{key}")
                splits_idx_keys[int(global_idx)].append(key)

                flattened[key] = {"dtype": tensor_splits[idx].dtype.name, 
                                  "shape": tensor_splits[idx].shape, 
                                  "data": _tobytes(tensor_splits[idx])}
    else:
        flattened = {k: {"dtype": v.dtype.name, "shape": v.shape, "data": _tobytes(v)} for k, v in tensors.items()}
    save_ed = time.time()
    print('Flattened_Time = %.4f'%(save_ed - save_st))
    # ReOrder
    reorder_st = time.time()

    ordered_params = OrderedDict()
    uid = 0
    for i in range(-1, TEMP_SPLITS):
        # print(f"len_{i}:{len(splits_idx_keys[i])}")
        for key in splits_idx_keys[i]:
            if (flattened[key]['dtype'] == 'int64'):
                uid_key = str(uid).zfill(UIDLEN) + key
                ordered_params[uid_key] = flattened[key]
                uid += 1
        for key in splits_idx_keys[i]:
            if (flattened[key]['dtype'] == 'float32'):
                uid_key = str(uid).zfill(UIDLEN) + key
                ordered_params[uid_key] = flattened[key]
                uid += 1

    reorder_ed = time.time()
    print('ReOrder_Time = %.4f'%(reorder_ed - reorder_st))
    print(f"total_keys:{len(ordered_params.keys())}")
    # exit(0)

    # Serialize
    serialize_st = time.time()
    serialize_stream = serialize(ordered_params, metadata=metadata)
    serialize_ed = time.time()
    print('Serialize_Time = %.4f'%(serialize_ed - serialize_st))

    # Save to HDFS
    to_hdfs_st = time.time()
    print(f"SAVE_FILE:{filename}")
    with fs.open(filename, 'wb') as hdfs_file:
        hdfs_file.write(serialize_stream)
    to_hdfs_ed = time.time()
    print('Save_to_HDFS_Time = %.4f'%(to_hdfs_ed - to_hdfs_st))

    save_ed = time.time()
    print('Safetensors_Save_Time = %.4f'%(save_ed - save_st))

def save_file_async_round(global_state_dict, local_device, filename,
                          archive_file, param_state_dict = None, 
                          first_used_device_index = 0,
                          ptr_dict = None, comm_dict = None):
    save_start = comm_dict['save_start']
    save_end = comm_dict['save_end']
    save_step = comm_dict['save_step']
    drop_step = comm_dict['drop_step']
    while True:
        if (save_start[0] == 0):
            time.sleep(0.5)
            continue
        save_start[0] = 0
        cur_step_dir = os.path.join(filename, "step" + str(save_step[0]))
        rm_step_dir = os.path.join(filename, "step" + str(drop_step[0]))
        cur_archive_file = os.path.join(
            cur_step_dir, archive_file
        )
        rm_archive_file = os.path.join(
            rm_step_dir, archive_file
        )
        print(f"SAVE_CKPT,"
          f"save_start:{save_start[0]}, "
          f"save_end:{save_end[0]}, "
          f"save_step:{save_step[0]}, "
          f"drop_step:{drop_step[0]}\n"
          f"filename:{filename},"
          f"archive_file:{archive_file}\n"
          f"cur_step_dir:{cur_step_dir}\n"
          f"rm_step_dir:{rm_step_dir}\n"
          f"cur_archive_file:{cur_archive_file}\n"
          f"rm_archive_file:{rm_archive_file}\n")
        st = time.time()
        device_index = local_device.global_index
        print(f"device_index:{device_index}")
        from multiprocessing import shared_memory
        shm_name = "/hetu_shared_memory" + str(device_index)
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        print(f"fsize:{existing_shm.size}")
        shm_dict = {}
        save_dict = {}
        for k, v in ptr_dict.items():
            try:
                bytes = v['size']
                offset = v['offset']
                shape = v['shape']
                dtype = hdtype2ndtype(v['dtype'])
                
                if offset + bytes > existing_shm.size:
                    raise ValueError("Requested data exceeds the bounds of the shared memory.")

                shm_dict[k] = np.ndarray(shape=shape, dtype=dtype, 
                                         buffer=existing_shm.buf[offset:offset + bytes])
                save_dict[k] = shm_dict[k]
                # print(f"key:{k}\nnd_f:{shm_dict[k].flatten()[:10]}"
                #     f"\nnd_t:{v['nd_t'].flatten()[:10]}")
            except:
                raise EOFError("Share memory error.")

        # 使用完毕后关闭共享内存
        existing_shm.close()
        ed = time.time()
        print(f"read_share_mem_time:{ed - st}")
        # for k, v in shm_dict.items():
        #     print(f"key:{k}, in_global_dict:{k in global_state_dict.keys()}")
        save_file(save_dict, cur_archive_file, metadata=None,
                  param_state_dict=param_state_dict, local_device=local_device)
        st_time = time.time()
        if os.path.exists(rm_step_dir):
            try:
                # shutil.rmtree(rm_step_dir)
                print(f"REMOVE:{rm_archive_file}.")
                os.remove(rm_archive_file)
                # print(f"REMOVE:{rm_archive_file}.")
            except OSError as e:
                print(f"Error: {e.strerror}, filename:{rm_archive_file}")
        ed_time = time.time()
        print(local_device, 'Remove_Checkpoint_Time = %.4f'%(ed_time - st_time))
        if local_device.global_index == first_used_device_index:
            # waiting for every rank remove its own checkpoint
            print(local_device, 'Waiting_Checkpoint_Time = %.4f'%(ed_time - st_time))
            # all checkpoints removed, the first device will remove the dir
            retry_times = 0
            while(os.path.exists(rm_step_dir)):
                try:
                    shutil.rmtree(rm_step_dir)
                    print(f"The folder {rm_step_dir} has been deleted.")
                except OSError as e:
                    print(f"Error: {e.strerror}, filename:{rm_step_dir},"
                            f"retry_times:{retry_times}.")
                time.sleep(0.5)
                retry_times += 1
            # only we remove the old dir, we renew the step.
            step_filename = filename + "/step.txt"
            with open(step_filename, "w") as step_file:
                step_file.write(str(save_step[0]))

        save_end[0] = 1

def save_file_async_hdfs(global_state_dict, local_device, filename,
                         archive_file, param_state_dict = None, 
                         first_used_device_index = 0,
                         ptr_dict = None, comm_dict = None, fs_type = SAVER_DST.SINGLE_DISK,
                         previous_copies = [], server_path = None, json_dict = {}):
    save_start = comm_dict['save_start']
    save_end = comm_dict['save_end']
    save_step = comm_dict['save_step']
    drop_step = comm_dict['drop_step']
    consumed_samples = comm_dict['consumed_samples']
    existing_shm = None
    save_disk_thread = None
    while True:
        if (save_start[0] == 0):
            time.sleep(0.5)
            continue
        save_start[0] = 0
        cur_step_dir = os.path.join(filename, "step" + str(save_step[0]))
        rm_step_dir = os.path.join(filename, "step" + str(drop_step[0]))
        cur_archive_file = os.path.join(
            cur_step_dir, archive_file
        )
        rm_archive_file = os.path.join(
            rm_step_dir, archive_file
        )
        print(f"SAVE_CKPT,"
          f"save_start:{save_start[0]}, "
          f"save_end:{save_end[0]}, "
          f"save_step:{save_step[0]}, "
          f"drop_step:{drop_step[0]}\n"
          f"filename:{filename}, "
          f"archive_file:{archive_file}\n"
          f"cur_step_dir:{cur_step_dir}\n"
          f"rm_step_dir:{rm_step_dir}\n"
          f"cur_archive_file:{cur_archive_file}\n"
          f"rm_archive_file:{rm_archive_file}\n")
        st = time.time()
        device_index = local_device.global_index
        from multiprocessing import shared_memory
        if existing_shm is None:
            shm_name = "/hetu_shared_memory" + str(device_index)
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            def cleanup(signum, frame):
                if existing_shm is not None:
                    existing_shm.close()
                print("Closing shared memory")
                exit(0)
            signal.signal(signal.SIGTERM, cleanup)
        print(f"device_index:{device_index}")
        print(f"fsize:{existing_shm.size}")
        shm_dict = {}
        save_dict = {}
        for k, v in ptr_dict.items():
            try:
                bytes = v['size']
                offset = v['offset']
                shape = v['shape']
                dtype = hdtype2ndtype(v['dtype'])
                
                if offset + bytes > existing_shm.size:
                    raise ValueError("Requested data exceeds the bounds of the shared memory.")

                shm_dict[k] = np.ndarray(shape=shape, dtype=dtype, 
                                         buffer=existing_shm.buf[offset:offset + bytes])
                # save_dict[k] = np.copy(shm_dict[k])
                save_dict[k] = shm_dict[k]
                # print(f"key:{k}\nnd_f:{shm_dict[k].flatten()[:10]}")
            except:
                raise EOFError("Share memory error.")

        ed = time.time()
        tensors = save_dict
        print(f"read_share_mem_time:{ed - st}")
        # for k, v in shm_dict.items():
        #     print(f"key:{k}, in_global_dict:{k in global_state_dict.keys()}")
        # save_file_hdfs(save_dict, cur_archive_file, fs, metadata=None,
        #                param_state_dict=param_state_dict, local_device=local_device)

        # currently we remove save_file_hdfs to the main function
        use_split = True
        save_st = time.time()
        if use_split:
            #TODO currently all tensors split in dim 0
            split_dim = 0
            flattened = OrderedDict()
            splits_idx_keys = OrderedDict()
            for i in range(-1, TEMP_SPLITS):
                splits_idx_keys[i] = []
            for k, v in param_state_dict.items():
                states_union = v['states_union']
                du = v['device_union']
                full = v['full']

                # Tensors without ZERO3, don't split
                if full:
                    full_k = k + "_full"
                    splits_idx_keys[-1].append(full_k)
                    flattened[full_k] = {"dtype": tensors[k].dtype.name, 
                                        "shape": tensors[k].shape, 
                                        "data": _tobytes(tensors[k])}
                    continue

                splits = TEMP_SPLITS

                for idx, dg in enumerate(du):
                    if local_device.global_index in dg:
                        states = states_union[idx]
                if split_dim in states.keys():
                    splits = splits // states[split_dim]
                partial_splits = TEMP_SPLITS 
                partial_splits = partial_splits // len(du)
                # splits = splits // len(du)
                start_idx = 0
                tidx = 0
                for idx, dg in enumerate(du):
                    if local_device.global_index in dg:
                        tidx = dg.index(local_device.global_index)
                        start_idx = start_idx + (tidx * splits) % partial_splits
                        end_idx = start_idx + splits
                        splits_idxs = [i for i in range(start_idx, end_idx)]
                        break
                    if (partial_splits < TEMP_SPLITS):
                        start_idx += partial_splits
                    
                # print(f"key:{k}, states:{states}, splits:{splits}, split_idxs:{splits_idxs}\ndu:{du}"
                #       f", partial_splits:{partial_splits}, start_idx:{start_idx}, lendu:{len(du)}, full:{full}")
                tensor_splits = np.split(tensors[k], splits, axis=0)
                for idx, global_idx in enumerate(splits_idxs):
                    key = k + "_split_" + str(global_idx)
                    # print(f"split_{global_idx}_append:{key}")
                    splits_idx_keys[int(global_idx)].append(key)

                    flattened[key] = {"dtype": tensor_splits[idx].dtype.name, 
                                    "shape": tensor_splits[idx].shape, 
                                    "data": _tobytes(tensor_splits[idx])}
        else:
            flattened = {k: {"dtype": v.dtype.name, "shape": v.shape, "data": _tobytes(v)} for k, v in tensors.items()}
        save_ed = time.time()
        print('Flattened_Time = %.4f'%(save_ed - save_st))
        # ReOrder
        reorder_st = time.time()

        ordered_params = OrderedDict()
        uid = 0
        for i in range(-1, TEMP_SPLITS):
            # print(f"len_{i}:{len(splits_idx_keys[i])}")
            for key in splits_idx_keys[i]:
                if (flattened[key]['dtype'] == 'int64'):
                    uid_key = str(uid).zfill(UIDLEN) + key
                    ordered_params[uid_key] = flattened[key]
                    uid += 1
            for key in splits_idx_keys[i]:
                if (flattened[key]['dtype'] == 'float32'):
                    uid_key = str(uid).zfill(UIDLEN) + key
                    ordered_params[uid_key] = flattened[key]
                    uid += 1

        reorder_ed = time.time()
        print('ReOrder_Time = %.4f'%(reorder_ed - reorder_st))
        print(f"total_keys:{len(ordered_params.keys())}")

        # Serialize
        serialize_st = time.time()
        serialize_stream = serialize(ordered_params, metadata=None)
        serialize_ed = time.time()
        print('Serialize_Time = %.4f'%(serialize_ed - serialize_st))

        if OVERLAP_CPU_AND_IO:
            while((save_disk_thread is not None) and (save_disk_thread.is_alive())):
                time.sleep(0.5)

        # copy params to thread
        st = time.time()
        cur_serialize_stream = serialize_stream
        cur_serialize_stream_f = cur_serialize_stream
        dir_name_f = filename
        cur_step_dir_f = cur_step_dir
        cur_archive_file_f = cur_archive_file
        rm_step_dir_f = rm_step_dir
        rm_archive_file_f = rm_archive_file 
        save_step_f = save_step[0]
        rm_step_f = drop_step[0]
        consumed_samples_f = consumed_samples[0]
        ed = time.time()
        print('Copy_Param_To_Thread_Time = %.4f'%(ed - st))

        # save_file_hdfs(save_dict, cur_archive_file, metadata=None,
        #                param_state_dict=param_state_dict, local_device=local_device)

        def saving(fs_type, stream, dir_name_t, cur_step_dir_t, filename_t, rm_step_dir_t, rm_archive_file_t, 
                   save_step_t, rm_step_t, consumed_samples_t, previous_copies_t, server_path_t, json_dict_t):
            # Save to HDFS
            file_system = None
            if fs_type == SAVER_DST.SINGLE_DISK:
                file_system = fsspec.filesystem('file')
            elif fs_type == SAVER_DST.HDFS:
                hdfs_conf = {
                    "dfs.client.socket-timeout": "300000",
                    "dfs.lease-renewer-interval-ms": "60000",
                }
                file_system = fsspec.filesystem('hdfs', host=HDFS_HOST, 
                                                user=HDFS_USER, extra_conf=hdfs_conf)
            if USE_COVER_STEP:
                channel = None
                stub = None
                st_time = time.time()
                for i in range(10):
                    try:
                        channel = grpc.insecure_channel(server_path_t)
                        stub = heturpc_pb2_grpc.DeviceControllerStub(channel)
                        print(f"connect to:{server_path_t}")
                        break
                    except:
                        print(f"error")
                        channel = None
                        stub = None
                if channel is None:
                    print(f"No connection error")
                    exit(0)
                ed_time = time.time()
                print('Get Connection Time = %.4f'%(ed_time - st_time))
                tmp_file = re.sub(r'step[0-9]+', TMP_STEP, filename_t)
                tmp_dir = re.sub(r'step[0-9]+', TMP_STEP, cur_step_dir_t)
                to_hdfs_st = time.time()
                print(f"SAVE_FILE:{tmp_file}, SAVE_DIR:{tmp_dir}")
                if local_device.global_index == first_used_device_index:
                    if file_system.exists(tmp_dir):
                        print(f"Directory exists: {tmp_dir}")
                    else:
                        file_system.mkdir(tmp_dir)
                        print(f"Directory does not exist: {tmp_dir}")
                stub.Consistent(heturpc_pb2.ConsistentRequest(rank=0, 
                                                              value=int(SAVING_BARRIER_VALUE),
                                                              world_rank=[]))
                for k, v in json_dict.items():
                    print(f"Write Json:{k}")
                    v["consumed_samples"] = consumed_samples_t
                    with file_system.open(k, 'w', encoding='utf-8') as file_obj:
                        json.dump(v, file_obj, ensure_ascii=False)
                print(f"Write to:{tmp_file}")
                with file_system.open(tmp_file, 'wb') as hdfs_file:
                    hdfs_file.write(stream)
                print(f"Write to:{tmp_file} finished.")
                stub.Consistent(heturpc_pb2.ConsistentRequest(rank=0, 
                                                              value=int(SAVING_BARRIER_VALUE),
                                                              world_rank=[]))
                to_hdfs_ed = time.time()
                print('Save_to_File_Sys_Time = %.4f'%(to_hdfs_ed - to_hdfs_st))
                
                st_time = time.time()
                if local_device.global_index == first_used_device_index:
                    if file_system.exists(cur_step_dir_t):
                        print(f"Destination directory already exists, removing it: {cur_step_dir_t}")
                    file_system.mv(tmp_dir, cur_step_dir_t, recursive=True)
                    print(f"MV {tmp_dir} TO {cur_step_dir_t}.")
                    if file_system.exists(rm_step_dir_t):
                        try:
                            # shutil.rmtree(rm_step_dir)
                            if file_system.exists(tmp_dir):
                                print(f"Destination directory already exists, removing it: {tmp_dir}")
                            print(f"MV {rm_step_dir_t} TO {tmp_dir}.")
                            file_system.mv(rm_step_dir_t, tmp_dir, recursive=True)
                            # print(f"REMOVE:{rm_archive_file}.")
                        except OSError as e:
                            print(f"Error: {e.strerror}, filename:{rm_archive_file_t}")

                        if (rm_step_t in previous_copies_t):
                            file_system.rm(tmp_dir, recursive=True)
                            file_system.mkdir(tmp_dir)
                            print(f"Exist in Previous Step {rm_step_t}, DEL {tmp_dir}.")
                ed_time = time.time()
                print(local_device, 'Remove_Checkpoint_Time = %.4f'%(ed_time - st_time))
                print(f"first_used_device_index:{first_used_device_index}")
                if local_device.global_index == first_used_device_index:
                    # waiting for every rank remove its own checkpoint
                    print(local_device, 'Waiting_Checkpoint_Time = %.4f'%(ed_time - st_time))
                    # all checkpoints removed, the first device will remove the dir
                    retry_times = 0
                    
                    # while(file_system.exists(rm_step_dir_t)):
                    #     try:
                    #         file_system.rm(rm_step_dir_t, recursive=True)
                    #         print(f"The folder {rm_step_dir_t} has been deleted.")
                    #     except OSError as e:
                    #         print(f"Error: {e.strerror}, filename:{rm_step_dir_t},"
                    #                 f"retry_times:{retry_times}.")
                    #     time.sleep(0.5)
                    #     retry_times += 1
                    # only we remove the old dir, we renew the step.
                    step_filename = dir_name_t + "/step.txt"
                    with file_system.open(step_filename, 'w') as step_file:
                        step_file.write(str(save_step_t))
                    print(f"write_step:{str(save_step_t)}")
                stub.Consistent(heturpc_pb2.ConsistentRequest(rank=0, 
                                                              value=int(SAVING_BARRIER_VALUE),
                                                              world_rank=[]))
                if channel is not None:
                    stub = None
                    channel.close()
                    channel = None
                
            else:
                to_hdfs_st = time.time()
                print(f"SAVE_FILE:{filename_t}")
                with file_system.open(filename_t, 'wb') as hdfs_file:
                    hdfs_file.write(stream)
                to_hdfs_ed = time.time()
                print('Save_to_File_Sys_Time = %.4f'%(to_hdfs_ed - to_hdfs_st))
                
                st_time = time.time()
                if file_system.exists(rm_step_dir_t):
                    try:
                        # shutil.rmtree(rm_step_dir)
                        print(f"REMOVE:{rm_archive_file_t}.")
                        file_system.rm(rm_archive_file_t)
                        # print(f"REMOVE:{rm_archive_file}.")
                    except OSError as e:
                        print(f"Error: {e.strerror}, filename:{rm_archive_file_t}")
                ed_time = time.time()
                print(local_device, 'Remove_Checkpoint_Time = %.4f'%(ed_time - st_time))
                print(f"first_used_device_index:{first_used_device_index}")
                if local_device.global_index == first_used_device_index:
                    # waiting for every rank remove its own checkpoint
                    print(local_device, 'Waiting_Checkpoint_Time = %.4f'%(ed_time - st_time))
                    # all checkpoints removed, the first device will remove the dir
                    retry_times = 0
                    while(file_system.exists(rm_step_dir_t)):
                        try:
                            file_system.rm(rm_step_dir_t, recursive=True)
                            print(f"The folder {rm_step_dir_t} has been deleted.")
                        except OSError as e:
                            print(f"Error: {e.strerror}, filename:{rm_step_dir_t},"
                                    f"retry_times:{retry_times}.")
                        time.sleep(0.5)
                        retry_times += 1
                    # only we remove the old dir, we renew the step.
                    step_filename = dir_name_t + "/step.txt"
                    with file_system.open(step_filename, 'w') as step_file:
                        step_file.write(str(save_step_t))
                    print(f"write_step:{str(save_step_t)}")

            file_system.invalidate_cache()
            if hasattr(file_system, 'close'):
                file_system.close()
        
        subthread_st = time.time()
    
        if (not OVERLAP_CPU_AND_IO):
            saving(fs_type, cur_serialize_stream_f, dir_name_f, cur_step_dir_f, cur_archive_file_f, 
                   rm_step_dir_f, rm_archive_file_f, save_step_f, rm_step_f, consumed_samples_f, 
                   previous_copies, server_path, json_dict)
        else:
            if fs_type == SAVER_DST.SINGLE_DISK:
                save_disk_thread = threading.Thread(target=saving,
                                                args=(fs_type, cur_serialize_stream_f, dir_name_f, cur_step_dir_f, cur_archive_file_f, 
                                                      rm_step_dir_f, rm_archive_file_f, save_step_f, rm_step_f, consumed_samples_f, 
                                                      previous_copies, server_path, json_dict))
            elif fs_type == SAVER_DST.HDFS:
                save_disk_thread = threading.Thread(target=saving,
                                                    args=(fs_type, cur_serialize_stream_f, dir_name_f, 
                                                            cur_step_dir_f, cur_archive_file_f, 
                                                            rm_step_dir_f, rm_archive_file_f, 
                                                            save_step_f, rm_step_f, consumed_samples_f,
                                                            previous_copies, server_path, json_dict))
            save_disk_thread.start()
            # save_disk_thread.join()
            
        # save_disk_thread.start()
        # if not OVERLAP_CPU_AND_IO:
        #     save_disk_thread.join()
        subthread_ed = time.time()
        print('\nSubThread_Time = %.4f'%(subthread_ed - subthread_st))
        save_ed = time.time()
        print('Safetensors_Save_Time = %.4f'%(save_ed - save_st))
        save_end[0] = 1

# distributed save without share disk
def save_file_async_dist(global_state_dict, local_device, filename,
                         archive_file, param_state_dict = None, 
                         first_used_device_index = 0,
                         ptr_dict = None, comm_dict = None,
                         node_idx = 0, nodes = None, json_files = []):
    save_start = comm_dict['save_start']
    save_end = comm_dict['save_end']
    save_step = comm_dict['save_step']
    drop_step = comm_dict['drop_step']
    while True:
        if (save_start[0] == 0):
            time.sleep(0.5)
            continue
        save_start[0] = 0
        cur_step_dir = os.path.join(filename, "step" + str(save_step[0]))
        rm_step_dir = os.path.join(filename, "step" + str(drop_step[0]))
        cur_archive_file = os.path.join(
            cur_step_dir, archive_file
        )
        rm_archive_file = os.path.join(
            rm_step_dir, archive_file
        )
        print(f"SAVE_CKPT,"
          f"save_start:{save_start[0]}, "
          f"save_end:{save_end[0]}, "
          f"save_step:{save_step[0]}, "
          f"drop_step:{drop_step[0]}\n"
          f"filename:{filename},"
          f"archive_file:{archive_file}\n"
          f"cur_step_dir:{cur_step_dir}\n"
          f"rm_step_dir:{rm_step_dir}\n"
          f"cur_archive_file:{cur_archive_file}\n"
          f"rm_archive_file:{rm_archive_file}\n")
        st = time.time()
        device_index = local_device.global_index
        print(f"device_index:{device_index}")
        from multiprocessing import shared_memory
        shm_name = "/hetu_shared_memory" + str(device_index)
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        print(f"fsize:{existing_shm.size}")
        shm_dict = {}
        save_dict = {}
        for k, v in ptr_dict.items():
            try:
                bytes = v['size']
                offset = v['offset']
                shape = v['shape']
                dtype = hdtype2ndtype(v['dtype'])
                
                if offset + bytes > existing_shm.size:
                    raise ValueError("Requested data exceeds the bounds of the shared memory.")

                shm_dict[k] = np.ndarray(shape=shape, dtype=dtype, 
                                         buffer=existing_shm.buf[offset:offset + bytes])
                save_dict[k] = np.copy(shm_dict[k])
                # print(f"key:{k}\nnd_f:{shm_dict[k].flatten()[:10]}"
                #     f"\nnd_t:{v['nd_t'].flatten()[:10]}")
            except:
                raise EOFError("Share memory error.")

        # 使用完毕后关闭共享内存
        existing_shm.close()
        ed = time.time()
        print(f"read_share_mem_time:{ed - st}")
        # for k, v in shm_dict.items():
        #     print(f"key:{k}, in_global_dict:{k in global_state_dict.keys()}")
        save_file(save_dict, cur_archive_file, metadata=None,
                  param_state_dict=param_state_dict, local_device=local_device)
        

        # we need do a ring scp in group
        num_nodes = len(nodes)
        current_directory = os.getcwd()
        dst_archive_file = os.path.join(current_directory, cur_archive_file)
        scp_st = time.time()
        for i in range(1, num_nodes):
            cur_dst_idx = (node_idx + i) % num_nodes
            cur_dst_addr = str(nodes[cur_dst_idx])
            sys_pssh_client = paramiko.SSHClient()
            sys_pssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy()) 
            try:
                sys_pssh_client.connect(cur_dst_addr, port=22, timeout=TIMEOUT)
                print(f"SUCCESS")
                with scp.SCPClient(sys_pssh_client.get_transport()) as scp_handler:
                    for json_file in json_files:
                        src_json_file = os.path.join(cur_step_dir, json_file)
                        dst_json_file = os.path.join(current_directory, src_json_file)
                        scp_handler.put(src_json_file, dst_json_file)
                        print(f"File {dst_json_file} successfully upload to {cur_dst_addr}")
                    scp_handler.put(cur_archive_file, dst_archive_file)
                    print(f"File {dst_archive_file} successfully upload to {cur_dst_addr}")

                sys_pssh_client.close()
            except Exception as e:
                print(f"NODE:{cur_dst_addr} disconnected, error:{e}")
                sys_pssh_client.close()
                continue    
        scp_ed = time.time()
        print(local_device, 'SCP_Time = %.4f'%(scp_ed - scp_st))

        st_time = time.time()
        if os.path.exists(rm_step_dir):
            try:
                # shutil.rmtree(rm_step_dir)
                print(f"REMOVE:{rm_archive_file}.")
                os.remove(rm_archive_file)
                # print(f"REMOVE:{rm_archive_file}.")
            except OSError as e:
                print(f"Error: {e.strerror}, filename:{rm_archive_file}")
        ed_time = time.time()
        print(local_device, 'Remove_Checkpoint_Time = %.4f'%(ed_time - st_time))
        if local_device.global_index == first_used_device_index:
            # waiting for every rank remove its own checkpoint
            print(local_device, 'Waiting_Checkpoint_Time = %.4f'%(ed_time - st_time))
            # all checkpoints removed, the first device will remove the dir
            retry_times = 0
            while(os.path.exists(rm_step_dir)):
                try:
                    shutil.rmtree(rm_step_dir)
                    print(f"The folder {rm_step_dir} has been deleted.")
                except OSError as e:
                    print(f"Error: {e.strerror}, filename:{rm_step_dir},"
                            f"retry_times:{retry_times}.")
                time.sleep(0.5)
                retry_times += 1
            # only we remove the old dir, we renew the step.
            step_filename = filename + "/step.txt"
            with open(step_filename, "w") as step_file:
                step_file.write(str(save_step[0]))

        save_end[0] = 1

class ModelSaver:
    def __init__(self, save_copies=2, config=None, 
                 local_device=None, 
                 all_devices=None,
                 save_dtype=hetu.float32,
                 base_step = 0,
                 additional_args = None,
                 min_interval_time = 20,
                 first_used_device_index = 0):
        # save copies (int) : the copies of checkpoint save, recommend to set >= 2
        self.save_copies = save_copies
        self.config = config
        self.local_device = local_device
        self.save_dtype = save_dtype
        self.additional_args = additional_args
        self.channel = None
        self.stub = None
        self.pid = None
        self.world_ranks = None
        self.base_step = base_step
        self.save_thread = None
        self.min_interval_time = min_interval_time
        self.last_save_time = 0
        self.first_used_device_index = first_used_device_index
        self.cur_copies = deque()
        self.shared_memory = OrderedDict()
        self.queue = None
        self.rm_ckpt = None
        self.save_ckpt = None
        self.comm_dict = OrderedDict()
        self.comm_dict['save_start'] = multiprocessing.Array("i", [0], lock=True)
        self.comm_dict['save_step'] = multiprocessing.Array("i", [0], lock=True)
        self.comm_dict['drop_step'] = multiprocessing.Array("i", [0], lock=True)
        self.comm_dict['save_end'] = multiprocessing.Array("i", [1], lock=True)
        self.comm_dict['consumed_samples'] = multiprocessing.Array("i", [0], lock=True)
        self.node_idx = additional_args.node_idx
        self.nodes = additional_args.nodes[1:-1].split(',')
        self.saver_dst = SAVER_DST.SINGLE_DISK
        self.previous_copies = []

        # set first used device index
        if all_devices is not None:
            if self.saver_dst == SAVER_DST.SINGLE_DISK:
                for i in range(all_devices.num_devices):
                    cur_device = all_devices.get(i)
                    if cur_device.local:
                        self.set_first_used_device_index(cur_device.global_index)
                        print(f"first_used_device_index:{cur_device.global_index}")
                        break
            elif self.saver_dst == SAVER_DST.HDFS:
                self.fs = fsspec.filesystem('hdfs', host=HDFS_HOST, user=HDFS_USER)
                min_global_index = 2 ** 32
                for i in range(all_devices.num_devices):
                    cur_device = all_devices.get(i)
                    if cur_device.is_cuda:
                        if cur_device.global_index < min_global_index:
                            min_global_index = cur_device.global_index
                self.set_first_used_device_index(min_global_index)
                print(f"first_used_device_index:{min_global_index}")
        # self.thread_pool = ProcessPoolExecutor(max_workers=1)
        self.connect()
    
    def __del__(self):
        if self.channel is not None:
            self.stub = None
            self.channel.close()
            self.channel = None
        if self.save_thread is not None:
            self.save_thread.terminate()
        if self.save_ckpt is not None:
            self.save_ckpt.terminate()
        print("save_thread killed.")

    def connect(self):
        st = time.time()
        server_path = self.additional_args.server_addr + ":" + \
                      self.additional_args.server_port
        pid = self.local_device.global_index
        self.pid = pid
        self.world_ranks = []
        for i in range(self.additional_args.global_ngpus):
            self.world_ranks.append(i)
        print(f"pid:{pid}, server_path:{server_path}")
        self.channel = grpc.insecure_channel(server_path)
        self.stub = heturpc_pb2_grpc.DeviceControllerStub(self.channel)
        response = self.stub.PutString(heturpc_pb2.PutStringRequest(key="SAVER", value=str(pid)))
        ed = time.time()
    
    def disconnect(self):
        if self.channel is not None:
            self.stub = None
            self.channel.close()
            self.channel = None
    
    def set_first_used_device_index(self, index):
        self.first_used_device_index = index
    
    def if_need_save(self, step):
        # if (self.save_thread is not None) and (not self.save_thread.is_alive()):
        #     self.save_thread = None
        # can_save = self.save_thread is None
        can_save = False
        if self.rm_ckpt is not None:
            if self.rm_ckpt.is_alive():
                can_save = False
            else:
                self.rm_ckpt = None
                can_save = bool(self.comm_dict['save_end'][0])
        else:
            if (self.save_ckpt is not None) and (not self.save_ckpt.is_alive()):
                exit(0)
            can_save = bool(self.comm_dict['save_end'][0])
        # print(f"Step:{step}, Can_Save:{can_save}")
        response = self.stub.Consistent(heturpc_pb2.ConsistentRequest(rank=self.pid, 
                                                                      value=int(can_save),
                                                                      world_rank=self.world_ranks))
        interval = time.time() - self.last_save_time
        need_save = (response.status) and (interval > self.min_interval_time) and \
                    (self.base_step + step >= 0)
        print(f"Step:{step}, Can_Save:{can_save}, Base step:{self.base_step}, Need_Save:{need_save}, Interval:{interval}")
        return need_save

    def save(self, model, optimizer, output_dir, step, save_step=0, consumed_samples=0):
        # self.stub.Barrier(heturpc_pb2.BarrierRequest(rank=self.pid, world_rank=self.world_ranks))
        need_save = self.if_need_save(step)
        if need_save:
            drop_step = -1
            if len(self.cur_copies) >= self.save_copies:
                drop_step = self.cur_copies[0]
                self.cur_copies.popleft()
            self.cur_copies.append(self.base_step + step)
            self.last_save_time = time.time()
            print("SAVE_BY_TRAINING:", self.local_device, "CONSUMED:", consumed_samples, 
                  "STEP:", self.base_step + step, "DropSTEP:", drop_step, "copies:", self.cur_copies)

            save_st = time.time()
            self.temp_save_split(model, optimizer, output_dir, 
                                 self.base_step + step, consumed_samples, 
                                 drop_step=drop_step,
                                 config=self.config, 
                                 local_device=self.local_device, 
                                 save_dtype=self.save_dtype)

            save_ed = time.time()
            print(f"cur_step:{step}, save_time:{save_ed - save_st}")
        return need_save

    
    def save_step_info_to_csv(self, csv_name, step, consumed_samples, loss):
        data = [str(step), str(consumed_samples), str(loss)]
        if (step == 0):
            with open(csv_name, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)
        else:
            with open(csv_name, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)

    def temp_save_split(
        self, model: hetu.nn.Module, optimizer, filename: str, 
        step = 0, consumed_samples = 0, drop_step = -1, config = None, 
        local_device = None, save_dtype = None,
        force_contiguous: bool = False,
        only_lora: bool = False,
        metadata: Optional[Dict[str, str]] = None
    ):
        total_st = time.time()
        save_st = time.time()
        json_files = []
        try:
            if self.saver_dst == SAVER_DST.HDFS:
                fs = self.fs
                fs.mkdir(filename, parents=True, exist_ok=True)
            elif self.saver_dst == SAVER_DST.SINGLE_DISK:
                fs = fsspec.filesystem('file')
                fs.mkdir(filename, parents=True, exist_ok=True)
            else:
                os.makedirs(filename, exist_ok=True)  # 如果目录已经存在，不会抛出异常
            print(f"Directories created: {filename}")
        except Exception as e:
            print(f"Error creating directories: {filename}")
        with hetu.graph("define_and_run"):

            cur_step_dir = os.path.join(filename, "step" + str(step))
            rm_step_dir = os.path.join(filename, "step" + str(drop_step))
            self.comm_dict['save_step'][0] = step
            self.comm_dict['drop_step'][0] = drop_step
            # if local_device.index == 0:
            if local_device.global_index == self.first_used_device_index:
                if self.saver_dst == SAVER_DST.HDFS or self.saver_dst == SAVER_DST.SINGLE_DISK:
                    if not fs.exists(cur_step_dir) and (not USE_COVER_STEP):
                        fs.mkdir(cur_step_dir, parents=True, exist_ok=True)
                else:
                    if not os.path.exists(cur_step_dir):
                        os.mkdir(cur_step_dir)
            global_state_dict = OrderedDict()
            param_state_dict = OrderedDict()
            all_device_groups = []
            all_device_group_unions = []
            state_dict = model.state_dict()
            ht_state_dict = model.state_dict(format='hetu')
            total_sum = len(state_dict.items())
            cur_sum = 0
            visit_tensors = set()
            ds_json = {}
            
            all_devices = hetu.global_device_group()

            for key, data in state_dict.items():
                # print(f"key:{key}, in_dict:{key in ht_state_dict.keys()}")
                # absmax need recompute
                if "absmax" in key:
                    continue
                # only_lora mode only save loraA and loraB 
                if only_lora and ("lora" not in key):
                    continue
                if (not only_lora) and ("lora" in key):
                    continue
                param = ht_state_dict[key]
                device_group_union = param.get_device_group_union()
                # print(type(device_group_union), device_group_union)
                device_group = device_group_union[0]
                # device_group = param.get_device_group()
                    
                if device_group not in all_device_groups:
                    all_device_groups.append(device_group)
                
                if device_group_union not in all_device_group_unions:
                    all_device_group_unions.append(device_group_union)
                
                # TODO: implement allgather_inter_group_param()
                
                # if not device_group.contains(local_device):
                #     continue
                
                local_device_in_union = False
                for single_device_group in device_group_union:
                    if single_device_group.contains(local_device):
                        local_device_in_union = True
                        break
                    
                if local_device_in_union == False:
                    continue
                
                global_value = param
                # shared tensors only compute once
                if param.id in visit_tensors:
                    continue
                visit_tensors.add(param.id)
                opt_state_dict = optimizer.get_states(param)
                global_state_dict[key] = data
                state = {}
                state["device_num"] = param.distributed_states.device_num
                state["order"] = param.distributed_states.order
                state["states"] = param.distributed_states.states
                state["states_union"] = get_states_union(param)
                if full_tensor(key):
                    state["full"] = True
                else:
                    state["full"] = False
                device_index = []
                device_index_union = []

                for single_device_group in param.get_device_group_union():
                    dg_num_devices = single_device_group.num_devices
                    device_index_union.append([])
                    for i in range(dg_num_devices):
                        device_index.append(all_devices.get_index(single_device_group.get(i)))
                        device_index_union[-1].append(all_devices.get_index(single_device_group.get(i)))
                state["device_group"] = device_index
                state["device_union"] = device_index_union
                ds_json_key = str(param.get_device_group_union())
                if ds_json_key not in ds_json.keys():
                    ds_json[ds_json_key] = {}
                ds_json[ds_json_key][key] = state
                param_state_dict[key] = state
                for k, state_param in opt_state_dict.items():
                    # if (k == "step"):
                    #     continue
                    state_key = key + "_" + k
                    state_data = state_param.get_data()
                    # global_value, abs_max = data_transform_for_store(state_key, state_param, state_data, config, 
                    #                                                  save_dtype, ht_state_dict, local_device)
                    # global_state_dict[state_key] = global_value.numpy(force=True, save=True)
                    global_state_dict[state_key] = state_data
                    state = {}
                    state["device_num"] = state_param.distributed_states.device_num
                    state["order"] = state_param.distributed_states.order
                    state["states"] = state_param.distributed_states.states
                    state["states_union"] = get_states_union(state_param)
                    if full_tensor(state_key):
                        state["full"] = True
                    else:
                        state["full"] = False
                    device_index = []
                    device_index_union = []
                    # for i in range(state["device_num"]):
                    #     device_index.append(all_devices.get_index(state_param.device_group.get(i)))
                    for single_device_group in state_param.get_device_group_union():
                        dg_num_devices = single_device_group.num_devices
                        device_index_union.append([])
                        for i in range(dg_num_devices):
                            device_index.append(all_devices.get_index(single_device_group.get(i)))
                            device_index_union[-1].append(all_devices.get_index(single_device_group.get(i)))
                    state["device_group"] = device_index
                    state["device_union"] = device_index_union
                    if ds_json_key not in ds_json.keys():
                        ds_json[ds_json_key] = {}
                    ds_json[ds_json_key][state_key] = state
                    param_state_dict[state_key] = state
                cur_sum += 1

            save_ed = time.time()
            print(local_device, 'Data_Transfer_Time = %.4f'%(save_ed - save_st))
            
            save_st = time.time()

            if force_contiguous:
                state_dict = {k: v.contiguous() for k, v in state_dict.items()}

            json_dict = {}
            
            for i, m_device_union in enumerate(all_device_group_unions):
                # print(f"Union{i}:{m_device_union}")
                need_save_json = False
                if len(m_device_union) > 0 and \
                m_device_union[0].contains(local_device) and \
                m_device_union[0].get_index(local_device) == 0:
                    need_save_json = True
                    
                if need_save_json:
                    if USE_COVER_STEP:
                        json_file = "param_states" + f'-{i + 1}-of-{len(all_device_group_unions)}' + ".json"
                        json_files.append(json_file)
                        tmp_dir = re.sub(r'step[0-9]+', TMP_STEP, cur_step_dir)
                        json_file = os.path.join(
                            tmp_dir, json_file
                        )

                        print("SAVE JSON:", local_device, json_file)
                        try:
                            import json
                            ds_json_for_dump = ds_json[str(m_device_union)]
                            # for k, v in ds_json_for_dump.items():
                            #     print(f"key:{k}, value:{v}")
                            ds_json_for_dump["consumed_samples"] = consumed_samples
                            self.comm_dict['consumed_samples'][0] = consumed_samples
                            if self.saver_dst == SAVER_DST.HDFS or self.saver_dst == SAVER_DST.SINGLE_DISK:
                                # with fs.open(json_file, 'w', encoding='utf-8') as file_obj:
                                #     json.dump(ds_json_for_dump, file_obj, ensure_ascii=False)
                                json_dict[json_file] = ds_json_for_dump
                            else:
                                file_obj=open(json_file,'w', encoding='utf-8')
                                json.dump(ds_json_for_dump, file_obj, ensure_ascii=False)
                            break
                        except ValueError as e:
                            msg = str(e)
                            msg += " Or use save_model(..., force_contiguous=True), read the docs for potential caveats."
                            raise ValueError(msg) 
                    else:
                        json_file = "param_states" + f'-{i + 1}-of-{len(all_device_group_unions)}' + ".json"
                        json_files.append(json_file)
                        json_file = os.path.join(
                            cur_step_dir, json_file
                        )
                        print("SAVE JSON:", local_device, json_file)
                        try:
                            import json
                            ds_json_for_dump = ds_json[str(m_device_union)]
                            # for k, v in ds_json_for_dump.items():
                            #     print(f"key:{k}, value:{v}")
                            ds_json_for_dump["consumed_samples"] = consumed_samples
                            if self.saver_dst == SAVER_DST.HDFS or self.saver_dst == SAVER_DST.SINGLE_DISK:
                                with fs.open(json_file, 'w', encoding='utf-8') as file_obj:
                                    json.dump(ds_json_for_dump, file_obj, ensure_ascii=False)
                            else:
                                file_obj=open(json_file,'w', encoding='utf-8')
                                json.dump(ds_json_for_dump, file_obj, ensure_ascii=False)
                            break
                        except ValueError as e:
                            msg = str(e)
                            msg += " Or use save_model(..., force_contiguous=True), read the docs for potential caveats."
                            raise ValueError(msg) 
            
            save_ed = time.time()
            print(local_device, 'SAVE_JSON_Time = %.4f'%(save_ed - save_st))
            
                    
            # print("all_devices:", all_devices)
            assert(all_devices.contains(local_device))
            num_devices = all_devices.num_devices
            d_index = all_devices.get_index(local_device)
            archive_file = WEIGHTS_NAME + f'-{d_index + 1}-of-{num_devices}' + WEIGHTS_FORMAT
            try:
                if self.save_ckpt is None:
                    save_st = time.time()
                    dict_for_trans = OrderedDict()
                    # for k, v in global_state_dict.items():
                    #     dict_for_trans[k] = v
                    #     print(f"key:{k}, ptr:{v.data_ptr}")
                    ptr_dict = OrderedDict()
                    def trans_to_ptr_dict(tensor_key, tensor):
                        nd = tensor.data
                        # nd2 = v.data
                        fsize = hdtype2size(nd.dtype)
                        for dim in nd.shape:
                            fsize *= dim
                        fsize = ((fsize - 1) // 16 + 1) * 16
                        # print(f"trans_to_ptr_dict:{tensor_key}")
                        print(f"key:{tensor_key}")
                        out = {"ptr": nd.data_ptr, "size": fsize,
                            "shape": nd.shape,
                            "index": local_device.global_index, 
                            "stream": hetu.stream(local_device, 8).ptr,
                            "offset":nd.shm_offset,
                            "dtype":nd.dtype}
                        return out
                    for k, v in ht_state_dict.items():
                        if k not in global_state_dict.keys():
                            continue
                        ptr_dict[k] = trans_to_ptr_dict(k, v)
                        opt_state_dict = optimizer.get_states(v)
                        for state_k, state_param in opt_state_dict.items():
                            state_key = k + "_" + state_k
                            # state_data = state_param.get_data()
                            ptr_dict[state_key] = trans_to_ptr_dict(state_key, state_param)
                    save_ed = time.time()
                    print(local_device, 'Trans_Time = %.4f'%(save_ed - save_st))
                    save_st = time.time()
                    flag = False
                    print(f"Current start method: {multiprocessing.get_start_method()}")
                    self.comm_dict['save_start'][0] = 1
                    self.comm_dict['save_end'][0] = 0
                    self.disconnect()
                    if (len(self.nodes) == 0):                    
                        self.save_ckpt = multiprocessing.Process(target=save_file_async_round, 
                                                                args=(dict_for_trans, local_device, filename,
                                                                      archive_file, param_state_dict, 
                                                                      self.first_used_device_index,
                                                                      ptr_dict, self.comm_dict))
                    else:
                        if self.saver_dst == SAVER_DST.HDFS or self.saver_dst == SAVER_DST.SINGLE_DISK:
                            server_path = self.additional_args.server_addr + ":" + self.additional_args.server_port
                            self.save_ckpt = multiprocessing.Process(target=save_file_async_hdfs, 
                                                                     args=(dict_for_trans, local_device, filename,
                                                                           archive_file, param_state_dict, 
                                                                           self.first_used_device_index,
                                                                           ptr_dict, self.comm_dict,
                                                                           self.saver_dst, self.previous_copies,
                                                                           server_path, json_dict))
                        else:
                            self.save_ckpt = multiprocessing.Process(target=save_file_async_dist, 
                                                                    args=(dict_for_trans, local_device, filename,
                                                                          archive_file, param_state_dict, 
                                                                          self.first_used_device_index,
                                                                          ptr_dict, self.comm_dict,
                                                                          self.node_idx, self.nodes, json_files))
                    self.save_ckpt.start()
                    save_mid = time.time()
                    self.connect()
                    print(local_device, 'Task_Add_Time = %.4f'%(save_mid - save_st))
                    del global_state_dict
                    del state_dict
                    del ht_state_dict
                else:
                    self.comm_dict['save_start'][0] = 1
                    self.comm_dict['save_end'][0] = 0
            
            except ValueError as e:
                msg = str(e)
                msg += " Or use save_model(..., force_contiguous=True), read the docs for potential caveats."
                raise ValueError(msg)
        save_ed = time.time()
        print(local_device, 'Pure_Save_Time = %.4f'%(save_ed - save_st))
        total_ed = time.time()
        print(local_device, 'Total_Save_Time = %.4f'%(total_ed - total_st))

    def rm_addtional_checkpoint(self, step_dirs, cur_step, filename, local_device) -> None:
        """
        Remove additional checkpoints.
        """
        if self.saver_dst == SAVER_DST.HDFS:
            fs = fsspec.filesystem('hdfs', host=HDFS_HOST, user=HDFS_USER)
        elif self.saver_dst == SAVER_DST.SINGLE_DISK:
            fs = fsspec.filesystem('file')
        rm_st = time.time()
        print(f"step_dirs:{step_dirs}")
        # delete incomplete ckpt
        for step_num in reversed(step_dirs):
            if step_num > cur_step:
                rm_step_dir = os.path.join(filename, "step" + str(step_num))
                try:
                    if self.saver_dst == SAVER_DST.HDFS or self.saver_dst == SAVER_DST.SINGLE_DISK:
                        fs.rm(rm_step_dir, recursive=True)
                    else:
                        shutil.rmtree(rm_step_dir)
                    print(f"The folder {rm_step_dir} has been deleted.")
                except OSError as e:
                    print(f"Error: {e.strerror}, {rm_step_dir}")
                step_dirs.pop()
            else:
                break

        for idx, step_num in enumerate(step_dirs):
            if len(step_dirs) - idx > self.save_copies:
                # remove additional checkpoints:
                rm_step_dir = os.path.join(filename, "step" + str(step_num))
                try:
                    if self.saver_dst == SAVER_DST.HDFS:
                        fs.rm(rm_step_dir, recursive=True)
                    else:
                        shutil.rmtree(rm_step_dir)
                    print(f"The folder {rm_step_dir} has been deleted.")
                except OSError as e:
                    print(f"Error: {e.strerror}, {rm_step_dir}")
            else:
                self.cur_copies.append(step_num)
        print(f"cur_copies:{self.cur_copies}")
        rm_ed = time.time()
        print(local_device, 'Rm_Additional_Time = %.4f'%(rm_ed - rm_st))

    def temp_load_split_fs(self, model: hetu.nn.Module, optimizer, 
                        filename: Union[str, os.PathLike],
                        config=None, local_device=None) -> Tuple[List[str], List[str]]:
        save_st = time.time()
        if self.saver_dst == SAVER_DST.SINGLE_DISK:
            # single disk
            fs = fsspec.filesystem('file')
        elif self.saver_dst == SAVER_DST.HDFS:
            fs = fsspec.filesystem('hdfs', host=HDFS_HOST, user=HDFS_USER)

        step_filename = filename + "/step.txt"
        if not fs.exists(step_filename):
            print("No checkpoint. Train with initialize.")
            self.base_step = 0
            return model, 0
        with fs.open(step_filename, "r") as step_file:
            cur_step = int(step_file.readline())
        self.base_step = cur_step + 1
        cdirs = fs.ls(filename)
        step_dirs = []
        for cdir in cdirs:
            if "txt" in cdir:
                continue
            # step_dirs.append(int(cdir.replace("step", "")))
            step_dirs.append(int(cdir.split("step")[-1]))
        step_dirs.sort()

        self.rm_ckpt = multiprocessing.Process(target=self.rm_addtional_checkpoint, 
                                                args=(step_dirs, cur_step, filename, local_device))
        self.rm_ckpt.start()

        # delete incomplete ckpt
        for step_num in reversed(step_dirs):
            if step_num > cur_step:
                rm_step_dir = os.path.join(filename, "step" + str(step_num))
                step_dirs.pop()
            else:
                break

        for idx, step_num in enumerate(step_dirs):
            if len(step_dirs) - idx > self.save_copies:
                # remove additional checkpoints:
                rm_step_dir = os.path.join(filename, "step" + str(step_num))
            else:
                self.cur_copies.append(step_num)
        
        self.previous_copies = self.cur_copies
        print(f"previous copies:{self.previous_copies}")

        # choose the current model checkpoint dir
        filename = os.path.join(filename, "step" + str(cur_step))
        print("BASE_STEP:", self.base_step)
        
        local_state_dict = model.state_dict(format='hetu')
        hetu_state_dict = model.state_dict(format='hetu')
        all_device_groups = []
        all_device_group_unions = []
        parameter_to_dtype = {}
        trans_strategy = {}
        state_dict = {}
        all_devices = hetu.global_device_group()
        
        save_ed = time.time()
        print(local_device, 'Get_Params_Time = %.4f'%(save_ed - save_st))

        ds_json = []
        d_index = local_device.global_index 
        num_devices = 0
        file_list = fs.ls(filename)
        print(f"ls_of_{filename}:{file_list}")
        ds_files = []
        archive_files = []
        local_hostname = hetu.device.get_local_hostname()
        for file in file_list:
            if ("param_states" in file and ".json" in file):
                ds_files.append(file)
            if (WEIGHTS_NAME in file and WEIGHTS_FORMAT in file):
                archive_files.append(file)
        # print("GYGYGYGY:", local_hostname, " ", ds_files)
        # print("DAFILE:", ds_files)
        consumed_samples = 0
        for i, json_file in enumerate(ds_files):
            json_file = os.path.join(
                filename, json_file
            )
            try:
                import json
                if self.saver_dst == SAVER_DST.HDFS:
                    file_obj = fs.open(json_file, 'r', encoding='utf-8')
                else:
                    file_obj=open(json_file,'r', encoding='utf-8')
                python_data=json.load(file_obj)
                if i == 0:
                    consumed_samples = python_data['consumed_samples']
                # exit(0)
                ds_json.append(python_data)
                file_obj.close()
                # if d_index == 0:
                #     print(json_file, "\n", python_data)
        
            except ValueError as e:
                msg = str(e)
                msg += " Or use save_model(..., force_contiguous=True), read the docs for potential caveats."
                raise ValueError(msg)

        need_switch = False
        param_state_dict = OrderedDict()
        for k in local_state_dict:
            param = local_state_dict[k]
            device_group_union = param.get_device_group_union()
            # device_group = param.get_device_group()
            device_group = device_group_union[0]
            if device_group not in all_device_groups:
                all_device_groups.append(device_group)
            if device_group_union not in all_device_group_unions:
                all_device_group_unions.append(device_group_union)
            # TODO: implement allgather_inter_group_param()
            # if not device_group.contains(local_device):
            #     continue
            local_device_in_union = False
            for single_device_group in device_group_union:
                if single_device_group.contains(local_device):
                    local_device_in_union = True
                    break
                
            if local_device_in_union == False:
                continue
            
            for i in range(len(ds_json)):
                if k in ds_json[i]:                
                    trans_strategy[k] = ds_json[i][k]
                    pre_states = {}
                    for sk, sv in trans_strategy[k]["states"].items():
                        pre_states[int(sk)] = sv
                    trans_strategy[k]["states"] = pre_states
                    # device_index = []
                    # for i in range(param.distributed_states.device_num):
                    #     device_index.append(all_devices.get_index(param.device_group.get(i)))
                    device_index = []
                    device_index_union = []
                    for single_device_group in param.get_device_group_union():
                        dg_num_devices = single_device_group.num_devices
                        device_index_union.append([])
                        for i in range(dg_num_devices):
                            device_index.append(all_devices.get_index(single_device_group.get(i)))
                            device_index_union[-1].append(all_devices.get_index(single_device_group.get(i)))
                    
                    if (param.distributed_states.device_num == trans_strategy[k]["device_num"] and
                        param.distributed_states.order == trans_strategy[k]["order"] and
                        param.distributed_states.states == trans_strategy[k]["states"] and
                        # get_states_union(param) == trans_strategy[k]["states_union"] and
                        device_index == trans_strategy[k]["device_group"] and
                        device_index_union == trans_strategy[k]["device_union"]):
                        pass
                    else:
                        need_switch = True
                    state = {}
                    state["device_num"] = param.distributed_states.device_num
                    state["order"] = param.distributed_states.order
                    state["states"] = param.distributed_states.states
                    state["states_union"] = get_states_union(param)
                    state["device_group"] = device_index
                    state["device_union"] = device_index_union
                    break
                
            if optimizer is not None:
                opt_state_dict = optimizer.get_states(param)
                # print("Opt_State_Dict:", opt_state_dict)
                for state_k, state_param in opt_state_dict.items():
                    # if (k == "step"):
                    #     continue
                    state_key = k + "_" + state_k
                    for i in range(len(ds_json)):
                        if state_key in ds_json[i]:                
                            trans_strategy[state_key] = ds_json[i][state_key]
                            pre_states = {}
                            for sk, sv in trans_strategy[state_key]["states"].items():
                                pre_states[int(sk)] = sv
                            trans_strategy[state_key]["states"] = pre_states
                            break
                    state_key = k + "_" + state_k
                    hetu_state_dict[state_key] = state_param
                    parameter_to_dtype[state_key] = state_param.dtype        

        save_st = time.time()
        # state_dict = load_file(archive_file, parameter_to_dtype)
        need_switch = True
        print("Need Switch:", need_switch)
        loading_split_time = 0
        open_time = 0
        if need_switch:
            archive_opens = {}
            archive_opens_keys = {}
            ptr = 0
            def archive_sort(file_name):
                return int(str(file_name).split("-")[-3])
            print(f"before sort{archive_files}")
            archive_files.sort(key=archive_sort)
            print(f"after sort{archive_files}")

            safetensors_metadata = OrderedDict()
            # read Metadata
            meta_time = 0
            st = time.time()
            for archive in archive_files:
                # archive_opens.append(safe_open(os.path.join(
                # filename, archive), framework="np"))
                pattern = re.compile(r'\-\d+\-of')
                match = pattern.search(archive)
                print("ARC:", archive)
                if match:
                    print(f"Match found: {match.group()}")
                else:
                    print("No match found")
                ac_device_idx = match.group().replace("-","").replace("of","")
                ost = time.time()
                print(f"archive:{archive}")
                hdfs_file = fs.open(archive, 'rb')
                header_len_struct = hdfs_file.read(8)
                header_len = struct.unpack('Q', header_len_struct)[0]
                print(f"header_len:{header_len}")
                # 读取JSON元数据
                header_data = hdfs_file.read(header_len)
                metadata = json.loads(header_data.decode('utf-8'))
                # for k, v in metadata.items():
                #     print(f"{k}:{v}")
                nums = 0
                pure_read_time = 0
                read_st = time.time()
                cache_dict = OrderedDict()
                partials = 0
                for k, v in metadata.items():
                    origin_key = k[8:]
                    if origin_key not in safetensors_metadata:
                        safetensors_metadata[origin_key] = {'file_idx': int(ac_device_idx) - 1,
                                                            'idx_key': k, 
                                                            'dtype': sttype2ndtype(v['dtype']), 
                                                            'shape': v['shape'], 
                                                            'offset': v['data_offsets'], 
                                                            'size': v['data_offsets'][1] - v['data_offsets'][0]}
                read_ed = time.time()

                fread_st = time.time()
                # hdfs_file.seek(0, 0)
                # data1 = hdfs_file.read()
                fread_ed = time.time()

                print(f"ftime:{fread_ed - fread_st}, Meta_Read_Time:{read_ed - read_st}, ptime:{pure_read_time}")

                # hdfs_file.seek(0, 0)
                archive_opens[int(ac_device_idx) - 1] = hdfs_file
                oed = time.time()
                open_time += (oed - ost)
                ptr += 1
            ed = time.time()
            meta_time = ed - st
            print(f"Meta_Time:{meta_time}")
            print(f"num_items:{len(safetensors_metadata.items())}")
            
            split1_sum = 0
            split2_sum = 0
            json_time = 0
            load_time = 0
            concat_time = 0
            split_time = 0

            file_uid_dict = OrderedDict()
            for i in archive_opens.keys():
                file_uid_dict[i] = {'uids':[], 'split_keys':{}, 'ori_keys':{}}

            for k in hetu_state_dict:
                st_time = time.time()
                param = hetu_state_dict[k]
                parameter_to_dtype[k] = param.dtype
                # device_group = param.get_device_group()
                # if device_group not in all_device_groups:
                #     all_device_groups.append(device_group)
                device_group_union = param.get_device_group_union()
                device_group = device_group_union[0]
                if device_group not in all_device_groups:
                    all_device_groups.append(device_group)
                if device_group_union not in all_device_group_unions:
                    all_device_group_unions.append(device_group_union)
                # if not device_group.contains(local_device):
                #     continue
                local_device_in_union = False
                for single_device_group in device_group_union:
                    if single_device_group.contains(local_device):
                        local_device_in_union = True
                        break
                
                if local_device_in_union == False:
                    continue
                
                shared_param = False
                if k not in trans_strategy.keys():
                    print("Shared:", param)
                    continue 
                device_num = trans_strategy[k]["device_num"]
                order = trans_strategy[k]["order"]
                states = trans_strategy[k]["states"]
                states_union = trans_strategy[k]["states_union"]
                device_index = trans_strategy[k]["device_group"]
                device_index_union = trans_strategy[k]["device_union"]
                # actually just dim -1, 
                tds = None
                for i, cur_device_group in enumerate(device_group_union):
                    if cur_device_group.contains(local_device):
                        dg_idx = i
                        tds_list = param.ds_hierarchy[0].ds_list
                        tds = tds_list[dg_idx]
                        break
                num_data_splits = TEMP_SPLITS
                param_order = tds.order
                param_states = tds.states
                start_idx = 0
                tidx = 0
                partial_splits = TEMP_SPLITS
                # if k[-4:] == "mean" or k[-8:] == "variance":
                #     partial_splits = partial_splits // len(device_group_union)
                if not full_tensor(k):
                    partial_splits = partial_splits // len(device_group_union)
                for dim in param_order:
                    if dim >= 0:
                        num_data_splits = num_data_splits // param_states[dim]
                split_idxs = None
                for dg in device_group_union:
                    if dg.contains(local_device):
                        tidx = dg.get_index(local_device)
                        start_idx = start_idx + (tidx * num_data_splits) % partial_splits
                        end_idx = start_idx + num_data_splits
                        split_idxs = [i for i in range(start_idx, end_idx)]
                        break
                    if (partial_splits < TEMP_SPLITS):
                        start_idx += partial_splits
                # print("GGG:", k, device_index, num_data_splits, archive_files)
                ptr = 0
                split_numpys = []
                file_idxs = []
                local_idxs = {}
                if full_tensor(k):
                    num_data_splits = 1
                    full_key = k + "_full"
                    split_idxs = [0]
                    split_numpys.append(safetensors_metadata[full_key])
                    file_idxs.append(safetensors_metadata[full_key]['file_idx'])
                    local_idxs[full_key] = 0
                    uid = int(safetensors_metadata[full_key]['idx_key'][:8])
                    file_uid_dict[file_idxs[-1]]['uids'].append(uid)
                    file_uid_dict[file_idxs[-1]]['split_keys'][uid] = full_key
                    file_uid_dict[file_idxs[-1]]['ori_keys'][uid] = k
                else:
                    for ptr in split_idxs:
                        tmp_key = k + "_split_" + str(ptr)
                        split_numpys.append(safetensors_metadata[tmp_key])
                        file_idxs.append(safetensors_metadata[tmp_key]['file_idx'])
                        local_idxs[tmp_key] = len(file_idxs) - 1
                        uid = int(safetensors_metadata[tmp_key]['idx_key'][:8])
                        # print(f"file_idxs[-1]:{file_idxs[-1]}, file_uid_dict:{file_uid_dict.keys()}")
                        file_uid_dict[file_idxs[-1]]['uids'].append(uid)
                        file_uid_dict[file_idxs[-1]]['split_keys'][uid] = tmp_key
                        file_uid_dict[file_idxs[-1]]['ori_keys'][uid] = k
                # print(f"key:{k}, split_idxs:{split_idxs}, tlen:{len(split_numpys)}, "
                #       f"param_shape:{param.shape}, gshape:{param.global_shape}, ds:{tds}, "
                #       f"dsl:{tds_list}, du:{device_group_union}")
                ed_time = time.time()
                json_time += (ed_time - st_time)
                # print(f"outshape:{out.shape}")
                tmp_info = OrderedDict()
                tmp_info['split_idxs'] = split_idxs
                tmp_info['split_numpys'] = split_numpys
                tmp_info['file_idxs'] = file_idxs
                tmp_info['local_idxs'] = local_idxs
                state_dict[k] = tmp_info
                ed_time = time.time()
                concat_time += (ed_time - st_time)
                st_time = time.time()


            construct_file_uid_dict_time = 0
            st = time.time()
            
            for i in archive_opens.keys():
                file_uid_dict[i]['uids'].sort()
                # file_uid_dict[i]['idx_keys'].sort()
                uids = file_uid_dict[i]['uids']
                pre_ptr = -1
                start_ptr = -1
                file_uid_dict[i]['uid_blocks'] = []
                for j in range(len(uids)):
                    if uids[j] != pre_ptr:
                        if start_ptr != -1:
                            file_uid_dict[i]['uid_blocks'].append([start_ptr, pre_ptr])
                        start_ptr = uids[j]
                        pre_ptr = uids[j] + 1
                    else:
                        pre_ptr = uids[j] + 1
                if start_ptr!= -1:
                    file_uid_dict[i]['uid_blocks'].append([start_ptr, pre_ptr])
                
                print(f"file_{i}, idxs:{file_uid_dict[i]['uids']}")
                print(f"file_{i}, uid_blocks:{file_uid_dict[i]['uid_blocks']}")
                print(f"file_{i}, split_keys:{file_uid_dict[i]['split_keys']}")
            cp_time = 0
            for i in archive_opens.keys():
                hdfs_file = archive_opens[i]
                print(hdfs_file.tell())
                for block in file_uid_dict[i]['uid_blocks']:
                    print(file_uid_dict[i]['split_keys'][block[0]])
                    start_key = file_uid_dict[i]['split_keys'][block[0]]
                    end_key = file_uid_dict[i]['split_keys'][block[1] - 1]
                    offset = safetensors_metadata[start_key]['offset'][0]
                    full_size = safetensors_metadata[end_key]['offset'][1] - offset
                    print(f"file_{i}, block:{block}, offset:{offset}, full_size:{full_size}")
                    print(f"block_st:{safetensors_metadata[start_key]}")
                    print(f"block_ed:{safetensors_metadata[end_key]}")
                    hdfs_file.seek(offset, 1)
                    pure_read_st = time.time()
                    data = hdfs_file.read(full_size)
                    full_data = memoryview(data)
                    print(f"buffer_length:{len(full_data)}")
                    for uid in range(block[0], block[1]):
                        dst_ori_key = file_uid_dict[i]['ori_keys'][uid]
                        dst_split_key = file_uid_dict[i]['split_keys'][uid]
                        dst_idx_key = safetensors_metadata[dst_split_key]['idx_key']
                        dst_dtype = safetensors_metadata[dst_split_key]['dtype']
                        dst_shape = safetensors_metadata[dst_split_key]['shape']
                        dst_start = safetensors_metadata[dst_split_key]['offset'][0] - offset
                        dst_end = safetensors_metadata[dst_split_key]['offset'][1] - offset
                        local_numpy_idx = state_dict[dst_ori_key]['local_idxs'][dst_split_key]
                        cp_st = time.time()
                        fs_data = np.frombuffer(full_data[dst_start:dst_end], dtype=dst_dtype).reshape(dst_shape)
                        # fs_data.flags.writeable = True
                        state_dict[dst_ori_key]['split_numpys'][local_numpy_idx] = fs_data
                        cp_ed = time.time()
                        cp_time += (cp_ed - cp_st)
                        # print(f"put {dst_idx_key} to {dst_ori_key}, local_numpy_idx:{local_numpy_idx}"
                        #       f"start:{dst_start}, end:{dst_end}")
                    pure_read_ed = time.time()
                    pure_read_time += (pure_read_ed - pure_read_st)
                    hdfs_file.seek(-(offset + full_size), 1)
                archive_opens[i].close()

            ed = time.time()
            construct_file_uid_dict_time += (ed - st)
            print(local_device, 'CP_Time = %.4f'%(cp_time))
            print(local_device, 'Construct_Uid_Dict_Time = %.4f'%(construct_file_uid_dict_time))
        else:
            for k in hetu_state_dict:
                param = hetu_state_dict[k]
                parameter_to_dtype[k] = param.dtype  
            d_index = all_devices.get_index(local_device)
            num_devices = all_devices.num_devices
            archive_file = WEIGHTS_NAME + f'-{d_index + 1}-of-{num_devices}' + WEIGHTS_FORMAT
            archive_file = os.path.join(
                filename, archive_file
            )
            print("ARC:", archive_file)
            state_dict = load_file(archive_file, parameter_to_dtype)
        save_ed = time.time()
        print(local_device, 'Opening_archives_Time = %.4f'%(open_time))
        print(local_device, 'Loading_splits_Time = %.4f'%(loading_split_time))
        print(local_device, 'Json_Time = %.4f'%(json_time))
        print(local_device, 'Load_Time = %.4f'%(load_time))
        print(local_device, 'Concat_Time = %.4f'%(concat_time))
        print(local_device, 'Split_Time = %.4f'%(split_time))
        load_params_time = save_ed - save_st
        print(local_device, 'Load_Params_Time = %.4f'%(load_params_time))

        
        
        save_st = time.time()
        total_mem = 0
        for k in hetu_state_dict:
            if k in state_dict:
                split_idxs = state_dict[k]['split_idxs']
                split_numpys = state_dict[k]['split_numpys']
                hetu_state_dict[k].reset_data_from_splits(split_numpys)
                for split_numpy in split_numpys:
                    total_mem += split_numpy.nbytes

        save_ed = time.time()
        print(local_device, 'Reset_Params_Time = %.4f'%(save_ed - save_st))
        return model, consumed_samples