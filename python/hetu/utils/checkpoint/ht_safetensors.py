from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import hetu
import numpy as np

from safetensors import deserialize, safe_open, serialize, serialize_file
from collections import OrderedDict

import shutil
import multiprocessing

WEIGHTS_NAME = 'hetu_pytorch_model'
WEIGHTS_FORMAT = '.safetensors'
TEMP_SPLITS = 8
SPLIT_DIMS = 2

base_step = 0

def get_states_union(param):
    tds_list = param.ds_hierarchy[0].ds_list
    states_union = []
    for tds in tds_list:
        states_union.append(tds.states)
    return states_union

def need_transfer(src_dtype, dst_dtype):
    if dst_dtype is None:
        return False
    if (src_dtype != hetu.float32) and \
       (src_dtype != hetu.float16) and \
       (src_dtype != hetu.bfloat16):
        return False
    if (src_dtype == dst_dtype):
        return False
    return True

def need_quantization(src_dtype, dst_dtype):
    if (not need_transfer(src_dtype, dst_dtype)):
        return False
    if (dst_dtype != hetu.float4) and (dst_dtype != hetu.nfloat4):
        return False
    return True

def data_transform_for_store(key, param, value, config, save_dtype, ht_state_dict, local_device):
    device_group = param.get_device_group()
    local_value = hetu.parallel_placeholder(param.dtype, global_shape=param.global_shape, 
                                            ds=param.distributed_states, device_group=device_group)
    feed_dict = {local_value: value}
    flag = 0
    if (param.dtype == hetu.float4) or (param.dtype == hetu.nfloat4):
        flag += 1
        key_absmax = key + "_absmax"
        weight_absmax = ht_state_dict[key_absmax]
        global_value = hetu.dequantization(local_value, weight_absmax, 
                                           config.dqtype, 64,
                                           device_group = device_group)
    else:
        global_value = local_value + 0
    key_to_model = ".".join(key.split("."))
    key_lora_A = key_to_model + ".lora_A.weight"
    key_lora_B = key_to_model + ".lora_B.weight"
    if key_lora_A in ht_state_dict:
        global_value = global_value + hetu.matmul(ht_state_dict[key_lora_A], 
                                                  ht_state_dict[key_lora_B],
                                                  device_group = device_group)
        
    if not param.distributed_states.is_pure_duplicate:
        flag += 2
        print("Param:", param, "Device:", local_device, "Dg:", param.device_group, 
              "Num:", param.distributed_states.device_num, "Order:", param.distributed_states.order, 
              "States:", param.distributed_states.states)
        num_devices = device_group.num_devices
        ds_dup = hetu.DistributedStates(num_devices, {-1: num_devices}, [-1])
        global_value = hetu.comm(global_value, ds_dup)
        global_value = global_value + 0 
    if ("qkv_dense") in key and ("lora" not in key) and ("absmax" not in key):
        flag += 4
        assert config != None, "There should be a config when using qkv_dense."
        num_heads, num_splits, hidden_size = 3, config.num_attention_heads, \
                                             config.hidden_size // config.num_attention_heads
        input_shape = global_value.shape
        saved_shape = [num_heads, num_splits, hidden_size] + input_shape[1:]
        global_value = global_value.reshape(saved_shape)
        perm = [i for i in range(len(saved_shape))]
        perm[0] = 1
        perm[1] = 0
        global_value = global_value.transpose(perm).contiguous()
        global_value = global_value.reshape(input_shape)
    if need_quantization(global_value.dtype, save_dtype):
        flag += 8
        global_values = global_value.quantization(save_dtype, 64)
        global_value = global_values[0]
        abs_max = global_values[1]
        abs_max = abs_max.to(dtype = abs_max.dtype, dev="cpu")
        global_value = global_value.to(dtype = global_value.dtype, dev="cpu")
        results = global_value.graph.run(global_value, [global_value, abs_max], 
                                         feed_dict=feed_dict, num_micro_batches=1,
                                         save_checkpoint=True)
        return results[0], results[1]
    if need_transfer(global_value.dtype, save_dtype):
        global_value = global_value.to(dtype = save_dtype, dev="cpu")
    else:
        global_value = global_value.to(dtype = global_value.dtype, dev="cpu")
    results = global_value.graph.run(global_value, [global_value], feed_dict=feed_dict, 
                                     num_micro_batches=1, save_checkpoint=True)
    return results[0], None

def change_query_key_value_ordering(param, num_splits, num_heads, hidden_size):
    # Permutes layout of param tensor to [num_heads * num_splits * hidden_size, :]
    # The original layout of param tensor is [num_splits * num_heads * hidden_size, :]
    input_shape = param.shape
    original_shape = [num_splits, num_heads, hidden_size] + input_shape[1:]
    ndim = len(original_shape)
    perm = [i for i in range(ndim)]
    perm[0] = 1 
    perm[1] = 0
    param = param.view(original_shape)
    param = param.transpose(perm)
    param = param.view(input_shape)
    return param

def storage_ptr(tensor: hetu.Tensor) -> int:
    return tensor.raw_data_ptr()

def _end_ptr(tensor: hetu.Tensor) -> int:
    if tensor.numel():
        stop = tensor.raw_data_ptr() + _SIZE[tensor.dtype] * tensor.numel()
    else:
        stop = tensor.raw_data_ptr()
    return stop

def storage_size(tensor: hetu.Tensor) -> int:
    return tensor.numel() * _SIZE[tensor.dtype]

def _filter_shared_not_shared(tensors: List[Set[str]], state_dict: Dict[str, hetu.Tensor]) -> List[Set[str]]:
    filtered_tensors = []
    for shared in tensors:
        if len(shared) < 2:
            filtered_tensors.append(shared)
            continue

        areas = []
        for name in shared:
            tensor = state_dict[name]
            areas.append((tensor.raw_data_ptr(), _end_ptr(tensor), name))
        areas.sort()

        _, last_stop, last_name = areas[0]
        filtered_tensors.append({last_name})
        for start, stop, name in areas[1:]:
            if start >= last_stop:
                filtered_tensors.append({name})
            else:
                filtered_tensors[-1].add(name)
            last_stop = stop

    return filtered_tensors

def _find_shared_tensors(state_dict: Dict[str, hetu.Tensor]) -> List[Set[str]]:
    tensors = defaultdict(set)
    for k, v in state_dict.items():
        if storage_ptr(v) != 0 and storage_size(v) != 0:
            # Need to add device as key because of multiple GPU.
            tensors[(v.device, storage_ptr(v), storage_size(v))].add(k)
    tensors = list(sorted(tensors.values()))
    tensors = _filter_shared_not_shared(tensors, state_dict)
    return tensors

def _is_complete(tensor: hetu.Tensor) -> bool:
    return tensor.raw_data_ptr() == storage_ptr(tensor) and tensor.numel() * _SIZE[tensor.dtype] == storage_size(tensor)

def _remove_duplicate_names(
    state_dict: Dict[str, hetu.Tensor],
    *,
    preferred_names: Optional[List[str]] = None,
    discard_names: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    if preferred_names is None:
        preferred_names = []
    preferred_names = set(preferred_names)
    if discard_names is None:
        discard_names = []
    discard_names = set(discard_names)

    shareds = _find_shared_tensors(state_dict)
    to_remove = defaultdict(list)
    for shared in shareds:
        complete_names = set([name for name in shared if _is_complete(state_dict[name])])
        if not complete_names:
            raise RuntimeError(
                "Error while trying to find names to remove to save state dict, but found no suitable name to keep"
                f" for saving amongst: {shared}. None is covering the entire storage.Refusing to save/load the model"
                " since you could be storing much more memory than needed. Please refer to"
                " https://huggingface.co/docs/safetensors/torch_shared_tensors for more information. Or open an"
                " issue."
            )

        keep_name = sorted(list(complete_names))[0]

        # Mechanism to preferentially select keys to keep
        # coming from the on-disk file to allow
        # loading models saved with a different choice
        # of keep_name
        preferred = complete_names.difference(discard_names)
        if preferred:
            keep_name = sorted(list(preferred))[0]

        if preferred_names:
            preferred = preferred_names.intersection(complete_names)
            if preferred:
                keep_name = sorted(list(preferred))[0]
        for name in sorted(shared):
            if name != keep_name:
                to_remove[keep_name].append(name)
    return to_remove


def save_model(
    model: hetu.nn.Module, filename: str, config=None, 
    local_device=None, save_dtype = None,
    force_contiguous: bool = False,
    only_lora: bool = False,
    metadata: Optional[Dict[str, str]] = None
):
    with hetu.graph("define_and_run", create_new=True, prefix="save_model"):
        global_state_dict = OrderedDict()
        all_device_groups = []
        state_dict = model.state_dict()
        ht_state_dict = model.state_dict(format='hetu')
        total_sum = len(state_dict.items())
        cur_sum = 0
        visit_tensors = set()
        import time
        save_st = time.time()

        for key, data in state_dict.items():
            # absmax need recompute
            if "absmax" in key:
                continue
            # only_lora mode only save loraA and loraB 
            if only_lora and ("lora" not in key):
                continue
            if (not only_lora) and ("lora" in key):
                continue
            param = ht_state_dict[key]
            device_group = param.get_device_group()
                
            if device_group not in all_device_groups:
                all_device_groups.append(device_group)
            # TODO: implement allgather_inter_group_param()
            if not device_group.contains(local_device):
                continue
            
            global_value = param
            # shared tensors only compute once
            if param.id in visit_tensors:
                continue
            visit_tensors.add(param.id)
            global_value, abs_max = data_transform_for_store(key, param, data, config, save_dtype, 
                                                             ht_state_dict, local_device)
            global_state_dict[key] = global_value.numpy(force=True, save=True)
            if ((save_dtype == hetu.float4) or (save_dtype == hetu.nfloat4)) and \
               (param.dtype != save_dtype):
                abs_max_key = key + "_absmax"
                global_state_dict[abs_max_key] = abs_max.numpy(force=True, save=True)
            cur_sum += 1

        save_ed = time.time()
        print('Data_Transfer_Time = %.4f'%(save_ed - save_st))

        if force_contiguous:
            state_dict = {k: v.contiguous() for k, v in state_dict.items()}
        for i, device_group in enumerate(all_device_groups):
            if device_group.contains(local_device):
                if device_group.get_index(local_device) == 0:
                    archive_file = WEIGHTS_NAME + f'-{i + 1}-of-{len(all_device_groups)}' + WEIGHTS_FORMAT
                    archive_file = os.path.join(
                        filename, archive_file
                    )
                    try:
                        save_file(global_state_dict, archive_file, metadata=metadata)
                    except ValueError as e:
                        msg = str(e)
                        msg += " Or use save_model(..., force_contiguous=True), read the docs for potential caveats."
                        raise ValueError(msg)

def temp_save(
    model: hetu.nn.Module, optimizer, filename: str, 
    step = 0, consumed_samples = 0, save_copies = 1, config = None, 
    local_device = None, save_dtype = None,
    force_contiguous: bool = False,
    only_lora: bool = False,
    metadata: Optional[Dict[str, str]] = None
):
    with hetu.graph("define_and_run"):
        import time
        save_st = time.time()
        print("SAVE_BY_TRAINING:", local_device, "CONSUMED:", consumed_samples, "STEP:", step)
        cur_step_dir = os.path.join(filename, "step" + str(step))
        rm_step_dir = os.path.join(filename, "step" + str(step - save_copies))
        if local_device.index == 0:
            if not os.path.exists(cur_step_dir):
                os.mkdir(cur_step_dir)
        global_state_dict = OrderedDict()
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
            # print("Opt_State_Dict:", opt_state_dict)
            # global_value, abs_max = data_transform_for_store(key, param, data, config, save_dtype, 
            #                                                  ht_state_dict, local_device)
            # global_state_dict[key] = global_value.numpy(force=True, save=True)
            
            # if ((save_dtype == hetu.float4) or (save_dtype == hetu.nfloat4)) and \
            #    (param.dtype != save_dtype):
            #     abs_max_key = key + "_absmax"
            #     global_state_dict[abs_max_key] = abs_max.numpy(force=True, save=True)
            global_state_dict[key] = data
            state = {}
            state["device_num"] = param.distributed_states.device_num
            state["order"] = param.distributed_states.order
            state["states"] = param.distributed_states.states
            state["states_union"] = get_states_union(param)
            device_index = []
            device_index_union = []
            # print("PAram:", param, 
            #       "\ndg_union:", param.get_device_group_union(),
            #       "\nds:", param.distributed_states,
            #       "\nds_hierarchy:", param.ds_hierarchy)
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
            cur_sum += 1

        save_ed = time.time()
        print(local_device, 'Data_Transfer_Time = %.4f'%(save_ed - save_st))
        
        save_st = time.time()

        if force_contiguous:
            state_dict = {k: v.contiguous() for k, v in state_dict.items()}
        
        for i, m_device_union in enumerate(all_device_group_unions):
            # print(f"Union{i}:{m_device_union}")
            need_save_json = False
            if len(m_device_union) > 0 and \
               m_device_union[0].contains(local_device) and \
               m_device_union[0].get_index(local_device) == 0:
                need_save_json = True
                 
            if need_save_json:
                # print("SAVE JSON:", local_device, "Union:", m_device_union,
                #       "\nJSON:", ds_json[str(m_device_union)])
                json_file = "param_states" + f'-{i + 1}-of-{len(all_device_group_unions)}' + ".json"
                # json_file = os.path.join(
                #     filename, json_file
                # )
                json_file = os.path.join(
                    cur_step_dir, json_file
                )
                print("SAVE JSON:", local_device, json_file)
                try:
                    import json
                    file_obj=open(json_file,'w',encoding='utf-8')
                    ds_json_for_dump = ds_json[str(m_device_union)]
                    ds_json_for_dump["consumed_samples"] = consumed_samples
                    json.dump(ds_json_for_dump, file_obj, ensure_ascii=False)
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
        # archive_file = os.path.join(
        #     filename, archive_file
        # )
        archive_file = os.path.join(
            cur_step_dir, archive_file
        )
        print(local_device, archive_file)
        save_st = time.time()
        try:
            # save_file(global_state_dict, archive_file, metadata=metadata)
            # if local_device.index == 0:
            #     step_filename = filename + "/step.txt"
            #     with open(step_filename, "w") as step_file:
            #         step_file.write(str(step))
            #     if os.path.exists(rm_step_dir):
            #         try:
            #             shutil.rmtree(rm_step_dir)
            #             print(f"The folder {rm_step_dir} has been deleted.")
            #         except OSError as e:
            #             print(f"Error: {e.strerror}")
            p = multiprocessing.Process(target=save_file_async, 
                                        args=(global_state_dict, archive_file, metadata,
                                              local_device, filename, step, rm_step_dir,))
            p.start()
            # save_file_async(global_state_dict, archive_file, metadata,
            #                 local_device, filename, step, rm_step_dir)
        except ValueError as e:
            msg = str(e)
            msg += " Or use save_model(..., force_contiguous=True), read the docs for potential caveats."
            raise ValueError(msg)
        save_ed = time.time()
        print(local_device, 'Pure_Save_Time = %.4f'%(save_ed - save_st))

def save_file_async(global_state_dict, archive_file, metadata,
                    local_device, filename, step, rm_step_dir):
    save_file(global_state_dict, archive_file, metadata=metadata)
    if local_device.index == 0:
        step_filename = filename + "/step.txt"
        with open(step_filename, "w") as step_file:
            step_file.write(str(step))
        if os.path.exists(rm_step_dir):
            try:
                shutil.rmtree(rm_step_dir)
                print(f"The folder {rm_step_dir} has been deleted.")
            except OSError as e:
                print(f"Error: {e.strerror}")

def temp_load(model: hetu.nn.Module, optimizer, filename: Union[str, os.PathLike],
              config=None, local_device=None, strict=True) -> Tuple[List[str], List[str]]:
    import time
    save_st = time.time()

    step_filename = filename + "/step.txt"
    if not os.path.exists(step_filename):
        print("No checkpoint. Train with initialize.")
        return model, 0
    with open(step_filename, "r") as step_file:
        cur_step = int(step_file.readline())
    global base_step
    base_step = cur_step + 1
    filename = os.path.join(filename, "step" + str(cur_step))
    print("BASE_STEP:", base_step)
    
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
    d_index = local_device.index 
    num_devices = 0
    file_list = os.listdir(filename)
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
            file_obj=open(json_file,'r',encoding='utf-8')
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
                    # print("Error:", k, "\nSDUnion:", get_states_union(param),
                    #       "\nTSD_UNION:", trans_strategy[k]["states_union"],
                    #       "\nST:", param.distributed_states.states,
                    #       "\nTST:", trans_strategy[k]["states"])
                    # exit(0)
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
    print("Need Switch:", need_switch)
    if need_switch:
        archive_opens = []
        ptr = 0
        def archive_sort(file_name):
            return int(str(file_name).split("-")[1])
        archive_files.sort(key=archive_sort)
        for archive in archive_files:
            archive_opens.append(safe_open(os.path.join(
            filename, archive), framework="np"))
            ptr += 1
        
        split1_sum = 0
        split2_sum = 0
        for k in hetu_state_dict:
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
            num_data_splits = int(device_num)
            for dim in order:
                if dim < 0:
                    num_data_splits = num_data_splits // states[dim]
            stride = int(1)
            split_numpys = []
            # print("GGG:", k, device_index, num_data_splits, archive_files)
            if (k[-4:] == "step"):
                num_data_splits = 1

            if num_data_splits == 1:
                split1_sum += 1
                # print(len(archive_opens))
                if k in archive_opens[device_index[0]].keys():
                    split_numpys.append(archive_opens[device_index[0]].get_tensor(k))
                    if (k == "transformer.h.10.attn.qkv_dense.weight_step"):
                        print("HHHHHHHH:", split_numpys)
                    shared_param = False
            else:
                split2_sum += 1
                for i in range(-1, -len(order) - 1, -1):
                    dim = order[i]
                    if dim < 0:
                        stride *= states[dim]
                    else:
                        if (len(split_numpys) == 0):
                            shared_param = False
                            ptr = 0
                            if (1 == 1) and ((k[-4:] == "mean") or (k[-8:] == "variance")):
                                ori_stride = 1
                                if "-1" in states_union[0].keys():
                                    ori_stride = states_union[0]["-1"]      

                                if "transformer.h.10.ln_1" in k:
                                    print("HJ:", local_device, device_index_union)
                                for j, dg in enumerate(device_index_union):
                                    ptr = 0
                                    cur_stride = 1
                                    if "-1" in states_union[j].keys():
                                        cur_stride = states_union[j]["-1"]
                                    while(ptr < len(dg)):
                                        if k in archive_opens[dg[ptr]].keys():
                                            if "transformer.h.10.ln_1" in k:
                                                print("K IN:", dg[ptr], -1 in states_union[j].keys(), states_union[j].keys(),
                                                      ori_stride, cur_stride)
                                            split_numpys.append(archive_opens[dg[ptr]].get_tensor(k))
                                            shared_param = False
                                        if stride == 1:
                                            ptr += stride
                                        else:
                                            ptr += (stride // ori_stride) * cur_stride
                                # print("ZERO SPLITS:", len(split_numpys), split_numpys)
                                if "transformer.h.10.ln_1" in k:
                                    for i, sl in enumerate(split_numpys):
                                        print("SL:", local_device, i, sl.shape, stride)
                                split_numpys = [np.concatenate(split_numpys, axis=dim)]
                                break
                            else:
                                for j in range(num_data_splits):
                                    if k in archive_opens[device_index[ptr]].keys():
                                        # if k == "transformer.wte.embedding_table_mean":
                                        #     print("KKKK:", trans_strategy[k]["device_num"], trans_strategy[k]["states"],
                                        #           "/n", ptr, device_index[ptr], 
                                        #           archive_opens[device_index[ptr]].get_tensor(k).shape)
                                        split_numpys.append(archive_opens[device_index[ptr]].get_tensor(k))
                                        shared_param = False
                                    ptr += stride
                                # print("HHJIJ:", len(split_numpys), num_data_splits, order, states)
                                assert(shared_param or (len(split_numpys) == num_data_splits))
                        
                        if (states[dim] > 1):           
                            split_numpys_ = []
                            assert(num_data_splits % states[dim] == 0)
                            # print(num_data_splits, " ", len(split_numpys))
                            for j in range(num_data_splits // states[dim]):
                                temp_storage = []
                                for l in range(states[dim]):
                                    temp_storage.append(split_numpys[j * states[dim] + l])
                                split_numpys_.append(np.concatenate(temp_storage, axis=dim))
                            split_numpys = split_numpys_
                            num_data_splits = num_data_splits // states[dim]
            # print(param, " shape:", param.shape, " t_shape:", split_numpys[0].shape, " order:", order, 
            #       " states", states, " num_splits", num_data_splits, " split_nps:", len(split_numpys))
            from hetu.nn.modules.module import parallel_data_provider
            # global_data = hetu.numpy_to_NDArray(split_numpys[0], parameter_to_dtype[k])
            global_data = split_numpys[0]

            cur_device_group_union = param.get_device_group_union()
            assert(len(cur_device_group_union) > 0)
            dg_idx = -1
            num_dgs = len(cur_device_group_union)
            # device_idx = 0
            for i, cur_device_group in enumerate(cur_device_group_union):
                if cur_device_group.contains(local_device):
                    dg_idx = i
                    if (k[-4:] == "step"):
                        state_dict[k] = global_data
                        break
                    # device_idx += cur_device_group.get_index(local_device)
                    device_idx = cur_device_group.num_devices * i
                    device_idx += cur_device_group.get_index(local_device)
                    tds_list = param.ds_hierarchy[0].ds_list
                    tds = tds_list[dg_idx]
                    # data = parallel_data_provider(global_data, param.distributed_states, device_idx)
                    # print(k, device_idx, global_data.shape, tds)
                    data = parallel_data_provider(global_data, tds, device_idx)
                    state_dict[k] = data
                    break
                # tdevice_idx += 
                # device_idx += cur_device_group.num_devices
            assert(dg_idx >= 0)
            # print("Reset:", "Device:", local_device, "Key:" , k, 
            #       "OriShape:", tuple(hetu_state_dict[k].shape), "RealShape:", tuple(state_dict[k].shape),
            #       "\nDS:", hetu_state_dict[k].distributed_states,
            #       "\nDSU:", hetu_state_dict[k].ds_hierarchy,
            #       "\nDGIDX:", dg_idx,
            #       "\nDVIDX:", device_idx,
            #       "\nTDS:", tds,
            #       "\nTDSLIST:", tds_list,
            #       "\nDGU:", hetu_state_dict[k].get_device_group_union())
            # 3D parallel: for pipeline situation, a device don't need to load all the checkpoint
            # if cur_device_group.contains(local_device):
            #     device_idx = cur_device_group.get_index(local_device)
            #     data = parallel_data_provider(global_data, param.distributed_states, device_idx)
            #     state_dict[k] = data
            # if (k == "transformer.h.10.attn.qkv_dense.weight_step"):
            #     print("GGGGGGGGG:", global_data,
            #           "\nDS:", param.distributed_states, 
            #           "\nIndex:", device_idx,
            #           "\nIIIIIII:", state_dict[k])
        print("SP1:", split1_sum, "SP>1:", split2_sum)    
        # for acv in archive_opens:    
        #     acv.close()
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
    print(local_device, 'Load_Params_Time = %.4f'%(save_ed - save_st))
    
    save_st = time.time()
    for k in hetu_state_dict:
        if k in state_dict:
            # if (k[-4:] == "step"):
            #     print("STEP:", k, ":", state_dict[k].shape, state_dict[k])
            # if tuple(hetu_state_dict[k].shape) != tuple(state_dict[k].shape):
            #     print("Reset:", "Device:", local_device, "Key:" , k, "OriShape:", tuple(hetu_state_dict[k].shape), 
            #           "RealShape:", tuple(state_dict[k].shape),
            #           "\nDSH:", hetu_state_dict[k].ds_hierarchy,
            #           "\nGSHAPE:", hetu_state_dict[k].global_shape)
            # assert(tuple(hetu_state_dict[k].shape) == tuple(state_dict[k].shape))
            # print("Reset:", "Device:", local_device, "Key:" , k, "OriShape:", tuple(hetu_state_dict[k].shape), "RealShape:", tuple(state_dict[k].shape))
            hetu_state_dict[k].reset_data(state_dict[k])
    save_ed = time.time()
    print(local_device, 'Reset_Params_Time = %.4f'%(save_ed - save_st))
    # exit(0)
    return model, consumed_samples

def load_by_training(
    model: hetu.nn.Module, optimizer, filename: str, 
    step=0, consumed_samples=0, save_copies=1, config=None, 
    local_device=None, save_dtype = None,
    force_contiguous: bool = False,
    only_lora: bool = False,
    metadata: Optional[Dict[str, str]] = None
):
    temp_save(model, optimizer, filename, base_step + step, consumed_samples, 
              save_copies, config, local_device, save_dtype, force_contiguous,
              only_lora, metadata)

def save_by_training(
    model: hetu.nn.Module, optimizer, filename: str, 
    step=0, consumed_samples=0, save_copies=1, config=None, 
    local_device=None, save_dtype = None,
    force_contiguous: bool = False,
    only_lora: bool = False,
    metadata: Optional[Dict[str, str]] = None,
):
    temp_save(model, optimizer, filename, base_step + step, consumed_samples, 
              save_copies, config, local_device, save_dtype, force_contiguous,
              only_lora, metadata)

def temp_save_split(
    model: hetu.nn.Module, optimizer, filename: str, config=None, 
    local_device=None, save_dtype = None,
    force_contiguous: bool = False,
    only_lora: bool = False,
    metadata: Optional[Dict[str, str]] = None
):
    with hetu.graph("define_and_run", create_new=True, prefix="save_model"):
        import time
        save_st = time.time()
        global_state_dict = OrderedDict()
        all_device_groups = []
        state_dict = model.state_dict()
        ht_state_dict = model.state_dict(format='hetu')
        total_sum = len(state_dict.items())
        cur_sum = 0
        visit_tensors = set()
        ds_json = {}

        for key, data in state_dict.items():
            # absmax need recompute
            if "absmax" in key:
                continue
            # only_lora mode only save loraA and loraB 
            if only_lora and ("lora" not in key):
                continue
            if (not only_lora) and ("lora" in key):
                continue
            param = ht_state_dict[key]
            device_group = param.get_device_group()
                
            if device_group not in all_device_groups:
                all_device_groups.append(device_group)
            # TODO: implement allgather_inter_group_param()
            if not device_group.contains(local_device):
                continue
            
            global_value = param
            # shared tensors only compute once
            if param.id in visit_tensors:
                continue
            visit_tensors.add(param.id)
            opt_state_dict = optimizer.get_states(param)
            # global_value, abs_max = data_transform_for_store(key, param, data, config, save_dtype, 
            #                                                  ht_state_dict, local_device)
            # global_state_dict[key] = global_value.numpy(force=True, save=True)
            
            # if ((save_dtype == hetu.float4) or (save_dtype == hetu.nfloat4)) and \
            #    (param.dtype != save_dtype):
            #     abs_max_key = key + "_absmax"
            #     global_state_dict[abs_max_key] = abs_max.numpy(force=True, save=True)
            
            # global_state_dict[key] = data
            state = {}
            state["device_num"] = param.distributed_states.device_num
            state["order"] = param.distributed_states.order
            state["states"] = param.distributed_states.states
            pre_states = {}
            for sk, sv in state["states"].items():
                pre_states[int(sk)] = sv
            state["states"] = pre_states
            state["device_group"] = param.device_group.device_index
            split_group = [0]
            split_data = [data]
            device_idx = param.device_group.get_index(local_device)
            split_dims = min(data.ndim, SPLIT_DIMS)
            for dim in range(split_dims - 1, -1, -1):
                temp_split_group = []
                temp_split_data = []
                split_dim = state["states"][dim] if dim in state["states"] else 1
                temp_splits = max(split_dim, TEMP_SPLITS)
                temp_splits = min(data.shape[dim], temp_splits)
                splits_per_device = temp_splits // split_dim
                block_idx = device_idx % split_dim
                device_idx = device_idx // split_dim
                start_pos =  block_idx * splits_per_device
                for idx, split_idx in enumerate(split_group):
                    for i in range(start_pos, start_pos + splits_per_device):
                        temp_split_group.append(split_idx * TEMP_SPLITS + i)
                    block_splits = np.split(split_data[idx], splits_per_device, axis=dim)
                    for blk in block_splits:
                        temp_split_data.append(blk)
                split_group = temp_split_group
                split_data = temp_split_data
            state["split_group"] = split_group
            for idx, split_idx in enumerate(split_group):
                split_key = key + "_split_" + str(split_idx)
                global_state_dict[split_key] = split_data[idx]
            ds_json[key] = state
            for k, state_param in opt_state_dict.items():
                # if (k == "step"):
                #     continue
                state_key = key + "_" + k
                state_data = state_param.get_data()
                state = {}
                state["device_num"] = state_param.distributed_states.device_num
                state["order"] = state_param.distributed_states.order
                state["states"] = state_param.distributed_states.states
                state["device_group"] = state_param.device_group.device_index
                split_group = [0]
                split_data = [state_data]
                device_idx = param.device_group.get_index(local_device)
                split_dims = min(state_data.ndim, SPLIT_DIMS)
                for dim in range(split_dims - 1, -1, -1):
                    temp_split_group = []
                    temp_split_data = []
                    split_dim = state["states"][dim] if dim in state["states"] else 1
                    temp_splits = max(split_dim, TEMP_SPLITS)
                    temp_splits = min(state_data.shape[dim], temp_splits)
                    splits_per_device = temp_splits // split_dim
                    block_idx = device_idx % split_dim
                    device_idx = device_idx // split_dim
                    start_pos =  block_idx * splits_per_device
                    for idx, split_idx in enumerate(split_group):
                        for i in range(start_pos, start_pos + splits_per_device):
                            temp_split_group.append(split_idx * temp_splits + i)
                        block_splits = np.split(split_data[idx], splits_per_device, axis=dim)
                        for blk in block_splits:
                            temp_split_data.append(blk)
                    split_group = temp_split_group
                    split_data = temp_split_data
                state["split_group"] = split_group
                for idx, split_idx in enumerate(split_group):
                    split_key = state_key + "_split_" + str(split_idx)
                    global_state_dict[split_key] = split_data[idx]
                ds_json[state_key] = state
            cur_sum += 1

        save_ed = time.time()
        print(local_device, 'Data_Transfer_Time = %.4f'%(save_ed - save_st))
        
        save_st = time.time()

        if force_contiguous:
            state_dict = {k: v.contiguous() for k, v in state_dict.items()}

        for i, device_group in enumerate(all_device_groups):
            if device_group.contains(local_device):
                if device_group.get_index(local_device) == 0:
                    json_file = "param_states" + f'-{i + 1}-of-{len(all_device_groups)}' + ".json"
                    json_file = os.path.join(
                        filename, json_file
                    )
                    try:
                        import json
                        file_obj=open(json_file,'w',encoding='utf-8')
                        json.dump(ds_json, file_obj,ensure_ascii=False)
                    except ValueError as e:
                        msg = str(e)
                        msg += " Or use save_model(..., force_contiguous=True), read the docs for potential caveats."
                        raise ValueError(msg)
        save_ed = time.time()
        print(local_device, 'Json_Save_Time = %.4f'%(save_ed - save_st))
        d_index = local_device.index 
        num_devices = 0
        for device_group in all_device_groups:
            num_devices += device_group.num_devices
        archive_file = WEIGHTS_NAME + f'-{d_index + 1}-of-{num_devices}' + WEIGHTS_FORMAT
        archive_file = os.path.join(
            filename, archive_file
        )
        save_st = time.time()
        try:
            save_file(global_state_dict, archive_file, metadata=metadata)
        except ValueError as e:
            msg = str(e)
            msg += " Or use save_model(..., force_contiguous=True), read the docs for potential caveats."
            raise ValueError(msg)
        save_ed = time.time()
        print(local_device, 'Pure_Save_Time = %.4f'%(save_ed - save_st))

def load_model(model: hetu.nn.Module, filename: Union[str, os.PathLike],
               config=None, local_device=None, strict=True) -> Tuple[List[str], List[str]]:
    local_state_dict = model.state_dict(format='hetu')
    hetu_state_dict = model.state_dict(format='hetu')
    all_device_groups = []
    parameter_to_dtype = {}
     
    # for k in local_state_dict:
    #     param = hetu_state_dict[k]
    #     parameter_to_dtype[k] = param.dtype
    #     device_group = param.get_device_group()
    #     if device_group not in all_device_groups:
    #         all_device_groups.append(device_group)
    #     # TODO: implement allgather_inter_group_param()
    #     if not device_group.contains(local_device):
    #         continue

    file_list = os.listdir(filename)
    archive_files = []
    for file in file_list:
        if (WEIGHTS_NAME in file and WEIGHTS_FORMAT in file):
            archive_files.append(file)

    archive_opens = []
    archive_keys = []
    ptr = 0
    archive_files.sort()
    for archive in archive_files:
        archive_opens.append(safe_open(os.path.join(
        filename, archive), framework="np"))
        archive_keys.append(archive_opens[ptr].keys())
        ptr += 1
    
    state_dict = {}
    ac_idx = -1 
    for k in local_state_dict:
        param = hetu_state_dict[k]
        parameter_to_dtype[k] = param.dtype
        device_group = param.get_device_group()
        if device_group not in all_device_groups:
            all_device_groups.append(device_group)
        # TODO: implement allgather_inter_group_param()
        if not device_group.contains(local_device):
            continue
        for i, archive in enumerate(archive_keys):
            if k in archive_keys[i]:
                state_dict[k] = hetu.numpy_to_NDArray(archive_opens[i].get_tensor(k), parameter_to_dtype[k])
                # break
        # if ac_idx == 0:
        #     for i, archive in enumerate(archive_opens):
        #         if k in archive_keys[i]:
        #             ac_idx = i
        # if k in archive_keys[ac_idx]:
        #     state_dict[k] = hetu.numpy_to_NDArray(archive_opens[ac_idx].get_tensor(k), parameter_to_dtype[k])
    
                

    for k in state_dict:
        # TODO: maybe implement it elsewhere
        # qkv_dense的weight和bias原先是[3 * num_heads * hidden_size, :]
        # 要先转化成为[num_heads * 3 * hidden_size, :]才可以
        # 因为会在num_heads这一维度上进行distributed tensor的split切分
        if ("qkv_dense" in k) and ("absmax" not in k):
            assert config != None, "There should be a config when using qkv_dense."
            state_dict[k] = change_query_key_value_ordering(state_dict[k], 
                                                            3, 
                                                            config.num_attention_heads, 
                                                            config.hidden_size // config.num_attention_heads)
    model.load_state_dict(state_dict, local_device, False)
    return model

def temp_load_split(model: hetu.nn.Module, optimizer, filename: Union[str, os.PathLike],
                    config=None, local_device=None, strict=True) -> Tuple[List[str], List[str]]:
    import time
    save_st = time.time()
    
    local_state_dict = model.state_dict(format='hetu')
    hetu_state_dict = model.state_dict(format='hetu')
    all_device_groups = []
    parameter_to_dtype = {}
    trans_strategy = {}
    state_dict = {}
    
    save_ed = time.time()
    print(local_device, 'Get_Params_Time = %.4f'%(save_ed - save_st))

    ds_json = []
    d_index = local_device.index 
    num_devices = 0
    file_list = os.listdir(filename)
    ds_files = []
    archive_files = []
    for file in file_list:
        if ("param_states" in file and ".json" in file):
            ds_files.append(file)
        if (WEIGHTS_NAME in file and WEIGHTS_FORMAT in file):
            archive_files.append(file)
    for i, json_file in enumerate(ds_files):
        json_file = os.path.join(
            filename, json_file
        )
        try:
            import json
            file_obj=open(json_file,'r',encoding='utf-8')
            python_data=json.load(file_obj)
            ds_json.append(python_data)
    
        except ValueError as e:
            msg = str(e)
            msg += " Or use save_model(..., force_contiguous=True), read the docs for potential caveats."
            raise ValueError(msg)

    need_switch = False
    for k in local_state_dict:
        param = local_state_dict[k]
        device_group = param.get_device_group()
        if device_group not in all_device_groups:
            all_device_groups.append(device_group)
        # TODO: implement allgather_inter_group_param()
        if not device_group.contains(local_device):
            continue
        for i in range(len(ds_json)):
            if k in ds_json[i]:                
                trans_strategy[k] = ds_json[i][k]
                pre_states = {}
                for sk, sv in trans_strategy[k]["states"].items():
                    pre_states[int(sk)] = sv
                trans_strategy[k]["states"] = pre_states
                if (param.distributed_states.device_num == trans_strategy[k]["device_num"] and
                    param.distributed_states.order == trans_strategy[k]["order"] and
                    param.distributed_states.states == trans_strategy[k]["states"] and
                    param.device_group.device_index == trans_strategy[k]["device_group"]):
                    pass
                else:
                    need_switch = True
                state = {}
                state["device_num"] = param.distributed_states.device_num
                state["order"] = param.distributed_states.order
                state["states"] = param.distributed_states.states
                state["device_group"] = param.device_group.device_index
                break
            
        if optimizer is not None:
            opt_state_dict = optimizer.get_states(param)
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
    if need_switch:
        archive_opens = []
        archive_keys = []
        ptr = 0
        archive_files.sort()
        for archive in archive_files:
            archive_opens.append(safe_open(os.path.join(
            filename, archive), framework="np"))
            archive_keys.append(archive_opens[ptr].keys())
            ptr += 1

        split1_sum = 0
        split2_sum = 0
        for k in hetu_state_dict:
            param = hetu_state_dict[k]
            device_group = param.get_device_group()
            if device_group not in all_device_groups:
                all_device_groups.append(device_group)
            # TODO: implement allgather_inter_group_param()
            if not device_group.contains(local_device):
                continue
            shared_param = False
            if k not in trans_strategy.keys():
                continue 
            parameter_to_dtype[k] = param.dtype
            
            pre_device_num = trans_strategy[k]["device_num"]
            pre_order = trans_strategy[k]["order"]
            pre_states = trans_strategy[k]["states"]
            pre_device_index = trans_strategy[k]["device_group"]
            
            device_num = param.distributed_states.device_num
            order = param.distributed_states.order
            states = param.distributed_states.states
            
            param = hetu_state_dict[k]
            device_group = param.get_device_group()
            
            state = trans_strategy[k]
            device_idx = param.device_group.get_index(local_device)
            split_data = []
            split_group = [0]
            split_dims = min(param.ndim, SPLIT_DIMS)
            
            for dim in range(split_dims - 1, -1, -1):
                temp_split_group = []
                split_dim = states[dim] if dim in states else 1
                temp_splits = max(split_dim, TEMP_SPLITS)
                temp_splits = min(param.shape[dim], temp_splits)
                splits_per_device = temp_splits // split_dim
                block_idx = device_idx % split_dim
                device_idx = device_idx // split_dim
                start_pos =  block_idx * splits_per_device
                for idx, split_idx in enumerate(split_group):
                    for i in range(start_pos, start_pos + splits_per_device):
                        temp_split_group.append(split_idx * temp_splits + i)
                split_group = temp_split_group
            
            num_data_splits = int(pre_device_num)
            for dim in pre_order:
                if dim < 0:
                    num_data_splits = num_data_splits // pre_states[dim]
                    
            if num_data_splits == 1:
                split1_sum += 1
                for split_idx in split_group:
                    split_key = k + "_split_" + str(split_idx) 
                    split_data.append(archive_opens[pre_device_index[0]].get_tensor(split_key))
            else:
                split2_sum += 1
                for split_idx in split_group:
                    split_key = k + "_split_" + str(split_idx)
                    for i in range(num_data_splits): 
                        if split_key in archive_keys[pre_device_index[i]]:
                            split_data.append(archive_opens[pre_device_index[i]].get_tensor(split_key))
                            break
                        
            split_dims = min(param.ndim, SPLIT_DIMS)
            for dim in range(split_dims - 1, -1, -1):
                temp_split_data = []
                split_dim = states[dim] if dim in states else 1
                temp_splits = max(split_dim, TEMP_SPLITS)
                temp_splits = min(param.shape[dim], temp_splits)
                splits_per_device = temp_splits // split_dim
                num_groups = len(split_data) // splits_per_device
                for i in range(num_groups):
                    concat_group = []
                    for j in range(splits_per_device):
                        concat_group.append(split_data[i * splits_per_device + j])
                    concat_data = np.concatenate(concat_group, axis=dim)
                    temp_split_data.append(concat_data)
                split_data = temp_split_data   
            assert(len(split_data) == 1)
            state_dict[k] = hetu.numpy_to_NDArray(split_data[0], parameter_to_dtype[k])
            
        save_ed = time.time()
        print(local_device, 'Load_Params_Time = %.4f'%(save_ed - save_st))
        
        save_st = time.time()
        for k in hetu_state_dict:
            if k in state_dict:
                hetu_state_dict[k].reset_data(state_dict[k])
        save_ed = time.time()
        print(local_device, 'Reset_Params_Time = %.4f'%(save_ed - save_st))    
        
    else:  
        d_index = local_device.index 
        num_devices = 0
        for device_group in all_device_groups:
            num_devices += device_group.num_devices
        archive_file = WEIGHTS_NAME + f'-{d_index + 1}-of-{num_devices}' + WEIGHTS_FORMAT
        archive_file = os.path.join(
            filename, archive_file
        )
        result = {}
        with safe_open(archive_file, framework="np") as f:
            for k, dtype in parameter_to_dtype.items():
                param = hetu_state_dict[k]
                device_group = param.get_device_group()
                if device_group not in all_device_groups:
                    all_device_groups.append(device_group)
                # TODO: implement allgather_inter_group_param()
                if not device_group.contains(local_device):
                    continue
                if k not in trans_strategy:
                    continue
                state = trans_strategy[k]
                device_idx = param.device_group.get_index(local_device)
                split_data = []
                split_group = [0]
                device_idx = param.device_group.get_index(local_device)
                split_dims = min(param.ndim, SPLIT_DIMS)
                for dim in range(split_dims - 1, -1, -1):
                    temp_split_group = []
                    split_dim = state["states"][dim] if dim in state["states"] else 1
                    temp_splits = max(split_dim, TEMP_SPLITS)
                    temp_splits = min(param.shape[dim], temp_splits)
                    splits_per_device = temp_splits // split_dim
                    block_idx = device_idx % split_dim
                    device_idx = device_idx // split_dim
                    start_pos =  block_idx * splits_per_device
                    for idx, split_idx in enumerate(split_group):
                        for i in range(start_pos, start_pos + splits_per_device):
                            temp_split_group.append(split_idx * temp_splits + i)
                    split_group = temp_split_group
                    
                for split_block_idx in split_group:
                    split_key = k + "_split_" + str(split_block_idx)
                    split_data.append(f.get_tensor(split_key))
                
                split_dims = min(hetu_state_dict[k].ndim, SPLIT_DIMS)
                for dim in range(split_dims - 1, -1, -1):
                    temp_split_data = []
                    split_dim = state["states"][dim] if dim in state["states"] else 1
                    temp_splits = max(split_dim, TEMP_SPLITS)
                    temp_splits = min(hetu_state_dict[k].shape[dim], temp_splits)
                    splits_per_device = temp_splits // split_dim
                    num_groups = len(split_data) // splits_per_device
                    for i in range(num_groups):
                        concat_group = []
                        for j in range(splits_per_device):
                            concat_group.append(split_data[i * splits_per_device + j])
                        concat_data = np.concatenate(concat_group, axis=dim)
                        temp_split_data.append(concat_data)
                    split_data = temp_split_data   
                assert(len(split_data) == 1)
                state_dict[k] = hetu.numpy_to_NDArray(split_data[0], parameter_to_dtype[k])
    save_ed = time.time()
    print(local_device, 'Load_Params_Time = %.4f'%(save_ed - save_st))
    
    save_st = time.time()
    for k in hetu_state_dict:
        if k in state_dict:
            hetu_state_dict[k].reset_data(state_dict[k])
    save_ed = time.time()
    print(local_device, 'Reset_Params_Time = %.4f'%(save_ed - save_st))
    return model

def save(tensors: Dict[str, torch.Tensor], metadata: Optional[Dict[str, str]] = None) -> bytes:
    """
    Saves a dictionary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, hetu.Tensor]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `bytes`: The raw bytes representing the format

    Example:

    ```python
    from hetu.utils.checkpoint.ht_safetensors import save
    import hetu

    tensors = {"embedding": hetu.zeros((512, 1024)), "attention": hetu.zeros((256, 256))}
    byte_data = save(tensors)
    ```
    """
    serialized = serialize(_flatten(tensors), metadata=metadata)
    result = bytes(serialized)
    return result

def save_file(
    tensors: Dict[str, hetu.NDArray],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
):
    """
    Saves a dictionary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, hetu.Tensor]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        filename (`str`, or `os.PathLike`)):
            The filename we're saving into.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `None`

    Example:

    ```python
    from hetu.utils.checkpoint.ht_safetensors import save_file
    import hetu

    tensors = {"embedding": hetu.zeros((512, 1024)), "attention": hetu.zeros((256, 256))}
    save_file(tensors, "model.safetensors")
    ```
    """
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            pass
    flattened = {k: {"dtype": str(v.dtype).split(".")[-1], "shape": v.shape, "data": _tobytes(v.numpy(save=True))} for k, v in tensors.items()}
    serialize_file(flattened, filename, metadata=metadata)

def load_file(
    filename: Union[str, os.PathLike],
    dtype: Optional[hetu.dtype] = None,
    device: str = "cpu",
) -> Dict[str, hetu.NDArray]:
    """
    Loads a safetensors file into hetu format.

    Args:
        filename (`str`, or `os.PathLike`):
            The name of the file which contains the tensors
        device (`Dict[str, any]`, *optional*, defaults to `cpu`):
            The device where the tensors need to be located after load.
            available options are all regular hetu device locations

    Returns:
        `Dict[str, hetu.Tensor]`: dictionary that contains name as key, value as `hetu.Tensor`

    Example:

    ```python
    from hetu.utils.checkpoint.ht_safetensors import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    """
    result = {}
    with safe_open(filename, framework="np") as f:
        for k in f.keys():
            # print(f.get_tensor(k))
            if (k in parameter_to_dtype):
                # print(parameter_to_dtype[k])
                # print("load:", k, f.get_tensor(k).dtype, parameter_to_dtype[k], f.get_tensor(k).flatten()[:10])
                print("Reset ", "Key:" , k, 
                      "\nPARAM:", f.get_tensor(k).flatten()[:20])
                result[k] = hetu.numpy_to_NDArray(f.get_tensor(k), parameter_to_dtype[k])
    return result

def load(data: bytes) -> Dict[str, torch.Tensor]:
    """
    Loads a safetensors file into hetu format from pure bytes.

    Args:
        data (`bytes`):
            The content of a safetensors file

    Returns:
        `Dict[str, hetu.Tensor]`: dictionary that contains name as key, value as `hetu.Tensor` on cpu

    Example:

    ```python
    from hetu.utils.checkpoint.ht_safetensors import load

    file_path = "./my_folder/bert.safetensors"
    with open(file_path, "rb") as f:
        data = f.read()

    loaded = load(data)
    ```
    """
    flat = deserialize(data)
    return _view2hetu(flat)

_SIZE = {
    hetu.int64: 8,
    hetu.float32: 4,
    hetu.int32: 4,
    hetu.bfloat16: 2,
    hetu.float16: 2,
    hetu.int16: 2,
    hetu.uint8: 1,
    hetu.int8: 1,
    hetu.bool: 1,
    hetu.float64: 8,
    hetu.float4: 1,
    hetu.nfloat4: 1,
}

_TYPES = {
    "F64": hetu.float64,
    "F32": hetu.float32,
    "F16": hetu.float16,
    "BF16": hetu.bfloat16,
    "I64": hetu.int64,
    # "U64": hetu.uint64,
    "I32": hetu.int32,
    # "U32": hetu.uint32,
    "I16": hetu.int16,
    # "U16": hetu.uint16,
    "I8": hetu.int8,
    "U8": hetu.uint8,
    "BOOL": hetu.bool,
    "F4": hetu.float4,
    "NF4": hetu.nfloat4,
}

def _getdtype(dtype_str: str) -> torch.dtype:
    return _TYPES[dtype_str]

def _view2torch(safeview) -> Dict[str, torch.Tensor]:
    result = {}
    for k, v in safeview:
        dtype = _getdtype(v["dtype"])
        arr = hetu.buffer_to_NDArray(v["data"], dtype=dtype, shape=v["shape"])
        result[k] = arr

    return result

def _is_little_endian(tensor: np.ndarray) -> bool:
    byteorder = tensor.dtype.byteorder
    if byteorder == "=":
        if sys.byteorder == "little":
            return True
        else:
            return False
    elif byteorder == "|":
        return True
    elif byteorder == "<":
        return True
    elif byteorder == ">":
        return False
    raise ValueError(f"Unexpected byte order {byteorder}")

def _tobytes(tensor: np.ndarray) -> bytes:
    if not _is_little_endian(tensor):
        tensor = tensor.byteswap(inplace=False)
    return tensor.tobytes()

def _flatten(tensors: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(tensors, dict):
        raise ValueError(f"Expected a dict of [str, hetu.NDArray] but received {type(tensors)}")

    for k, v in tensors.items():
        if not isinstance(v, np.ndarray):
            raise ValueError(f"Key `{k}` is invalid, expected hetu.NDArray but received {type(v)}")

    return {
        k: {
            "dtype": str(v.dtype).split(".")[-1],
            "shape": v.shape,
            "data": _tobytes(v, k),
        }
        for k, v in tensors.items()
    }