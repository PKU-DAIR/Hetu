from __future__ import annotations

import os
import gc
import re
import copy
import json
import hetu
import logging
import numpy as np
from packaging import version
from zipfile import is_zipfile
from collections import defaultdict
from hetu.models.utils.config_utils import PreTrainedConfig
from hetu.models.utils import SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME, TORCH_WEIGHTS_NAME, TORCH_WEIGHTS_INDEX_NAME
from hetu.models.utils.hub import is_remote_url
from hetu.utils.checkpoint.ht_safetensors import load_file
from typing import Optional, Union, OrderedDict
from hetu.utils.checkpoint import save_file
from hetu.utils.parallel import get_dg_from_union
from hetu.models.utils.common_utils import split_hetu_state_dict_into_shards

def _add_variant(weights_name: str, variant: Optional[str] = None) -> str:
    if variant is not None:
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = ".".join(splits)

    return weights_name

def simple_check_blocks_range(ds_parallel_configs, num_layers, model='llama'):
    """
    Simple check to verify that the transformer blocks range configuration is valid.
    
    Args:
        ds_parallel_configs: Distributed parallel configuration.
        num_layers: Number of transformer layers in the model.
        model: Model type, defaults to 'llama'.
    
    Raises:
        AssertionError: If the blocks range doesn't match the number of layers.
    """
    # simple check for transformer blocks range
    ranges = []
    for _, block_config in ds_parallel_configs[0][model]['blocks'].items():
        ranges.append(block_config['range'])
    assert ranges[0][0] == 0 and ranges[-1][-1] == num_layers-1, \
        f"{model} blocks range: {ranges} is conflict with num_hidden_layers: {num_layers}!"

def get_state_dict_dtype(state_dict):
    """
    Get the data type of the state dictionary by finding the first floating point tensor.
    
    Args:
        state_dict: Model state dictionary.
    
    Returns:
        Data type of the state dictionary.
    """
    for value in state_dict.values():
        if value.is_floating_point():
            return value.dtype
    return next(iter(state_dict.values())).dtype

def get_parameter_dtype(parameter: hetu.nn.Module):
    """
    Get the data type of a module by finding the first floating point parameter or buffer.
    
    Args:
        parameter: Module to get the data type from.
    
    Returns:
        Data type of the module.
    """
    dtype = None
    for t in parameter.parameters():
        dtype = t.dtype
        if t.is_floating_point():
            return dtype
        
    if dtype is not None:
        return dtype

    # fall back to buffer dtype
    for t in parameter.buffers():
        last_dtype = t.dtype
        if t.is_floating_point():
            return t.dtype
    return last_dtype

def allgather_intra_group_param(param, value, local_device, idx, dtype):
    """
    Gather parameter values across all devices within a device group.
    
    Args:
        param: The parameter to gather.
        value: The local value of the parameter.
        local_device: The local device.
        idx: Index or name identifier for the parameter.
        dtype: Data type for the operation.
    
    Returns:
        The gathered parameter values.
    """
    with hetu.graph("define_and_run", create_new=True, prefix="save_" + param.name + str(idx)):
        with hetu.subgraph(name="save"):
            with hetu.autocast(dtype):
                device_group_union = param.get_device_group_union()
                _, device_group = get_dg_from_union(local_device, device_group_union)
                local_value = hetu.parallel_placeholder(
                    param.dtype,
                    global_shape=param.global_shape, 
                    ds_hierarchy=param.ds_hierarchy,
                    device_group_hierarchy=param.dg_hierarchy,
                    name='local_device'
                )
                hetu.add_to_subgraph(local_value)
                if not param.distributed_states.is_pure_duplicate:
                    hetero_size = len(device_group_union)
                    ds_list_dup = [hetu.DistributedStates(device_group_union[i].num_devices * hetero_size, {-1: device_group_union[i].num_devices * hetero_size}, [-1])
                                for i in range(hetero_size)]
                    ds_union_dup = hetu.DistributedStatesUnion(ds_list_dup, -1 if hetero_size > 1 else -3)
                    global_value = hetu.comm(local_value, [ds_union_dup])
                    # TODO: 目前graph中必须有parameter来获取used_ranks
                    dummy = hetu.parallel_parameter(eval(f'hetu.zeros_initializer()'), 
                                                    [0], [ds_union_dup], [device_group.get_index(local_device)], 
                                                    dtype=dtype, requires_grad=False, 
                                                    device_group_hierarchy=param.dg_hierarchy, name='dummy')    
                    # TODO: 目前无法fetch被替换的comm op 
                    bias = hetu.parallel_placeholder(param.dtype, global_shape=global_value.shape, ds_hierarchy=[ds_union_dup],
                                                     device_group_hierarchy=param.dg_hierarchy, name='zero_bias')    
                    hetu.add_to_subgraph(bias)
                    global_value = global_value + bias
                else:
                    dummy = hetu.parallel_parameter(eval(f'hetu.zeros_initializer()'), 
                                                    [0], param.ds_hierarchy, [device_group.get_index(local_device)], 
                                                    dtype=dtype, requires_grad=False, 
                                                    device_group_hierarchy=param.dg_hierarchy, name='dummy')
                    bias = hetu.parallel_placeholder(param.dtype, global_shape=local_value.shape, ds_hierarchy=param.ds_hierarchy,
                                                     device_group_hierarchy=param.dg_hierarchy, name='zero_bias')
                    hetu.add_to_subgraph(bias)
                    global_value = local_value + bias
    local_bias = np.zeros(value.shape, dtype=value.dtype)
    feed_dict = {local_value: value, bias: local_bias}
    with hetu.autocast(dtype):
        results = global_value.graph.run(global_value, [global_value], feed_dict=feed_dict)[0]
    return results[0].numpy(force=True, save=True)

def get_gathered_state_dict_to_save(model: hetu.nn.Module, local_device, dtype=None):
    """
    Gather the state dictionary from all devices for saving.
    
    Args:
        model: The model to get the state dictionary from.
        local_device: The local device.
        dtype: Optional data type to cast the state dictionary to.
    
    Returns:
        The gathered state dictionary.
    """
    gathered_state_dict = OrderedDict()
    local_state_dict = model.state_dict(format='hetu')
    for name, param in local_state_dict.items():
        _, device_group = get_dg_from_union(local_device, param.get_device_group_union())
        if device_group is None:
            continue
        
        global_value = allgather_intra_group_param(param, param.get_data(), local_device, name, dtype)
        if device_group.get_index(local_device) == 0:
            gathered_state_dict[name] = hetu.numpy_to_NDArray(global_value)
    return gathered_state_dict

def get_peft_model_state_dict(model: hetu.nn.Module, local_device, gather_param=True):
    """
    Get the state dictionary of a PEFT (Parameter-Efficient Fine-Tuning) model.
    
    Args:
        model: The PEFT model to get the state dictionary from.
        local_device: The local device.
        gather_param: Whether to gather distributed parameters.
    
    Returns:
        The PEFT model state dictionary.
    """
    peft_model_state_dict = OrderedDict()
    local_state_dict = model.state_dict(format='hetu')
    for name, param in local_state_dict.items():
        if "lora_" not in name:
            continue
        _, device_group = get_dg_from_union(local_device, param.get_device_group_union())
        if device_group is None:
            continue
        global_value = param.get_data()
        if gather_param and not param.distributed_states.is_pure_duplicate:
            global_value = allgather_intra_group_param(param, param.get_data(), local_device, name)
        if device_group.get_index(local_device) == 0:
            peft_model_state_dict[name] = hetu.from_numpy(global_value)
    return peft_model_state_dict

def get_all_device_groups(model: hetu.nn.Module, dp_rank: int):
    """
    Get all unique device groups in a model for a specific data parallel rank.
    
    Args:
        model: The model to get device groups from.
        dp_rank: Data parallel rank.
    
    Returns:
        Set of all unique device groups in the model.
    """
    local_state_dict = model.state_dict(format='hetu')
    all_device_groups = set()
    for _, param in local_state_dict.items():
        device_group = param.get_device_group_union()[dp_rank]
        if device_group not in all_device_groups:
            all_device_groups.add(device_group)
    return all_device_groups

def load_state_dict(
    checkpoint_file: Union[str, os.PathLike],
    dtype: Optional[hetu.dtype] = None,
    weights_only: bool = True,
):
    """
    Load a state dictionary from a checkpoint file.
    
    Args:
        checkpoint_file: Path to the checkpoint file.
        dtype: Optional data type to cast the state dictionary to.
        weights_only: Whether to load only weights.
    
    Returns:
        Loaded state dictionary.
    
    Raises:
        ImportError: If PyTorch is not installed.
        ValueError: If unable to load weights from the checkpoint file.
    """
    if checkpoint_file.endswith(".safetensors"):
        return load_file(checkpoint_file, dtype=dtype)
    else:
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required to load model weights in PyTorch format")

        def shared_pointers(tensors):
            ptrs = defaultdict(list)
            for k, v in tensors.items():
                ptrs[v.data_ptr()].append(k)
            return [names for names in ptrs.values() if len(names) > 1]

        extra_args = {}
        # mmap can only be used with files serialized with zipfile-based format.
        if (
            version.parse(torch.__version__) >= version.parse("2.1.0") and
            is_zipfile(checkpoint_file)
        ):
            extra_args = {"mmap": True}
        weights_only_kwarg = {"weights_only": weights_only}
        try:
            loaded = torch.load(
                checkpoint_file,
                map_location="cpu",
                **weights_only_kwarg,
                **extra_args,
            )
            loaded = loaded.get("state_dict", loaded)            
            shared = shared_pointers(loaded)

            for shared_weights in shared:
                for name in shared_weights[1:]:
                    loaded.pop(name)

            # Convert the state dict to Hetu tensors
            state_dict = {}
            for key, value in loaded.items():
                if value.dtype == torch.bfloat16:
                    numpy_value = value.float().numpy()
                else:
                    numpy_value = value.numpy()
                state_dict[key] = hetu.from_numpy(numpy_value)
                if dtype is not None and not numpy_value.dtype == str(dtype).split(".")[1]:
                   state_dict[key] = state_dict[key].to(dtype, dev="cpu")
            return state_dict
        except Exception as e:
            raise ValueError(f"Unable to load weights from checkpoint file {checkpoint_file}: {e}")

def _load_state_dict_into_model(
    model,
    state_dict,
    start_prefix,
    local_device,
):
    """
    Load a state dictionary into a model.
    
    Args:
        model: The model to load the state dictionary into.
        state_dict: The state dictionary to load.
        start_prefix: Prefix for the state dictionary keys.
        local_device: The local device.
    
    Returns:
        List of error messages encountered during loading.
    """
    error_msgs = []
    state_dict = state_dict.copy()
    
    def load(module: hetu.nn.Module, state_dict, prefix=""):
        module._load_from_state_dict(
            state_dict, local_device, prefix, True, [], [], error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".")
    
    load(model, state_dict, prefix=start_prefix)
    del state_dict
    
    return error_msgs

def _get_tied_weight_keys(module: hetu.nn.Module, prefix=""):
    tied_weight_keys = []
    if getattr(module, "_tied_weights_keys", None) is not None:
        names = [f"{prefix}.{k}" if prefix else k for k in module._tied_weights_keys]
        tied_weight_keys.extend(names)
    for name, submodule in module.named_children():
        local_prefix = f"{prefix}.{name}" if prefix else name
        tied_weight_keys.extend(_get_tied_weight_keys(submodule, prefix=local_prefix))
    return tied_weight_keys

class PreTrainedModel(hetu.nn.Module):
    config_class = None
    base_model_prefix = ""
    
    # a list of `re` patterns of `state_dict` keys that should be removed from the list of missing
    # keys we find (keys inside the model but not in the checkpoint) and avoid unnecessary warnings.
    _keys_to_ignore_on_load_missing = None
    # a list of `re` patterns of `state_dict` keys that should be removed from the list of
    # unexpected keys we find (keys inside the checkpoint but not the model) and avoid unnecessary
    # warnings.
    _keys_to_ignore_on_load_unexpected = None
    # a list of `state_dict` keys to ignore when saving the model (useful for keys that aren't
    # trained, but which are either deterministic or tied variables)
    _keys_to_ignore_on_save = None
    # a list of `state_dict` keys that are potentially tied to another key in the state_dict.
    _tied_weights_keys = None
    # whether the model is a PEFT model
    _is_peft_config_loaded = False
    
    def __init__(self, config: PreTrainedConfig, **kwargs):
        super().__init__()
        self.config = config
        self.name_or_path = config.name_or_path
    
    @staticmethod
    def _fix_state_dict_key_on_load(key):
        is_changed = False
        fix_structure = {"language_model.":"",
                         "embedding.word_embeddings.weight":"transformer.wte.embedding_table",
                         "embedding.position_embeddings.weight":"transformer.wpe.embedding_table"}
        fix_name = {"encoder" : "transformer", "layers": "h", "self_attention" : "attn", "linear_proj" : "dense", 
                    "linear_qkv" : "qkv_dense", "final_layernorm":"ln_f", "input_norm":"ln_1",
                    "post_attention_norm":"ln_2", "query_key_value":"qkv_dense",
                    "final_norm": "ln_f"}
        for k, v in fix_structure.items():
            if k in key:
                key = key.replace(k, v)
                is_changed = True
        key_split = key.split(".")
        key_new = '.'.join([fix_name.get(k, k) for k in key_split])
        if key_new != key:
            is_changed = True
        return key_new, is_changed
    
    @classmethod
    def _fix_state_dict_keys_on_load(cls, state_dict):
        """
        Fix state dictionary key names during loading.
        
        Args:
            state_dict: Original state dictionary.
        
        Returns:
            Tuple of fixed state dictionary and warning messages.
        """
        renamed_keys = {}
        warning_msgs = []
        state_dict_keys = list(state_dict.keys())
        for key in state_dict_keys:
            new_key, is_changed = cls._fix_state_dict_key_on_load(key)
            if is_changed:
                state_dict[new_key] = state_dict.pop(key)
                renamed_keys[key] = new_key
        
        if renamed_keys:
            for old_key, new_key in renamed_keys.items():
                warning_msg = f"* `{old_key}` -> `{new_key}`"
                warning_msgs.append(warning_msg)
        return state_dict, warning_msgs
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        ds_parallel_configs,
        config: Optional[Union[PreTrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        weights_only: bool = True,
        use_safetensors: bool = True,
        **kwargs,
    ):
        """
        Load a pre-trained model from a checkpoint.
        
        Args:
            pretrained_model_name_or_path: Path to the pre-trained model or checkpoint.
            ds_parallel_configs: Distributed parallel configuration.
            config: Optional configuration for the model.
            cache_dir: Optional cache directory.
            ignore_mismatched_sizes: Whether to ignore mismatched sizes.
            weights_only: Whether to load only weights.
            use_safetensors: Whether to use safetensors format.
            **kwargs: Additional arguments.
        
        Returns:
            Loaded pre-trained model.
        """
        model_dtype = kwargs.pop("model_dtype", None)
        subfolder = kwargs.pop("subfolder", "")
        state_dict = kwargs.pop("state_dict", None)
        variant = kwargs.pop("variant", None)
        output_loading_info = kwargs.pop("output_loading_info", False)

        # read from model config file (cls.config_class.from_pretrained)
        if not isinstance(config, PreTrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config = cls.config_class.from_pretrained(config_path, cache_dir=cache_dir, **kwargs)
        else:
            config = copy.deepcopy(config)
        # 根据model_name_or_path判断是否从本地读取，以及是否为sharded存储（远端存储）
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        is_sharded = False
        if is_local:
            if use_safetensors and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant))
            ):
                archive_file = os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant))
            elif use_safetensors and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))
            ):
                archive_file = os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))
                is_sharded = True
            elif not use_safetensors and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(TORCH_WEIGHTS_NAME, variant))
            ):
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(TORCH_WEIGHTS_NAME, variant)
                )
            elif not use_safetensors and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(TORCH_WEIGHTS_INDEX_NAME, variant))
            ):
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(TORCH_WEIGHTS_INDEX_NAME, variant)
                )
                is_sharded = True
            elif use_safetensors:
                raise EnvironmentError(
                    f"Error no file named {_add_variant(SAFE_WEIGHTS_NAME, variant)} found in directory"
                    f" {pretrained_model_name_or_path}."
                )
            else:
                raise EnvironmentError(
                    f"Error no file named {_add_variant(TORCH_WEIGHTS_NAME, variant)}, {_add_variant(SAFE_WEIGHTS_NAME, variant)}"
                    f" found in directory {pretrained_model_name_or_path}."
                )
        elif is_remote_url(pretrained_model_name_or_path):
            # TODO: support remote download
            raise NotImplementedError
        else:
            ValueError("Model name or path is not valid")
        if is_local:
            resolved_archive_file = archive_file
        else:
            # TODO: support remote download
            raise NotImplementedError
        # 如果是sharded存储，需要读取metadata和sharded文件名
        if is_sharded:
            if use_safetensors:
                index_file = os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))
            else:
                index_file = os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(TORCH_WEIGHTS_INDEX_NAME, variant))
            with open(index_file, "r") as f:
                index = json.loads(f.read())
            shard_filenames = sorted(set(index["weight_map"].values()))
            sharded_metadata = index["metadata"]
            sharded_metadata["all_checkpoint_keys"] = list(index["weight_map"].keys())
            sharded_metadata["weight_map"] = index["weight_map"].copy()
            
            if is_local:
                resolved_archive_file = [os.path.join(pretrained_model_name_or_path, subfolder, f) for f in shard_filenames]
            else:
                # TODO: download from remote
                raise NotImplementedError
        
        if model_dtype is not None:
            if isinstance(model_dtype, str):
                if model_dtype == "auto":
                    # 从state_dict中获取dtype
                    if is_sharded and "dtype" in sharded_metadata:
                        model_dtype = sharded_metadata["dtype"]
                    elif not is_sharded:
                        if state_dict is None:
                            state_dict = load_state_dict(resolved_archive_file, weights_only=weights_only)
                        model_dtype = get_state_dict_dtype(state_dict)
                    else:
                        one_state_dict = load_state_dict(resolved_archive_file[0], weights_only=weights_only)
                        model_dtype = get_state_dict_dtype(one_state_dict)
                        del one_state_dict  # free CPU memory
                    config.model_dtype = model_dtype
                    for sub_config_key in config.sub_configs.keys():
                        value = getattr(config, sub_config_key)
                        value.model_dtype = default_dtype
                elif hasattr(hetu, model_dtype):
                    model_dtype = getattr(hetu, model_dtype)
                for sub_config_key in config.sub_configs.keys():
                    sub_config = getattr(config, sub_config_key)
                    sub_config.model_dtype = model_dtype
            elif isinstance(model_dtype, hetu.dtype):
                config.model_dtype = model_dtype
                for sub_config_key in config.sub_configs.keys():
                    value = getattr(config, sub_config_key)
                    value.model_dtype = default_dtype
            elif isinstance(model_dtype, dict):
                for key, dtype in model_dtype.items():
                    if hasattr(config, key):
                        value = getattr(config, key)
                        value.model_dtype = dtype
                model_dtype = model_dtype.get("")
                if isinstance(model_dtype, str) and hasattr(hetu, model_dtype):
                    model_dtype = getattr(hetu, model_dtype)
                elif model_dtype is None:
                    model_dtype = hetu.float32
                config.model_dtype = model_dtype
            else:
                ValueError("model_dtype should be str, hetu.dtype or dict")
        else:
            default_dtype = "float32"
            model_dtype = getattr(hetu, default_dtype)
            config.model_dtype = model_dtype
            for sub_config_key in config.sub_configs.keys():
                value = getattr(config, sub_config_key)
                value.model_dtype = default_dtype
        
        config.name_or_path = pretrained_model_name_or_path
        
        if state_dict is None and not is_sharded:
            state_dict = load_state_dict(resolved_archive_file, weights_only=weights_only, dtype=model_dtype)
        
        # 如果是单文件存储，直接读取state_dict，否则只从metadata获取loaded key
        if not is_sharded:
            loaded_state_dict_keys = list(state_dict.keys())
        else:
            loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]

        with hetu.graph("define_and_run", num_strategy=len(ds_parallel_configs)):
            with hetu.autocast(model_dtype):
                model = cls(config, ds_parallel_configs=ds_parallel_configs)
        
        # 加载state_dict
        (
            model,
            missing_keys,
            unexpected_keys,
            mismatched_keys,
            error_msgs,
        ) = cls._load_pretrained_model(
            model,
            state_dict,
            loaded_state_dict_keys,
            resolved_archive_file,
            pretrained_model_name_or_path,
            sharded_metadata=sharded_metadata,
            dtype=model_dtype,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            weights_only=weights_only,
        )
        logging.info("State dict loaded successfully")
        
        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info
        
        return model
    
    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict,
        loaded_keys,
        resolved_archive_file,
        pretrained_model_name_or_path,
        sharded_metadata=None,
        dtype=None,
        ignore_mismatched_sizes=False,
        weights_only=True,
    ):
        is_sharded = sharded_metadata is not None
        
        # Retrieve missing and unexpected keys
        model_state_dict = model.state_dict(format='hetu')
        expected_keys = list(model_state_dict.keys())
        prefix = model.base_model_prefix
        
        if len(prefix) > 0:
            load_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
            expect_prefix_module = any(s.startswith(prefix) for s in expected_keys)
        else:
            load_prefix_module = False
            expect_prefix_module = False
        
        original_loaded_keys = loaded_keys
        
        # re-name keys of the newly created model
        # instead of loaded keys
        remove_prefix_from_model = not load_prefix_module and expect_prefix_module
        add_prefix_to_model = load_prefix_module and not expect_prefix_module
        
        if remove_prefix_from_model:
            _prefix = f"{prefix}."
            expected_keys_not_prefixed = [k for k in expected_keys if not k.startswith(_prefix)]
            expected_keys = [k[len(_prefix): ] if k.startswith(_prefix) else k for k in expected_keys]
        elif add_prefix_to_model:
            expected_keys = [".".join([prefix, k]) for k in expected_keys]
        
        missing_keys = sorted(set(expected_keys) - set(loaded_keys))
        unexpected_keys = set(loaded_keys) - set(expected_keys)
        
        # Remove model buffer names from unexpected keys
        model_buffer_keys = {n for n, _ in model.named_buffers()}
        if remove_prefix_from_model:
            model_buffer_keys = {k[len(_prefix): ] if k.startswith(_prefix) else k for k in model_buffer_keys}
        elif add_prefix_to_model:
            model_buffer_keys = [".".join([prefix, k]) for k in model_buffer_keys]
        unexpected_keys = sorted(unexpected_keys - model_buffer_keys)
        
        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]
        
        start_prefix = ""
        model_to_load = model
        if len(cls.base_model_prefix) > 0 and not hasattr(model, cls.base_model_prefix) and load_prefix_module:
            start_prefix = cls.base_model_prefix + "."
        elif len(cls.base_model_prefix) > 0 and hasattr(model, cls.base_model_prefix) and not load_prefix_module:
            model_to_load = getattr(model, cls.base_model_prefix)
            base_model_expected_keys = list(model_to_load.state_dict(format='hetu').keys())
            if any(key in expected_keys_not_prefixed and key not in base_model_expected_keys for key in loaded_keys):
                raise ValueError(
                    f"Meet unexpected weights for base model {cls.base_model_prefix}. "
                    "The state dictionary of the model you are trying to load is corrupted. Are you sure it was "
                    "properly saved?"
                )
        
        def _find_mismatch_keys(
            ckpt_state_dict,
            model_state_dict,
            loaded_keys,
            original_loaded_keys,
            remove_prefix_from_model,
            add_prefix_to_model,
            ignore_mismatched_sizes,
        ):
            mismatch_keys = []
            if not ignore_mismatched_sizes:
                for ckpt_key, model_key in zip(original_loaded_keys, loaded_keys):
                    if ckpt_key not in ckpt_state_dict:
                        continue
                    if remove_prefix_from_model:
                        model_key = f"{prefix}.{model_key}"
                    elif add_prefix_to_model:
                        model_key = ".".join(model_key.split(".")[1:])
                    
                    if (
                        model_key in model_state_dict
                        and ckpt_state_dict[ckpt_key].shape != list(model_state_dict[model_key].global_shape)
                    ):
                        mismatch_keys.append(
                            (ckpt_key, ckpt_state_dict[ckpt_key].shape, list(model_state_dict[model_key].global_shape))
                        )
                        del ckpt_state_dict[ckpt_key]
            return mismatch_keys
        
        local_device = hetu.local_device()
        
        _tied_weights_keys = _get_tied_weight_keys(model_to_load)
        # remove tied weights from state_dict and get them by comm ops
        for key in _tied_weights_keys:
            if key in state_dict:
                del state_dict[key]
                state_dict.pop(key)

        if not is_sharded:
            mismatched_keys = _find_mismatch_keys(
                state_dict,
                model_state_dict,
                loaded_keys,
                original_loaded_keys,
                remove_prefix_from_model,
                add_prefix_to_model,
                ignore_mismatched_sizes,
            )
            fixed_state_dict, rename_msgs = cls._fix_state_dict_keys_on_load(state_dict)
            error_msgs = _load_state_dict_into_model(
                model_to_load, fixed_state_dict, start_prefix, local_device
            )
        else:
            error_msgs = []
            rename_msgs = []
            mismatched_keys = []
            for shard_file in resolved_archive_file:
                state_dict = load_state_dict(shard_file, weights_only=weights_only, dtype=dtype)
                mismatched_keys += _find_mismatch_keys(
                    state_dict,
                    model_state_dict,
                    loaded_keys,
                    original_loaded_keys,
                    remove_prefix_from_model,
                    add_prefix_to_model,
                    ignore_mismatched_sizes,
                )
                fixed_state_dict, sub_rename_msgs = cls._fix_state_dict_keys_on_load(state_dict)
                rename_msgs += sub_rename_msgs
                error_msgs += _load_state_dict_into_model(
                    model_to_load, fixed_state_dict, start_prefix, local_device
                )
                del state_dict
                gc.collect()
        
        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if "size mismatch" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

        if len(unexpected_keys) > 0:
            logging.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logging.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}")
        
        if len(missing_keys) > 0:
            logging.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logging.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )
        else:
            logging.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
            )
        
        if len(rename_msgs) > 0:
            rename_warning = "\n".join(rename_msgs)
            logging.warning(
                f"A pretrained model of type `{cls.__name__}` "
                f"contains parameters that have been renamed internally (a few are listed below but more are present in the model):\n"
                f"{rename_warning}\n"
            )
        
        return model, missing_keys, unexpected_keys, mismatched_keys, error_msgs        

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        state_dict: Optional[dict] = None,
        max_shard_size: Union[int, str] = "5GB",
        variant: Optional[str] = None,
        **kwargs,
    ):
        """
        Save the pre-trained model to a directory.
        
        Args:
            save_directory: Directory to save the model to.
            state_dict: Optional state dictionary to save.
            max_shard_size: Maximum shard size for saving.
            variant: Optional variant for the model.
            **kwargs: Additional arguments.
        
        Raises:
            AssertionError: If the provided path is a file instead of a directory.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)
        
        dtype = kwargs.get("dtype", None)
        if dtype is None:
            dtype = get_parameter_dtype(self)
        
        self.config.model_dtype = str(dtype).split(".")[1]
        
        local_device = hetu.local_device()
        
        # Get dp rank
        local_state_dict = self.state_dict(format='hetu')
        dp_rank = None
        for value in local_state_dict.values():
            if dp_rank is not None:
                break
            ds_union = value.get_ds_union()
            dg_union = value.get_device_group_union()
            device_group_index, device_group = get_dg_from_union(local_device, dg_union)
            local_device_idx = device_group.get_index(local_device)
            ds = ds_union.get_local(device_group_index)
            dp_rank = ds.get_dup_group_index(local_device_idx)
        del local_state_dict
        if dp_rank is None:
            raise ValueError("Cannot find dp rank")
        
        is_main_process = False
        if dp_rank == 0:
            all_device_groups = get_all_device_groups(self, dp_rank)
            save_devices = set()
            for device_group in all_device_groups:
                save_devices.add(hash(device_group.get(0)))
            save_devices = sorted(save_devices)
            num_save_devices = len(save_devices)
            local_save_index = save_devices.index(hash(local_device))
            is_main_process = (local_save_index == 0)
            
            # Save config
            if is_main_process:
                if not self._is_peft_config_loaded:
                    self.config.save_pretrained(save_directory)
                else:
                    state_dict = get_peft_model_state_dict(self, local_device=local_device)
                    current_peft_config = self.peft_config
                    current_peft_config.save_pretrained(save_directory)
                
            # Save the model
            if state_dict is None:
                state_dict = get_gathered_state_dict_to_save(self, local_device=local_device, dtype=dtype)
            
            # Handle the case where some state_dict keys shouldn't be saved
            if state_dict is not None and self._keys_to_ignore_on_save is not None:
                for ignore_key in self._keys_to_ignore_on_save:
                    if ignore_key in state_dict.keys():
                        del state_dict[ignore_key]
            
            # Safetensors does not allow tensor aliasing.
            # We're going to remove aliases before saving
            _tied_weights_keys = _get_tied_weight_keys(self)
            for key in _tied_weights_keys:
                if key in state_dict:
                    del state_dict[key]
                    state_dict.pop(key)
            
            # Shard the model if it is too big.
            if any(
                device_group.contains(local_device) and device_group.get_index(local_device) == 0
                for device_group in all_device_groups
            ):
                weights_name = _add_variant(SAFE_WEIGHTS_NAME, variant)
                filename_pattern = weights_name.replace(
                    ".safetensors",
                    f".{local_save_index}-rank-{num_save_devices}." + "{suffix}.safetensors"
                )
                state_dict_split = split_hetu_state_dict_into_shards(
                    state_dict, filename_pattern=filename_pattern, max_shard_size=max_shard_size
                )
                index = {
                    "metadata": state_dict_split.metadata,
                    "weight_map": state_dict_split.tensor_to_filename,
                }

                # Clean the folder from a previous save
                if is_main_process:
                    for filename in os.listdir(save_directory):
                        full_filename = os.path.join(save_directory, filename)
                        # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
                        # in distributed settings to avoid race conditions.
                        weights_no_suffix = weights_name.replace(".safetensors", "")

                        # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
                        filename_no_suffix = filename.replace(".safetensors", "")
                        reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

                        if (
                            filename.startswith(weights_no_suffix)
                            and os.path.isfile(full_filename)
                            and filename not in state_dict_split.filename_to_tensors.keys()
                            and is_main_process
                            and reg.fullmatch(filename_no_suffix) is not None
                        ):
                            os.remove(full_filename)
                
                # Save the model
                for shard_file, tensors in state_dict_split.filename_to_tensors.items():
                    shard = {}
                    for tensor in tensors:
                        shard[tensor] = state_dict[tensor].contiguous()
                        del state_dict[tensor]
                    save_file(
                        shard,
                        os.path.join(save_directory, shard_file),
                        metadata={"format": "ht"},
                    )
                del state_dict
                
                if num_save_devices == 1 and not state_dict_split.is_sharded:
                    path_to_weights = os.path.join(save_directory, weights_name)
                    logging.info(f"Model weights saved in {path_to_weights}")
                else:
                    save_index_file = SAFE_WEIGHTS_INDEX_NAME.replace(".index", f".{local_save_index}-rank-{num_save_devices}.index")
                    save_index_file = os.path.join(save_directory, _add_variant(save_index_file, variant))
                    # Save the index as well
                    with open(save_index_file, "w", encoding="utf-8") as f:
                        content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                        f.write(content)

        # hetu.global_comm_barrier_rpc()
        if is_main_process:
            # merge index shards
            merged_index = {"metadata": {}, "weight_map": {}}
            for save_index in range(num_save_devices):
                index_file = SAFE_WEIGHTS_INDEX_NAME.replace(".index", f".{save_index}-rank-{num_save_devices}.index")
                index_file = os.path.join(save_directory, _add_variant(index_file, variant))
                with open(index_file, "r") as f:
                    index = json.loads(f.read())
                merged_index["metadata"].update(index["metadata"])
                merged_index["weight_map"].update(index["weight_map"])
            # save merged index file
            save_index_file = os.path.join(save_directory, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(merged_index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            # delete index shards
            for save_index in range(num_save_devices):
                index_file = SAFE_WEIGHTS_INDEX_NAME.replace(".index", f".{save_index}-rank-{num_save_devices}.index")
                index_file = os.path.join(save_directory, _add_variant(index_file, variant))
                os.remove(index_file)

            print(
                f"The model is saved by {num_save_devices} devices in {save_directory}. "
                f"You can find where each parameters has been saved in the index located at {save_index_file}."
            )
