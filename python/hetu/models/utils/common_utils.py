import hetu
from huggingface_hub.serialization import split_state_dict_into_shards_factory
from typing import Dict, Union

SAFETENSORS_WEIGHTS_FILE_PATTERN = "model{suffix}.safetensors"
MAX_SHARD_SIZE = "5GB"

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

def get_hetu_storage_size(tensor: "hetu.NDArray") -> int:
    return tensor.numel() * _SIZE[tensor.dtype]

def get_hetu_storage_id(tensor: "hetu.NDArray") -> int:
    return tensor.device, tensor.data_ptr, get_hetu_storage_size(tensor)

def split_hetu_state_dict_into_shards(
    state_dict: Dict[str, "hetu.NDArray"],
    *,
    filename_pattern: str = SAFETENSORS_WEIGHTS_FILE_PATTERN,
    max_shard_size: Union[int, str] = MAX_SHARD_SIZE,
):
    return split_state_dict_into_shards_factory(
        state_dict,
        max_shard_size=max_shard_size,
        filename_pattern=filename_pattern,
        get_storage_size=get_hetu_storage_size,
        get_storage_id=get_hetu_storage_id,
    )
