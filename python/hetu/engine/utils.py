import hetu as ht
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Args:
    @staticmethod
    def print_args_list(args_list: List["Args"]):
        for item in args_list:
            print(item)
            
    def _type_check(self, value, *types):
        for cur_type in types:
            if isinstance(value, cur_type):
                return value
        raise TypeError(f"Value {value} must in types {types}.")

@dataclass
class TrainerCtxs(Args):
    bf16: bool
    hetero_tp_alpha: List[float]
    hetero_tp_weight: List[float]
    normal_layers: int
    normal_mbn: int
    normal_compute_time: int
    memory_k: List[float]
    memory_embedding: float
    memory_extra: float
    memory_bound: float
    memory_safe_gap: float
    straggler_threshold: float
    straggler_safe_gap: float
    top_k: int

@dataclass
class TrainerDatasetArgs(Args):
    dataset: Any
    consumed_samples: int
    steps: int
    epochs: int
    step: int
    epoch: int

@dataclass
class TrainerStrategyArgs(Args):
    dp: int
    tp: int
    pp: int
    zero: bool
    rank_to_device_mapping: Dict[int, Any]
    suspended_rank_list: List[int]
    unused_rank_list: List[int]
    hetero_data: bool
    hetero_layers: List[List[int]]
    hetero_stages: List[int]
    hetero_micro_batch_num_list: List[int]

@dataclass
class TrainerCommArgs(Args):
    input_ds_union: "ht.DistributedStatesUnion"
    input_device_group_union: List["ht.DeviceGroup"]
    label_ds_union: "ht.DistributedStatesUnion"
    label_device_group_union: List["ht.DeviceGroup"]
    local_device: "ht.device"
    all_devices: "ht.DeviceGroup"

@dataclass
class TrainerCommAllArgs(Args):
    input_ds_hierarchy: "ht.DistributedStatesHierarchy"
    input_dg_hierarchy: "ht.DeviceGroupHierarchy"
    label_ds_hierarchy: "ht.DistributedStatesHierarchy"
    label_dg_hierarchy: "ht.DeviceGroupHierarchy"
    local_device: "ht.device"
    all_devices: "ht.DeviceGroup"

@dataclass
class TrainerEnvs(Args):
    run_straggler_experiment: bool
    run_memory_experiment: bool
    straggler_file: str
    memory_file: str
    elastic: bool
