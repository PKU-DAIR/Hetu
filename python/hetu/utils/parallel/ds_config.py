from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Any

# generated recompute config
@dataclass
class RecomputeConfig:
    """
    Configuration for recomputing specific layers during training to save memory.
    
    Attributes:
        recompute_granularity: List of granularity for each DCP (DeepSpeed Computation Partition).
        recompute_layer_idxs_list: List of layer indices to recompute for each DCP.
        blocks_recompute: Boolean flags for recomputing each layer within each DCP.
        blocks_output_recompute: Boolean flags for recomputing outputs of each layer within each DCP.
    """
    recompute_granularity: List[Optional[str]]  # dcp_size
    recompute_layer_idxs_list: List[List[int]]  # dcp_size * [recompute_layer_1, recompute_layer_2, ..., recompute_layer_dcp_k]
    blocks_recompute: List[List[bool]]  # total_layers * dcp_size
    blocks_output_recompute: List[List[bool]]  # total_layers * dcp_size

@dataclass
class RecomputeStrategy:
    """
    Strategy for configuring recomputation settings.
    
    Attributes:
        recompute_granularity: Granularity level for recomputation.
        recompute_method: Method used for recomputation.
        recompute_num_layers: Number of layers to recompute.
        recompute_layer_idxs_list: Specific layer indices to recompute.
    """
    recompute_granularity: Any = field(
        default=None,
        metadata={"help": "Recompute granularity"},
    )
    recompute_method: Any = field(
        default=None,
        metadata={"help": "Recompute method"},
    )
    recompute_num_layers: Any = field(
        default=None,
        metadata={"help": "Recompute number of layers"},
    )
    recompute_layer_idxs_list: Any = field(
        default=None,
        metadata={"help": "Recompute layer index list"},
    )
    
    def __post_init__(self):
        """
        Validate the recompute strategy configuration after initialization.
        
        Raises:
            TypeError: If any of the configuration values have incorrect types.
        """
        if self.recompute_num_layers is not None:
            if not isinstance(self.recompute_num_layers, (int, list)):
                raise TypeError("recompute_num_layers should be an integer or a list of integers")
            if isinstance(self.recompute_num_layers, list):
                for item in self.recompute_num_layers:
                    if not isinstance(item, int):
                        raise TypeError("recompute_num_layers should be an integer or a list of integers")
        
        if self.recompute_layer_idxs_list is not None:
            if not isinstance(self.recompute_layer_idxs_list, (int, list)):
                raise TypeError("recompute_layer_idxs_list should be an integer, a list of integers or a list of lists of integers")
            if isinstance(self.recompute_layer_idxs_list, list):
                # Support nested lists List[List[int]]
                for item in self.recompute_layer_idxs_list:
                    if isinstance(item, list):
                        if not all(isinstance(x, int) for x in item):
                            raise TypeError("recompute_layer_idxs_list contains a list with non-integer elements")
                    elif not isinstance(item, int):
                        raise TypeError("recompute_layer_idxs_list should contain only integers or lists of integers")
        
        if self.recompute_method is not None:
            if not isinstance(self.recompute_method, (str, list)):
                raise TypeError("recompute_method should be a string or a list of strings")
        
        if self.recompute_granularity is not None:
            if not isinstance(self.recompute_granularity, (str, list)):
                raise TypeError("recompute_granularity should be a string or a list of strings")

@dataclass
class StrategyConfig:
    """
    Configuration for distributed training strategy.
    
    Attributes:
        hetero: Whether to use heterogeneous training across devices.
        dp: Number of data parallel stages.
        cp: Number of context parallel stages.
        tp: Number of tensor parallel stages.
        pp: Number of pipeline parallel stages.
        zero: Whether to use ZeRO optimizer.
        num_layers: Number of model layers.
        num_gpus: Number of GPUs to use.
        ds_parallel_config_path: Path to DeepSpeed parallel configuration file.
        ds_parallel_config_name: Name of DeepSpeed parallel configuration.
        rank_to_device_mapping: Mapping from ranks to device IDs for heterogeneous training.
        unused_rank: List of unused ranks for heterogeneous training.
        gpus_per_stage: Number of GPUs per pipeline stage for heterogeneous training.
        hetero_layers: Layer mapping for heterogeneous training.
        micro_batch_num_list: Micro batch number list for heterogeneous training.
        seq_len_list: Sequence length list for heterogeneous training.
        cp_list: CP degree list for heterogeneous training.
        recompute: Recomputation strategy configuration.
    """
    hetero: bool = field(
        default=False,
        metadata={"help": "Whether to use heterogeneous training"},
    )
    
    dp: int = field(
        default=1,
        metadata={"help": "Number of data parallel stages"},
    )
    
    cp: Optional[int] = field(
        default=1,
        metadata={"help": "Number of context parallel stages"},
    )
    
    tp: int = field(
        default=1,
        metadata={"help": "Number of tensor parallel stages"},
    )
    
    pp: int = field(
        default=1,
        metadata={"help": "Number of pipeline parallel stages"},
    )
    
    zero: bool = field(
        default=False,
        metadata={"help": "Whether to spread zero"},
    )
    
    num_layers: int = field(
        default=1,
        metadata={"help": "Number of layers"},
    )
    
    num_gpus: int = field(
        default=1,
        metadata={"help": "Number of GPUs"},
    )
    
    ds_parallel_config_path: Optional[str] = field(
        default=None,
        metadata={"help": "Data parallel configuration path for distributed training"},
    )
    
    ds_parallel_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Data parallel configuration name for distributed training"},
    )
    
    rank_to_device_mapping: Optional[Dict[int, int]] = field(
        default=None,
        metadata={"help": "Rank to device mapping for heterogeneous training"},
    )
    
    unused_rank: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Unused ranks for heterogeneous training"},
    )
    
    gpus_per_stage: Optional[int] = field(
        default=None,
        metadata={"help": "Number of GPUs per pipeline stage for heterogeneous training"},
    )
    
    hetero_layers: Optional[List[List[int]]] = field(
        default=None,
        metadata={"help": "Layer mapping for heterogeneous training"},
    )
    
    micro_batch_num_list: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Micro batch number list for heterogeneous training"},
    )
    
    seq_len_list: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Sequence length list for heterogeneous training"},
    )
    
    cp_list: Optional[List[int]] = field(
        default=None,
        metadata={"help": "CP degree list for heterogeneous training"},
    )
    
    recompute: Optional[RecomputeStrategy] = field(
        default=None,
        metadata={"help": "Recompute configuration for heterogeneous training"},
    )
    
    def __post_init__(self):
        """
        Validate and set default values after initialization.
        
        Validates heterogeneous configuration when enabled and initializes cp_list if not provided.
        """
        if self.hetero:
            assert self.rank_to_device_mapping is not None
            assert self.hetero_layers is not None
        if self.cp_list is None:
            self.cp_list = [self.cp] * self.dp