from hetu.data.dataloader import DataLoadLevel
from hetu.utils.parallel import StrategyConfig
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrainingConfig:
    """
    Configuration class for training settings.
    
    This class defines all common parameters for model training, including
    output directories, optimization settings, data loading parameters and
    profiling options.
    """
    
    output_dir: str = field(
        default="./output",
        metadata={"help": "Directory to save the training outputs"},
    )
    
    overwrite_output_dir: bool = field(
        default=False,
        metadata={"help": "Whether to overwrite the output directory"},
    )
    
    plot_loss: bool = field(
        default=False,
        metadata={"help": "Whether to plot the loss"},
    )
    
    plot_update_freq: int = field(
        default=10,
        metadata={"help": "Frequency to plot the loss"},
    )
    
    bf16: bool = field(
        default=False,
        metadata={"help": "Whether to use bfloat16"},
    )
    
    packing: bool = field(
        default=True,
        metadata={"help": "Whether to pack the input sequences for faster training"},
    )
    
    micro_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Micro batch size for training, used in padding mode"},
    )
    
    ds_parallel: Optional[StrategyConfig] = field(
        default=None,
        metadata={"help": "Data parallel configuration"},
    )
    
    global_load_size: int = field(
        default=64,
        metadata={"help":
            "Global load size for data loading, depending on the data loading level. "
            "If the data loading level is 'sample', this is the number of samples. "
            "If the data loading level is 'token', this is the number of tokens."
        },
    )
    
    data_load_level: DataLoadLevel = field(
        default=DataLoadLevel.SAMPLE,
        metadata={"help": "Data loading level (sample, token)"},
    )
    
    torch_profile: bool = field(
        default=False,
        metadata={"help": "Whether to profile PyTorch operations"},
    )
    
    start_profile_step: int = field(
        default=1,
        metadata={"help": "Step to start profiling"},
    )
    
    end_profile_step: int = field(
        default=5,
        metadata={"help": "Step to end profiling"},
    )
    
    profile_save_path: str = field(
        default="./trace",
        metadata={"help": "Path to save the profiling results"},
    )
    
    train_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the training dataset"},
    )
    
    dataset_text_field: Optional[str] = field(
        default=None,
        metadata={"help": "Field name for text content in the dataset"},
    )
    
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum sequence length for input sequences"},
    )
    
    steps: int = field(
        default=1000,
        metadata={"help": "Number of training steps"},
    )

    learning_rate: float = field(
        default=0.001,
        metadata={"help": "Learning rate"},
    )