import torch
import numpy as np
from enum import Enum
from typing import List, Optional, Callable, Union, Dict, Any
from torch.utils.data import Dataset

class DataLoadLevel(str, Enum):
    SAMPLE = "sample"
    TOKEN = "token"

class BatchSampleLevel(str, Enum):
    GLOBAL = "global"
    LOCAL = "local"

def truncate_fn(
    batch: List[Union[np.ndarray, Dict[str, np.ndarray]]],
    pad_id: int,
    global_token_num: int
) -> List[Union[np.ndarray, Dict[str, np.ndarray]]]:
    """Truncate the last sequence in the batch to match the global token num

    Args:
        batch (List[np.ndarray]): global batch
        pad_id (int): the pad token id of the tokenizer
        global_token_num (int): target global batch token num

    Returns:
        List[np.ndarray]: Truncated batch whose total token num equals to `global_token_num`
    """

    valid_token_num = sum([np.sum(seq != pad_id) for seq in batch])
    if isinstance(batch[-1], dict):
        last_seq_valid_token = np.sum(batch[-1]["input_ids"] != pad_id)
    else:
        last_seq_valid_token = np.sum(batch[-1] != pad_id)
    assert (
        valid_token_num >= global_token_num and 
        valid_token_num - global_token_num < last_seq_valid_token
    ), "cannot truncate the last seq"
    if isinstance(batch[-1], dict):
        batch[-1]["input_ids"][last_seq_valid_token - (valid_token_num - global_token_num):] = pad_id
    else:
        batch[-1][last_seq_valid_token - (valid_token_num - global_token_num):] = pad_id
    return batch

def build_data_loader(
    dataset: Dataset,
    consumed_samples: int,
    global_load_size: int,
    data_load_level: DataLoadLevel = DataLoadLevel.SAMPLE,
    data_collator: Optional[Callable] = None,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader with specified batch sampling strategy.

    This function creates a DataLoader that supports two loading levels:
    - sample level: batch size is determined by number of samples
    - token level: batch size is determined by total number of tokens

    Args:
        dataset (Dataset): The dataset to load
        consumed_samples int: Number of samples already consumed.
        global_load_size int: Global batch size (samples) or token number (tokens).
        data_load_level (DataLoadLevel, optional): Loading strategy, either "sample" or "token". 
            Defaults to "sample"
        data_collator (Callable, optional): Custom collator function. Defaults to None

    Returns:
        torch.utils.data.DataLoader: The configured data loader

    Example:
        >>> loader = build_data_loader(
        ...     dataset=my_dataset,
        ...     consumed_samples=0,
        ...     global_load_size=2048,
        ...     data_load_level="token"
        ... )
    """

    if data_load_level == DataLoadLevel.SAMPLE:
        collate_fn = data_collator
        batch_sampler = SampleLevelBatchSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            batch_size=global_load_size,
        )
    elif data_load_level == DataLoadLevel.TOKEN:
        collate_fn = lambda batch: data_collator(truncate_fn(batch, dataset.pad_id(), global_load_size))
        batch_sampler = TokenLevelBatchSampler(
            dataset=dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            token_num=global_load_size,
        )

    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )

def build_dist_data_loader(
    dataset: Dataset,
    consumed_samples: int,
    micro_batch_size: int,
    dp_rank: int,
    dp_size: int,
    data_load_level: DataLoadLevel = DataLoadLevel.SAMPLE,
) -> torch.utils.data.DataLoader:
    """Build a distributed DataLoader for parallel training.

    This function creates a DataLoader suitable for distributed training scenarios. 
    It currently only supports sample-level batch sampling.

    Args:
        dataset (Dataset): The dataset to load
        consumed_samples (int): Number of samples already consumed
        micro_batch_size (int): Size of micro batches for each data parallel worker
        dp_rank (int): Rank of current data parallel process
        dp_size (int): Total number of data parallel processes
        data_load_level (DATA_LOAD_LEVEL, optional): Loading strategy, must be "sample".
            Defaults to "sample"

    Returns:
        torch.utils.data.DataLoader: The configured distributed data loader

    Example:
        >>> loader = build_dist_data_loader(
        ...     dataset=my_dataset,
        ...     consumed_samples=0,
        ...     micro_batch_size=32,
        ...     dp_rank=0,
        ...     dp_size=8,
        ...     data_load_level="sample"
        ... )

    Raises:
        AssertionError: If data_load_level is not "sample"
    """

    assert data_load_level == "sample", \
        "only sample level is supported for distributed dataloader"
    
    batch_sampler = SampleLevelBatchSampler(
        total_samples=len(dataset),
        consumed_samples=consumed_samples,
        batch_size=micro_batch_size,
        data_parallel_rank=dp_rank,
        data_parallel_size=dp_size,
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

class SampleLevelBatchSampler:
    """A batch sampler that yields indices based on sample count.

    This sampler creates batches with fixed number of samples, regardless of 
    the token count in each sample. It supports partial batch dropping.

    Args:
        total_samples (int): Total number of samples in dataset
        consumed_samples (int): Number of samples already consumed
        batch_size (int): Number of samples in each batch
        data_parallel_rank (Optional[int], optional): Rank of the data parallel process.
            Defaults to None
        data_parallel_size (Optional[int], optional): Number of data parallel processes.
            Defaults to None
        drop_last (bool, optional): Whether to drop the last incomplete batch. 
            Defaults to True

    Example:
        >>> sampler = SampleLevelBatchSampler(
        ...     total_samples=1000,
        ...     consumed_samples=0,
        ...     batch_size=32
        ... )
    """

    def __init__(
        self,
        total_samples: int,
        consumed_samples: int,
        batch_size: int,
        data_parallel_rank: Optional[int] = None,
        data_parallel_size: Optional[int] = None,
        drop_last: bool = True,
    ):
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        if data_parallel_rank is not None and data_parallel_size is not None:
            self.batch_sample_level = BatchSampleLevel.LOCAL
            self.data_parallel_size = data_parallel_size
            self.data_parallel_rank = data_parallel_rank
            self.micro_batch_size = batch_size
            self.sample_batch_size = self.micro_batch_size * data_parallel_size
        else:
            self.batch_sample_level = BatchSampleLevel.GLOBAL
            self.sample_batch_size = batch_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, \
            f"no sample to consume: {self.total_samples}"
        assert self.consumed_samples < self.total_samples, \
            f"no samples left to consume: {self.consumed_samples}, {self.total_samples}"

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.sample_batch_size:
                if self.batch_sample_level == BatchSampleLevel.LOCAL:
                    start_idx, end_idx = self.get_start_end_idx()
                    yield batch[start_idx:end_idx]
                elif self.batch_sample_level == BatchSampleLevel.GLOBAL:
                    yield batch
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            if self.batch_sample_level == BatchSampleLevel.LOCAL:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
            elif self.batch_sample_level == BatchSampleLevel.GLOBAL:
                yield batch
            
class TokenLevelBatchSampler:
    """A batch sampler that yields indices based on token count.

    This sampler creates dynamic-sized batches where the total token count in each batch
    is approximately equal to the specified global token number. (May be higher than the target
    due to the fixed sample length. Need to truncate the last sequence by `truncate_fn` to match the target.)
    It supports partial batch dropping.

    Args:
        dataset (Dataset): The dataset containing samples
        total_samples (int): Total number of samples in dataset
        consumed_samples (int): Number of samples already consumed
        token_num (int): Target number of tokens in each batch
        drop_last (bool, optional): Whether to drop the last incomplete batch.
            Defaults to True

    Example:
        >>> sampler = TokenLevelBatchSampler(
        ...     dataset=my_dataset,
        ...     total_samples=1000,
        ...     consumed_samples=0,
        ...     token_num=2048
        ... )
    """

    def __init__(
        self,
        dataset: Dataset,
        total_samples: int,
        consumed_samples: int,
        token_num: int,
        drop_last: bool = True,
    ):
        assert hasattr(dataset, 'data'), "dataset does not have 'data' attribute"
        assert hasattr(dataset, 'pad_id'), "dataset does not have 'pad_id' method"
        assert callable(getattr(dataset, 'pad_id')), "dataset 'pad_id' method is not callable"
        
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.token_num = token_num
        self.drop_last = drop_last

        # Sanity checks
        assert self.total_samples > 0, \
            f"no sample to consume: {self.total_samples}"
        assert self.consumed_samples < self.total_samples, \
            f"no samples left to consume: {self.consumed_samples}, {self.total_samples}"

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        batch = []
        current_token_count = 0
        for idx in range(self.consumed_samples, self.total_samples):
            sample = self.dataset.data[idx]
            sample_token_count = np.sum(np.array(sample) != self.dataset.pad_id()) # 获取当前样本的token数
            batch.append(idx)
            if current_token_count + (sample_token_count - 1) >= self.token_num:
                yield batch
                batch = []
                current_token_count = 0
            else:
                current_token_count += (sample_token_count - 1)

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            yield batch