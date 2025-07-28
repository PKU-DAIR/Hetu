import os
import json
import pickle
import time
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from .indexed_dataset import MMapIndexedDataset
from .blendedHetuDatasetConfig import BlendedHetuDatasetConfig
from .hetuDataset import HetuDataset
from typing import Dict
import torch

from .utils import Split
import sys
from dataclasses import dataclass

@dataclass
class LLaMaDatasetConfig(BlendedHetuDatasetConfig):

    reset_position_ids: bool = None

    reset_attention_mask: bool = None

    eod_mask_loss: bool = None

    vocab_size: int = sys.maxsize

    def __post_init__(self) -> None:
        """Do asserts and set fields post init
        """
        super().__post_init__()

        assert self.reset_position_ids is not None
        assert self.reset_attention_mask is not None
        assert self.eod_mask_loss is not None



class LLaMAJsonDataset(HetuDataset):
    def __init__(
        self,
        indexed_dataset: MMapIndexedDataset,
        dataset_path: str,
        indexed_indices: np.ndarray,
        num_samples: int,
        index_split: Split,
        config: LLaMaDatasetConfig,
    ) -> None:
        super().__init__(
            indexed_dataset, dataset_path, indexed_indices, num_samples, index_split, config
        )

        self.vocab_size = config.vocab_size 
        self.sample_index = self._build_sample_shuffle_indices()

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: MMapIndexedDataset) -> int:
        return low_level_dataset.sequence_lengths.shape[0]


    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: LLaMaDatasetConfig) -> MMapIndexedDataset:
        return MMapIndexedDataset(dataset_path, False)

    def __len__(self) -> int:
        return self.sample_index.shape[0]


    def _build_sample_shuffle_indices(
        self,
    ) -> np.ndarray:

        path_to_cache = self.config.path_to_cache
        if path_to_cache is None:
            path_to_cache = os.path.join(
                self.dataset.path_prefix, "cache", f"{type(self).__name__}_indices"
            )

        get_path_to = lambda suffix: os.path.join(
            path_to_cache, f"{self.unique_description_hash}-{type(self).__name__}-{suffix}"
        )
        path_to_sample_index = get_path_to("sample_index.npy")


        
        if os.path.isfile(path_to_sample_index) == False:
            print(f"\tBuild and save the sample index to {os.path.basename(path_to_sample_index)}")
            t_beg = time.time()
            sequence_length = self.config.sequence_length
            num_whole_dataset = self.num_samples // self.indices.shape[0]
            random_size = self.num_samples % self.indices.shape[0]
            
            full_repeats = np.tile(self.indices, num_whole_dataset)
            numpy_random_state = np.random.RandomState(self.config.random_seed)
            random_elements = numpy_random_state.choice(self.indices, random_size, replace=True)

            sample_index = np.concatenate((full_repeats, random_elements))
            np.random.shuffle(sample_index)

            os.makedirs(path_to_cache, exist_ok=True)
            np.save(path_to_sample_index, sample_index, allow_pickle=True)
            print("save sample_index to", path_to_sample_index, os.path.isfile(path_to_sample_index))
            t_end = time.time()
            print(f"\t> time elapsed: {t_end - t_beg:4f} seconds")
            
            return sample_index

        sample_index = np.load(path_to_sample_index, allow_pickle=True, mmap_mode='r')
        return sample_index

    def __getitem__(self, idx):
        ids = self.sample_index[idx]
        sequence = self.dataset[ids]

        sequence_length = self.config.sequence_length + 1
        if sequence.shape[0] > sequence_length:
            sequence = sequence[:sequence_length]
        elif sequence.shape[0] < sequence_length:
            sequence = list(sequence) 
            sequence += [self.config.tokenizer.pad] * (sequence_length - len(sequence))
            sequence = np.array(sequence)
        sequence = sequence.astype(np.int64)
        return sequence

