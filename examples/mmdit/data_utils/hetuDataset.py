import hashlib
import json
from abc import abstractmethod
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Union

import numpy
import torch

from .blendedHetuDatasetConfig import BlendedHetuDatasetConfig
from .indexed_dataset import MMapIndexedDataset
from .utils import Split

LowLevelDataset = Union[MMapIndexedDataset, Iterable]


class HetuDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: str,
        indices: numpy.ndarray,
        num_samples: int,
        index_split: Split,
        config: BlendedHetuDatasetConfig,
    ) -> None:
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.indices = indices
        self.num_samples = num_samples
        self.index_split = index_split
        self.config = config


        self.unique_identifiers = OrderedDict()
        self.unique_identifiers["class"] = type(self).__name__
        self.unique_identifiers["dataset_path"] = self.dataset_path
        self.unique_identifiers["num_samples"] = self.num_samples
        self.unique_identifiers["index_split"] = self.index_split.name
        for attr in self._key_config_attributes():
            self.unique_identifiers[attr] = getattr(self.config, attr)

        self.unique_description = json.dumps(
            self.unique_identifiers, indent=4, default=lambda obj: obj.unique_identifiers
        )
        self.unique_description_hash = hashlib.md5(
            self.unique_description.encode("utf-8")
        ).hexdigest()

        self._finalize()

    def _finalize(self) -> None:
        pass

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        raise NotImplementedError

    @staticmethod
    def build_low_level_dataset(
        dataset_path: str, config: BlendedHetuDatasetConfig
    ) -> LowLevelDataset:
        raise NotImplementedError

    @staticmethod
    def _key_config_attributes() -> List[str]:
        return ["random_seed", "sequence_length", "split", "split_matrix"]

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, numpy.ndarray]]:
        pass