import math
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union

import numpy
import torch

from .blendedDataset import BlendedDataset
from .blendedHetuDatasetConfig import BlendedHetuDatasetConfig
from .hetuDataset import LowLevelDataset, HetuDataset
from .utils import Split, normalize


MidLevelDataset = Union[HetuDataset]

TopLevelDataset = Union[BlendedDataset, MidLevelDataset]

DistributedDataset = Union[
    TopLevelDataset, MidLevelDataset, LowLevelDataset, torch.utils.data.Dataset
]


class BlendedHetuDatasetBuilder(object):
    def __init__(
        self, cls: Type[MidLevelDataset], sizes: List[int], config: BlendedHetuDatasetConfig,
    ):
        self.cls = cls
        self.sizes = sizes
        self.config = config

    def build(self) -> List[Optional[TopLevelDataset]]:
        return self._build_blended_dataset_splits()

    def _build_blended_dataset_splits(self,) -> List[Optional[TopLevelDataset]]:
        
        # All splits come from the same distribution
        if self.config.blend:
            blend = self.config.blend
            split = self.config.split_matrix

            # Blend consists of a single prefix
            if len(blend) == 1:
                return self._build_hetu_dataset_splits(blend[0], split, self.sizes)

            # Blend consists of multiple weights and prefixes
            print("blend", blend)
            print("sizes", self.sizes)
            (
                prefix_per_dataset,
                weight_per_dataset,
                sizes_per_dataset,
            ) = _get_prefixes_weights_and_sizes_for_blend(blend, self.sizes)

            print("prefix_per_dataset", prefix_per_dataset)
            print("weight_per_dataset", weight_per_dataset)
            print("sizes_per_dataset", sizes_per_dataset)
            hetu_datasets = [[] for _ in range(len(Split))]

            for i in range(len(prefix_per_dataset)):
                hetu_datasets_split = self._build_hetu_dataset_splits(
                    prefix_per_dataset[i], split, sizes_per_dataset[i]
                )
                for j in range(len(hetu_datasets_split)):
                    hetu_datasets[j].append(hetu_datasets_split[j])

            # Sum over all contributing datasets, per split
            size_per_split = list(map(sum, zip(*sizes_per_dataset)))
            print("size_per_split", size_per_split)

            blended_datasets = []

            for i in range(len(hetu_datasets)):
                is_none = map(lambda _: _ is None, hetu_datasets[i])

                if split[i] is None:
                    assert all(is_none)
                    blended_datasets.append(None)
                else:
                    assert all(is_none) or not any(is_none)
                    blended_datasets.append(
                        self.build_generic_dataset(
                            BlendedDataset,
                            hetu_datasets[i],
                            weight_per_dataset,
                            size_per_split[i],
                            self.config,
                        )
                    )

            return blended_datasets

        # Each split comes from a separate distribution
        else:
            blended_datasets = []
            for i in range(len(Split)):
                blend = self.config.blend_per_split[i]

                # Blend is not provided
                if not blend:
                    blended_datasets.append(None)
                    continue

                split_spoof = [None] * len(Split)
                split_spoof[i] = (0.0, 1.0)
                sizes_spoof = [0] * len(Split)
                sizes_spoof[i] = self.sizes[i]

                # Blend consists of a sigle prefix
                if len(blend) == 1:
                    blended_datasets.append(
                        self._build_hetu_dataset_splits(blend[0], split_spoof, sizes_spoof)[i]
                    )

                # Blend consists of multiple weights and prefixes
                else:
                    (
                        prefix_per_dataset,
                        weight_per_dataset,
                        sizes_per_dataset,
                    ) = _get_prefixes_weights_and_sizes_for_blend(blend, sizes_spoof)

                    hetu_datasets = []
                    for j in range(len(prefix_per_dataset)):
                        hetu_datasets.append(
                            self._build_hetu_dataset_splits(
                                prefix_per_dataset[j], split_spoof, sizes_per_dataset[j],
                            )[i]
                        )

                    size_per_split = list(map(sum, zip(*sizes_per_dataset)))

                    blended_datasets.append(
                        self.build_generic_dataset(
                            BlendedDataset,
                            self.config.is_built_on_rank,
                            hetu_datasets,
                            weight_per_dataset,
                            size_per_split[i],
                            self.config,
                        )
                    )

            return blended_datasets



    def _build_hetu_dataset_splits(
        self, dataset_path: Optional[str], split: List[float], sizes: List[int],
    ) -> List[Optional[MidLevelDataset]]:

        if issubclass(self.cls, HetuDataset):
            low_level_dataset = self.cls.build_low_level_dataset(dataset_path, self.config)
        else:
            raise NotImplementedError

        if low_level_dataset is not None:
            num_elements = self.cls.numel_low_level_dataset(low_level_dataset)
            split_indices = []
            for i, _ in enumerate(Split):
                if split[i] is not None:
                    beg = int(round(split[i][0] * float(num_elements)))
                    end = int(round(split[i][1] * float(num_elements)))
                    split_indices.append(
                        numpy.arange(start=beg, stop=end, step=1, dtype=numpy.int32)
                    )
                else:
                    split_indices.append(None)
        else:
            split_indices = [None for _ in Split]

        # Build the mid level dataset
        mid_level_datasets = []
        for i, _split in enumerate(Split):
            mid_level_datasets.append(
                self.build_generic_dataset(
                    self.cls,
                    low_level_dataset,
                    dataset_path,
                    split_indices[i],
                    sizes[i],
                    _split,
                    self.config,
                )
            )

        return mid_level_datasets

    @staticmethod
    def build_generic_dataset(
        cls: Union[Type[DistributedDataset], Callable], *args: Any
    ) -> Optional[Union[DistributedDataset, Iterable]]:

        dataset = None

        try:
            dataset = cls(*args)
        except OSError as err:
            log = (
                f"Failed to write dataset materials to the data cache directory. "
                + f"Please supply a directory to which you have write access via "
                + f"the path_to_cache attribute in BlendedHetuDatasetConfig and "
                + f"retry. Refer to the preserved traceback above for more information."
            )
            raise Exception(log) from err

        return dataset



def _get_prefixes_weights_and_sizes_for_blend(
    blend: List[str], target_num_samples_per_split: List[int]
) -> Tuple[List[str], List[float], List[List[int]]]:
    weights, prefixes = zip(
        *[(float(blend[i]), blend[i + 1].strip()) for i in range(0, len(blend), 2)]
    )

    weights = normalize(weights)

    # Use 0.5% target margin to ensure we satiate the network
    sizes_per_dataset = [
        [
            int(math.ceil(target_num_samples * weight * 1.005))
            for target_num_samples in target_num_samples_per_split
        ]
        for weight in weights
    ]

    return prefixes, weights, sizes_per_dataset
