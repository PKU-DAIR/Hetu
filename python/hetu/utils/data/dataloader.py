# TODO: The following implementation is incomplete

import hetu
from . import Dataset, IterableDataset
import functools
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)
class DataLoader():
    # dataset: Dataset
    batch_size: Optional[int]
    num_workers: int
    drop_last: bool
    __initialized = False

    def __init__(self, dataset, batch_size: Optional[int] = 1,
                 shuffle: bool = False, num_workers: int = 0, 
                 pin_memory: bool = False, drop_last: bool = False):
        if num_workers < 0:
            raise ValueError('num_workers option should be non-negative; '
                             'use num_workers=0 to disable multiprocessing.')

        self.dataset = dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataloader_c = hetu.Dataloader(dataset, batch_size, num_workers,
                                            "default", shuffle, drop_last)
        print(self.dataloader_c.batch_num)

        # Arg-check dataset related before checking samplers because we want to
        # tell users that iterable-style datasets are incompatible with custom
        # samplers first, so that they don't learn that this combo doesn't work
        # after spending time fixing the custom sampler errors.
        if isinstance(dataset, IterableDataset):
            # self._dataset_kind = _DatasetKind.Iterable
            if shuffle is not False:
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "shuffle option, but got shuffle={}".format(shuffle))

            batch_size = None
            drop_last = False
        elif batch_size is None:
            # no auto_collation
            if drop_last:
                raise ValueError('batch_size=None option disables auto-batching '
                                 'and is mutually exclusive with drop_last')


        self.batch_size = batch_size
        self.drop_last = drop_last



        self.__initialized = True
        self._IterableDataset_len_called = None 

        self._iterator = None

    # def _get_iterator(self):
    #     if self.num_workers == 0:
    #         return _SingleProcessDataLoaderIter(self)
    #     else:
    #         self.check_worker_number_rationality()
    #         return _MultiProcessingDataLoaderIter(self)

    def __setattr__(self, attr, val):
        if self.__initialized and attr in (
                'batch_size', 'batch_sampler', 'sampler', 'drop_last', 'dataset', 'persistent_workers'):
            raise ValueError('{} attribute should not be set after {} is '
                             'initialized'.format(attr, self.__class__.__name__))

        super(DataLoader, self).__setattr__(attr, val)

    # We quote '_BaseDataLoaderIter' since it isn't defined yet and the definition can't be moved up
    # since '_BaseDataLoaderIter' references 'DataLoader'.
    def __iter__(self):
        return self.dataloader_c

    def __len__(self) -> int:
        return self.dataloader_c.sample_num