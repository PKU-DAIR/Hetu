# TODO: The following implementation is incomplete

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
class Dataset():
    functions: Dict[str, Callable] = {}

    def __getitem__(self, index):
        raise NotImplementedError

    # def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
    #     return ConcatDataset([self, other])


    def __getattr__(self, attribute_name):
        if attribute_name in Dataset.functions:
            function = functools.partial(Dataset.functions[attribute_name], self)
            return function
        else:
            raise AttributeError

class IterableDataset(Dataset):
    functions: Dict[str, Callable] = {}
    reduce_ex_hook : Optional[Callable] = None

    def __iter__(self) -> Iterator:
        raise NotImplementedError

    # def __add__(self, other: Dataset[T_co]):
    #     return ChainDataset([self, other])

    # No `def __len__(self)` default? Subclasses raise `TypeError` when needed.
    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]

    def __getattr__(self, attribute_name):
        if attribute_name in IterableDataset.functions:
            function = functools.partial(IterableDataset.functions[attribute_name], self)
            return function
        else:
            raise AttributeError

    def __reduce_ex__(self, *args, **kwargs):
        if IterableDataset.reduce_ex_hook is not None:
            try:
                return IterableDataset.reduce_ex_hook(self)
            except NotImplementedError:
                pass
        return super().__reduce_ex__(*args, **kwargs)
