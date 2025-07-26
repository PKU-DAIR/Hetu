"""Dataloaders."""

import torch
import torch.utils
import torch.utils.data

def build_pretraining_data_loader(dataset, consumed_samples, mbs, dp_rank, dp_size):
    """
        Buld dataloader given an input dataset.
        args: mbs, num_workers
    """

    if dataset is None:
        return None

    # batch sampler
    batch_sampler = GPTBatchSampler(
        total_samples=len(dataset),
        consumed_samples=consumed_samples,
        micro_batch_size=mbs,
        data_parallel_rank=dp_rank,
        data_parallel_size=dp_size)

    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       shuffle=False,
                                       num_workers=0, # num_workers>0 exists bugs with mpirun
                                       pin_memory=False)

def build_dynamic_data_loader(dataset, consumed_samples, mbs, dp_rank, dp_size, micro_batch_num_list=None):
    """
        Buld dataloader given an input dataset.
        args: mbs, num_workers
    """

    if dataset is None:
        return None

    # batch sampler
    print("LEN:", len(dataset),
          "micro_batch_size:", mbs,
          "data_parallel_rank:", dp_rank,
          "data_parallel_size:", dp_size)
    if micro_batch_num_list is None:
        batch_sampler = DynamicBatchSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=mbs,
            data_parallel_rank=dp_rank,
            data_parallel_size=dp_size)
    else:
        batch_sampler = DynamicHetoroBatchSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=mbs,
            data_parallel_rank=dp_rank,
            data_parallel_size=dp_size,
            micro_batch_num_list=micro_batch_num_list)

    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       shuffle=False,
                                       num_workers=0, # num_workers>0 exists bugs with mpirun
                                       pin_memory=False)

class GPTBatchSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]

class DynamicBatchSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        # for idx in range(self.consumed_samples, self.total_samples):
        idx = self.consumed_samples
        while True:
            batch.append(idx)
            idx += 1
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                print("Get BAtch:", batch, batch[start_idx:end_idx])
                # yield batch[:end_idx - start_idx]
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]


class DynamicHetoroBatchSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, micro_batch_num_list,
                 drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.micro_batch_num_list = micro_batch_num_list
        self.gbs = sum(micro_batch_num_list) * self.micro_batch_size
        self.mbn = micro_batch_num_list[self.data_parallel_rank]
        self.dp_start = 0
        self.dp_end = 0
        for i, bsz in enumerate(micro_batch_num_list):
            if i == self.data_parallel_rank:
                self.dp_end += self.micro_batch_size * bsz
                break
            self.dp_start += self.micro_batch_size * bsz
            self.dp_end += self.micro_batch_size * bsz   
        print(f"gbs:{self.gbs}, st:{self.dp_start}, ed:{self.dp_end}, mbn:{self.mbn}")
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        # for idx in range(self.consumed_samples, self.total_samples):
        idx = self.consumed_samples
        while True:
            batch.append(idx)
            idx += 1
            if len(batch) == self.gbs:
                for i in range(self.mbn):
                    st_idx = self.dp_start + i * self.micro_batch_size
                    ed_idx = st_idx + self.micro_batch_size

                    yield batch[st_idx:ed_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]