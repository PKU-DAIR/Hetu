"""Dataloaders."""

import torch
import numpy as np

def numpy_collate_fn(batch):
    return np.array(batch)

def build_bucket_global_data_loader(dataset, consumed_samples, gbs,
                                    min_seq_length=256, max_seq_length=8192):
    """
        Buld dataloader given an input dataset.
        args: mbs, num_workers
    """

    if dataset is None:
        return None

    seq_bucket = {}
    seq_num = {}
    dataset_size = len(dataset)
    pad_id = dataset.encoder.pad_id()
    # 分析数据集的分布，得到seq_distribution
    for idx, tokens in enumerate(dataset.data):
        effective_len = len(tokens) - tokens.count(pad_id)
        padded_len = max(min(2 ** (int(effective_len).bit_length()), max_seq_length), min_seq_length)
        seq_bucket.setdefault(padded_len, []).append(idx)
        seq_num[padded_len] = seq_num.get(padded_len, 0) + 1
    seq_distribution = {}
    for seq_len in seq_num.keys():
        seq_distribution[seq_len] = seq_num[seq_len] / dataset_size

    # batch sampler
    batch_sampler = BucketGlobalBatchSampler(
        total_samples=dataset_size,
        consumed_samples=consumed_samples,
        seq_distribution=seq_distribution,
        seq_bucket=seq_bucket,
        global_batch_size=gbs,
        prefetch_batch_num=dataset_size // gbs)

    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       shuffle=False,
                                       collate_fn=numpy_collate_fn,
                                       num_workers=0, # num_workers>0 exists bugs with mpirun
                                       pin_memory=False)

class BucketGlobalBatchSampler:
    def __init__(self, total_samples, consumed_samples, seq_distribution, seq_bucket,
                 global_batch_size, prefetch_batch_num=8, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.seq_bucket = seq_bucket
        self.global_batch_size = global_batch_size
        self.prefetch_batch_num = prefetch_batch_num
        self.drop_last = drop_last
        
        self.seq_prefetch_size = {}
        self.prefetch_num = prefetch_batch_num * global_batch_size
        seq_num = 0
        seq_with_max_num = max(seq_distribution.keys(), key=lambda x: seq_distribution[x])
        for l, p in seq_distribution.items():
            self.seq_prefetch_size[l] = round(p * self.prefetch_num)
            seq_num += self.seq_prefetch_size[l]
        self.seq_prefetch_size[seq_with_max_num] += (self.prefetch_num - seq_num)
        self.seq_counter = {seq_len: 0 for seq_len in self.seq_prefetch_size.keys()}

        # Sanity checks.
        assert self.drop_last == True, \
            'drop_last shold be True for GPTBucketGlobalBatchSampler'
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        batch = []
        for _ in range(0, self.total_samples - self.consumed_samples, self.prefetch_num):
            prefetch_div_bucket = {}
            prefetch_rem_bucket = {}
            prefetch_rem_list = []
            prefetch_rem_idx = 0

            # check if all samples are consumed
            exit_flag = False
            for seq_len, num in self.seq_prefetch_size.items():
                if self.seq_counter[seq_len] + num > len(self.seq_bucket[seq_len]):
                    exit_flag = True
            if exit_flag:
                break

            for seq_len, num in self.seq_prefetch_size.items():
                prefetch_div_bucket.setdefault(seq_len, []) \
                                   .extend(self.seq_bucket[seq_len][self.seq_counter[seq_len]: self.seq_counter[seq_len] + num - num % self.prefetch_batch_num])
                prefetch_rem_bucket.setdefault(seq_len, []) \
                                   .extend(self.seq_bucket[seq_len][self.seq_counter[seq_len] + num - num % self.prefetch_batch_num: self.seq_counter[seq_len] + num])
                self.seq_counter[seq_len] += num
            prefetch_div_keys = sorted(prefetch_div_bucket.keys())
            prefetch_rem_keys = sorted(prefetch_rem_bucket.keys(), reverse=True)
            prefetch_rem_num = np.sum([len(prefetch_rem_bucket[seq_len]) for seq_len in prefetch_rem_keys])
            while len(prefetch_rem_list) < prefetch_rem_num:
                for seq_len in prefetch_rem_keys:
                    if len(prefetch_rem_bucket[seq_len]) > 0:
                        prefetch_rem_list.append(prefetch_rem_bucket[seq_len].pop(0))
            
            for i in range(self.prefetch_batch_num):
                for seq_len in prefetch_div_keys:
                    batch_seq_num = len(prefetch_div_bucket[seq_len]) // self.prefetch_batch_num
                    batch.extend(prefetch_div_bucket[seq_len][i * batch_seq_num: (i + 1) * batch_seq_num])
                if len(batch) < self.global_batch_size:
                    rem_size = self.global_batch_size - len(batch)
                    batch.extend(prefetch_rem_list[prefetch_rem_idx: prefetch_rem_idx + rem_size])
                    prefetch_rem_idx += rem_size
                yield batch
                batch = []
