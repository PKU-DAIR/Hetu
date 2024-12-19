import numpy as np
import bisect
import math
from types import SimpleNamespace
from torch.utils.data import Dataset, DataLoader
from data_utils.gpt_load_dataset import build_encoder, load_truncated_dataset

class GPTJsonDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        key,
        max_seq_len,
        vocab_file,
        merge_file,
        encoder=None,
        root_folder='data'
    ):
        if encoder is None or encoder.args.key != key:
            self.encoder = build_encoder(key, vocab_file, merge_file)
        else:
            self.encoder = encoder
        cache_path = f'{root_folder}/{dataset_name}/{dataset_name}_cache.pkl'
        self.max_seq_len = max_seq_len
        self.data = load_truncated_dataset(dataset_name, self.encoder, max_seq_len, cache_path, root_folder)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        doc_ids = self.data[idx]
        max_seq_len = self.max_seq_len + 1
        if len(doc_ids) > max_seq_len:
            doc_ids = doc_ids[:max_seq_len]
        elif len(doc_ids) < max_seq_len:
            doc_ids += [self.encoder.pad_id()] * (max_seq_len - len(doc_ids))
        tokens = np.array(doc_ids)
        return tokens
    
    def get_length_distribution(self, min_seq_len=256, max_seq_len=16384, buckets=None):
        # Get length distribution
        seq_len_distribution = {}
        for doc_ids in self.data:
            sample_len = len(doc_ids) - doc_ids.count(self.encoder.pad_id())
            if buckets is None:
                padded_len = max(min(2 ** (sample_len.bit_length()), max_seq_len), min_seq_len)
            else:
                padded_len = buckets[bisect.bisect_left(buckets, sample_len)]
            seq_len_distribution[padded_len] = seq_len_distribution.get(padded_len, 0) + 1
        total_num = len(self.data)
        for seq_len, num in seq_len_distribution.items():
            seq_len_distribution[seq_len] = num / total_num
        return seq_len_distribution

    def get_aligned_buckets(self, alignment=16):
        buckets = set()
        for doc_ids in self.data:
            sample_len = len(doc_ids) - doc_ids.count(self.encoder.pad_id())
            padded_len = (sample_len + alignment - 1) // alignment * alignment
            buckets.add(padded_len)
        return buckets

    def get_distribution_of_fine_grained_buckets(self, gbs, seq_len_distribution, fine_grained_seq_len_distribution):
        seq_len_num_distribution = {k: math.ceil(p * gbs) if p * gbs < 1 else round(p * gbs) for k, p in seq_len_distribution.items()}
        seq_len_num_distribution = dict(sorted(seq_len_num_distribution.items(), key=lambda x: x[0]))
        fine_grained_seq_len_distribution = dict(sorted(fine_grained_seq_len_distribution.items(), key=lambda x: x[0]))
        fine_grained_seq_len_num_distribution = {}
        previous_bound = 0
        for k, v in seq_len_num_distribution.items():
            cum_p = 0
            dispatch_num = 0
            max_s = 0
            for s, p in fine_grained_seq_len_distribution.items():
                if s <= k and s > previous_bound:
                    cum_p += p
                    if max_s < s:
                        max_s = s
                elif s > k:
                    break
            for s, p in fine_grained_seq_len_distribution.items():
                if s <= k and s > previous_bound:
                    fine_grained_seq_len_num_distribution[s] = fine_grained_seq_len_num_distribution.get(s, 0) + round(v * p / cum_p)
                    dispatch_num += round(v * p / cum_p)
                    if dispatch_num > v:
                        fine_grained_seq_len_num_distribution[s] = fine_grained_seq_len_num_distribution.get(s, 0) - (dispatch_num - v)
                        dispatch_num = v
                elif s > k:
                    break
            if dispatch_num < v:
                fine_grained_seq_len_num_distribution[max_s] = fine_grained_seq_len_num_distribution.get(max_s, 0) + (v - dispatch_num)
            previous_bound = k
        return fine_grained_seq_len_num_distribution

def get_mask_and_position_ids(tokens, pad):
    batch_size, seq_length = tokens.shape
    attention_mask = np.not_equal(tokens, pad)
    position_ids = np.arange(0, seq_length, dtype=np.int64) # [1, seq_len]
    position_ids = np.tile(position_ids, [batch_size, 1]) # [batch_size, seq_len]
    return attention_mask, position_ids

def get_position_ids(gbs_per_dp, seq_len): 
    position_ids = np.arange(0, seq_len, dtype=np.int64) # [1, seq_len]
    position_ids = np.tile(position_ids, [gbs_per_dp, 1]) # [dp_size, seq_len]
    return position_ids

if __name__ == '__main__':
    root_folder = 'data'
    test_dataset = GPTJsonDataset(
        dataset_name='web/refinedweb0',
        key='content',
        max_seq_len=16384,
        vocab_file=f'{root_folder}/vocab.json',
        merge_file=f'{root_folder}/merges.txt')
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    for idx, tokens in enumerate(test_dataloader):
        if idx > 4:
            break
        attention_mask, position_ids = get_mask_and_position_ids(tokens, test_dataset.encoder.pad_id())
        print(f'batch {idx}: shape = {tokens.shape}\ntokens = {tokens}\nattention_mask={attention_mask}\nposition_ids={position_ids}')