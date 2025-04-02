import os
import numpy as np
import logging
from typing import List, Optional, Union, Dict
from collections import defaultdict
from hetu.data import IGNORE_INDEX

class Bucket:
    """Bucket is a container for a batch of data, which is used for padding and packing.
    The inner batch is a dictionary of numpy arrays, where the key is the field name
    such as 'input_ids' and 'labels'.
    
    Args:
        pad_id (int): The padding token id.
        max_seqlen (int): The maximum sequence length.
        alignment (int): The alignment for packing.
    
    """
    def __init__(
        self,
        pad_id: Union[int, Dict[str, int]],
        max_seqlen: int,
        alignment: int
    ):
        if isinstance(pad_id, int):
            self._pad_id = {"input_ids": pad_id}
        else:
            self._pad_id = pad_id
        self._max_seqlen = max_seqlen
        self._alignment = alignment

        self._batch = defaultdict(list)
        self._cu_seqlens_list = []
        self._padded_batch = None
        self._padded_cu_seqlens_list = None
        self._packed_batch = None
        self._packed_cu_seqlens_list = None
        self._cp_packed_batch = None
        self._cp_packed_cu_seqlens_list = None

    def add_data(self, data: Union[np.ndarray, Dict[str, np.ndarray]], truncate_length: Optional[int] = None):
        """Add data to the bucket."""
        
        if isinstance(data, dict):
            input_ids = data["input_ids"]
        else:
            input_ids = data
        seq_length = np.sum(input_ids != self._pad_id["input_ids"])
        if truncate_length is not None:
            if truncate_length > seq_length:
                logging.warning(
                    f"Adding data to bucket with truncate_length {truncate_length}, "
                    f"which is larger than sequence length {seq_length}. It will be truncated to {seq_length}."
                )
                truncate_length = seq_length
        else:
            truncate_length = seq_length

        if isinstance(data, dict):
            for key in data.keys():
                self._batch[key].append(data[key][:truncate_length])
        else:
            self._batch["input_ids"].append(data[:truncate_length])

        self._cu_seqlens_list.append(np.array([0, truncate_length], dtype=np.int32))
        
    def pad_data(self):
        """Pad data in the bucket to max seqlen."""

        padded_batch = defaultdict(list)
        padded_cu_seqlens_list = []
        
        for idx, data in enumerate(self._batch["input_ids"]):
            pad_seqlen = self._max_seqlen - len(data)
            padded_cu_seqlens_list.append(self._max_seqlen)
            for key in self._batch.keys():
                if pad_seqlen > 0:
                    padded_batch[key].append(np.concatenate([self._batch[key][idx], np.array([self._pad_id[key]] * pad_seqlen)]))
                else:
                    padded_batch[key].append(self._batch[key][idx][: self._max_seqlen])

        self._padded_batch = padded_batch
        self._padded_cu_seqlens_list = padded_cu_seqlens_list

    # 已经默认batch中的数据按照从短到长排序
    def pack_data(
        self,
        batching_option_matrix: Optional[np.ndarray] = None,
        static_shape: bool = False,
        sorted: bool = True,
    ):
        """Pack data in the bucket to micro batches.
        
        There are two packing strategies: workload balance and greedy.
        The workload balance strategy is controlled by the batching_option_matrix.
        The greedy strategy is controlled by the static_shape and sorted.
        

        Args:
            batching_option_matrix (np.ndarray, optional):
                Workload balance seq-batch mapping matrix.
                batching_option_matrix[i][j] = 1 means the i-th seq is put into the j-th micro batch.
                Defaults to None.
            static_shape (bool, optional):
                Whether use static shape (max seqlen) for all micro batches.
                If True, padding to align the sequence length to alignment.
                Defaults to False.
            sorted (bool, optional):
                Whether sort the packed data.
                If True, sort by the number of non-padding tokens in descending order.
                Defaults to True.
        """
        packed_batch = defaultdict(list)
        packed_cu_seqlens_list = []
        # 负载均衡的packing策略
        if batching_option_matrix is not None:
            assert len(batching_option_matrix.shape) == 2, f"{batching_option_matrix} is not a 2 dim matrix"
            for micro_batch_id in range(batching_option_matrix.shape[1]):
                packed_seqs = defaultdict(list)
                cur_cu_seqlen = 0
                cu_seqlens = [0]
                for seq_id in range(batching_option_matrix.shape[0]):
                    if batching_option_matrix[seq_id][micro_batch_id]:
                        for key in self._batch.keys():
                            packed_seqs[key].append(self._batch[key][seq_id])
                        cur_cu_seqlen += len(self._batch["input_ids"][seq_id])
                        cu_seqlens.append(cur_cu_seqlen)
                # pad to the nearest number that the sequence parallel degree can divide evenly
                if cur_cu_seqlen % self._alignment != 0:
                    pad_seqlen = self._alignment - (cur_cu_seqlen % self._alignment) 
                    for key in self._batch.keys():
                        packed_seqs[key].append(np.array([self._pad_id[key]] * pad_seqlen))
                    # padding tokens也加入cu seqlens中
                    cur_cu_seqlen += pad_seqlen
                    cu_seqlens.append(cur_cu_seqlen)
                for key in self._batch.keys():
                    packed_batch[key].append(np.concatenate(packed_seqs[key]))
                packed_cu_seqlens_list.append(np.array(cu_seqlens, dtype=np.int32))   
        # 简单的贪心packing策略
        else:
            is_visited = set()
            input_ids = self._batch["input_ids"]
            for i in range(len(input_ids)):
                if i in is_visited:
                    continue
                packed_seqs = defaultdict(list)
                for key in self._batch.keys():
                    packed_seqs[key] = [self._batch[key][i]]
                cur_cu_seqlen = len(input_ids[i])
                cu_seqlens = [0, cur_cu_seqlen]
                is_visited.add(i)
                for j in reversed(range(i + 1, len(input_ids))):
                    if j not in is_visited and cur_cu_seqlen + len(input_ids[j]) <= self._max_seqlen:
                        for key in self._batch.keys():
                            packed_seqs[key].append(self._batch[key][j])
                        cur_cu_seqlen += len(input_ids[j])
                        cu_seqlens.append(cur_cu_seqlen)
                        is_visited.add(j)
                # already support multi shape micro batch
                if static_shape:
                    # pad to max_seqlen
                    if cur_cu_seqlen < self._max_seqlen:
                        for key in self._batch.keys():
                            packed_seqs[key].append(np.array([self._pad_id[key]] * (self._max_seqlen - cur_cu_seqlen)))
                        cu_seqlens.append(self._max_seqlen)
                else:
                    # pad to the nearest number that the sequence parallel degree can divide evenly
                    if cur_cu_seqlen % self._alignment != 0:
                        pad_seqlen = self._alignment - (cur_cu_seqlen % self._alignment) 
                        for key in self._batch.keys():
                            packed_seqs[key].append(np.array([self._pad_id[key]] * pad_seqlen))
                        cu_seqlens.append(cur_cu_seqlen + pad_seqlen)
                        cur_cu_seqlen += pad_seqlen
                for key in self._batch.keys():
                    packed_batch[key].append(np.concatenate(packed_seqs[key]))
                packed_cu_seqlens_list.append(np.array(cu_seqlens, dtype=np.int32))
        assert packed_batch["input_ids"], "No data in the bucket, please check the input data."
        if sorted:
            input_ids = packed_batch["input_ids"]
            non_pad_counts = [np.sum(batch != self._pad_id) for batch in input_ids]
            sorted_indices = np.argsort(non_pad_counts)[::-1]  # 从大到小排序
            for key in packed_batch.keys():
                packed_batch[key] = [packed_batch[key][i] for i in sorted_indices]
            packed_cu_seqlens_list = [packed_cu_seqlens_list[i] for i in sorted_indices]
        
        self._packed_batch = packed_batch
        self._packed_cu_seqlens_list = packed_cu_seqlens_list
    
    def generate_cp_pad_data(self):
        # TODO: Implement this function
        raise NotImplementedError
    
    def generate_cp_pack_data(
        self,
        cp_size: int,
        cp_lens_rate: Optional[List[float]] = None
    ):
        """Generate CP packed data for context parallel.
        
        Args:
            cp_size (int): The number of CP ranks.
            cp_lens_rate (List[float], optional): The rate of sequence length for each CP rank.
                Defaults to None.
        """
        assert self._packed_cu_seqlens_list is not None, (
            "please ensure you have packed the bucket by `pack_data()`."
        )
        assert self._cp_packed_cu_seqlens_list is None, (
            "`generate_cp_pack_data()` can only call once"
        )

        # 目前只写了SYM情形下的packing
        split_pattern = os.environ.get('HETU_PARALLEL_ATTN_SPLIT_PATTERN')
        assert split_pattern == "SYM", (
            f"Unsupported HETU_PARALLEL_ATTN_SPLIT_PATTERN value: {split_pattern}. "
            f"Only 'SYM' is supported when using CP + packing."
        )
        if cp_lens_rate is None:
            cp_lens_rate = [1 / cp_size] * cp_size
        else:
            assert len(cp_lens_rate) == cp_size and sum(cp_lens_rate) == 1, (
                f"{cp_lens_rate} is invalid because the sum of cp_lens_rate "
                f"should be 1 and the length should be equal to cp_size."
            )
        # cp_packed_batch是一个每个cp_rank到其各自的多个micro_batch（即packed sequence）的映射
        # cp_packed_cu_seqlens_list则是多个micro_batch的cu_seqlens
        # 注意这里增加cp后cu_seqlens的shape从原先的[packing_num,]变成[cp, packing_num]
        # 每个cp_rank上是各自local qkv的[packing_num,]的cu_seqlens
        # 这么设计是为了让C++端的parallel attn算子比较方便实现   
        self._cp_packed_batch = {cp_idx: defaultdict(list) for cp_idx in range(cp_size)}
        self._cp_packed_cu_seqlens_list = [] 
        for packed_seq_idx in range(len(self._packed_cu_seqlens_list)):
            packed_cu_seqlens = self._packed_cu_seqlens_list[packed_seq_idx]
            cp_packed_cu_seqlens = np.zeros((cp_size, len(packed_cu_seqlens)), dtype=packed_cu_seqlens.dtype)
            cp_packed_seq = [defaultdict(list) for _ in range(cp_size)] # 每个cp_idx都要存一个 
            for original_seq_idx in range(len(packed_cu_seqlens) - 1):
                original_seq = {
                    key: packed_seq_of_key[packed_seq_idx][
                        packed_cu_seqlens[original_seq_idx]: packed_cu_seqlens[original_seq_idx + 1]
                    ]
                    for key, packed_seq_of_key in self._packed_batch.items()
                }
                original_seqlen = len(original_seq["input_ids"])
                begin_idx = 0
                end_idx = original_seqlen
                for cp_idx in range(cp_size):
                    # TODO: 考虑没法整除的情形
                    cp_seqlen = int(original_seqlen * cp_lens_rate[cp_idx])
                    cp_packed_cu_seqlens[cp_idx, original_seq_idx + 1] = cp_packed_cu_seqlens[cp_idx, original_seq_idx] + cp_seqlen
                    # SYM对半切要将其再分割成两部分
                    for key in self._packed_batch.keys():
                        first_half = original_seq[key][begin_idx:begin_idx + int(cp_seqlen / 2)]
                        second_half = original_seq[key][end_idx - (cp_seqlen - int(cp_seqlen / 2)):end_idx]
                        cp_packed_seq[cp_idx][key].append(first_half)
                        cp_packed_seq[cp_idx][key].append(second_half)
                    
                    begin_idx = begin_idx + int(cp_seqlen / 2)
                    end_idx = end_idx - (cp_seqlen - int(cp_seqlen / 2))
                    assert begin_idx <= end_idx, "something wrong"
            # TODO: alignment对齐
            for cp_idx in range(cp_size):
                cp_packed_seqlen = cp_packed_cu_seqlens[cp_idx, -1]
                for key in self._packed_batch.keys():
                    cp_packed_seq_numpy = np.concatenate(cp_packed_seq[cp_idx][key])
                    assert len(cp_packed_seq_numpy) == cp_packed_seqlen, \
                        f"length mismatches for {key}: {len(cp_packed_seq_numpy)} vs {cp_packed_seqlen}"
                    self._cp_packed_batch[cp_idx][key].append(cp_packed_seq_numpy)
            self._cp_packed_cu_seqlens_list.append(cp_packed_cu_seqlens)

    def packed_batch_size(self):
        """Return the size of packed batch."""
        
        assert self._packed_batch is not None, "please ensure you have packed the bucket by `pack_data()`."
        return len(self._packed_batch["input_ids"])
    
    def padded_batch_size(self):
        """Return the size of padded batch."""
        
        assert self._padded_batch is not None, "please ensure you have padded the bucket by `pad_data()`."
        return len(self._padded_batch["input_ids"])

    def original_batch_size(self):
        """Return the size of original batch."""
        
        return len(self._batch["input_ids"])
    
    def padded_batch(self):
        """Return the padded batch."""
        
        assert self._padded_batch is not None, "please ensure you have padded the bucket by `pad_data()`."
        return self._padded_batch
    
    def padded_cu_seqlens_list(self):
        """Return the padded cu seqlens list."""
        
        assert self._padded_cu_seqlens_list is not None, "please ensure you have padded the bucket by `pad_data()`."
        return self._padded_cu_seqlens_list 

    def packed_batch(self):
        """Return the packed batch."""
        
        assert self._packed_batch is not None, "please ensure you have packed the bucket by `pack_data()`."
        return self._packed_batch
    
    def packed_cu_seqlens_list(self):
        """Return the packed cu seqlens list."""
        
        assert self._packed_cu_seqlens_list is not None, "please ensure you have packed the bucket by `pack_data()`."
        return self._packed_cu_seqlens_list 
    
    def cp_packed_batch(self, cp_idx):
        """Return the CP packed batch of the specified CP rank `cp_idx`."""
        
        assert self._cp_packed_batch is not None, \
            "please ensure you have packed the bucket and generate the cp packed data by `generate_cp_pack_data()`."
        # TODO: check it
        # assert cp_idx in self._cp_packed_batch, f"CP rank {cp_idx} out of index"
        return self._cp_packed_batch[cp_idx]
    
    def cp_packed_cu_seqlens_list(self):
        """Return the CP packed cu seqlens list."""
        
        assert self._cp_packed_cu_seqlens_list is not None, \
            "please ensure you have packed the bucket and generate the cp packed data by `generate_cp_pack_data()`."
        return self._cp_packed_cu_seqlens_list 
    
    def cp_packed_seqlen_list(self):
        """Return the dict of CP packed seqlen list for each CP rank."""
        
        assert self._cp_packed_batch is not None, \
            "please ensure you have packed the bucket and generate the cp packed data by `generate_cp_pack_data()`."
        return {cp_idx: [len(packed_seq) for packed_seq in packed_batch["input_ids"]] for cp_idx, packed_batch in self._cp_packed_batch.items()}

def get_sorted_batch_and_len(global_batch: Union[np.ndarray, Dict[str, np.ndarray]], pad_id: int):
    """Sort the global batch by the number of non-padding tokens in ascending order.

    Args:
        global_batch (Union[np.ndarray, Dict[str, np.ndarray]]): global batch.
        pad_id (int): pad token id of the tokenizer.

    Returns:
        Tuple of sorted global batch and related valid seq lengths.
    """
    
    if isinstance(global_batch, dict):
        input_ids = global_batch["input_ids"]
        other_fields = {key: global_batch[key] for key in global_batch if key != "input_ids"}
        non_pad_counts = np.sum(input_ids != pad_id, axis=1)
        sorted_indices = np.argsort(non_pad_counts)
        sorted_input_ids = input_ids[sorted_indices]
        sorted_valid_tokens = non_pad_counts[sorted_indices]
        sorted_other_fields = {key: other_fields[key][sorted_indices] for key in other_fields}
        sorted_global_batch = {"input_ids": sorted_input_ids, **sorted_other_fields}
    else:
        non_pad_counts = np.sum(global_batch != pad_id, axis=1)
        sorted_indices = np.argsort(non_pad_counts)
        sorted_global_batch = global_batch[sorted_indices]
        sorted_valid_tokens = non_pad_counts[sorted_indices]
    return sorted_global_batch, sorted_valid_tokens

# 从global_batch中取出batch_indices的seqs并进行截断后分别构成两个buckets
# alignment来保证之后seqlen会padding到alignment的倍数
# valid_alignment来控制原始seqlen是alignment的倍数
# 目前只有CP时候为了整除会用到valid_alignment
def get_input_and_label_buckets(
    global_batch: Union[np.ndarray, Dict[str, np.ndarray]],
    pad_id: Union[int, Dict[str, int]],
    batch_indices: List[int],
    max_seqlen: int,
    alignment: int = 128,
    valid_alignment: int = 1,
):
    """Get input and label buckets from the global batch.
    
    Args:
        global_batch (Union[np.ndarray, Dict[str, np.ndarray]]): global batch.
        pad_id (Union[int, Dict[str, int]]): pad token id of the tokenizer.
        batch_indices (List[int]): batch indices.
        max_seqlen (int): max sequence length.
        alignment (int, optional): alignment for packing. Defaults to 128.
        valid_alignment (int, optional): valid alignment for packing. Defaults to 1.
    
    Returns:
        A bucket with both input_ids and labels.
    """
    if isinstance(pad_id, int):
        pad_id = {"input_ids": pad_id, "labels": IGNORE_INDEX}

    if isinstance(global_batch, dict):
        # global_batch has `input_ids` and `labels` fields, we need to put them into two buckets
        bucket = Bucket(pad_id, max_seqlen, alignment)
        bucket_valid_tokens = np.sum(global_batch["input_ids"][batch_indices] != pad_id["input_ids"], axis=1)
        for idx, valid_tokens in enumerate(bucket_valid_tokens):
            aligned_valid_tokens = (valid_tokens - 1) // valid_alignment * valid_alignment
            cur_seq = {key: global_batch[key][batch_indices][idx] for key in global_batch}
            bucket.add_data(cur_seq, aligned_valid_tokens)
    else:
        bucket = Bucket(pad_id, max_seqlen, alignment)
        bucket_valid_tokens = np.sum(global_batch["input_ids"][batch_indices] != pad_id["input_ids"], axis=1)
        for idx, valid_tokens in enumerate(bucket_valid_tokens):
            aligned_valid_tokens = (valid_tokens - 1) // valid_alignment * valid_alignment
            cur_seq = {
                "input_ids": global_batch[batch_indices][idx],
                "labels": global_batch[batch_indices][idx][1:],
            }
            bucket.add_data(cur_seq, aligned_valid_tokens)
    return bucket
