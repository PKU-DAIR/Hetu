import os
import numpy as np
from typing import List

class Bucket:
    # 这里的max_seqlen表示的是显存支持的最大值
    def __init__(self, pad_token: int, max_seqlen: int, alignment: int):
        self._pad_token = pad_token
        self._max_seqlen = max_seqlen
        self._alignment = alignment
        self._batch = []
        self._cu_seqlens_list = []
        self._padded_batch = None
        self._padded_cu_seqlens_list = None
        self._packed_batch = None
        self._packed_cu_seqlens_list = None
        self._cp_packed_batch = None
        self._cp_packed_cu_seqlens_list = None

    def add_data(self, padded_sequence, valid_tokens):
        self._batch.append(padded_sequence[:valid_tokens])
        self._cu_seqlens_list.append(np.array([0, valid_tokens], dtype=np.int32))
        
    def pad_data(self):
        padded_batch = []
        padded_cu_seqlens_list = []
        for i in range(len(self._batch)):
            pad_seqlen = self._max_seqlen - len(self._batch[i])
            if pad_seqlen > 0:
                padded_batch.append(np.concatenate([self._batch[i], np.array([self._pad_token] * pad_seqlen)]))
            else:
                padded_batch.append(self._batch[i])
            # padded_cu_seqlens_list.append(self._cu_seqlens_list[i])
            padded_cu_seqlens_list.append(self._max_seqlen) # 应该使用pad后的max_seqlen
        self._padded_batch = padded_batch
        self._padded_cu_seqlens_list = padded_cu_seqlens_list

    # 已经默认batch中的数据按照从短到长排序
    def pack_data(self, batching_option_matrix=None, static_shape=False, sorted=True):
        packed_batch = []
        packed_cu_seqlens_list = []
        # 负载均衡的packing策略
        # batching_option_matrix的第i行第j列表示是否将第i个seq放入第j个micro batch
        if isinstance(batching_option_matrix, np.ndarray):
            assert len(batching_option_matrix.shape) == 2, f"{batching_option_matrix} is not a 2 dim matrix"
            for micro_batch_id in range(batching_option_matrix.shape[1]):
                packed_seqs = []
                cur_cu_seqlen = 0
                cu_seqlens = [0]
                for seq_id in range(batching_option_matrix.shape[0]):
                    if batching_option_matrix[seq_id][micro_batch_id]:
                        packed_seqs.append(self._batch[seq_id])
                        cur_cu_seqlen += len(self._batch[seq_id])
                        cu_seqlens.append(cur_cu_seqlen)
                # pad to the nearest number that the sequence parallel degree can divide evenly
                if cur_cu_seqlen % self._alignment != 0:
                    pad_seqlen = self._alignment - (cur_cu_seqlen % self._alignment) 
                    packed_seqs.append(np.array([self._pad_token] * pad_seqlen))
                    # padding tokens也加入cu seqlens中
                    cur_cu_seqlen += pad_seqlen
                    cu_seqlens.append(cur_cu_seqlen)
                packed_batch.append(np.concatenate(packed_seqs))
                packed_cu_seqlens_list.append(np.array(cu_seqlens, dtype=np.int32))   
        # 简单的贪心packing策略
        else:
            # static shape开关表示是否支持每个micro batch的shape都动态
            is_visited = set()
            for i in range(len(self._batch)):
                if i in is_visited:
                    continue
                packed_seqs = [self._batch[i]]
                cur_cu_seqlen = len(self._batch[i])
                cu_seqlens = [0, cur_cu_seqlen]
                is_visited.add(i)
                for j in reversed(range(i + 1, len(self._batch))):
                    if j not in is_visited and cur_cu_seqlen + len(self._batch[j]) <= self._max_seqlen:
                        packed_seqs.append(self._batch[j])
                        cur_cu_seqlen += len(self._batch[j])
                        cu_seqlens.append(cur_cu_seqlen)
                        is_visited.add(j)
                # already support multi shape micro batch
                if static_shape:
                    # pad to max_seqlen
                    if cur_cu_seqlen < self._max_seqlen:
                        packed_seqs.append(np.array([self._pad_token] * (self._max_seqlen - cur_cu_seqlen)))
                        cu_seqlens.append(self._max_seqlen)
                else:
                    # pad to the nearest number that the sequence parallel degree can divide evenly
                    if cur_cu_seqlen % self._alignment != 0:
                        pad_seqlen = self._alignment - (cur_cu_seqlen % self._alignment) 
                        packed_seqs.append(np.array([self._pad_token] * pad_seqlen))
                        cu_seqlens.append(cur_cu_seqlen + pad_seqlen)
                packed_batch.append(np.concatenate(packed_seqs))
                packed_cu_seqlens_list.append(np.array(cu_seqlens, dtype=np.int32))
        assert len(packed_batch) > 0, "currently not support no data after packing"
        if sorted:
            non_pad_counts = [np.sum(batch != self._pad_token) for batch in packed_batch]
            sorted_indices = np.argsort(non_pad_counts)[::-1]  # 从大到小排序
            packed_batch = [packed_batch[i] for i in sorted_indices]
            packed_cu_seqlens_list = [packed_cu_seqlens_list[i] for i in sorted_indices]
        self._packed_batch = packed_batch
        self._packed_cu_seqlens_list = packed_cu_seqlens_list
    
    def generate_cp_pad_data(self):
        raise NotImplementedError
    
    def generate_cp_pack_data(self, cp_size, cp_lens_rate=None):
        assert self._packed_cu_seqlens_list != None, "please ensure you have packed the bucket"
        assert self._cp_packed_cu_seqlens_list == None, "generate_cp_pack_data() can only call once"
        # 目前只写了SYM情形下的packing
        split_pattern = os.environ.get('HETU_PARALLEL_ATTN_SPLIT_PATTERN')
        assert split_pattern == "SYM", f"Unsupported HETU_PARALLEL_ATTN_SPLIT_PATTERN value: {split_pattern}. Only 'SYM' is supported when using CP + packing."
        if cp_lens_rate == None:
            cp_lens_rate = [1 / cp_size] * cp_size
        else:
            assert len(cp_lens_rate) == cp_size and sum(cp_lens_rate) == 1, f"{cp_lens_rate} is invalid"
        # cp_packed_batch是一个每个cp_rank到其各自的多个micro_batch（即packed sequence）的映射
        # cp_packed_cu_seqlens_list则是多个micro_batch的cu_seqlens
        # 注意这里增加cp后cu_seqlens的shape从原先的[packing_num,]变成[cp, packing_num]
        # 每个cp_rank上是各自local qkv的[packing_num,]的cu_seqlens
        # 这么设计是为了让C++端的parallel attn算子比较方便实现   
        self._cp_packed_batch = {cp_idx: [] for cp_idx in range(cp_size)}
        self._cp_packed_cu_seqlens_list = [] 
        for packed_seq_idx in range(len(self._packed_cu_seqlens_list)):
            packed_seq = self._packed_batch[packed_seq_idx]
            packed_cu_seqlens = self._packed_cu_seqlens_list[packed_seq_idx]
            cp_packed_cu_seqlens = np.zeros((cp_size, len(packed_cu_seqlens)), dtype=packed_cu_seqlens.dtype)
            cp_packed_seq = [[] for _ in range(cp_size)] # 每个cp_idx都要存一个 
            for original_seq_idx in range(len(packed_cu_seqlens) - 1):
                original_seq = packed_seq[packed_cu_seqlens[original_seq_idx]: packed_cu_seqlens[original_seq_idx + 1]]
                original_seqlen = len(original_seq)
                begin_idx = 0
                end_idx = original_seqlen
                for cp_idx in range(cp_size):
                    # TODO: 考虑没法整除的情形
                    cp_seqlen = int(original_seqlen * cp_lens_rate[cp_idx])
                    cp_packed_cu_seqlens[cp_idx, original_seq_idx + 1] = cp_packed_cu_seqlens[cp_idx, original_seq_idx] + cp_seqlen
                    # SYM对半切要将其再分割成两部分
                    cp_packed_seq[cp_idx].append(original_seq[begin_idx: begin_idx + int(cp_seqlen / 2)])
                    cp_packed_seq[cp_idx].append(original_seq[end_idx - (cp_seqlen - int(cp_seqlen / 2)): end_idx])
                    begin_idx = begin_idx + int(cp_seqlen / 2)
                    end_idx = end_idx - (cp_seqlen - int(cp_seqlen / 2))
                    assert begin_idx <= end_idx, "something wrong"
            # TODO: alignment对齐
            for cp_idx in range(cp_size):
                cp_packed_seqlen = cp_packed_cu_seqlens[cp_idx, -1]
                cp_packed_seq_numpy = np.concatenate(cp_packed_seq[cp_idx])
                assert len(cp_packed_seq_numpy) == cp_packed_seqlen, "length mismatches"
                self._cp_packed_batch[cp_idx].append(cp_packed_seq_numpy)  
            self._cp_packed_cu_seqlens_list.append(cp_packed_cu_seqlens)

    def packed_batch_size(self):
        assert self._packed_batch != None, "please ensure you have packed the bucket"
        return len(self._packed_batch)
    
    def padded_batch_size(self):
        assert self._paded_batch != None, "please ensure you have padded the bucket"
        return len(self._padded_batch)

    def original_batch_size(self):
        return len(self._batch)
    
    def padded_batch(self):
        assert self._padded_batch != None, "please ensure you have padded the bucket"
        return self._padded_batch
    
    def padded_cu_seqlens_list(self):
        assert self._padded_cu_seqlens_list != None, "please ensure you have padded the bucket"
        return self._padded_cu_seqlens_list 

    def packed_batch(self):
        assert self._packed_batch != None, "please ensure you have packed the bucket"
        return self._packed_batch
    
    def packed_cu_seqlens_list(self):
        assert self._packed_cu_seqlens_list != None, "please ensure you have packed the bucket"
        return self._packed_cu_seqlens_list 
    
    def cp_packed_batch(self, cp_idx):
        assert self._cp_packed_batch != None, "please ensure you have packed the bucket and generate the cp packed data"
        assert cp_idx in self._cp_packed_batch, f"CP rank {cp_idx} out of index"
        return self._cp_packed_batch[cp_idx]
    
    def cp_packed_cu_seqlens_list(self):
        assert self._cp_packed_cu_seqlens_list != None, "please ensure you have packed the bucket and generate the cp packed data"
        return self._cp_packed_cu_seqlens_list 
    
    def cp_packed_seqlen_list(self):
        assert self._cp_packed_batch != None, "please ensure you have packed the bucket and generate the cp packed data"
        return {cp_idx: [len(packed_seq) for packed_seq in packed_batch] for cp_idx, packed_batch in self._cp_packed_batch.items()}
 
# 对global_batch中的seq按从小到大的顺序进行排序   
def get_sorted_batch_and_len(global_batch: np.ndarray, pad_token: int):
    # print(f"global_batch shape is {global_batch.shape}, pad_token is {pad_token}")
    non_pad_counts = np.sum(global_batch != pad_token, axis=1)
    sorted_indices = np.argsort(non_pad_counts)
    sorted_global_batch = global_batch[sorted_indices]
    sorted_valid_tokens = non_pad_counts[sorted_indices]
    return sorted_global_batch, sorted_valid_tokens

# 制造一个虚假的data
def build_fake_batch_and_len(fake_seqlens: List[int], pad_token: int):
    # 对fake_seqlens进行排序
    sorted_seqlens = sorted(fake_seqlens)
    # 获取最大长度
    max_len = sorted_seqlens[-1]
    # 初始化返回的ndarray
    result = np.full((len(sorted_seqlens), max_len), pad_token, dtype=int)
    # 填充0
    assert pad_token != 0, "please change a number to fill the fake batch"
    for i, seq_len in enumerate(sorted_seqlens):
        result[i, :seq_len] = 0
    return result, sorted_seqlens

# 从global_batch中取出batch_indices的seqs并进行截断后分别构成两个buckets
# alignment来保证之后seqlen会padding到alignment的倍数
# valid_alignment来控制原始seqlen是alignment的倍数
# 目前只有CP时候为了整除会用到valid_alignment
def get_input_and_label_buckets(global_batch: np.ndarray, pad_token: int, batch_indices: List[int], max_seqlen: int, alignment: int = 128, valid_alignment: int = 1):
    bucket_batch = global_batch[batch_indices]
    bucket_valid_tokens = np.sum(bucket_batch != pad_token, axis=1)
    input_bucket = Bucket(pad_token, max_seqlen, alignment)
    label_bucket = Bucket(pad_token, max_seqlen, alignment)
    for seq, vailid_tokens in zip(bucket_batch, bucket_valid_tokens):
        aligned_valid_tokens = (vailid_tokens - 1) // valid_alignment * valid_alignment
        input_bucket.add_data(seq, aligned_valid_tokens)
        label_bucket.add_data(seq[1:], aligned_valid_tokens)
    return input_bucket, label_bucket

if __name__ == '__main__':
    pass