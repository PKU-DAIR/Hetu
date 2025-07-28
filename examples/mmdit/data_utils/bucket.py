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
        self._packed_batch = None
        self._packed_cu_seqlens_list = None
        self._padded_batch = None
        self._padded_cu_seqlens_list = None

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
            padded_cu_seqlens_list.append(self._cu_seqlens_list[i])
        self._padded_batch = padded_batch
        self._padded_cu_seqlens_list = padded_cu_seqlens_list

    # 已经默认batch中的数据按照从短到长排序
    def pack_data(self, batching_option_matrix, static_shape: bool):
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
                else:
                    # pad to the nearest number that the sequence parallel degree can divide evenly
                    if cur_cu_seqlen % self._alignment != 0:
                        pad_seqlen = self._alignment - (cur_cu_seqlen % self._alignment) 
                        packed_seqs.append(np.array([self._pad_token] * pad_seqlen))
                        # cu_seqlens[-1] += pad_seqlen
                packed_batch.append(np.concatenate(packed_seqs))
                packed_cu_seqlens_list.append(np.array(cu_seqlens, dtype=np.int32))
        assert len(packed_batch) > 0, "currently not support no data after packing"
        self._packed_batch = packed_batch
        self._packed_cu_seqlens_list = packed_cu_seqlens_list

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
def get_input_and_label_buckets(global_batch: np.ndarray, pad_token: int, batch_indices: List[int], max_seqlen: int, alignment: int):
    bucket_batch = global_batch[batch_indices]
    bucket_valid_tokens = np.sum(bucket_batch != pad_token, axis=1)
    input_bucket = Bucket(pad_token, max_seqlen, alignment)
    label_bucket = Bucket(pad_token, max_seqlen, alignment)
    for seq, vailid_tokens in zip(bucket_batch, bucket_valid_tokens):
        input_bucket.add_data(seq, vailid_tokens - 1)
        label_bucket.add_data(seq[1:], vailid_tokens - 1)
    return input_bucket, label_bucket

if __name__ == '__main__':
    pass