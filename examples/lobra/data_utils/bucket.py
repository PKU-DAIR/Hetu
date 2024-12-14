import numpy as np
from typing import List

class Bucket:
    # 这里的max_seqlen表示的是显存支持的最大值
    def __init__(self, pad_token: int, max_seqlen: int, min_seqlen: int, alignment: int):
        self._pad_token = pad_token
        self._max_seqlen = max_seqlen
        self._min_seqlen = min_seqlen
        self._alignment = alignment
        self._batch = []
        self._cu_seqlens_list = []
        self._task_indices = []
        self._packed_batch = None
        self._packed_cu_seqlens_list = None
        self._padded_batch = None
        self._padded_cu_seqlens_list = None
        self._task_seq_lens_list = None
    
    def _align_pad_len(self, valid_seq_len, alignment):
        return (valid_seq_len + alignment - 1) // alignment * alignment - valid_seq_len

    def add_data(self, padded_sequence, valid_tokens, task_indice=0):
        if valid_tokens > self._min_seqlen and valid_tokens <= self._max_seqlen:
            self._batch.append(padded_sequence[:valid_tokens])
            self._cu_seqlens_list.append(np.array([0, valid_tokens], dtype=np.int32))
            self._task_indices.append(task_indice)
            return True
        else:
            return False
        
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
    def pack_data(self, batching_option_matrix, tp, static_shape: bool, train_task_num=1, fuse_multi_task=False):
        packed_batch = []
        packed_cu_seqlens_list = []
        task_seq_lens_list = []
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
                cur_task_indices = self._task_indices[i]
                seq_lens = [len(self._batch[i])]
                task_indices = [self._task_indices[i]]
                task_seq_lens_dict = {self._task_indices[i]: len(self._batch[i])}
                task_pad_lens_dict = {self._task_indices[i]: self._align_pad_len(len(self._batch[i]), tp)}
                is_visited.add(i)
                for j in reversed(range(i + 1, len(self._batch))):
                    # cur_cu_seqlen + 所有task在task_pad_lens_dict中的pad长度 - 当前task的pad长度 + 当前task的seq长度 + 加上当前task seq后的pad长度 <= max_seqlen
                    if fuse_multi_task and j not in is_visited and \
                            cur_cu_seqlen + sum(task_pad_lens_dict.values()) - task_pad_lens_dict.get(self._task_indices[j], 0) + \
                            len(self._batch[j]) + self._align_pad_len(task_pad_lens_dict.get(self._task_indices[j], 0) + \
                            len(self._batch[j]), tp) <= self._max_seqlen:
                        packed_seqs.append(self._batch[j])
                        cur_cu_seqlen += len(self._batch[j])
                        # cu_seqlens.append(cur_cu_seqlen)
                        seq_lens.append(len(self._batch[j]))
                        task_seq_lens_dict[self._task_indices[j]] = task_seq_lens_dict.get(self._task_indices[j], 0) + len(self._batch[j])
                        task_pad_lens_dict[self._task_indices[j]] = self._align_pad_len(task_seq_lens_dict[self._task_indices[j]], tp)
                        task_indices.append(self._task_indices[j])
                        is_visited.add(j)
                    elif not fuse_multi_task and j not in is_visited and cur_task_indices == self._task_indices[j] and cur_cu_seqlen + len(self._batch[j]) <= self._max_seqlen:
                        packed_seqs.append(self._batch[j])
                        cur_cu_seqlen += len(self._batch[j])
                        # cu_seqlens.append(cur_cu_seqlen)
                        seq_lens.append(len(self._batch[j]))
                        task_seq_lens_dict[self._task_indices[j]] = task_seq_lens_dict.get(self._task_indices[j], 0) + len(self._batch[j])
                        task_indices.append(self._task_indices[j])
                        is_visited.add(j)
                sorted_seq_len_indices = np.lexsort((-np.array(seq_lens), task_indices))
                if fuse_multi_task:
                    # 在每个task的最后一条seq后面加上pad，重新构造sorted_seq_lens和sorted_packed_seqs
                    sorted_seq_lens = []
                    sorted_packed_seqs = []
                    for i in range(len(sorted_seq_len_indices)):
                        sorted_packed_seqs.append(packed_seqs[sorted_seq_len_indices[i]])
                        if i < len(sorted_seq_len_indices) - 1 and task_indices[sorted_seq_len_indices[i]] != task_indices[sorted_seq_len_indices[i + 1]]:
                            sorted_packed_seqs.append(np.array([self._pad_token] * task_pad_lens_dict[task_indices[sorted_seq_len_indices[i]]]))
                            sorted_seq_lens.append(seq_lens[sorted_seq_len_indices[i]] + task_pad_lens_dict[task_indices[sorted_seq_len_indices[i]]])
                            cur_cu_seqlen += task_pad_lens_dict[task_indices[sorted_seq_len_indices[i]]]
                            task_seq_lens_dict[task_indices[sorted_seq_len_indices[i]]] += task_pad_lens_dict[task_indices[sorted_seq_len_indices[i]]]
                        else:
                            sorted_seq_lens.append(seq_lens[sorted_seq_len_indices[i]])
                else:
                    sorted_seq_lens = [seq_lens[i] for i in sorted_seq_len_indices]
                    sorted_packed_seqs = [packed_seqs[i] for i in sorted_seq_len_indices]
                # 根据sorted_seq_lens重新计算cu_seqlens
                cu_seqlens = [0]
                for seq_len in sorted_seq_lens:
                    cu_seqlens.append(cu_seqlens[-1] + seq_len)
                # already support multi shape micro batch
                if static_shape:
                    # pad to max_seqlen
                    if cur_cu_seqlen < self._max_seqlen:
                        sorted_packed_seqs.append(np.array([self._pad_token] * (self._max_seqlen - cur_cu_seqlen)))
                        # 给task_seq_lens中task_id最大的task添加pad
                        task_seq_lens_dict[max(task_seq_lens_dict.keys())] += self._max_seqlen - cur_cu_seqlen
                else:
                    # pad to the nearest number that the sequence parallel degree can divide evenly
                    if cur_cu_seqlen % self._alignment != 0:
                        pad_seqlen = self._alignment - (cur_cu_seqlen % self._alignment) 
                        sorted_packed_seqs.append(np.array([self._pad_token] * pad_seqlen))
                        # cu_seqlens[-1] += pad_seqlen
                task_seq_lens = [task_seq_lens_dict.get(task_id, 0) for task_id in range(train_task_num)]
                # print(f"task_seq_lens = {task_seq_lens}")
                packed_batch.append(np.concatenate(sorted_packed_seqs))
                packed_cu_seqlens_list.append(np.array(cu_seqlens, dtype=np.int32))
                task_seq_lens_list.append(task_seq_lens)
        assert len(packed_batch) > 0, "currently not support no data after packing"
        self._packed_batch = packed_batch
        self._packed_cu_seqlens_list = packed_cu_seqlens_list
        self._task_seq_lens_list = task_seq_lens_list
        # print(f"packed_cu_seqlens_list = {packed_cu_seqlens_list}")
        # print(f"task_seq_lens_list = {task_seq_lens_list}")

    def make_dummy_batch(self):
        # actually do nothing when exists empty bucket
        self._max_seq_len = 128
        self._packed_batch = [np.array([self._pad_token] * self._max_seq_len)] * 8
        self._cu_seqlens_list = [np.array([0], dtype=np.int32)] * 8

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

    def packed_task_seqlens_list(self):
        assert self._task_seq_lens_list != None, "please ensure you have packed the bucket"
        return self._task_seq_lens_list

    def max_seq_len(self):
        return self._max_seqlen
    
    def min_seq_len(self):
        return self._min_seqlen
 
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

# 从global_batch中取出batch_indices的seqs后分别构成两个buckets
def get_input_and_label_buckets(global_batch: np.ndarray, pad_token: int, bucket_sizes: List[int], alignment: int = 128):
    non_pad_counts = np.sum(global_batch != pad_token, axis=1)
    # print(f'non_pad_counts = {non_pad_counts}, sum = {np.sum(non_pad_counts)}')
    sorted_indices = np.argsort(-non_pad_counts)
    sorted_global_batch = global_batch[sorted_indices]
    sorted_valid_tokens = non_pad_counts[sorted_indices]

    buckets = []
    assert len(bucket_sizes) >= 2, 'len(bucket size) must >= 2'
    left, right = 0, 1
    input_bucket = Bucket(pad_token, bucket_sizes[left], bucket_sizes[right], alignment)
    label_bucket = Bucket(pad_token, bucket_sizes[left], bucket_sizes[right], alignment)
    buckets.append((input_bucket, label_bucket))
    
    for seq, valid_tokens in zip(sorted_global_batch, sorted_valid_tokens):
        is_add = input_bucket.add_data(seq, valid_tokens - 1) \
            and label_bucket.add_data(seq[1:], valid_tokens - 1)
        while not is_add and right < len(bucket_sizes) - 1:
            left += 1
            right += 1
            input_bucket = Bucket(pad_token, bucket_sizes[left], bucket_sizes[right], alignment)
            label_bucket = Bucket(pad_token, bucket_sizes[left], bucket_sizes[right], alignment)
            buckets.append((input_bucket, label_bucket))
            is_add = input_bucket.add_data(seq, valid_tokens - 1) \
                and label_bucket.add_data(seq[1:], valid_tokens - 1)

    return buckets

if __name__ == '__main__':
    pass