import cython
import numpy as np
from libc.math cimport INFINITY
from cython.parallel import prange
cimport numpy as np
import time

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def get_buckets_dp(np.ndarray[np.int32_t, ndim=1] global_batch_seqlen_list, 
                   np.ndarray[np.int32_t, ndim=1] bucket_candidates, 
                   int bucket_limit):
    cdef double s_time = time.time()
    cdef int i, j, k, bucket_num
    cdef long cur_bucket_size
    cdef double pad_num
    cdef int n = global_batch_seqlen_list.shape[0]
    cdef int bucket_limit_actual = min(bucket_limit, bucket_candidates.shape[0])

    cdef double[:,:] pad = np.zeros((bucket_limit_actual + 1, n + 1), dtype=np.float64)
    cdef int[:,:] path = np.zeros((bucket_limit_actual + 1, n + 1), dtype=np.int32)
    cdef int[:,:] bucket_trace = np.zeros((bucket_limit_actual + 1, n + 1), dtype=np.int32)
    cdef int[:] bucket_for_global_batch = np.zeros(n, dtype=np.int32)
    cdef long[:] prefix_sum_of_global_batch = np.cumsum(np.hstack(([0], global_batch_seqlen_list)))

    for i in range(n):
        bucket_for_global_batch[i] = bucket_candidates[np.searchsorted(bucket_candidates, global_batch_seqlen_list[i], side='left')]

    for i in range(1, n + 1):
        pad[0, i] = INFINITY
        bucket_trace[0, i] = -1

    for j in range(1, bucket_limit_actual + 1):
        pad[j, 0] = INFINITY

    for j in range(1, bucket_limit_actual + 1):
        with cython.nogil:
            for i in prange(1, n + 1):
                pad[j, i] = INFINITY
                bucket_trace[j, i] = -1
                for k in range(i):
                    if j == 1 and k > 0:
                        continue
                    if bucket_trace[j - 1, k] == -1:
                        continue

                    cur_bucket_size = bucket_for_global_batch[i - 1]
                    if cur_bucket_size <= bucket_trace[j - 1, k]:
                        break

                    pad_num = pad[j - 1, k] + (cur_bucket_size * (i - k) - (prefix_sum_of_global_batch[i] - prefix_sum_of_global_batch[k]))
                    if pad_num < pad[j, i]:
                        pad[j, i] = pad_num
                        path[j, i] = k
                        bucket_trace[j, i] = cur_bucket_size

    min_pad_num = INFINITY
    bucket_num = -1
    for j in range(bucket_limit_actual + 1):
        if pad[j, n] < min_pad_num:
            min_pad_num = pad[j, n]
            bucket_num = j

    min_pad_num = pad[bucket_num, n]

    bucket_indices = []
    i = n
    while i > 0:
        bucket_indices.append(int(bucket_trace[bucket_num, i]))
        i = int(path[bucket_num, i])
        bucket_num -= 1

    bucket_indices.sort()
    effective_sum = np.sum(global_batch_seqlen_list)
    e_time = time.time()

    print(f"dynamic programming time = {e_time - s_time:.3f}s")
    print(f"min_pad_num = {min_pad_num}, effective_sum = {effective_sum}, pad ratio = {min_pad_num / (min_pad_num + effective_sum)}")
    return bucket_indices

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def get_all_buckets_dp(np.ndarray[np.int32_t, ndim=1] global_batch_seqlen_list, 
                       np.ndarray[np.int32_t, ndim=1] bucket_candidates, 
                       int bucket_limit):
    cdef double s_time = time.time()
    cdef int i, j, k, bucket_num, cur_bucket_size
    cdef double pad_num
    cdef int n = global_batch_seqlen_list.shape[0]
    cdef int bucket_limit_actual = min(bucket_limit, bucket_candidates.shape[0])
    
    cdef double[:,:] pad = np.zeros((bucket_limit_actual + 1, n + 1), dtype=np.float64)
    cdef int[:,:] path = np.zeros((bucket_limit_actual + 1, n + 1), dtype=np.int32)
    cdef int[:,:] bucket_trace = np.zeros((bucket_limit_actual + 1, n + 1), dtype=np.int32)
    cdef int[:] bucket_for_global_batch = np.zeros(n, dtype=np.int32)
    cdef long[:] prefix_sum_of_global_batch = np.cumsum(np.hstack(([0], global_batch_seqlen_list)))

    for i in range(n):
        bucket_for_global_batch[i] = bucket_candidates[np.searchsorted(bucket_candidates, global_batch_seqlen_list[i], side='left')]

    for i in range(1, n + 1):
        pad[0, i] = INFINITY
        bucket_trace[0, i] = -1

    for j in range(1, bucket_limit_actual + 1):
        pad[j, 0] = INFINITY

    for j in range(1, bucket_limit_actual + 1):
        with cython.nogil:
            for i in prange(1, n + 1):
                pad[j, i] = INFINITY
                bucket_trace[j, i] = -1
                for k in range(i):
                    if j == 1 and k > 0:
                        continue
                    if bucket_trace[j - 1, k] == -1:
                        continue

                    cur_bucket_size = bucket_for_global_batch[i - 1]
                    if cur_bucket_size <= bucket_trace[j - 1, k]:
                        break

                    pad_num = pad[j - 1, k] + (cur_bucket_size * (i - k) - (prefix_sum_of_global_batch[i] - prefix_sum_of_global_batch[k]))
                    if pad_num < pad[j, i]:
                        pad[j, i] = pad_num
                        path[j, i] = k
                        bucket_trace[j, i] = cur_bucket_size
    bucket_num = -1

    bucket_indices_list = []
    for j in range(1, bucket_limit_actual + 1):
        i = n
        bucket_num = j
        bucket_indices = []
        while i > 0:
            bucket_indices.append(int(bucket_trace[bucket_num, i]))
            i = int(path[bucket_num, i])
            bucket_num -= 1
        bucket_indices.sort()
        bucket_indices_list.append(bucket_indices)

    e_time = time.time()

    effective_sum = np.sum(global_batch_seqlen_list)
    pad_ratio_list = [pad[j, n] / (pad[j, n] + effective_sum) for j in range(1, bucket_limit_actual + 1)]
    return bucket_indices_list, pad_ratio_list
