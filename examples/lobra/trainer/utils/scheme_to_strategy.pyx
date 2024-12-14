import cython
from libcpp.vector cimport vector

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def combine_scheme_to_strategy_candidates(int gpu_num, int strategy_pool_size, int[:] strategy_pool):
    cdef int i, j, strategy_gpu
    cdef vector[int] dp = vector[int](gpu_num + 1, 0)
    dp[0] = 1
    cdef vector[vector[vector[int]]] paths = vector[vector[vector[int]]](gpu_num + 1)
    paths[0].push_back(vector[int]())

    for i in range(strategy_pool_size):
        strategy_gpu = strategy_pool[i]
        for j in range(strategy_gpu, gpu_num + 1):
            if dp[j - strategy_gpu] > 0:
                dp[j] += dp[j - strategy_gpu]
                update_paths(paths, j, j - strategy_gpu, i)

    cdef list py_paths = []
    for path in paths[gpu_num]:
        py_paths.append([p for p in path])

    return dp[gpu_num], py_paths

cdef void update_paths(vector[vector[vector[int]]] &paths, int target, int source, int strategy_index):
    cdef vector[int] new_path
    for path in paths[source]:
        new_path = vector[int](path)
        new_path.push_back(strategy_index)
        paths[target].push_back(new_path)
