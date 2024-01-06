#pragma once

#include "hetu/impl/utils/ndarray_utils.h"
#include "hetu/impl/utils/dispatch.h"

namespace hetu {
namespace impl {

template<typename spec_t, int vec_size>
struct alignas(sizeof(spec_t) * vec_size) aligned_vector {
  spec_t val[vec_size];
};

inline int GetThreadNum(int cnt) {
  if (cnt >= 1048576)
    return 1024;
  if (cnt >= 262144)
    return 512;
  if (cnt >= 65536)
    return 256;
  if (cnt >= 16384)
    return 128;
  if (cnt >= 256)
    return 64;
  return 32;
}

inline int64_t get_index(int64_t idx, int64_t ndims, const int64_t* stride, const int64_t* c_shape) {
  int64_t i_idx = 0;
  int64_t t = idx;
  for (int i = ndims - 1; i >= 0; --i) {
    int64_t ratio = t % c_shape[i];
    t /= c_shape[i];
    i_idx += ratio * stride[i];
  }
  return i_idx;
}

} // namespace impl
} // namespace hetu
